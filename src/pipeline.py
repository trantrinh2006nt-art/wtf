"""
pipeline.py — VERPipeline: Bộ điều phối chính cho hệ thống Video Evidence Retrieval.

Kiến trúc 3 tầng:
  Tầng 1: Query Analyzer — phân rã query thành sub-queries đa phương thức
  Tầng 2: Multi-Index Retriever — truy xuất song song từ 5 chỉ mục
  Tầng 3: Track-based Processing:
    • Auto Track  (<2s): BGE-Reranker → kết quả nhanh
    • Semi Track  (<15s): LVLM Reranker + CRAG + Verifier → kết quả chính xác

GPU: RTX 4060 8GB → mỗi engine load → xử lý → unload tuần tự.
"""

import sys
import os
import json
import time
import glob
import torch
from pathlib import Path
from typing import Optional

# Đảm bảo import từ cùng thư mục src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, format_time,
    extract_frame, extract_frames_batch, extract_video_clips,
    get_video_duration, tokenize_vietnamese,
)


# ============================================================
# PIPELINE CHÍNH
# ============================================================

class VERPipeline:
    """
    Bộ điều phối chính cho hệ thống Video Evidence Retrieval.

    Hỗ trợ 2 track:
      - 'auto': Tự động hoàn toàn, tối ưu tốc độ (<2s)
      - 'semi': Bán tự động, tối ưu chất lượng (<15s), có CRAG correction
    """

    def __init__(self, track: str = "auto"):
        """
        Khởi tạo pipeline với tất cả engines cần thiết.

        Args:
            track: 'auto' (tự động, nhanh) hoặc 'semi' (bán tự động, chính xác)
        """
        self.track = track
        self._init_start = time.perf_counter()

        log_step(
            "VER Pipeline",
            f"═══ Khởi tạo hệ thống (track={track}) ═══"
        )

        # --- Tầng 1: Query Analyzer ---
        self.query_analyzer = self._init_query_analyzer()

        # --- Tầng 2: Multi-Index Retriever ---
        self.retriever = self._init_retriever()

        # --- Tầng 3: Track Processing ---
        self.auto_reranker = None
        self.lvlm_reranker = None
        self.crag_engine = None
        self.verifier = None

        if track == "auto":
            self.auto_reranker = self._init_auto_reranker()
        else:
            self.lvlm_reranker = self._init_lvlm_reranker()
            self.crag_engine = self._init_crag()
            self.verifier = self._init_verifier()

        # --- Module đánh giá ---
        self.evaluator = self._init_evaluator()

        # --- Thống kê latency ---
        self._latency_log: list[dict] = []

        init_time = time.perf_counter() - self._init_start
        log_success(f"Pipeline sẵn sàng trong {init_time:.1f}s")
        log_info(f"Track: {track} | Device: {DEVICE}")
        log_vram("sau khởi tạo")

    # ================================================================
    # KHỞI TẠO TỪNG ENGINE (graceful degradation)
    # ================================================================

    def _init_query_analyzer(self):
        """Khởi tạo Query Analyzer — Tầng 1."""
        try:
            # Thử import từ online/ trước
            from online.query_analyzer import QueryAnalyzer
            analyzer = QueryAnalyzer()
            log_success("QueryAnalyzer: ✓")
            return analyzer
        except ImportError:
            log_warning(
                "QueryAnalyzer chưa tồn tại (online/query_analyzer.py). "
                "Sẽ dùng fallback phân tích cơ bản."
            )
            return None
        except Exception as e:
            log_error(f"QueryAnalyzer: Lỗi khởi tạo — {e}")
            return None

    def _init_retriever(self):
        """Khởi tạo Multi-Index Retriever — Tầng 2."""
        try:
            from online.retriever import MultiIndexRetriever
            retriever = MultiIndexRetriever()
            log_success("MultiIndexRetriever: ✓")
            return retriever
        except ImportError:
            log_warning("MultiIndexRetriever chưa tồn tại. Sẽ bỏ qua bước retrieval.")
            return None
        except Exception as e:
            log_error(f"MultiIndexRetriever: Lỗi — {e}")
            return None

    def _init_auto_reranker(self):
        """Khởi tạo Auto Reranker — BGE-Reranker (nhanh, cho auto track)."""
        try:
            from online.auto_track import AutoTrack
            track = AutoTrack()
            log_success("AutoTrack: ✓")
            return track
        except ImportError:
            log_warning(
                "AutoTrack chưa tồn tại (online/auto_track.py). "
                "Auto track sẽ dùng score gốc."
            )
            return None
        except Exception as e:
            log_error(f"AutoTrack: Lỗi — {e}")
            return None

    def _init_lvlm_reranker(self):
        """Khởi tạo LVLM Reranker — Gemini Vision (cho semi track)."""
        try:
            from reranker import LVLMReranker
            reranker = LVLMReranker(api_key=GEMINI_API_KEY)
            log_success("LVLMReranker: ✓")
            return reranker
        except ImportError:
            log_warning("LVLMReranker chưa tồn tại.")
            return None
        except Exception as e:
            log_error(f"LVLMReranker: Lỗi — {e}")
            return None

    def _init_crag(self):
        """Khởi tạo CRAG Engine — Corrective RAG."""
        try:
            from crag import CRAGEngine
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            engine = CRAGEngine(client=client)
            log_success("CRAGEngine: ✓")
            return engine
        except ImportError:
            log_warning("CRAGEngine chưa tồn tại.")
            return None
        except Exception as e:
            log_error(f"CRAGEngine: Lỗi — {e}")
            return None

    def _init_verifier(self):
        """Khởi tạo Verifier Agent."""
        try:
            from verifier import VerifierAgent
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            verifier = VerifierAgent(client=client)
            log_success("VerifierAgent: ✓")
            return verifier
        except ImportError:
            log_warning("VerifierAgent chưa tồn tại.")
            return None
        except Exception as e:
            log_error(f"VerifierAgent: Lỗi — {e}")
            return None

    def _init_evaluator(self):
        """Khởi tạo Evaluation Engine."""
        try:
            from evaluation import EvaluationEngine
            evaluator = EvaluationEngine()
            log_success("EvaluationEngine: ✓")
            return evaluator
        except Exception as e:
            log_warning(f"EvaluationEngine: Lỗi — {e}")
            return None

    # ================================================================
    # TẦNG 1: QUERY ANALYSIS
    # ================================================================

    def _analyze_query(self, query: str) -> AnalyzedQuery:
        """
        Phân tích query thành các sub-queries đa phương thức.

        Nếu QueryAnalyzer không khả dụng, tạo AnalyzedQuery cơ bản
        với tất cả sub-queries = query gốc.
        """
        t0 = time.perf_counter()
        log_step("Tầng 1 — Query Analysis", f'"{query}"')

        if self.query_analyzer is not None:
            try:
                analyzed = self.query_analyzer.analyze(query)
                latency = time.perf_counter() - t0
                log_success(f"Phân tích xong trong {latency:.2f}s")
                log_info(f"Query type: {analyzed.query_type}")
                log_info(f"Entities: {analyzed.entities}")
                self._log_latency("query_analysis", latency)
                return analyzed
            except Exception as e:
                log_error(f"Lỗi phân tích query: {e}")

        # Fallback: dùng query gốc cho tất cả modalities
        analyzed = AnalyzedQuery(
            original_query=query,
            visual_query=query,
            temporal_query=query,
            audio_query=query,
            ocr_query=query,
            text_query=query,
            query_type="general",
        )
        latency = time.perf_counter() - t0
        self._log_latency("query_analysis", latency)
        log_info(f"Dùng fallback query analysis ({latency:.2f}s)")
        return analyzed

    # ================================================================
    # TẦNG 2: MULTI-INDEX RETRIEVAL
    # ================================================================

    def _retrieve(self, analyzed_query: AnalyzedQuery) -> list[SearchResult]:
        """
        Truy xuất candidates từ tất cả các indexes.

        Trả về Top TOP_K_RETRIEVAL (100) candidates đã fusion.
        """
        t0 = time.perf_counter()
        log_step("Tầng 2 — Multi-Index Retrieval", f"Top {TOP_K_RETRIEVAL} candidates")

        results = []
        try:
            if hasattr(self.retriever, 'retrieve'):
                # MultiIndexRetriever mới
                results = self.retriever.retrieve(analyzed_query, top_k=TOP_K_RETRIEVAL)
            else:
                log_warning("Không có retriever nào khả dụng")
        except Exception as e:
            log_error(f"Lỗi retrieval: {e}")

        latency = time.perf_counter() - t0
        self._log_latency("retrieval", latency)
        log_success(f"Truy xuất {len(results)} candidates trong {latency:.2f}s")
        return results

    # ================================================================
    # TẦNG 3: TRACK PROCESSING
    # ================================================================

    def _process_auto_track(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Auto Track: Xử lý nhanh (<2s).

        Pipeline: BGE-Reranker → Top-K → Done
        """
        t0 = time.perf_counter()
        log_step("Tầng 3 — Auto Track", f"Budget: {AUTO_TRACK_BUDGET_SEC}s")

        results = candidates
        if self.auto_reranker is not None:
            try:
                results = self.auto_reranker.process(query, candidates)
                log_success(f"Auto rerank: {len(results)} kết quả")
            except Exception as e:
                log_error(f"Auto reranker lỗi: {e}")
                # Fallback: giữ nguyên thứ tự candidates
                results = candidates
        else:
            # Không có auto reranker → sắp xếp theo score gốc
            results = sorted(candidates, key=lambda r: r.score, reverse=True)
            log_info("Không có AutoTrack, dùng score gốc")

        latency = time.perf_counter() - t0
        self._log_latency("auto_track", latency)

        if latency > AUTO_TRACK_BUDGET_SEC:
            log_warning(f"Auto track vượt budget: {latency:.2f}s > {AUTO_TRACK_BUDGET_SEC}s")

        return results

    def _process_semi_track(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Semi Track: Xử lý chính xác (<15s).

        Pipeline: LVLM Reranker → CRAG Evaluation → Verifier
        Nếu CRAG trả về AMBIGUOUS/INCORRECT → reformulate và re-search.
        """
        t0 = time.perf_counter()
        log_step("Tầng 3 — Semi Track", f"Budget: {SEMI_TRACK_BUDGET_SEC}s")

        results = candidates

        # --- Bước 3a: LVLM Reranking (Gemini Vision) ---
        if self.lvlm_reranker is not None:
            try:
                reranked = self.lvlm_reranker.rerank(
                    query=query,
                    candidates=results[:TOP_K_RERANK],
                    video_dir=str(RAW_VIDEO_DIR),
                )
                # Ghép kết quả đã rerank với phần còn lại
                remaining = results[TOP_K_RERANK:]
                results = reranked + remaining
                log_success(f"LVLM Rerank: top {len(reranked)} đã chấm điểm lại")
            except Exception as e:
                log_error(f"LVLM Reranker lỗi: {e}")

        # --- Bước 3b: CRAG Evaluation + Correction Loop ---
        if self.crag_engine is not None:
            try:
                verdict, score = self.crag_engine.evaluate_results(query, results[:TOP_K_FINAL])
                log_info(f"CRAG verdict: {verdict} (score={score:.4f})")

                if verdict != "CORRECT":
                    log_warning(f"CRAG: {verdict} — bắt đầu correction loop...")
                    results = self._crag_correction_loop(query, results, verdict)
            except Exception as e:
                log_error(f"CRAG lỗi: {e}")

        # --- Bước 3c: Verification ---
        if self.verifier is not None:
            try:
                results = self.verifier.verify_results(query, results[:TOP_K_FINAL])
                log_success(f"Verification hoàn tất: {len(results)} kết quả")
            except Exception as e:
                log_error(f"Verifier lỗi: {e}")

        latency = time.perf_counter() - t0
        self._log_latency("semi_track", latency)

        if latency > SEMI_TRACK_BUDGET_SEC:
            log_warning(f"Semi track vượt budget: {latency:.2f}s > {SEMI_TRACK_BUDGET_SEC}s")

        return results

    def _crag_correction_loop(
        self,
        original_query: str,
        current_results: list[SearchResult],
        initial_verdict: str,
    ) -> list[SearchResult]:
        """
        Vòng lặp tự sửa lỗi CRAG.

        Reformulate query → Re-search → Re-evaluate.
        Tối đa CRAG_MAX_RETRIES lần.
        """
        best_results = current_results
        best_score = 0.0

        for attempt in range(1, CRAG_MAX_RETRIES + 1):
            log_info(f"CRAG correction lần {attempt}/{CRAG_MAX_RETRIES}")

            try:
                # Reformulate query
                new_query = self.crag_engine.reformulate_query(
                    original_query=original_query,
                    failed_results=current_results[:5],
                    attempt=attempt,
                )
                log_info(f'Query mới: "{new_query}"')

                # Re-search
                analyzed = self._analyze_query(new_query)
                new_candidates = self._retrieve(analyzed)

                if not new_candidates:
                    log_warning("Re-search không có kết quả")
                    continue

                # Re-evaluate
                verdict, score = self.crag_engine.evaluate_results(
                    original_query, new_candidates[:TOP_K_FINAL]
                )
                log_info(f"Re-eval: {verdict} (score={score:.4f})")

                if score > best_score:
                    best_score = score
                    best_results = new_candidates

                if verdict == "CORRECT":
                    log_success(f"CRAG: CORRECT sau {attempt} lần correction")
                    return best_results

            except Exception as e:
                log_error(f"Lỗi correction loop #{attempt}: {e}")
                break

        log_info(f"Hết lượt correction, trả về kết quả tốt nhất (score={best_score:.4f})")
        return best_results

    # ================================================================
    # PHƯƠNG THỨC TÌM KIẾM CHÍNH
    # ================================================================

    @timer
    def search(self, query: str, top_k: int = None) -> list[SearchResult]:
        """
        THE MAIN SEARCH METHOD — Chạy toàn bộ pipeline 3 tầng.

        Args:
            query: Câu truy vấn (tiếng Việt hoặc tiếng Anh).
            top_k: Số kết quả trả về (mặc định TOP_K_FINAL=5).

        Returns:
            Danh sách SearchResult đã xếp hạng, tối đa top_k.
        """
        if top_k is None:
            top_k = TOP_K_FINAL

        pipeline_start = time.perf_counter()
        self._latency_log.clear()

        log_step("╔══ VER SEARCH", f'Query: "{query}" | Track: {self.track}')

        # ─── Tầng 1: Query Analysis ───
        analyzed_query = self._analyze_query(query)

        # ─── Tầng 2: Multi-Index Retrieval → Top 100 ───
        candidates = self._retrieve(analyzed_query)

        if not candidates:
            log_warning("Không tìm thấy kết quả nào")
            return []

        # ─── Tầng 3: Track Processing ───
        if self.track == "auto":
            final_results = self._process_auto_track(query, candidates)
        else:
            final_results = self._process_semi_track(query, candidates)

        # ─── Trả về top-K ───
        top_results = final_results[:top_k]

        # ─── Log tổng kết ───
        pipeline_time = time.perf_counter() - pipeline_start
        self._log_latency("total_pipeline", pipeline_time)

        log_step("╚══ KẾT QUẢ", f"{len(top_results)} kết quả trong {pipeline_time:.2f}s")

        # Hiển thị kết quả
        for i, result in enumerate(top_results, 1):
            time_str = format_time(result.start_time)
            text_preview = (result.text[:100] + "...") if result.text and len(result.text) > 100 else (result.text or "")
            log_info(
                f"  #{i} [{time_str}] score={result.score:.4f} "
                f"({result.source}) — {text_preview}"
            )

        # Log latency breakdown
        self._print_latency_breakdown()

        return top_results

    # ================================================================
    # TIỀN XỬ LÝ OFFLINE
    # ================================================================

    @timer
    def preprocess(
        self,
        video_path: str = None,
        video_dir: str = None,
    ) -> dict:
        """
        Chạy pipeline tiền xử lý offline đầy đủ.

        Quy trình cho mỗi video (tuần tự để quản lý 8GB VRAM):
          1. ASR: PhoWhisper → transcript
          2. Audio: CLAP → audio events + embeddings
          3. OCR: PaddleOCR + VietOCR → text từ frames
          4. Video: InternVideo2 → action embeddings
          5. Frame: SigLIP2 → frame embeddings
          6. Temporal Alignment → unified segments
          7. Build indexes

        Args:
            video_path: Đường dẫn 1 video (ưu tiên nếu cung cấp).
            video_dir: Thư mục chứa videos (mặc định RAW_VIDEO_DIR).

        Returns:
            Dict thống kê quá trình tiền xử lý.
        """
        preprocess_start = time.perf_counter()
        log_step("OFFLINE PREPROCESSING", "═══ Bắt đầu tiền xử lý dữ liệu ═══")

        # Thu thập danh sách video
        video_files = self._collect_video_files(video_path, video_dir)
        if not video_files:
            log_error("Không tìm thấy video nào để xử lý")
            return {"error": "No videos found"}

        log_info(f"Tìm thấy {len(video_files)} video cần xử lý")

        # Kết quả tổng hợp
        all_segments: list[SegmentSchema] = []
        stats = {
            "total_videos": len(video_files),
            "processed_videos": 0,
            "total_segments": 0,
            "total_duration_sec": 0.0,
            "errors": [],
        }

        # Xử lý từng video
        for idx, vpath in enumerate(video_files, 1):
            video_name = Path(vpath).name
            log_step("Progress", f"[{idx}/{len(video_files)}] {video_name}")

            try:
                segments = self._preprocess_single_video(vpath)
                all_segments.extend(segments)
                stats["processed_videos"] += 1
                stats["total_segments"] += len(segments)

                duration = get_video_duration(vpath)
                stats["total_duration_sec"] += duration

                log_success(
                    f"{video_name}: {len(segments)} segments, "
                    f"duration={format_time(duration)}"
                )
            except Exception as e:
                log_error(f"Lỗi xử lý {video_name}: {e}")
                stats["errors"].append({"video": video_name, "error": str(e)})

        # --- Build tất cả indexes ---
        if all_segments:
            log_step("Build Indexes", "Xây dựng chỉ mục đa phương thức")
            self._build_all_indexes(all_segments)

            # --- Lưu segments ra disk ---
            self._save_segments(all_segments)

        # Tổng kết
        total_time = time.perf_counter() - preprocess_start
        stats["total_time_sec"] = round(total_time, 1)

        log_step("HOÀN TẤT", "═══ Tiền xử lý xong ═══")
        log_info(f"Videos xử lý: {stats['processed_videos']}/{stats['total_videos']}")
        log_info(f"Tổng segments: {stats['total_segments']}")
        log_info(f"Tổng thời lượng video: {format_time(stats['total_duration_sec'])}")
        log_info(f"Thời gian xử lý: {total_time:.1f}s")

        if stats["errors"]:
            log_warning(f"Có {len(stats['errors'])} lỗi trong quá trình xử lý")

        return stats

    def _collect_video_files(
        self, video_path: str = None, video_dir: str = None
    ) -> list[str]:
        """Thu thập danh sách đường dẫn video cần xử lý."""
        if video_path:
            vp = str(Path(video_path).resolve())
            if os.path.exists(vp):
                return [vp]
            log_error(f"Video không tồn tại: {vp}")
            return []

        search_dir = video_dir or str(RAW_VIDEO_DIR)
        extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(search_dir, ext)))
        return sorted(files)

    def _preprocess_single_video(self, video_path: str) -> list[SegmentSchema]:
        """
        Tiền xử lý 1 video qua 5 "giác quan".
        Mỗi engine: load → process → unload (tuần tự để quản lý VRAM).
        Cuối cùng gọi temporal_align để đồng bộ hóa thành list[SegmentSchema].
        """
        video_name = Path(video_path).name

        # ─── 1. ASR: PhoWhisper ───
        asr_segments = self._run_asr(video_path)
        log_vram("sau ASR")
        free_vram()

        # ─── 2. Audio Events: CLAP ───
        audio_events = self._run_audio_extraction(video_path)
        log_vram("sau Audio")
        free_vram()

        # ─── 3. OCR: PaddleOCR + VietOCR ───
        ocr_data = self._run_ocr(video_path)
        log_vram("sau OCR")
        free_vram()

        # ─── 4. Video Encoder: InternVideo2 ───
        video_clips = self._run_video_encoding(video_path)
        log_vram("sau Video Encoder")
        free_vram()

        # ─── 5. Frame Encoder: SigLIP2 ───
        frame_data = self._run_frame_encoding(video_path)
        log_vram("sau Frame Encoder")
        free_vram()

        # ─── 6. Temporal Alignment ───
        log_step("Temporal Alignment", f"Đồng bộ hóa các luồng cho {video_name}")
        duration = get_video_duration(video_path)
        segments = self._temporal_align(
            video_path,
            duration, asr_segments, audio_events, ocr_data, video_clips, frame_data
        )

        # ─── THÊM ĐOẠN CODE NÀY VÀO TRƯỚC KHI RETURN ───
        import cv2
        log_step("Save Keyframes", f"Đang lưu thumbnails cho {video_name}")
        for seg in segments:
            kf_path = Path(KEYFRAMES_DIR) / f"{seg.segment_id}.jpg"
            if not kf_path.exists():
                mid_time = (seg.start_time + seg.end_time) / 2
                try:
                    # Gọi hàm từ utils.py để cắt frame tại thời điểm mid_time
                    frame_img = extract_frame(video_path, mid_time) 
                    if frame_img is not None:
                        import numpy as np
                        # Convert PIL Image về format OpenCV (BGR) để lưu xuống đĩa
                        cv2_img = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(kf_path), cv2_img)
                except Exception as e:
                    log_warning(f"Lỗi không thể lưu keyframe {seg.segment_id}: {e}")

        return segments

    # ─── Engine runners (tuần tự, lazy load) ───

    def _run_asr(self, video_path: str) -> list[dict]:
        """Chạy ASR engine trên video. Trả về list[{start, end, text}]."""
        log_step("ASR", f"Transcribe: {Path(video_path).name}")

        # Kiểm tra cache transcript
        video_stem = Path(video_path).stem
        transcript_path = TRANSCRIPT_DIR / f"{video_stem}_transcript.json"
        if transcript_path.exists():
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log_success(f"Loaded cached transcript: {len(data)} segments")
                return data
            except Exception as e:
                log_warning(f"Cache lỗi, chạy lại ASR: {e}")

        try:
            from offline.asr_engine import ASREngine
            engine = ASREngine()
            segments = engine.transcribe(video_path)

            # Lưu transcript cache
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            log_success(f"ASR: {len(segments)} segments → {transcript_path.name}")

            # Unload model
            del engine
            free_vram()
            return segments
        except ImportError:
            log_warning("ASREngine chưa tồn tại (offline/asr_engine.py)")
            # Fallback: thử whisper trực tiếp
            return self._run_asr_fallback(video_path)
        except Exception as e:
            log_error(f"ASR lỗi: {e}")
            return []

    def _run_asr_fallback(self, video_path: str) -> list[dict]:
        """Fallback ASR dùng openai-whisper trực tiếp."""
        try:
            import whisper
            model = whisper.load_model("medium", device=DEVICE)
            result = model.transcribe(
                str(video_path), language="vi", verbose=False,
                fp16=(DEVICE == "cuda"),
            )
            segments = [
                {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in result.get("segments", [])
            ]
            del model
            free_vram()
            log_success(f"ASR fallback: {len(segments)} segments")
            return segments
        except Exception as e:
            log_error(f"ASR fallback lỗi: {e}")
            return []

    def _run_audio_extraction(self, video_path: str) -> list[dict]:
        """Trích xuất audio events + embeddings bằng CLAP."""
        log_step("Audio", f"Extract audio events: {Path(video_path).name}")
        try:
            from offline.audio_engine import AudioEventEngine
            engine = AudioEventEngine()
            results = engine.extract_audio_events(video_path)
            del engine
            free_vram()
            log_success(f"Audio: extracted {len(results)} events")
            return results
        except ImportError:
            log_warning("AudioEngine chưa tồn tại (offline/audio_engine.py)")
            return []
        except Exception as e:
            log_error(f"Audio extraction lỗi: {e}")
            return []

    def _run_ocr(self, video_path: str) -> list[dict]:
        """Trích xuất OCR text từ keyframes."""
        log_step("OCR", f"Extract text from frames: {Path(video_path).name}")
        try:
            from offline.ocr_engine import OCREngine
            engine = OCREngine()
            results = engine.extract_text_from_frames(video_path)
            del engine
            free_vram()
            log_success(f"OCR: extracted text from {len(results)} frames")
            return results
        except ImportError:
            log_warning("OCREngine chưa tồn tại (offline/ocr_engine.py)")
            return []
        except Exception as e:
            log_error(f"OCR lỗi: {e}")
            return []

    def _run_video_encoding(self, video_path: str) -> tuple:
        """Tạo video embeddings bằng InternVideo2."""
        log_step("Video Encoder", f"Encode clips: {Path(video_path).name}")
        try:
            from offline.video_encoder import VideoEncoder
            encoder = VideoEncoder()
            results = encoder.encode_clips(video_path)
            del encoder
            free_vram()
            log_success(f"Video Encoder: encoded {len(results[0])} clips")
            return results
        except ImportError:
            log_warning("VideoEncoder chưa tồn tại (offline/video_encoder.py)")
            return ([], [])
        except Exception as e:
            log_error(f"Video encoding lỗi: {e}")
            return ([], [])

    def _run_frame_encoding(self, video_path: str) -> tuple:
        """Tạo frame embeddings bằng SigLIP2."""
        log_step("Frame Encoder", f"Encode frames: {Path(video_path).name}")
        try:
            from offline.frame_encoder import FrameEncoder
            encoder = FrameEncoder()
            results = encoder.encode_frames(video_path)
            del encoder
            free_vram()
            log_success(f"Frame Encoder: encoded {len(results[0])} frames")
            return results
        except ImportError:
            log_warning("FrameEncoder chưa tồn tại (offline/frame_encoder.py)")
            return ([], [])
        except Exception as e:
            log_error(f"Frame encoding lỗi: {e}")
            return ([], [])

    # ─── Alignment & Indexing ───

    def _temporal_align(
        self, video_file: str, duration: float, asr_segments: list[dict], 
        audio_events: list[dict], ocr_data: list[dict], video_clips: tuple, frame_data: tuple
    ) -> list[SegmentSchema]:
        """Đồng bộ hóa temporal alignment giữa các luồng để tạo unified segments."""
        try:
            from offline.temporal_align import TemporalAligner
            aligner = TemporalAligner()
            aligned = aligner.align(video_file, duration, asr_segments, audio_events, ocr_data, video_clips, frame_data)
            return aligned
        except ImportError:
            log_warning("TemporalAligner chưa tồn tại — không thể align")
            return []
        except Exception as e:
            log_error(f"Temporal alignment lỗi: {e}")
            return []

    def _build_all_indexes(self, segments: list[SegmentSchema]):
        """Xây dựng tất cả indexes từ unified segments."""
        try:
            # FIX: Gọi trực tiếp UnifiedIndexer từ thư mục offline
            from offline.indexer import UnifiedIndexer
            builder = UnifiedIndexer()
            # FIX: Gọi đúng tên hàm build_all_indexes
            builder.build_all_indexes(segments)
            log_success("Đã build tất cả indexes thành công")
        except Exception as e:
            log_error(f"Build indexes lỗi: {e}")

    def _save_segments(self, segments: list[SegmentSchema]):
        """Lưu unified segments ra disk."""
        try:
            seg_dicts = [s.to_dict() for s in segments]
            Path(SEGMENTS_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
                json.dump(seg_dicts, f, ensure_ascii=False, indent=2)
            log_success(f"Đã lưu {len(seg_dicts)} segments → {SEGMENTS_PATH}")
        except Exception as e:
            log_error(f"Lỗi lưu segments: {e}")

    # ================================================================
    # TRẠNG THÁI & TIỆN ÍCH
    # ================================================================

    def set_track(self, track: str):
        """Thay đổi track xử lý tại runtime."""
        if track not in ["auto", "semi"]:
            log_warning(f"Track không hợp lệ: {track}, giữ nguyên {self.track}")
            return
        
        if self.track == track:
            return
            
        log_info(f"Chuyển pipeline sang track: {track}")
        self.track = track
        
        # Load các engines của track mới nếu chưa load
        if track == "auto" and self.auto_reranker is None:
            self.auto_reranker = self._init_auto_reranker()
        elif track == "semi":
            if self.lvlm_reranker is None:
                self.lvlm_reranker = self._init_lvlm_reranker()
            if self.crag_engine is None:
                self.crag_engine = self._init_crag()
            if self.verifier is None:
                self.verifier = self._init_verifier()

    def get_status(self) -> dict:
        """Trả về trạng thái tổng hợp của pipeline."""
        try:
            import psutil
            ram_mb = round(psutil.Process().memory_info().rss / 1024**2, 1)
        except ImportError:
            ram_mb = 0

        # Thông tin GPU
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }

        # Kiểm tra indexes
        indexes = {
            "text_dense": os.path.exists(TEXT_DENSE_INDEX_PATH),
            "text_sparse": os.path.exists(TEXT_SPARSE_INDEX_DIR),
            "visual": os.path.exists(VISUAL_INDEX_PATH),
            "temporal": os.path.exists(TEMPORAL_INDEX_PATH),
            "audio": os.path.exists(AUDIO_INDEX_PATH),
            "segments": os.path.exists(SEGMENTS_PATH),
        }

        # Đếm segments
        segment_count = 0
        if os.path.exists(SEGMENTS_PATH):
            try:
                with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
                    segment_count = len(json.load(f))
            except Exception:
                pass

        return {
            "track": self.track,
            "device": DEVICE,
            "cuda_available": CUDA_AVAILABLE,
            "gpu": gpu_info,
            "ram_mb": ram_mb,
            "engines": {
                "query_analyzer": self.query_analyzer is not None,
                "retriever": self.retriever is not None,
                "auto_reranker": self.auto_reranker is not None,
                "lvlm_reranker": self.lvlm_reranker is not None,
                "crag_engine": self.crag_engine is not None,
                "verifier": self.verifier is not None,
                "evaluator": self.evaluator is not None,
            },
            "indexes": indexes,
            "segment_count": segment_count,
            "config": {
                "top_k_retrieval": TOP_K_RETRIEVAL,
                "top_k_rerank": TOP_K_RERANK,
                "top_k_final": TOP_K_FINAL,
                "crag_max_retries": CRAG_MAX_RETRIES,
                "auto_budget_sec": AUTO_TRACK_BUDGET_SEC,
                "semi_budget_sec": SEMI_TRACK_BUDGET_SEC,
            },
            "latency_log": self._latency_log.copy(),
        }

    def _log_latency(self, step: str, latency: float):
        """Ghi nhận latency cho một bước."""
        self._latency_log.append({
            "step": step,
            "latency_sec": round(latency, 3),
        })

    def _print_latency_breakdown(self):
        """In breakdown latency ra console."""
        if not self._latency_log:
            return
        log_step("Latency Breakdown", "")
        for entry in self._latency_log:
            log_info(f"  {entry['step']:25s}: {entry['latency_sec']:.3f}s")