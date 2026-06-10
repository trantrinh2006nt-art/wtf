"""
retriever.py — Tầng 2: Multi-Index Retrieval + RRF Fusion.

Truy xuất đồng thời từ 5 index:
1. Text Sparse (BM25)    — tìm kiếm từ khóa trong transcript
2. Text Dense (BGE-M3)   — tìm kiếm ngữ nghĩa trong transcript  
3. Visual (SigLIP2)      — tìm kiếm hình ảnh
4. Audio (CLAP)          — tìm kiếm âm thanh
5. Temporal (InternVideo2) — tìm kiếm hành động

Kết quả được hợp nhất bằng Weighted Reciprocal Rank Fusion (RRF),
cho phép kết hợp điểm mạnh của từng modality.

VRAM-safe: Lazy-load model, free_vram() sau khi encode.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, tokenize_vietnamese
)

try:
    import faiss
except ImportError:
    faiss = None
    log_warning("faiss chưa cài. Chạy: pip install faiss-cpu hoặc faiss-gpu")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    log_warning("rank_bm25 chưa cài. Chạy: pip install rank-bm25")


class MultiIndexRetriever:
    """
    Tầng 2 — Multi-Index Retrieval với RRF Fusion.
    
    Truy xuất từ nhiều index song song, sau đó dùng Weighted RRF
    để hợp nhất kết quả. Trọng số mỗi modality lấy từ config.
    
    Text dense/sparse chiếm ưu thế (~55%) vì ASR tiếng Việt là
    nguồn thông tin chính trong video bằng chứng.
    """

    def __init__(self):
        """Khởi tạo retriever: load indexes + segments từ disk."""
        log_step("MultiIndexRetriever", "Khởi tạo retriever...")

        # Segments metadata
        self.segments: list[SegmentSchema] = []
        self.segment_map: dict[str, SegmentSchema] = {}

        # FAISS indexes — sẽ load từ disk
        self.text_dense_index = None
        self.text_dense_meta: list[str] = []  # segment_id list

        self.visual_index = None
        self.visual_meta: list[str] = []

        self.audio_index = None
        self.audio_meta: list[str] = []

        self.temporal_index = None
        self.temporal_meta: list[str] = []

        # BM25
        self.bm25 = None
        self.bm25_segment_ids: list[str] = []
        self.bm25_corpus: list[list[str]] = []

        # Models — lazy load
        self._bge_model = None
        self._siglip_model = None
        self._siglip_processor = None
        self._clap_model = None
        self._clap_processor = None

        # Load dữ liệu
        self._load_segments()
        self._load_indexes()

    # ================================================================
    # INIT HELPERS
    # ================================================================

    def _load_segments(self):
        """Load unified segments từ disk."""
        segments_path = Path(SEGMENTS_PATH)
        if not segments_path.exists():
            log_warning(f"Segments file không tồn tại: {SEGMENTS_PATH}")
            return

        try:
            with open(segments_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for d in data:
                seg = SegmentSchema.from_dict(d)
                self.segments.append(seg)
                self.segment_map[seg.segment_id] = seg

            log_success(f"Loaded {len(self.segments)} segments")
        except Exception as e:
            log_error(f"Lỗi load segments: {e}")

    def _load_indexes(self):
        """Load tất cả FAISS indexes + BM25 từ disk."""
        # Text Dense Index
        self._load_faiss_index(
            TEXT_DENSE_INDEX_PATH, TEXT_DENSE_META_PATH,
            "text_dense"
        )
        # Visual Index
        self._load_faiss_index(
            VISUAL_INDEX_PATH, VISUAL_META_PATH,
            "visual"
        )
        # Audio Index
        self._load_faiss_index(
            AUDIO_INDEX_PATH, AUDIO_META_PATH,
            "audio"
        )
        # Temporal Index
        self._load_faiss_index(
            TEMPORAL_INDEX_PATH, TEMPORAL_META_PATH,
            "temporal"
        )
        # BM25 (text sparse)
        self._load_bm25()

    def _load_faiss_index(self, index_path: str, meta_path: str, name: str):
        """Load 1 FAISS index + metadata từ disk."""
        if faiss is None:
            log_warning(f"faiss không khả dụng, bỏ qua {name} index")
            return

        idx_file = Path(index_path)
        meta_file = Path(meta_path)

        if not idx_file.exists():
            log_info(f"{name} index chưa được build: {index_path}")
            return

        try:
            index = faiss.read_index(str(idx_file))

            meta = []
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
                    # SỬA LỖI Ở ĐÂY: Chỉ lấy mảng segment_ids thay vì lấy cả dict
                    meta = meta_dict.get("segment_ids", [])

            # Gán vào attribute tương ứng
            if name == "text_dense":
                self.text_dense_index = index
                self.text_dense_meta = meta
            elif name == "visual":
                self.visual_index = index
                self.visual_meta = meta
            elif name == "audio":
                self.audio_index = index
                self.audio_meta = meta
            elif name == "temporal":
                self.temporal_index = index
                self.temporal_meta = meta

            log_success(f"{name} index loaded: {index.ntotal} vectors")
        except Exception as e:
            log_error(f"Lỗi load {name} index: {e}")

    def _load_bm25(self):
        """Load/Build BM25 index từ segments."""
        if BM25Okapi is None:
            log_warning("rank_bm25 không khả dụng, bỏ qua BM25")
            return

        # Thử load BM25 corpus đã lưu
        bm25_corpus_path = Path(TEXT_SPARSE_INDEX_DIR) / "bm25_corpus.json"
        if bm25_corpus_path.exists():
            try:
                with open(bm25_corpus_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                self.bm25_corpus = saved.get("corpus", [])
                self.bm25_segment_ids = saved.get("segment_ids", [])
                if self.bm25_corpus:
                    self.bm25 = BM25Okapi(
                        self.bm25_corpus,
                        k1=BM25_K1,
                        b=BM25_B
                    )
                    log_success(f"BM25 loaded: {len(self.bm25_corpus)} documents")
                    return
            except Exception as e:
                log_warning(f"Lỗi load BM25 corpus: {e}")

        # Fallback: build BM25 từ segments hiện có
        if not self.segments:
            log_info("Không có segments để build BM25")
            return

        log_info("Building BM25 index từ segments...")
        corpus = []
        seg_ids = []
        for seg in self.segments:
            text = seg.get_full_text()
            if text.strip():
                tokens = tokenize_vietnamese(text)
                corpus.append(tokens)
                seg_ids.append(seg.segment_id)

        if corpus:
            self.bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
            self.bm25_corpus = corpus
            self.bm25_segment_ids = seg_ids
            log_success(f"BM25 built: {len(corpus)} documents")

    # ================================================================
    # PUBLIC: retrieve()
    # ================================================================

    @timer
    def retrieve(self, analyzed_query: AnalyzedQuery,
                 top_k: int = TOP_K_RETRIEVAL) -> list[SearchResult]:
        """
        Truy xuất đa phương thức + RRF fusion.

        Args:
            analyzed_query: Query đã được phân rã bởi QueryAnalyzer
            top_k: Số kết quả trả về (default: TOP_K_RETRIEVAL=100)

        Returns:
            list[SearchResult] đã sắp xếp theo điểm fusion
        """
        log_step("Multi-Index Retrieval", f"Truy xuất top-{top_k}")

        results_lists = []
        weights = []

        # 1. Text Sparse (BM25)
        sparse_results = self._text_sparse_search(
            analyzed_query.text_query,
            analyzed_query.hyde_document,
            top_k
        )
        if sparse_results:
            results_lists.append(sparse_results)
            weights.append(WEIGHT_TEXT_SPARSE)
            log_info(f"BM25: {len(sparse_results)} kết quả")

        # 2. Text Dense (BGE-M3)
        dense_results = self._text_dense_search(
            analyzed_query.text_query,
            top_k
        )
        if dense_results:
            results_lists.append(dense_results)
            weights.append(WEIGHT_TEXT_DENSE)
            log_info(f"Dense text: {len(dense_results)} kết quả")

        # 3. Visual (SigLIP2)
        visual_results = self._visual_search(
            analyzed_query.visual_query,
            top_k
        )
        if visual_results:
            results_lists.append(visual_results)
            weights.append(WEIGHT_VISUAL)
            log_info(f"Visual: {len(visual_results)} kết quả")

        # 4. Audio (CLAP)
        if analyzed_query.audio_query:
            audio_results = self._audio_search(
                analyzed_query.audio_query,
                top_k
            )
            if audio_results:
                results_lists.append(audio_results)
                weights.append(WEIGHT_AUDIO)
                log_info(f"Audio: {len(audio_results)} kết quả")

        # 5. Temporal (InternVideo2)
        if analyzed_query.temporal_query:
            temporal_results = self._temporal_search(
                analyzed_query.temporal_query,
                top_k
            )
            if temporal_results:
                results_lists.append(temporal_results)
                weights.append(WEIGHT_TEMPORAL)
                log_info(f"Temporal: {len(temporal_results)} kết quả")

        # Không có kết quả từ bất kỳ index nào
        if not results_lists:
            log_warning("Không có kết quả từ bất kỳ index nào!")
            return []

        # RRF Fusion
        fused = self._rrf_fusion(results_lists, weights)

        # Lấy top_k
        fused = fused[:top_k]
        log_success(f"RRF Fusion: {len(fused)} kết quả (từ {len(results_lists)} indexes)")

        return fused

    # ================================================================
    # PRIVATE: Text Sparse Search (BM25)
    # ================================================================

    def _text_sparse_search(self, query: str, hyde_doc: str = "",
                            top_k: int = 100) -> list[SearchResult]:
        """
        BM25 search trong transcript + OCR text.
        
        Kết hợp query gốc + HyDE document để tăng recall:
        HyDE document chứa nhiều từ khóa liên quan hơn query ngắn.
        """
        if self.bm25 is None:
            return []

        try:
            # Tokenize query
            query_tokens = tokenize_vietnamese(query)

            # Nếu có HyDE, kết hợp tokens
            if hyde_doc:
                hyde_tokens = tokenize_vietnamese(hyde_doc)
                # Trộn: query tokens quan trọng hơn (lặp lại 2 lần)
                combined_tokens = query_tokens * 2 + hyde_tokens
            else:
                combined_tokens = query_tokens

            if not combined_tokens:
                return []

            # BM25 scoring
            scores = self.bm25.get_scores(combined_tokens)

            # Lấy top_k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            # Chuẩn hóa scores
            max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0

            for idx in top_indices:
                if scores[idx] <= 0:
                    break

                seg_id = self.bm25_segment_ids[idx]
                seg = self.segment_map.get(seg_id)

                result = SearchResult(
                    segment_id=seg_id,
                    video_file=seg.video_file if seg else "",
                    start_time=seg.start_time if seg else 0.0,
                    end_time=seg.end_time if seg else 0.0,
                    text=seg.get_full_text() if seg else "",
                    score=float(scores[idx] / max_score),  # Chuẩn hóa 0-1
                    source="text_sparse",
                    sparse_score=float(scores[idx] / max_score),
                    keyframe_path=seg.keyframe_path if seg else "",
                )
                results.append(result)

            return results

        except Exception as e:
            log_error(f"BM25 search lỗi: {e}")
            return []

    # ================================================================
    # PRIVATE: Text Dense Search (BGE-M3)
    # ================================================================

    def _text_dense_search(self, query: str,
                           top_k: int = 100) -> list[SearchResult]:
        """Encode query bằng BGE-M3 → FAISS search."""
        if self.text_dense_index is None:
            return []

        try:
            import torch

            # Lazy load BGE-M3
            if self._bge_model is None:
                self._load_bge_model()

            if self._bge_model is None:
                return []

            # Encode query
            with torch.no_grad():
                query_embedding = self._bge_model.encode(
                    [query],
                    normalize_embeddings=True,
                    batch_size=1
                )

            query_vec = np.array(query_embedding, dtype=np.float32)
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)

            # FAISS search
            actual_k = min(top_k, self.text_dense_index.ntotal)
            scores, indices = self.text_dense_index.search(query_vec, actual_k)

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])  # FIX 1: Ép kiểu về int chuẩn của Python
                if idx < 0:
                    continue

                score = float(scores[0][i])
                # Cosine similarity đã normalize → score trong [-1, 1]
                # Chuyển về [0, 1]
                norm_score = (score + 1.0) / 2.0

                seg_id = self.text_dense_meta[idx] if idx < len(self.text_dense_meta) else ""
                seg = self.segment_map.get(seg_id)

                result = SearchResult(
                    segment_id=seg_id,
                    video_file=seg.video_file if seg else "",
                    start_time=seg.start_time if seg else 0.0,
                    end_time=seg.end_time if seg else 0.0,
                    text=seg.get_full_text() if seg else "",
                    score=norm_score,
                    source="text_dense",
                    dense_score=norm_score,
                    keyframe_path=seg.keyframe_path if seg else "",
                )
                results.append(result)

            return results

        except Exception as e:
            log_error(f"Text dense search lỗi: {e}")
            return []

    # ================================================================
    # PRIVATE: Visual Search (SigLIP2)
    # ================================================================

    def _visual_search(self, query: str,
                       top_k: int = 100) -> list[SearchResult]:
        """Encode visual query bằng SigLIP2 text encoder → FAISS search."""
        if self.visual_index is None:
            return []

        try:
            import torch

            # Lazy load SigLIP2
            if self._siglip_model is None:
                self._load_siglip_model()

            if self._siglip_model is None or self._siglip_processor is None:
                return []

            # Encode text query bằng SigLIP2 text encoder
            with torch.no_grad():
                inputs = self._siglip_processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()
                          if k in ['input_ids', 'attention_mask']}
                text_features = self._siglip_model.get_text_features(**inputs)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_vec = text_features.cpu().numpy().astype(np.float32)

             # FAISS search
            actual_k = min(top_k, self.visual_index.ntotal)
            scores, indices = self.visual_index.search(query_vec, actual_k)

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])  # FIX 1: Ép kiểu về int chuẩn của Python
                if idx < 0:
                    continue

                score = float(scores[0][i])
                norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0))

                seg_id = self.visual_meta[idx] if idx < len(self.visual_meta) else ""
                seg = self.segment_map.get(seg_id)

                result = SearchResult(
                    segment_id=seg_id,
                    video_file=seg.video_file if seg else "",
                    start_time=seg.start_time if seg else 0.0,
                    end_time=seg.end_time if seg else 0.0,
                    text=seg.get_full_text() if seg else "",
                    score=norm_score,
                    source="visual",
                    visual_score=norm_score,
                    keyframe_path=seg.keyframe_path if seg else "",
                )
                results.append(result)

            return results

        except Exception as e:
            log_error(f"Visual search lỗi: {e}")
            return []

    # ================================================================
    # PRIVATE: Audio Search (CLAP)
    # ================================================================

    def _audio_search(self, query: str,
                      top_k: int = 100) -> list[SearchResult]:
        """Encode audio query bằng CLAP text encoder → FAISS search."""
        if self.audio_index is None:
            return []

        try:
            import torch

            # Lazy load CLAP
            if self._clap_model is None:
                self._load_clap_model()

            if self._clap_model is None or self._clap_processor is None:
                return []

            # Encode text query bằng CLAP text encoder
            with torch.no_grad():
                inputs = self._clap_processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                text_features = self._clap_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_vec = text_features.cpu().numpy().astype(np.float32)

            # FAISS search
            actual_k = min(top_k, self.audio_index.ntotal)
            scores, indices = self.audio_index.search(query_vec, actual_k)

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])  # FIX 1: Ép kiểu về int chuẩn của Python
                if idx < 0:
                    continue

                score = float(scores[0][i])
                norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0))

                seg_id = self.audio_meta[idx] if idx < len(self.audio_meta) else ""
                seg = self.segment_map.get(seg_id)

                result = SearchResult(
                    segment_id=seg_id,
                    video_file=seg.video_file if seg else "",
                    start_time=seg.start_time if seg else 0.0,
                    end_time=seg.end_time if seg else 0.0,
                    text=seg.get_full_text() if seg else "",
                    score=norm_score,
                    source="audio",
                    audio_score=norm_score,
                    keyframe_path=seg.keyframe_path if seg else "",
                )
                results.append(result)

            return results

        except Exception as e:
            log_error(f"Audio search lỗi: {e}")
            return []

    # ================================================================
    # PRIVATE: Temporal Search (InternVideo2)
    # ================================================================

    def _temporal_search(self, query: str,
                         top_k: int = 100) -> list[SearchResult]:
        """
        Temporal search dùng text embedding encode temporal_query
        → FAISS search trên temporal index (video_embedding).
        
        Vì InternVideo2 text encoder phức tạp và nặng, ta sẽ dùng
        BGE-M3 encode temporal_query nếu temporal index lưu bằng
        text embedding, hoặc skip nếu index chưa có.
        """
        if self.temporal_index is None:
            return []

        try:
            import torch

            # FIX 2: Dùng SigLIP2 (768d) thay vì BGE-M3 (1024d) để đồng bộ số chiều với Temporal Index
            if self._siglip_model is None:
                self._load_siglip_model()

            if self._siglip_model is None or self._siglip_processor is None:
                return []

            with torch.no_grad():
                inputs = self._siglip_processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()
                          if k in ['input_ids', 'attention_mask']}
                text_features = self._siglip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_vec = text_features.cpu().numpy().astype(np.float32)

            # Kiểm tra dimension phù hợp
            if query_vec.shape[1] != self.temporal_index.d:
                log_warning(
                    f"Temporal index dim mismatch: query={query_vec.shape[1]}, "
                    f"index={self.temporal_index.d}. Bỏ qua temporal search."
                )
                return []

            # FAISS search
            actual_k = min(top_k, self.temporal_index.ntotal)
            scores, indices = self.temporal_index.search(query_vec, actual_k)

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])  # FIX 1: Ép kiểu về int chuẩn của Python
                if idx < 0:
                    continue

                score = float(scores[0][i])
                norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0))

                seg_id = self.temporal_meta[idx] if idx < len(self.temporal_meta) else ""
                seg = self.segment_map.get(seg_id)

                result = SearchResult(
                    segment_id=seg_id,
                    video_file=seg.video_file if seg else "",
                    start_time=seg.start_time if seg else 0.0,
                    end_time=seg.end_time if seg else 0.0,
                    text=seg.get_full_text() if seg else "",
                    score=norm_score,
                    source="temporal",
                    temporal_score=norm_score,
                    keyframe_path=seg.keyframe_path if seg else "",
                )
                results.append(result)

            return results

        except Exception as e:
            log_error(f"Temporal search lỗi: {e}")
            return []

    # ================================================================
    # PRIVATE: RRF Fusion
    # ================================================================

    def _rrf_fusion(self, results_lists: list[list[SearchResult]],
                    weights: list[float]) -> list[SearchResult]:
        """
        Weighted Reciprocal Rank Fusion.
        
        RRF score = Σ weight_i / (RRF_K + rank_i)
        
        Tại sao RRF mà không phải score fusion?
        → RRF robust hơn vì chỉ dùng rank, không bị ảnh hưởng
          bởi scale khác nhau giữa các modality scores.

        Args:
            results_lists: list chứa list[SearchResult] từ mỗi index
            weights: trọng số tương ứng cho mỗi index

        Returns:
            list[SearchResult] đã sort theo RRF score
        """
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return []
        norm_weights = [w / total_weight for w in weights]

        # Tính RRF score cho mỗi segment
        rrf_scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, SearchResult] = {}
        component_scores: dict[str, dict] = defaultdict(lambda: {
            "dense_score": 0.0, "sparse_score": 0.0,
            "visual_score": 0.0, "audio_score": 0.0,
            "temporal_score": 0.0
        })

        for result_list, weight in zip(results_lists, norm_weights):
            for rank, result in enumerate(result_list, start=1):
                seg_id = result.segment_id
                if not seg_id:
                    continue

                # RRF formula
                rrf_contrib = weight / (RRF_K + rank)
                rrf_scores[seg_id] += rrf_contrib

                # Lưu component scores
                if result.source == "text_dense":
                    component_scores[seg_id]["dense_score"] = max(
                        component_scores[seg_id]["dense_score"], result.score
                    )
                elif result.source == "text_sparse":
                    component_scores[seg_id]["sparse_score"] = max(
                        component_scores[seg_id]["sparse_score"], result.score
                    )
                elif result.source == "visual":
                    component_scores[seg_id]["visual_score"] = max(
                        component_scores[seg_id]["visual_score"], result.score
                    )
                elif result.source == "audio":
                    component_scores[seg_id]["audio_score"] = max(
                        component_scores[seg_id]["audio_score"], result.score
                    )
                elif result.source == "temporal":
                    component_scores[seg_id]["temporal_score"] = max(
                        component_scores[seg_id]["temporal_score"], result.score
                    )

                # Lưu result tốt nhất cho segment (ưu tiên text vì có nội dung)
                if seg_id not in best_result or result.text:
                    best_result[seg_id] = result

        # Build final results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for seg_id in sorted_ids:
            base = best_result[seg_id]
            comp = component_scores[seg_id]

            result = SearchResult(
                segment_id=seg_id,
                video_file=base.video_file,
                start_time=base.start_time,
                end_time=base.end_time,
                text=base.text,
                score=rrf_scores[seg_id],
                source="fused",
                dense_score=comp["dense_score"],
                sparse_score=comp["sparse_score"],
                visual_score=comp["visual_score"],
                audio_score=comp["audio_score"],
                temporal_score=comp["temporal_score"],
                keyframe_path=base.keyframe_path,
                metadata={
                    "rrf_score": rrf_scores[seg_id],
                    "num_sources": sum(1 for v in comp.values() if v > 0),
                },
            )
            fused_results.append(result)

        return fused_results

    # ================================================================
    # MODEL LOADING — Lazy, VRAM-safe
    # ================================================================

    def _load_bge_model(self):
        """Lazy load BGE-M3 cho dense text encoding."""
        try:
            from sentence_transformers import SentenceTransformer
            log_info(f"Loading BGE-M3: {TEXT_EMBEDDING_MODEL}...")
            log_vram("Trước load BGE-M3")

            self._bge_model = SentenceTransformer(
                TEXT_EMBEDDING_MODEL,
                device=DEVICE,
            )
            # Chuyển sang fp16 nếu GPU
            if DEVICE == "cuda" and USE_FP16:
                self._bge_model.half()

            log_success("BGE-M3 loaded")
            log_vram("Sau load BGE-M3")
        except Exception as e:
            log_error(f"Lỗi load BGE-M3: {e}")
            self._bge_model = None

    def _load_siglip_model(self):
        """Lazy load SigLIP2 cho visual text encoding."""
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            log_info(f"Loading SigLIP2: {FRAME_ENCODER_MODEL}...")
            log_vram("Trước load SigLIP2")

            self._siglip_processor = AutoProcessor.from_pretrained(
                FRAME_ENCODER_MODEL,
                use_fast=True
            )
            self._siglip_model = AutoModel.from_pretrained(
                FRAME_ENCODER_MODEL,
                dtype=torch.float16 if USE_FP16 else torch.float32,
            ).to(DEVICE).eval()

            log_success("SigLIP2 loaded")
            log_vram("Sau load SigLIP2")
        except Exception as e:
            log_error(f"Lỗi load SigLIP2: {e}")
            self._siglip_model = None
            self._siglip_processor = None

    def _load_clap_model(self):
        """Lazy load CLAP cho audio text encoding."""
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            log_info(f"Loading CLAP: {AUDIO_MODEL}...")
            log_vram("Trước load CLAP")

            self._clap_processor = ClapProcessor.from_pretrained(AUDIO_MODEL)
            self._clap_model = ClapModel.from_pretrained(
                AUDIO_MODEL,
                dtype=torch.float16 if USE_FP16 else torch.float32,
            ).to(DEVICE).eval()

            log_success("CLAP loaded")
            log_vram("Sau load CLAP")
        except Exception as e:
            log_error(f"Lỗi load CLAP: {e}")
            self._clap_model = None
            self._clap_processor = None

    # ================================================================
    # CLEANUP
    # ================================================================

    def unload_models(self):
        """Giải phóng tất cả models khỏi VRAM."""
        import torch

        models_to_unload = [
            ("_bge_model", "BGE-M3"),
            ("_siglip_model", "SigLIP2"),
            ("_siglip_processor", "SigLIP2 processor"),
            ("_clap_model", "CLAP"),
            ("_clap_processor", "CLAP processor"),
        ]

        for attr, name in models_to_unload:
            model = getattr(self, attr, None)
            if model is not None:
                del model
                setattr(self, attr, None)
                log_info(f"Unloaded {name}")

        free_vram()
        log_success("Tất cả retriever models đã được giải phóng")
        log_vram("Sau unload")
