"""
auto_track.py — Tầng 3A: Track Tự Động (Bot vs Bot).

Mục tiêu: Latency < 2 giây.

Flow:
1. Nhận top TOP_K_RERANK candidates từ Tầng 2 (retriever)
2. BGE-Reranker: cross-encoder scoring (query, passage) pairs
3. Sort theo rerank_score
4. Fast CRAG: heuristic kiểm tra confidence (KHÔNG gọi LLM)
5. Trả top TOP_K_FINAL kết quả

Fast CRAG (Corrective RAG đơn giản):
- CORRECT:   top score > CRAG_THRESHOLD_CORRECT → tin tưởng
- AMBIGUOUS: top score > CRAG_THRESHOLD_AMBIGUOUS → cảnh báo
- INCORRECT: top score thấp → đánh dấu, caller có thể retry

VRAM: BGE-Reranker ~1GB VRAM (fp16), lazy-load, unload sau khi dùng.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SearchResult, log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram
)


class AutoTrack:
    """
    Tầng 3A — Track Tự Động cho thi đấu Bot vs Bot.
    
    Dùng cross-encoder reranker (BGE-Reranker-v2-M3) để rerank 
    candidates, rồi Fast CRAG để đánh giá confidence.
    
    Tối ưu speed: 
    - Batch inference cho reranker
    - Không gọi LLM (khác với SemiTrack)
    - Heuristic CRAG thay vì LLM-based CRAG
    """

    def __init__(self):
        """Khởi tạo AutoTrack — reranker được lazy-load."""
        self._reranker = None
        self._reranker_loaded = False
        log_info("AutoTrack: Khởi tạo (reranker sẽ lazy-load khi cần)")

    # ================================================================
    # MODEL LOADING
    # ================================================================

    def _load_reranker(self):
        """
        Load BGE-Reranker-v2-M3 bằng sentence_transformers CrossEncoder.
        
        Model này ~560M params, fp16 ≈ 1.1GB VRAM.
        Dùng cross-encoder nên chính xác hơn bi-encoder nhưng chậm hơn.
        """
        if self._reranker_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder

            log_info(f"Loading reranker: {AUTO_RERANKER_MODEL}...")
            log_vram("Trước load reranker")

            # Load CrossEncoder
            self._reranker = CrossEncoder(
                AUTO_RERANKER_MODEL,
                max_length=512,
                device=DEVICE,
            )

            # Chuyển sang fp16 nếu GPU
            if DEVICE == "cuda" and USE_FP16:
                self._reranker.model.half()

            self._reranker_loaded = True
            log_success(f"Reranker loaded: {AUTO_RERANKER_MODEL}")
            log_vram("Sau load reranker")

        except Exception as e:
            log_error(f"Lỗi load reranker: {e}")
            self._reranker = None
            self._reranker_loaded = False

    # ================================================================
    # PUBLIC: process()
    # ================================================================

    @timer
    def process(self, query: str,
                candidates: list[SearchResult]) -> list[SearchResult]:
        """
        Rerank + Fast CRAG cho Auto Track.

        Args:
            query: Query gốc từ người dùng
            candidates: list[SearchResult] từ Tầng 2 (retriever)

        Returns:
            list[SearchResult] đã rerank, top TOP_K_FINAL kết quả
        """
        log_step("Auto Track", f"Rerank {len(candidates)} candidates")

        if not candidates:
            log_warning("Không có candidates để rerank")
            return []

        # Lấy top TOP_K_RERANK candidates
        candidates = candidates[:TOP_K_RERANK]
        log_info(f"Xử lý top-{len(candidates)} candidates")

        # Bước 1: Rerank bằng cross-encoder
        reranked = self._rerank(query, candidates)

        # Bước 2: Fast CRAG — heuristic confidence check
        verdict, final_results = self._fast_crag(query, reranked)

        # Lấy top TOP_K_FINAL
        final_results = final_results[:TOP_K_FINAL]

        log_success(
            f"Auto Track hoàn thành: {len(final_results)} kết quả, "
            f"verdict={verdict}"
        )

        return final_results

    # ================================================================
    # PRIVATE: _rerank() — Cross-Encoder Reranking
    # ================================================================

    def _rerank(self, query: str,
                candidates: list[SearchResult]) -> list[SearchResult]:
        """
        Rerank candidates bằng BGE-Reranker cross-encoder.
        
        Cross-encoder nhận (query, passage) pair và output score trực tiếp,
        chính xác hơn bi-encoder nhưng O(n) thay vì O(1) per query.
        
        Args:
            query: Query string
            candidates: list[SearchResult] cần rerank

        Returns:
            list[SearchResult] đã sort theo rerank_score (giảm dần)
        """
        # Lazy load reranker
        if not self._reranker_loaded:
            self._load_reranker()

        if self._reranker is None:
            log_warning("Reranker không khả dụng, giữ thứ tự ban đầu")
            return candidates

        try:
            # Tạo (query, passage) pairs
            pairs = []
            for cand in candidates:
                # Passage = full text của segment
                passage = cand.text if cand.text else cand.segment_id
                pairs.append([query, passage])

            # Batch scoring
            with torch.no_grad():
                raw_scores = self._reranker.predict(
                    pairs,
                    batch_size=min(len(pairs), 32),  # Batch nhỏ cho VRAM
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

            # Normalize scores: sigmoid → [0, 1]
            # BGE-Reranker output logits, cần sigmoid
            scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

            # Gán rerank_score cho mỗi candidate
            for i, cand in enumerate(candidates):
                cand.rerank_score = float(scores[i])
                cand.score = float(scores[i])  # Cập nhật score chính
                cand.metadata["rerank_raw"] = float(raw_scores[i])

            # Sort theo rerank_score (giảm dần)
            candidates.sort(key=lambda x: x.rerank_score, reverse=True)

            log_info(
                f"Reranked: top score={candidates[0].rerank_score:.4f}, "
                f"bottom score={candidates[-1].rerank_score:.4f}"
            )

            return candidates

        except Exception as e:
            log_error(f"Rerank lỗi: {e}")
            return candidates

    # ================================================================
    # PRIVATE: _fast_crag() — Fast Corrective RAG
    # ================================================================

    def _fast_crag(self, query: str,
                   results: list[SearchResult]) -> tuple[str, list[SearchResult]]:
        """
        Fast CRAG: kiểm tra confidence bằng heuristic (không gọi LLM).
        
        Logic:
        - CORRECT:   top result score > CRAG_THRESHOLD_CORRECT (0.7)
          → Tin tưởng kết quả, trả về nguyên
        - AMBIGUOUS: top score > CRAG_THRESHOLD_AMBIGUOUS (0.4)  
          → Cảnh báo, nhưng vẫn trả về (auto track không retry)
        - INCORRECT: top score thấp
          → Đánh dấu metadata, caller có thể quyết định retry

        Thêm heuristic bổ sung:
        - Score gap: nếu top-1 >> top-2 → confident
        - Multi-source: kết quả xuất hiện ở nhiều index → confident

        Args:
            query: Query string
            results: list[SearchResult] đã rerank

        Returns:
            (verdict, results) — verdict ∈ {"CORRECT", "AMBIGUOUS", "INCORRECT"}
        """
        if not results:
            return "INCORRECT", []

        top_score = results[0].rerank_score

        # Heuristic 1: Ngưỡng score tuyệt đối
        if top_score >= CRAG_THRESHOLD_CORRECT:
            verdict = "CORRECT"
        elif top_score >= CRAG_THRESHOLD_AMBIGUOUS:
            verdict = "AMBIGUOUS"
        else:
            verdict = "INCORRECT"

        # Heuristic 2: Score gap — top-1 vượt trội hẳn
        if len(results) >= 2:
            score_gap = results[0].rerank_score - results[1].rerank_score
            if score_gap > 0.3 and top_score >= CRAG_THRESHOLD_AMBIGUOUS:
                # Top-1 vượt trội → tăng confidence
                if verdict == "AMBIGUOUS":
                    verdict = "CORRECT"
                    log_info("CRAG: score gap lớn → nâng lên CORRECT")

        # Heuristic 3: Multi-source evidence
        # Nếu top result xuất hiện ở nhiều index → tin tưởng hơn
        if results[0].metadata.get("num_sources", 1) >= 3:
            if verdict == "AMBIGUOUS":
                verdict = "CORRECT"
                log_info("CRAG: multi-source → nâng lên CORRECT")

        # Gán metadata
        for r in results:
            r.metadata["crag_verdict"] = verdict
            r.metadata["crag_method"] = "fast_heuristic"

        # Log kết quả
        log_info(f"Fast CRAG: verdict={verdict}, top_score={top_score:.4f}")
        if verdict == "INCORRECT":
            log_warning(
                "CRAG verdict=INCORRECT: kết quả có thể không chính xác. "
                "Caller có thể thử reformulate query."
            )

        return verdict, results

    # ================================================================
    # CLEANUP
    # ================================================================

    def unload(self):
        """Giải phóng reranker model khỏi VRAM."""
        if self._reranker is not None:
            del self._reranker
            self._reranker = None
            self._reranker_loaded = False
            free_vram()
            log_success("Reranker đã được giải phóng")
            log_vram("Sau unload reranker")
