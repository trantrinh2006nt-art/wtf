"""
indexer.py — Build và Load tất cả FAISS indexes cho hệ thống VER.

Chức năng:
- Build 5 loại index từ SegmentSchema:
  1. Sparse Text (BM25) — tìm kiếm từ khóa truyền thống
  2. Dense Text (FAISS + BGE-M3) — tìm kiếm ngữ nghĩa text
  3. Visual (FAISS + SigLIP2 embeddings) — tìm kiếm hình ảnh
  4. Audio (FAISS + CLAP embeddings) — tìm kiếm âm thanh
  5. Temporal (FAISS + InternVideo2 embeddings) — tìm kiếm hành động
- Load indexes đã build từ disk
- Save/Load SegmentSchema dưới dạng JSON

GPU: BGE-M3 (~1.5GB fp16) — load chỉ khi cần, giải phóng sau khi xong.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, format_time,
    extract_frame, extract_frames_batch, extract_video_clips,
    get_video_duration, tokenize_vietnamese,
)


class UnifiedIndexer:
    """
    Build và quản lý tất cả indexes cho hệ thống retrieval.

    Index types:
    - text_sparse: BM25 (rank_bm25) — dùng tokenize_vietnamese
    - text_dense: FAISS IndexFlatIP + BGE-M3 1024d
    - visual: FAISS IndexFlatIP + SigLIP2 768d
    - audio: FAISS IndexFlatIP + CLAP 512d
    - temporal: FAISS IndexFlatIP + InternVideo2 768d
    """

    def __init__(self):
        # Lazy load — chỉ load BGE-M3 khi cần encode text
        self.text_encoder = None
        self._text_encoder_loaded = False

    def _load_text_encoder(self):
        """
        Load BGE-M3 (BAAI/bge-m3) bằng SentenceTransformer.
        Dùng float16 để tiết kiệm VRAM (~1.5GB thay vì ~3GB).
        """
        if self._text_encoder_loaded and self.text_encoder is not None:
            return

        log_step("Indexer", "Đang load BGE-M3 text encoder...")
        log_vram("trước khi load BGE-M3")

        try:
            from sentence_transformers import SentenceTransformer

            self.text_encoder = SentenceTransformer(
                TEXT_EMBEDDING_MODEL,
                device=DEVICE,
            )

            # Chuyển sang float16 nếu có thể
            if USE_FP16 and DEVICE == "cuda":
                self.text_encoder.half()

            self._text_encoder_loaded = True
            log_success(f"BGE-M3 loaded trên {DEVICE}")
            log_vram("sau khi load BGE-M3")

        except Exception as e:
            log_error(f"Không thể load BGE-M3: {e}")
            self.text_encoder = None
            self._text_encoder_loaded = False

    def _unload_text_encoder(self):
        """Giải phóng BGE-M3 khỏi VRAM."""
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
            self._text_encoder_loaded = False
            free_vram()
            log_info("BGE-M3 đã được giải phóng")
            log_vram("sau khi unload BGE-M3")

    # ==========================================================
    # BUILD ALL INDEXES
    # ==========================================================

    @timer
    def build_all_indexes(self, segments: list[SegmentSchema]):
        """
        Build tất cả indexes từ danh sách segments.

        Thứ tự build được tối ưu cho VRAM:
        1. Sparse text (BM25) — không cần GPU
        2. Dense text (FAISS + BGE-M3) — cần GPU, unload sau khi xong
        3. Visual (FAISS) — chỉ copy embeddings, không cần model
        4. Audio (FAISS) — chỉ copy embeddings
        5. Temporal (FAISS) — chỉ copy embeddings
        """
        if not segments:
            log_warning("Không có segments để build index")
            return

        log_step("Indexer", f"Build indexes cho {len(segments)} segments")

        # 1. Sparse Text Index (BM25)
        self._build_sparse_text_index(segments)

        # 2. Dense Text Index (FAISS + BGE-M3)
        self._build_dense_text_index(segments)

        # 3. Visual Index (FAISS)
        self._build_visual_index(segments)

        # 4. Audio Index (FAISS)
        self._build_audio_index(segments)

        # 5. Temporal Index (FAISS)
        self._build_temporal_index(segments)

        log_success("✅ Build tất cả indexes hoàn tất!")

    # ==========================================================
    # 1. SPARSE TEXT INDEX (BM25)
    # ==========================================================

    @timer
    def _build_sparse_text_index(self, segments: list[SegmentSchema]):
        """
        Build BM25 index từ full text của mỗi segment.

        Lưu:
        - corpus.json: danh sách texts gốc
        - tokenized.json: danh sách texts đã tokenize
        - segment_ids.json: mapping index → segment_id
        """
        log_step("Indexer", "Building Sparse Text Index (BM25)...")

        # Tạo thư mục
        sparse_dir = Path(TEXT_SPARSE_INDEX_DIR)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        corpus = []
        tokenized_corpus = []
        segment_ids = []

        for seg in segments:
            full_text = seg.get_full_text().strip()
            if not full_text:
                # Vẫn thêm entry rỗng để giữ alignment
                corpus.append("")
                tokenized_corpus.append([])
                segment_ids.append(seg.segment_id)
                continue

            corpus.append(full_text)
            tokens = tokenize_vietnamese(full_text)
            tokenized_corpus.append(tokens)
            segment_ids.append(seg.segment_id)

        # Lưu ra file JSON
        with open(sparse_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=None)

        with open(sparse_dir / "tokenized.json", "w", encoding="utf-8") as f:
            json.dump(tokenized_corpus, f, ensure_ascii=False, indent=None)

        with open(sparse_dir / "segment_ids.json", "w", encoding="utf-8") as f:
            json.dump(segment_ids, f, ensure_ascii=False, indent=None)

        # Test BM25 có build được không
        non_empty = sum(1 for t in tokenized_corpus if t)
        log_success(
            f"BM25 index: {len(corpus)} entries ({non_empty} có text), "
            f"saved to {sparse_dir}"
        )

    # ==========================================================
    # 2. DENSE TEXT INDEX (FAISS + BGE-M3)
    # ==========================================================

    @timer
    def _build_dense_text_index(self, segments: list[SegmentSchema]):
        """
        Encode text bằng BGE-M3 → build FAISS IndexFlatIP.

        BGE-M3 output: 1024 chiều, đã L2 normalized → dùng Inner Product
        tương đương Cosine Similarity.
        """
        import faiss

        log_step("Indexer", "Building Dense Text Index (BGE-M3 + FAISS)...")

        # Chuẩn bị texts
        texts = []
        segment_ids = []
        valid_indices = []  # Chỉ encode segments có text

        for idx, seg in enumerate(segments):
            full_text = seg.get_full_text().strip()
            if full_text:
                texts.append(full_text)
                segment_ids.append(seg.segment_id)
                valid_indices.append(idx)

        if not texts:
            log_warning("Không có text nào để build dense index, bỏ qua")
            return

        # Load BGE-M3
        self._load_text_encoder()
        if self.text_encoder is None:
            log_error("BGE-M3 không khả dụng, bỏ qua dense text index")
            return

        # Encode texts theo batch
        log_info(f"Encoding {len(texts)} texts bằng BGE-M3...")
        try:
            embeddings = self.text_encoder.encode(
                texts,
                batch_size=128,  # Batch size tối ưu hơn cho text
                show_progress_bar=True,
                normalize_embeddings=True,  # BGE-M3 cần normalize
            )
            embeddings = np.array(embeddings, dtype=np.float32)

            # Verify dimensions
            if embeddings.shape[1] != TEXT_EMBEDDING_DIM:
                log_warning(
                    f"BGE-M3 output dim={embeddings.shape[1]}, "
                    f"expected {TEXT_EMBEDDING_DIM}"
                )

        except Exception as e:
            log_error(f"Lỗi encode text: {e}")
            self._unload_text_encoder()
            return

        # Unload BGE-M3 ngay sau khi encode xong — giải phóng VRAM
        self._unload_text_encoder()

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Lưu index
        faiss.write_index(index, TEXT_DENSE_INDEX_PATH)

        # Lưu metadata
        metadata = {
            "segment_ids": segment_ids,
            "dim": dim,
            "count": len(segment_ids),
            "texts": texts,  # Giữ text gốc để debug/display
        }
        with open(TEXT_DENSE_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_success(
            f"Dense text index: {index.ntotal} vectors × {dim}d → "
            f"{TEXT_DENSE_INDEX_PATH}"
        )

    # ==========================================================
    # 3. VISUAL INDEX (FAISS)
    # ==========================================================

    @timer
    def _build_visual_index(self, segments: list[SegmentSchema]):
        """
        Build FAISS index từ frame embeddings (SigLIP2 768d).
        Embeddings đã có sẵn trong SegmentSchema.frame_embedding.
        """
        import faiss

        log_step("Indexer", "Building Visual Index (SigLIP2 + FAISS)...")

        embeddings = []
        segment_ids = []

        for seg in segments:
            if seg.frame_embedding and len(seg.frame_embedding) > 0:
                embeddings.append(seg.frame_embedding)
                segment_ids.append(seg.segment_id)

        if not embeddings:
            log_warning("Không có frame embeddings, bỏ qua visual index")
            return

        # Chuyển sang numpy
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Verify dimensions
        dim = embeddings_np.shape[1]
        if dim != FRAME_EMBEDDING_DIM:
            log_warning(
                f"Frame embedding dim={dim}, expected {FRAME_EMBEDDING_DIM}"
            )

        # Build FAISS index
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)

        # Lưu index
        faiss.write_index(index, VISUAL_INDEX_PATH)

        # Lưu metadata
        metadata = {
            "segment_ids": segment_ids,
            "dim": dim,
            "count": len(segment_ids),
        }
        with open(VISUAL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_success(
            f"Visual index: {index.ntotal} vectors × {dim}d → "
            f"{VISUAL_INDEX_PATH}"
        )

    # ==========================================================
    # 4. AUDIO INDEX (FAISS)
    # ==========================================================

    @timer
    def _build_audio_index(self, segments: list[SegmentSchema]):
        """
        Build FAISS index từ audio embeddings (CLAP 512d).
        Embeddings đã có sẵn trong SegmentSchema.audio_embedding.
        """
        import faiss

        log_step("Indexer", "Building Audio Index (CLAP + FAISS)...")

        embeddings = []
        segment_ids = []

        for seg in segments:
            if seg.audio_embedding and len(seg.audio_embedding) > 0:
                embeddings.append(seg.audio_embedding)
                segment_ids.append(seg.segment_id)

        if not embeddings:
            log_warning("Không có audio embeddings, bỏ qua audio index")
            return

        # Chuyển sang numpy
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Verify dimensions
        dim = embeddings_np.shape[1]
        if dim != AUDIO_EMBEDDING_DIM:
            log_warning(
                f"Audio embedding dim={dim}, expected {AUDIO_EMBEDDING_DIM}"
            )

        # Build FAISS index
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)

        # Lưu index
        faiss.write_index(index, AUDIO_INDEX_PATH)

        # Lưu metadata
        metadata = {
            "segment_ids": segment_ids,
            "dim": dim,
            "count": len(segment_ids),
        }
        with open(AUDIO_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_success(
            f"Audio index: {index.ntotal} vectors × {dim}d → "
            f"{AUDIO_INDEX_PATH}"
        )

    # ==========================================================
    # 5. TEMPORAL INDEX (FAISS)
    # ==========================================================

    @timer
    def _build_temporal_index(self, segments: list[SegmentSchema]):
        """
        Build FAISS index từ video embeddings (InternVideo2 768d).
        Embeddings đã có sẵn trong SegmentSchema.video_embedding.
        """
        import faiss

        log_step("Indexer", "Building Temporal Index (InternVideo2 + FAISS)...")

        embeddings = []
        segment_ids = []

        for seg in segments:
            if seg.video_embedding and len(seg.video_embedding) > 0:
                embeddings.append(seg.video_embedding)
                segment_ids.append(seg.segment_id)

        if not embeddings:
            log_warning("Không có video embeddings, bỏ qua temporal index")
            return

        # Chuyển sang numpy
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Verify dimensions
        dim = embeddings_np.shape[1]
        if dim != VIDEO_EMBEDDING_DIM:
            log_warning(
                f"Video embedding dim={dim}, expected {VIDEO_EMBEDDING_DIM}"
            )

        # Build FAISS index
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_np)

        # Lưu index
        faiss.write_index(index, TEMPORAL_INDEX_PATH)

        # Lưu metadata
        metadata = {
            "segment_ids": segment_ids,
            "dim": dim,
            "count": len(segment_ids),
        }
        with open(TEMPORAL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_success(
            f"Temporal index: {index.ntotal} vectors × {dim}d → "
            f"{TEMPORAL_INDEX_PATH}"
        )

    # ==========================================================
    # LOAD ALL INDEXES
    # ==========================================================

    @timer
    def load_all_indexes(self) -> dict:
        """
        Load tất cả indexes đã build từ disk.

        Returns:
            Dict chứa tất cả indexes và metadata:
            {
                "text_sparse": {"corpus", "tokenized", "segment_ids", "bm25"},
                "text_dense": {"index", "segment_ids", "texts"},
                "visual": {"index", "segment_ids"},
                "audio": {"index", "segment_ids"},
                "temporal": {"index", "segment_ids"},
            }
        """
        log_step("Indexer", "Loading tất cả indexes từ disk...")
        result = {}

        # 1. Sparse Text (BM25)
        result["text_sparse"] = self._load_sparse_index()

        # 2. Dense Text (FAISS)
        result["text_dense"] = self._load_faiss_index(
            TEXT_DENSE_INDEX_PATH, TEXT_DENSE_META_PATH, "text_dense"
        )

        # 3. Visual (FAISS)
        result["visual"] = self._load_faiss_index(
            VISUAL_INDEX_PATH, VISUAL_META_PATH, "visual"
        )

        # 4. Audio (FAISS)
        result["audio"] = self._load_faiss_index(
            AUDIO_INDEX_PATH, AUDIO_META_PATH, "audio"
        )

        # 5. Temporal (FAISS)
        result["temporal"] = self._load_faiss_index(
            TEMPORAL_INDEX_PATH, TEMPORAL_META_PATH, "temporal"
        )

        # Thống kê
        loaded = [k for k, v in result.items() if v is not None]
        missing = [k for k, v in result.items() if v is None]

        if loaded:
            log_success(f"Loaded: {', '.join(loaded)}")
        if missing:
            log_warning(f"Missing: {', '.join(missing)}")

        return result

    def _load_sparse_index(self) -> Optional[dict]:
        """Load BM25 sparse index từ disk."""
        sparse_dir = Path(TEXT_SPARSE_INDEX_DIR)

        corpus_path = sparse_dir / "corpus.json"
        tokenized_path = sparse_dir / "tokenized.json"
        segment_ids_path = sparse_dir / "segment_ids.json"

        if not all(p.exists() for p in [corpus_path, tokenized_path, segment_ids_path]):
            log_warning("Sparse text index chưa được build")
            return None

        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            with open(tokenized_path, "r", encoding="utf-8") as f:
                tokenized = json.load(f)
            with open(segment_ids_path, "r", encoding="utf-8") as f:
                segment_ids = json.load(f)

            # Build BM25 object từ tokenized corpus
            from rank_bm25 import BM25Okapi

            # Lọc bỏ entries rỗng khi build BM25 nhưng giữ mapping
            # BM25 cần ít nhất 1 document có token
            non_empty_tokenized = [t if t else [""] for t in tokenized]
            bm25 = BM25Okapi(
                non_empty_tokenized,
                k1=BM25_K1,
                b=BM25_B,
            )

            log_success(f"BM25 loaded: {len(corpus)} documents")
            return {
                "corpus": corpus,
                "tokenized": tokenized,
                "segment_ids": segment_ids,
                "bm25": bm25,
            }

        except Exception as e:
            log_error(f"Lỗi load sparse index: {e}")
            return None

    def _load_faiss_index(
        self,
        index_path: str,
        meta_path: str,
        name: str,
    ) -> Optional[dict]:
        """Load một FAISS index + metadata từ disk."""
        if not os.path.exists(index_path):
            log_warning(f"{name} index chưa được build ({index_path})")
            return None

        try:
            import faiss

            index = faiss.read_index(index_path)

            # Load metadata
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            segment_ids = metadata.get("segment_ids", [])
            texts = metadata.get("texts", [])

            log_success(
                f"{name} index loaded: {index.ntotal} vectors × "
                f"{metadata.get('dim', '?')}d"
            )

            result = {
                "index": index,
                "segment_ids": segment_ids,
            }
            if texts:
                result["texts"] = texts

            return result

        except Exception as e:
            log_error(f"Lỗi load {name} index: {e}")
            return None

    # ==========================================================
    # SEGMENT I/O
    # ==========================================================

    @timer
    def save_segments(self, segments: list[SegmentSchema]):
        """
        Lưu danh sách SegmentSchema thành JSON.

        Lưu ý: embeddings KHÔNG được lưu cùng segments (quá lớn).
        Chúng được lưu riêng trong FAISS indexes.
        """
        log_step("Indexer", f"Saving {len(segments)} segments...")

        # Đảm bảo thư mục tồn tại
        Path(SEGMENTS_PATH).parent.mkdir(parents=True, exist_ok=True)

        data = []
        for seg in segments:
            d = seg.to_dict()
            # Thêm flag cho biết có embedding hay không (không lưu vector)
            d["has_frame_embedding"] = bool(seg.frame_embedding)
            d["has_video_embedding"] = bool(seg.video_embedding)
            d["has_audio_embedding"] = bool(seg.audio_embedding)
            data.append(d)

        with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log_success(f"Saved {len(data)} segments → {SEGMENTS_PATH}")

    @timer
    def load_segments(self) -> list[SegmentSchema]:
        """
        Load danh sách SegmentSchema từ JSON.

        Lưu ý: embeddings KHÔNG nằm trong JSON —
        chúng cần được load riêng từ FAISS indexes.
        """
        log_step("Indexer", "Loading segments từ disk...")

        if not os.path.exists(SEGMENTS_PATH):
            log_warning(f"Segments file chưa tồn tại: {SEGMENTS_PATH}")
            return []

        try:
            with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = []
            for d in data:
                seg = SegmentSchema.from_dict(d)
                segments.append(seg)

            log_success(f"Loaded {len(segments)} segments từ {SEGMENTS_PATH}")
            return segments

        except Exception as e:
            log_error(f"Lỗi load segments: {e}")
            return []
