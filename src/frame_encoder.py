"""
frame_encoder.py — SigLIP2 Frame Embedding Engine.

Chức năng:
- Encode frames video thành dense vectors (768d) bằng SigLIP2
- Encode text queries cho visual search (SigLIP2 text encoder)
- Batch processing với FRAME_BATCH_SIZE=16 để tối ưu VRAM
- L2 normalize embeddings cho cosine similarity search
- Lazy loading model (RTX 4060 8GB)
"""

import sys
import numpy as np
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

import torch
from PIL import Image


class FrameEncoder:
    """
    SigLIP2 Frame Encoder — Visual embedding cho frames video.
    
    Model: google/siglip2-base-patch16-224
    Output: 768-dimensional L2-normalized embeddings
    
    Hỗ trợ:
    - Encode batch frames → visual embeddings
    - Encode text query → text embedding (cùng không gian SigLIP2)
    - VRAM: ~600MB khi load
    """

    def __init__(self):
        """Khởi tạo — KHÔNG load model ở đây (lazy loading)."""
        self.model = None
        self.processor = None
        log_info("FrameEncoder khởi tạo (lazy — chưa load model)")

    def _load_model(self):
        """Lazy load SigLIP2 model và processor."""
        if self.model is not None:
            return  # Đã load rồi

        log_step("FrameEncoder", f"Đang load model {FRAME_ENCODER_MODEL}...")
        log_vram("trước khi load FrameEncoder")

        try:
            from transformers import AutoModel, AutoProcessor

            # Xác định dtype
            dtype = torch.float16 if USE_FP16 and CUDA_AVAILABLE else torch.float32

            # Load processor (tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(FRAME_ENCODER_MODEL, use_fast=True)

            # Load model với fp16
            self.model = AutoModel.from_pretrained(
                FRAME_ENCODER_MODEL,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(DEVICE)
            self.model.eval()

            log_success(
                f"SigLIP2 loaded trên {DEVICE} "
                f"(embedding dim={FRAME_EMBEDDING_DIM})"
            )
            log_vram("sau khi load FrameEncoder")

        except Exception as e:
            log_error(f"Không thể load SigLIP2: {e}")
            raise RuntimeError(f"FrameEncoder load thất bại: {e}")

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings theo hàng."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Tránh chia cho 0
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    @timer
    @torch.no_grad()
    def encode_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
    ) -> tuple[list[list[float]], list[float]]:
        """
        Encode frames từ video thành visual embeddings.
        
        Args:
            video_path: Đường dẫn video
            sample_rate: Số frames/giây cần trích xuất (default: 1fps)
            
        Returns:
            Tuple of:
            - embeddings: List[List[float]] — mỗi embedding 768d
            - timestamps: List[float] — timestamp tương ứng
        """
        log_step("FrameEncoder", f"Encoding frames: {Path(video_path).name}")

        # Bước 1: Load model nếu chưa load
        self._load_model()

        # Bước 2: Trích xuất frames
        try:
            frames, timestamps = extract_frames_batch(
                str(video_path),
                sample_rate=sample_rate,
            )
        except Exception as e:
            log_error(f"Không thể trích xuất frames: {e}")
            return [], []

        if not frames:
            log_warning(f"Không tìm thấy frame nào trong {video_path}")
            return [], []

        log_info(f"Đã trích xuất {len(frames)} frames @ {sample_rate}fps")

        # Bước 3: Encode theo batch
        all_embeddings = []
        batch_size = FRAME_BATCH_SIZE

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            try:
                # Xử lý ảnh qua processor
                inputs = self.processor(
                    images=batch_frames,
                    return_tensors="pt",
                    padding=True,
                )

                # Move inputs lên GPU
                inputs = {
                    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Chỉ lấy vision features — dùng vision_model
                # SigLIP2 có get_image_features() method
                image_features = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )

                # Chuyển về numpy
                batch_emb = image_features.cpu().float().numpy()
                all_embeddings.append(batch_emb)

            except Exception as e:
                log_warning(f"Lỗi batch {batch_start}-{batch_end}: {e}")
                # Tạo zero embeddings cho batch lỗi
                zero_emb = np.zeros(
                    (len(batch_frames), FRAME_EMBEDDING_DIM),
                    dtype=np.float32,
                )
                all_embeddings.append(zero_emb)

            # Log tiến trình
            if (batch_start // batch_size + 1) % 5 == 0:
                progress = min(batch_end, len(frames))
                log_info(f"  Encoded {progress}/{len(frames)} frames")

        # Bước 4: Gộp và normalize
        if not all_embeddings:
            return [], []

        embeddings_np = np.concatenate(all_embeddings, axis=0)
        embeddings_np = self._l2_normalize(embeddings_np)

        # Chuyển thành list[list[float]]
        embeddings_list = embeddings_np.tolist()

        log_success(
            f"Frame encoding hoàn thành: {len(embeddings_list)} embeddings × "
            f"{FRAME_EMBEDDING_DIM}d"
        )

        # Giải phóng frames khỏi bộ nhớ
        del frames
        free_vram()

        return embeddings_list, timestamps

    @torch.no_grad()
    def encode_text(self, text: str) -> list[float]:
        """
        Encode text query thành embedding trong không gian SigLIP2.
        Dùng cho visual text-to-image search.
        
        Args:
            text: Text query cần encode
            
        Returns:
            List[float] — embedding 768d, L2-normalized
        """
        if not text or not text.strip():
            log_warning("Text query rỗng — trả về zero vector")
            return [0.0] * FRAME_EMBEDDING_DIM

        # Load model nếu chưa
        self._load_model()

        try:
            # Tokenize text
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )

            # Move lên GPU
            inputs = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Lấy text features qua SigLIP2 text encoder
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

            # Normalize và chuyển về list
            emb = text_features.cpu().float().numpy().squeeze()
            emb = emb / max(np.linalg.norm(emb), 1e-8)

            return emb.tolist()

        except Exception as e:
            log_error(f"Lỗi encode text: {e}")
            return [0.0] * FRAME_EMBEDDING_DIM

    @torch.no_grad()
    def encode_single_frame(self, image: Image.Image) -> list[float]:
        """
        Encode một frame đơn lẻ (PIL Image).
        
        Args:
            image: PIL Image cần encode
            
        Returns:
            List[float] — embedding 768d, L2-normalized
        """
        if image is None:
            return [0.0] * FRAME_EMBEDDING_DIM

        self._load_model()

        try:
            inputs = self.processor(
                images=[image],
                return_tensors="pt",
            )
            inputs = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            image_features = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            emb = image_features.cpu().float().numpy().squeeze()
            emb = emb / max(np.linalg.norm(emb), 1e-8)
            return emb.tolist()

        except Exception as e:
            log_error(f"Lỗi encode single frame: {e}")
            return [0.0] * FRAME_EMBEDDING_DIM

    def unload(self):
        """Giải phóng model khỏi VRAM."""
        log_info("FrameEncoder: Đang giải phóng model...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        free_vram()
        log_success("FrameEncoder: Đã giải phóng VRAM")
