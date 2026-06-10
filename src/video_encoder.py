"""
video_encoder.py — Trích xuất đặc trưng video bằng InternVideo2.

Chức năng:
- Encode video clips thành embeddings 768 chiều (InternVideo2-Stage2_1B)
- Fallback sang SigLIP2 (frame giữa mỗi clip) nếu InternVideo2 không load được
- Quản lý VRAM chặt chẽ: lazy load, float16, giải phóng sau khi dùng

GPU: RTX 4060 8GB → InternVideo2 ~4GB, phải unload trước khi dùng model khác.
"""

import sys
import torch
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


class VideoEncoder:
    """
    Trích xuất video embeddings bằng InternVideo2 (action recognition).

    Chiến lược:
    1. Thử load InternVideo2-Stage2_1B-224p-f4 (~4GB VRAM ở fp16)
    2. Nếu thất bại → fallback sang SigLIP2 (encode frame giữa mỗi clip)
    3. Nếu cả hai đều thất bại → trả về empty list (không crash pipeline)
    """

    def __init__(self):
        # Lazy load — chưa load model khi khởi tạo
        self.model = None
        self.processor = None
        self.available = False        # InternVideo2 sẵn sàng?
        self.fallback_mode = False    # Đang dùng SigLIP2 thay thế?
        self._fallback_encoder = None # FrameEncoder instance (nếu fallback)

    def _load_model(self):
        """
        Load InternVideo2 lên GPU với float16.
        Nếu thất bại (thiếu thư viện, hết VRAM, lỗi tải) → chuyển sang fallback.
        """
        if self.available or self.fallback_mode:
            return

        log_step("VideoEncoder", "Đang load InternVideo2...")
        log_vram("trước khi load InternVideo2")

        try:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                VIDEO_ENCODER_MODEL,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                VIDEO_ENCODER_MODEL,
                dtype=torch.float16 if USE_FP16 else torch.float32,
                trust_remote_code=True,
            ).to(DEVICE).eval()

            self.available = True
            log_success(f"InternVideo2 loaded thành công trên {DEVICE}")
            log_vram("sau khi load InternVideo2")

        except Exception as e:
            log_warning(f"Không thể load InternVideo2: {e}")
            log_info("Chuyển sang fallback SigLIP2 (encode frame giữa mỗi clip)")
            self.model = None
            self.processor = None
            self.available = False
            self._init_fallback()

    def _init_fallback(self):
        """
        Khởi tạo fallback SigLIP2 encoder.
        Import FrameEncoder từ offline package — nếu chưa có thì tự build mini encoder.
        """
        try:
            from offline.frame_encoder import FrameEncoder
            self._fallback_encoder = FrameEncoder()
            self.fallback_mode = True
            log_success("Fallback FrameEncoder (SigLIP2) sẵn sàng")
        except ImportError:
            # FrameEncoder chưa được tạo — tự build mini SigLIP2 encoder
            try:
                self._build_mini_siglip_fallback()
                self.fallback_mode = True
                log_success("Fallback mini SigLIP2 encoder sẵn sàng")
            except Exception as e2:
                log_error(f"Không thể tạo fallback encoder: {e2}")
                self.fallback_mode = False

    def _build_mini_siglip_fallback(self):
        """
        Tự build SigLIP2 encoder đơn giản nếu FrameEncoder chưa tồn tại.
        Nhẹ hơn nhiều so với InternVideo2 (~400MB vs ~4GB).
        """
        from transformers import AutoModel, AutoProcessor

        log_step("VideoEncoder", "Đang load SigLIP2 cho fallback...")
        self._fallback_processor = AutoProcessor.from_pretrained(
            FRAME_ENCODER_MODEL,
            trust_remote_code=True,
        )
        self._fallback_model = AutoModel.from_pretrained(
            FRAME_ENCODER_MODEL,
            dtype=torch.float16 if USE_FP16 else torch.float32,
            trust_remote_code=True,
        ).to(DEVICE).eval()

        log_vram("sau khi load SigLIP2 fallback")

    @timer
    @torch.no_grad()
    def encode_clips(
        self, video_path: str
    ) -> tuple[list[list[float]], list[tuple[float, float]]]:
        """
        Encode tất cả clips từ một video.

        Args:
            video_path: Đường dẫn đến file video

        Returns:
            (embeddings, clip_time_ranges):
            - embeddings: list các vector 768 chiều (đã L2 normalize)
            - clip_time_ranges: list các (start_sec, end_sec) tương ứng
        """
        # Đảm bảo model đã được load
        self._load_model()

        # Nếu không có model nào khả dụng → trả về empty
        if not self.available and not self.fallback_mode:
            log_warning("Không có video encoder khả dụng, trả về empty")
            return [], []

        try:
            # Bước 1: Trích xuất clips từ video
            log_info(f"Đang trích xuất clips từ {Path(video_path).name}...")
            clips, clip_times = extract_video_clips(
                video_path, clip_length=VIDEO_CLIP_LENGTH
            )

            if not clips:
                log_warning(f"Không trích xuất được clip nào từ {video_path}")
                return [], []

            log_info(f"Trích xuất được {len(clips)} clips")

            # Bước 2: Encode tùy theo mode
            if self.available:
                embeddings = self._encode_with_internvideo(clips)
            else:
                embeddings = self._encode_with_fallback(clips)

            log_success(
                f"Encode xong {len(embeddings)}/{len(clips)} clips "
                f"→ {VIDEO_EMBEDDING_DIM}d vectors"
            )
            return embeddings, clip_times

        except Exception as e:
            log_error(f"Lỗi khi encode video {video_path}: {e}")
            return [], []

    def _encode_with_internvideo(
        self, clips: list[list]
    ) -> list[list[float]]:
        """
        Encode clips bằng InternVideo2.

        Mỗi clip là list 4 PIL Images → processor → model → embedding 768d.
        Xử lý theo batch nhỏ để tránh OOM trên 8GB VRAM.
        """
        embeddings = []
        batch_size = max(1, FRAME_BATCH_SIZE // 4)  # Mỗi clip có 4 frames

        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i : i + batch_size]

            try:
                # InternVideo2 nhận video frames dưới dạng tensor
                inputs = self.processor(
                    videos=batch_clips,
                    return_tensors="pt",
                ).to(DEVICE)

                # Forward pass
                outputs = self.model(**inputs)

                # Lấy video embedding — tùy kiến trúc model
                if hasattr(outputs, "video_embeds"):
                    batch_embeds = outputs.video_embeds
                elif hasattr(outputs, "pooler_output"):
                    batch_embeds = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # Global average pooling trên sequence
                    batch_embeds = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback: lấy attribute đầu tiên có shape phù hợp
                    batch_embeds = self._extract_embedding_from_output(outputs)

                # L2 normalize
                batch_embeds = torch.nn.functional.normalize(
                    batch_embeds.float(), p=2, dim=-1
                )

                # Chuyển về CPU list
                for emb in batch_embeds:
                    vec = emb.cpu().numpy().tolist()
                    # Kiểm tra dimension
                    if len(vec) != VIDEO_EMBEDDING_DIM:
                        log_warning(
                            f"Embedding dim {len(vec)} != expected {VIDEO_EMBEDDING_DIM}, "
                            f"sẽ pad/truncate"
                        )
                        vec = self._adjust_dim(vec, VIDEO_EMBEDDING_DIM)
                    embeddings.append(vec)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_warning(f"OOM tại batch {i}, giảm batch size và thử lại")
                    free_vram()
                    # Encode từng clip riêng
                    for clip in batch_clips:
                        single_emb = self._encode_single_clip_internvideo(clip)
                        if single_emb is not None:
                            embeddings.append(single_emb)
                else:
                    log_error(f"Lỗi encode batch {i}: {e}")

        return embeddings

    def _encode_single_clip_internvideo(
        self, clip: list
    ) -> Optional[list[float]]:
        """Encode 1 clip đơn lẻ bằng InternVideo2 (dùng khi OOM batch)."""
        try:
            free_vram()
            inputs = self.processor(
                videos=[clip],
                return_tensors="pt",
            ).to(DEVICE)

            outputs = self.model(**inputs)

            if hasattr(outputs, "video_embeds"):
                emb = outputs.video_embeds[0]
            elif hasattr(outputs, "pooler_output"):
                emb = outputs.pooler_output[0]
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)[0]
            else:
                emb = self._extract_embedding_from_output(outputs)[0]

            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            vec = emb.cpu().numpy().tolist()
            vec = self._adjust_dim(vec, VIDEO_EMBEDDING_DIM)
            return vec

        except Exception as e:
            log_error(f"Lỗi encode single clip: {e}")
            return None

    def _encode_with_fallback(
        self, clips: list[list]
    ) -> list[list[float]]:
        """
        Fallback: encode frame giữa mỗi clip bằng SigLIP2.

        Chiến lược: lấy frame thứ 2 (giữa clip 4 frames) để đại diện clip.
        Output vẫn là 768d (FRAME_EMBEDDING_DIM == VIDEO_EMBEDDING_DIM).
        """
        embeddings = []

        for clip_idx, clip_frames in enumerate(clips):
            try:
                # Lấy frame giữa clip (index 1 hoặc 2 trong 4 frames)
                mid_idx = len(clip_frames) // 2
                mid_frame = clip_frames[mid_idx]

                if self._fallback_encoder is not None:
                    # Dùng FrameEncoder nếu có
                    emb = self._fallback_encoder.encode_single_frame(mid_frame)
                    if emb is not None:
                        embeddings.append(emb)
                    continue

                # Dùng mini SigLIP2 encoder tự build
                if hasattr(self, "_fallback_model") and self._fallback_model is not None:
                    inputs = self._fallback_processor(
                        images=mid_frame,
                        return_tensors="pt",
                    ).to(DEVICE)

                    outputs = self._fallback_model(**inputs)

                    # SigLIP2 trả về pooler_output hoặc last_hidden_state
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        emb = outputs.pooler_output[0]
                    else:
                        emb = outputs.last_hidden_state[:, 0, :][0]

                    emb = torch.nn.functional.normalize(
                        emb.float(), p=2, dim=-1
                    )
                    vec = emb.cpu().numpy().tolist()
                    vec = self._adjust_dim(vec, VIDEO_EMBEDDING_DIM)
                    embeddings.append(vec)

            except Exception as e:
                log_warning(f"Lỗi encode fallback clip {clip_idx}: {e}")

        return embeddings

    def _extract_embedding_from_output(self, outputs) -> torch.Tensor:
        """
        Trích xuất embedding từ output model khi không biết chính xác cấu trúc.
        Duyệt qua các attributes phổ biến để tìm tensor phù hợp.
        """
        # Thử các attribute phổ biến theo thứ tự ưu tiên
        candidates = [
            "video_embeds", "image_embeds", "pooler_output",
            "logits", "last_hidden_state",
        ]
        for attr in candidates:
            if hasattr(outputs, attr):
                val = getattr(outputs, attr)
                if val is not None and isinstance(val, torch.Tensor):
                    if val.dim() == 2:
                        return val
                    elif val.dim() == 3:
                        return val.mean(dim=1)

        # Cuối cùng, thử iterate qua outputs nếu nó là tuple/list
        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.dim() >= 2:
                    if item.dim() == 2:
                        return item
                    return item.mean(dim=1)

        raise ValueError("Không thể trích xuất embedding từ model output")

    @staticmethod
    def _adjust_dim(vec: list[float], target_dim: int) -> list[float]:
        """Pad hoặc truncate vector về target_dim."""
        if len(vec) == target_dim:
            return vec
        elif len(vec) > target_dim:
            return vec[:target_dim]
        else:
            # Zero-pad
            return vec + [0.0] * (target_dim - len(vec))

    def unload(self):
        """Giải phóng model khỏi VRAM."""
        log_step("VideoEncoder", "Đang unload model...")

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Unload fallback nếu có
        if self._fallback_encoder is not None:
            if hasattr(self._fallback_encoder, "unload"):
                self._fallback_encoder.unload()
            del self._fallback_encoder
            self._fallback_encoder = None

        if hasattr(self, "_fallback_model") and self._fallback_model is not None:
            del self._fallback_model
            self._fallback_model = None
        if hasattr(self, "_fallback_processor") and self._fallback_processor is not None:
            del self._fallback_processor
            self._fallback_processor = None

        self.available = False
        self.fallback_mode = False

        free_vram()
        log_success("VideoEncoder đã được giải phóng")
        log_vram("sau khi unload VideoEncoder")
