"""
ocr_engine.py — OCR Engine cho Video (Sử dụng EasyOCR).

Chức năng:
- Phát hiện và nhận dạng text trên video frames bằng EasyOCR (hỗ trợ tiếng Việt).
- Tự động tương thích với PyTorch và tối ưu chạy trên GPU.
- Trích xuất frames theo sample_rate (mặc định 0.5fps = 1 frame mỗi 2 giây).
- Lazy loading models.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Thêm thư mục gốc vào sys.path để import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, format_time,
    extract_frame, extract_frames_batch, extract_video_clips,
    get_video_duration, tokenize_vietnamese,
)

# Kiểm tra thư viện EasyOCR có sẵn không
_EASYOCR_AVAILABLE = False
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    pass


class OCREngine:
    """
    OCR Engine — Phát hiện và nhận dạng text trên video frames bằng EasyOCR.
    
    Fallback:
    - Nếu không có EasyOCR → trả về kết quả rỗng + warning
    """

    # Cấu hình
    MIN_TEXT_CONFIDENCE = 0.35  # Ngưỡng confidence tối thiểu (EasyOCR đôi khi confidence thấp nhưng text vẫn đúng)

    def __init__(self):
        """Khởi tạo — KHÔNG load model ở đây (lazy loading)."""
        self.reader = None
        self._easyocr_available = _EASYOCR_AVAILABLE
        self._loaded = False
        self._mode = "none"         # "easyocr", "none"
        log_info("OCREngine khởi tạo (lazy — chưa load model)")

    def _load_models(self):
        """
        Lazy load EasyOCR model.
        """
        if self._loaded:
            return

        if not self._easyocr_available:
            log_warning(
                "Không tìm thấy thư viện EasyOCR! "
                "Vui lòng cài đặt: uv pip install easyocr"
            )
            self._mode = "none"
            self._loaded = True
            return

        log_step("OCR", "Đang load EasyOCR models...")
        
        try:
            # Khởi tạo EasyOCR, hỗ trợ tiếng Việt ('vi') và tiếng Anh ('en')
            # gpu=CUDA_AVAILABLE sẽ tự động bắt GPU RTX 4060 nếu PyTorch thấy GPU
            self.reader = easyocr.Reader(['vi', 'en'], gpu=CUDA_AVAILABLE)
            
            self._mode = "easyocr"
            log_success(f"EasyOCR loaded thành công (GPU: {CUDA_AVAILABLE})!")
        except Exception as e:
            log_warning(f"Không thể load EasyOCR: {e}")
            self.reader = None
            self._mode = "none"

        self._loaded = True

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> dict | None:
        """
        Xử lý một frame bằng EasyOCR.
        
        Returns:
            Dict {timestamp, texts} hoặc None nếu không có text
        """
        if self._mode != "easyocr" or self.reader is None:
            return None

        texts = []

        try:
            # Chuyển BGR (OpenCV mặc định) sang RGB vì EasyOCR tối ưu với RGB hơn
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # EasyOCR trả về list các tuple: (bbox, text, confidence)
            results = self.reader.readtext(frame_rgb)
            
            for bbox, text, conf in results:
                if conf >= self.MIN_TEXT_CONFIDENCE:
                    # Text từ EasyOCR đã là string
                    text_clean = str(text).strip()
                    if text_clean:
                        texts.append(text_clean)

        except Exception as e:
            log_warning(f"Lỗi khi đọc text tại timestamp {timestamp:.1f}: {e}")
            return None

        # Lọc text trùng lặp và quá ngắn
        texts = self._filter_texts(texts)

        if not texts:
            return None

        return {
            "timestamp": round(timestamp, 2),
            "texts": texts,
        }

    @staticmethod
    def _filter_texts(texts: list[str]) -> list[str]:
        """Lọc và deduplicate OCR texts."""
        if not texts:
            return []

        filtered = []
        seen = set()

        for text in texts:
            text = text.strip()

            # Bỏ text quá ngắn (< 2 ký tự)
            if len(text) < 2:
                continue

            # Bỏ text chỉ có số hoặc ký tự đặc biệt
            # (thường là noise từ OCR)
            alpha_count = sum(1 for c in text if c.isalpha())
            if alpha_count == 0 and len(text) < 4:
                continue

            # Deduplicate (case-insensitive)
            text_lower = text.lower()
            if text_lower in seen:
                continue
            seen.add(text_lower)

            filtered.append(text)

        return filtered

    @timer
    def extract_text_from_frames(
        self,
        video_path: str,
        sample_rate: float = 0.5,
    ) -> list[dict]:
        """
        Trích xuất text từ video frames.
        
        Args:
            video_path: Đường dẫn video
            sample_rate: Số frames/giây (0.5 = 1 frame mỗi 2 giây)
            
        Returns:
            List[dict] với mỗi dict chứa:
            - timestamp: float
            - texts: List[str]
        """
        log_step("OCR", f"Trích xuất text: {Path(video_path).name}")

        # Bước 1: Load models
        self._load_models()

        # Kiểm tra mode
        if self._mode == "none":
            log_warning(
                "Không có OCR engine nào — trả về kết quả rỗng. "
                "Cài đặt: uv pip install easyocr"
            )
            return []

        # Bước 2: Trích xuất frames
        results = []
        try:
            # Dùng OpenCV trực tiếp vì sample_rate có thể < 1
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                log_error(f"Không thể mở video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                log_error(f"FPS không hợp lệ: {fps}")
                cap.release()
                return []

            # Tính frame interval từ sample_rate
            # sample_rate=0.5 → 1 frame mỗi 2 giây → interval = fps * 2
            frame_interval = max(1, int(fps / sample_rate))
            expected_frames = total_frames // frame_interval
            log_info(
                f"Video: {total_frames / fps:.1f}s @ {fps:.0f}fps → "
                f"~{expected_frames} frames (sample_rate={sample_rate}fps)"
            )

            frame_count = 0
            processed_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps

                    try:
                        result = self._process_frame(frame, timestamp)
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        log_warning(
                            f"Lỗi OCR frame {processed_count} "
                            f"(t={timestamp:.1f}s): {e}"
                        )

                    processed_count += 1

                    # Log tiến trình
                    if processed_count % 20 == 0:
                        log_info(
                            f"  Đã xử lý {processed_count}/{expected_frames} frames, "
                            f"tìm thấy {len(results)} frames có text"
                        )

                frame_count += 1

            cap.release()

            # Thống kê
            total_texts = sum(len(r["texts"]) for r in results)
            log_success(
                f"OCR hoàn thành: {processed_count} frames xử lý, "
                f"{len(results)} frames có text, "
                f"{total_texts} đoạn text tổng cộng"
            )

        except Exception as e:
            log_error(f"Lỗi OCR {Path(video_path).name}: {e}")

        return results

    def unload(self):
        """Giải phóng models."""
        log_info("OCREngine: Đang giải phóng models...")

        if self.reader is not None:
            del self.reader
            self.reader = None

        self._loaded = False
        self._mode = "none"

        free_vram()
        log_success("OCREngine: Đã giải phóng bộ nhớ")