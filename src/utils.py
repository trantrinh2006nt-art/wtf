"""
utils.py — Cấu trúc dữ liệu chuẩn và tiện ích cho Pipeline Lai Tạo VER.

Core dataclasses:
- SegmentSchema: Unified segment chứa tất cả thông tin đa phương thức
- AnalyzedQuery: Query đã được LLM phân rã thành sub-queries
- SearchResult: Kết quả tìm kiếm chuẩn hóa
"""

import cv2
import gc
import re
import sys
import time
import torch
import functools
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


def configure_stdio(encoding: str = "utf-8"):
    """Use UTF-8 console output so Windows code pages do not crash on Vietnamese text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding=encoding, errors="replace")
            except Exception:
                pass


configure_stdio()


# ============================================================
# 1. CẤU TRÚC DỮ LIỆU — UNIFIED SEGMENT SCHEMA
# ============================================================

@dataclass
class SegmentSchema:
    """
    Schema thống nhất cho mỗi đoạn video — gộp tất cả 5 "giác quan".
    Đây là đơn vị cơ bản của hệ thống, mỗi segment ~1-5 giây.
    """
    segment_id: str = ""                    # ID duy nhất: "video01_00042"
    video_file: str = ""                    # Tên file video gốc
    start_time: float = 0.0                 # Thời điểm bắt đầu (giây)
    end_time: float = 0.0                   # Thời điểm kết thúc (giây)

    # --- Giác quan 1: ASR Transcript ---
    asr_text: str = ""                      # Nội dung phiên âm (PhoWhisper)

    # --- Giác quan 2: Audio Events ---
    audio_events: list = field(default_factory=list)  # ["tiếng súng", "còi xe"]
    audio_embedding: list = field(default_factory=list)  # CLAP vector

    # --- Giác quan 3: OCR Text ---
    ocr_texts: list = field(default_factory=list)  # ["BREAKING NEWS", "Hà Nội"]

    # --- Giác quan 4: Video Actions ---
    action_labels: list = field(default_factory=list)  # ["running", "fighting"]
    video_embedding: list = field(default_factory=list)  # InternVideo2 vector

    # --- Giác quan 5: Frame Visual ---
    frame_embedding: list = field(default_factory=list)  # SigLIP2 vector
    keyframe_path: str = ""                 # Đường dẫn keyframe đã lưu

    def get_full_text(self) -> str:
        """Gộp tất cả text sources: ASR + OCR + Actions."""
        parts = []
        if self.asr_text:
            parts.append(self.asr_text)
        if self.ocr_texts:
            parts.append(" ".join(self.ocr_texts))
        if self.action_labels:
            parts.append(" ".join(self.action_labels))
        if self.audio_events:
            parts.append(" ".join(self.audio_events))
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "video_file": self.video_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "asr_text": self.asr_text,
            "audio_events": self.audio_events,
            "ocr_texts": self.ocr_texts,
            "action_labels": self.action_labels,
            "keyframe_path": self.keyframe_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentSchema":
        return cls(
            segment_id=d.get("segment_id", ""),
            video_file=d.get("video_file", ""),
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
            asr_text=d.get("asr_text", ""),
            audio_events=d.get("audio_events", []),
            ocr_texts=d.get("ocr_texts", []),
            action_labels=d.get("action_labels", []),
            keyframe_path=d.get("keyframe_path", ""),
        )


# ============================================================
# 2. CẤU TRÚC DỮ LIỆU — ANALYZED QUERY
# ============================================================

@dataclass
class AnalyzedQuery:
    """
    Query đã được LLM phân rã thành các sub-queries theo modality.
    Output của Tầng 1 (Query Analyzer).
    """
    original_query: str = ""                # Query gốc từ người dùng
    visual_query: str = ""                  # Sub-query cho visual search
    temporal_query: str = ""                # Sub-query cho action/temporal search
    audio_query: str = ""                   # Sub-query cho audio event search
    ocr_query: str = ""                     # Sub-query cho OCR text search
    text_query: str = ""                    # Sub-query cho ASR text search
    entities: list = field(default_factory=list)  # Thực thể phát hiện được
    entity_knowledge: str = ""              # Tri thức bổ sung từ Entity Grounding
    hyde_document: str = ""                 # Hypothetical document (HyDE)
    query_type: str = "general"             # "entity_specific" | "action" | "audio" | "general"


# ============================================================
# 3. CẤU TRÚC DỮ LIỆU — SEARCH RESULT
# ============================================================

@dataclass
class SearchResult:
    """
    Kết quả tìm kiếm chuẩn hóa. Mọi engine đều trả về list[SearchResult].
    """
    segment_id: str = ""                    # Tham chiếu đến SegmentSchema
    video_file: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""                          # ASR + OCR text
    score: float = 0.0                      # Điểm tổng hợp (0.0 - 1.0)
    source: str = ""                        # "text_dense", "text_sparse", "visual", "audio", "fused"
    # Điểm thành phần
    dense_score: float = 0.0
    sparse_score: float = 0.0
    visual_score: float = 0.0
    audio_score: float = 0.0
    temporal_score: float = 0.0
    rerank_score: float = 0.0
    # Metadata bổ sung
    metadata: dict = field(default_factory=dict)
    keyframe_path: str = ""

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "video_file": self.video_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "time_display": format_time(self.start_time),
            "keyframe_path": self.keyframe_path,
        }


# ============================================================
# 4. VRAM MANAGEMENT — Quan trọng cho RTX 4060 8GB
# ============================================================

def free_vram():
    """Giải phóng VRAM bằng cách chạy garbage collection + empty cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_usage() -> dict:
    """Trả về thông tin sử dụng VRAM hiện tại."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
    }


def log_vram(label: str = ""):
    """Log VRAM usage ra console."""
    info = get_vram_usage()
    if info.get("available"):
        log_info(
            f"VRAM {label}: {info['allocated_mb']:.0f}MB / "
            f"{info['total_mb']:.0f}MB allocated"
        )


# ============================================================
# 5. LOGGING CÓ MÀU SẮC
# ============================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def log_step(step_name: str, message: str = ""):
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ [{step_name}]{Colors.END} {message}")

def log_success(message: str):
    print(f"  {Colors.GREEN}✓{Colors.END} {message}")

def log_warning(message: str):
    print(f"  {Colors.YELLOW}⚠{Colors.END} {message}")

def log_error(message: str):
    print(f"  {Colors.RED}✗{Colors.END} {message}")

def log_info(message: str):
    print(f"  {Colors.DIM}ℹ{Colors.END} {message}")

def log_result(rank: int, result: SearchResult):
    time_str = format_time(result.start_time)
    print(f"  {Colors.BOLD}#{rank}{Colors.END} "
          f"[{Colors.YELLOW}{time_str}{Colors.END}] "
          f"Score: {Colors.GREEN}{result.score:.4f}{Colors.END} "
          f"({result.source})")
    if result.text:
        display = result.text[:120] + "..." if len(result.text) > 120 else result.text
        print(f"     {Colors.DIM}{display}{Colors.END}")


# ============================================================
# 6. FORMAT THỜI GIAN
# ============================================================

def format_time(seconds: float) -> str:
    if seconds < 0:
        return "00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ============================================================
# 7. VIDEO FRAME EXTRACTION
# ============================================================

def extract_frame(video_path: str, timestamp_sec: float) -> Optional[Image.Image]:
    """Cắt 1 khung hình từ video tại timestamp. Trả về PIL Image."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None
    except Exception:
        return None


def extract_frames_batch(video_path: str, sample_rate: int = 1):
    """Trích xuất frames từ video theo sample_rate (frames/giây)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"FPS không hợp lệ: {fps}")

    frame_interval = max(1, int(fps / sample_rate))
    frames, timestamps = [], []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
            timestamps.append(count / fps)
        count += 1

    cap.release()
    return frames, timestamps


def extract_video_clips(video_path: str, clip_length: float = 4.0, stride: float = 2.0):
    """
    Trích xuất video clips cho InternVideo2.
    Mỗi clip dài clip_length giây, bước nhảy stride giây.
    Trả về (list of frame_lists, list of (start, end) timestamps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    clips = []
    clip_times = []
    num_frames_per_clip = 4  # InternVideo2 thường dùng 4 hoặc 8 frames

    t = 0.0
    while t + clip_length <= duration:
        clip_frames = []
        # Sample num_frames_per_clip frames đều trong clip
        for i in range(num_frames_per_clip):
            frame_time = t + (i * clip_length / num_frames_per_clip)
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ret, frame = cap.read()
            if ret:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                clip_frames.append(pil_img)

        if len(clip_frames) == num_frames_per_clip:
            clips.append(clip_frames)
            clip_times.append((t, t + clip_length))

        t += stride

    cap.release()
    return clips, clip_times


def get_video_duration(video_path: str) -> float:
    """Trả về thời lượng video (giây)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0.0


# ============================================================
# 8. TIMER DECORATOR
# ============================================================

def timer(func):
    """Decorator đo thời gian thực thi."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log_info(f"{func.__name__} hoàn thành trong {elapsed:.2f}s")
        return result
    return wrapper


# ============================================================
# 9. TEXT PROCESSING
# ============================================================

def tokenize_vietnamese(text: str) -> list[str]:
    """Tách từ cơ bản cho tiếng Việt."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()
