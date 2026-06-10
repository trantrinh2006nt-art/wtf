"""
temporal_align.py — Căn chỉnh 5 luồng dữ liệu lên cùng trục thời gian.

Chức năng:
- Tạo lưới thời gian thống nhất (mỗi ô = segment_duration giây)
- Gộp ASR, Audio Events, OCR, Video Clips, Frame data vào từng segment
- Xử lý overlap/gap giữa các nguồn dữ liệu bất đồng bộ
- Output: list[SegmentSchema] — sẵn sàng cho indexing

Đây là bước QUAN TRỌNG nhất trong offline pipeline:
biến 5 dòng dữ liệu riêng lẻ thành 1 dòng thống nhất.
"""

import sys
import math
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, format_time,
    extract_frame, extract_frames_batch, extract_video_clips,
    get_video_duration, tokenize_vietnamese,
)


class TemporalAligner:
    """
    Căn chỉnh 5 luồng dữ liệu lên cùng trục thời gian thống nhất.

    Ý tưởng: chia video thành các segment đều nhau (mặc định 1 giây),
    rồi gán dữ liệu từ mỗi "giác quan" vào segment tương ứng dựa trên
    overlap thời gian.

    Input (5 luồng):
    1. asr_segments: [{"start": float, "end": float, "text": str}, ...]
    2. audio_events: [{"start": float, "end": float, "event": str, "embedding": list}, ...]
    3. ocr_data: [{"timestamp": float, "texts": list[str]}, ...]
    4. video_clips: (embeddings: list[list[float]], times: list[tuple[float,float]])
    5. frame_data: (embeddings: list[list[float]], timestamps: list[float])

    Output: list[SegmentSchema]
    """

    def __init__(self, segment_duration: float = 1.0):
        """
        Args:
            segment_duration: Độ dài mỗi segment (giây).
                - 1.0s → chi tiết cao, nhiều segments
                - 2.0s → cân bằng
                - Nên dùng 1.0s cho video ngắn, 2.0s cho video dài
        """
        self.segment_duration = max(0.1, segment_duration)  # Tối thiểu 0.1s

    @timer
    def align(
        self,
        video_path: str,
        duration: float,
        asr_segments: list[dict] | None = None,
        audio_events: list[dict] | None = None,
        ocr_data: list[dict] | None = None,
        video_clips: tuple[list, list] | None = None,
        frame_data: tuple[list, list] | None = None,
    ) -> list[SegmentSchema]:
        """
        Căn chỉnh tất cả dữ liệu lên lưới thời gian thống nhất.

        Args:
            video_path: Tên file video (ví dụ "video_001.mp4")
            duration: Thời lượng video (giây)
            asr_segments: Kết quả ASR — list dict có start, end, text
            audio_events: Kết quả audio analysis — list dict có start, end, event, embedding
            ocr_data: Kết quả OCR — list dict có timestamp, texts
            video_clips: Tuple (embeddings, clip_times) từ VideoEncoder
            frame_data: Tuple (embeddings, timestamps) từ FrameEncoder

        Returns:
            list[SegmentSchema]: Các segment đã căn chỉnh xong
        """
        # Chuẩn hóa input — tránh NoneType errors
        asr_segments = asr_segments or []
        audio_events = audio_events or []
        ocr_data = ocr_data or []

        video_embeddings, video_times = ([], [])
        if video_clips and len(video_clips) == 2:
            video_embeddings, video_times = video_clips

        frame_embeddings, frame_timestamps = ([], [])
        if frame_data and len(frame_data) == 2:
            frame_embeddings, frame_timestamps = frame_data

        # Tạo video stem cho segment_id
        video_stem = Path(video_path).stem

        # Tính số segment
        if duration <= 0:
            log_warning(f"Duration <= 0 cho {video_path}, bỏ qua")
            return []

        num_segments = math.ceil(duration / self.segment_duration)
        log_step(
            "TemporalAligner",
            f"Chia {video_path} ({format_time(duration)}) thành "
            f"{num_segments} segments × {self.segment_duration}s"
        )

        # Log thống kê input
        log_info(
            f"Input: ASR={len(asr_segments)} | Audio={len(audio_events)} | "
            f"OCR={len(ocr_data)} | Video={len(video_embeddings)} clips | "
            f"Frame={len(frame_embeddings)} frames"
        )

        segments = []

        for i in range(num_segments):
            t_start = i * self.segment_duration
            t_end = min((i + 1) * self.segment_duration, duration)

            segment = SegmentSchema(
                segment_id=f"{video_stem}_{int(t_start):05d}",
                video_file=Path(video_path).name,
                start_time=round(t_start, 3),
                end_time=round(t_end, 3),
            )

            # --- 1. ASR: gộp text từ các đoạn overlap ---
            segment.asr_text = self._merge_asr(
                asr_segments, t_start, t_end
            )

            # --- 2. Audio Events: gộp labels + chọn embedding tốt nhất ---
            events, audio_emb = self._merge_audio(
                audio_events, t_start, t_end
            )
            segment.audio_events = events
            segment.audio_embedding = audio_emb

            # --- 3. OCR: gộp text từ các frame trong khoảng thời gian ---
            segment.ocr_texts = self._merge_ocr(
                ocr_data, t_start, t_end
            )

            # --- 4. Video Clips: chọn embedding + action labels ---
            vid_emb, actions = self._merge_video_clips(
                video_embeddings, video_times, t_start, t_end
            )
            segment.video_embedding = vid_emb
            segment.action_labels = actions

            # --- 5. Frame: chọn frame gần nhất và CẮT THUMBNAIL ảnh vật lý ---
            segment.frame_embedding = self._find_closest_frame(
                frame_embeddings, frame_timestamps, t_start, t_end
            )
            
            # Logic lấy ảnh thực tế:
            if frame_timestamps:
                midpoint = (t_start + t_end) / 2.0
                # Tìm timestamp thực tế gần điểm giữa của segment nhất
                closest_ts = min(frame_timestamps, key=lambda ts: abs(ts - midpoint))
                
                # Trích xuất khung hình dạng PIL Image từ video gốc
                img = extract_frame(video_path, closest_ts)
                if img is not None:
                    # Đặt tên ảnh trùng khớp với segment_id hệ thống tìm kiếm
                    out_path = Path(KEYFRAMES_DIR) / f"{segment.segment_id}.jpg"
                    # Lưu xuống đĩa dưới định dạng JPEG
                    img.save(out_path, "JPEG")
                    # Gán đường dẫn vào metadata của segment
                    segment.keyframe_path = str(out_path)

            segments.append(segment)

        # Thống kê kết quả
        non_empty_asr = sum(1 for s in segments if s.asr_text)
        non_empty_audio = sum(1 for s in segments if s.audio_events)
        non_empty_ocr = sum(1 for s in segments if s.ocr_texts)
        non_empty_video = sum(1 for s in segments if s.video_embedding)
        non_empty_frame = sum(1 for s in segments if s.frame_embedding)

        log_success(
            f"Tạo xong {len(segments)} segments: "
            f"ASR={non_empty_asr} | Audio={non_empty_audio} | "
            f"OCR={non_empty_ocr} | Video={non_empty_video} | "
            f"Frame={non_empty_frame}"
        )

        return segments

    # ==========================================================
    # CÁC PHƯƠNG THỨC MERGE CHO TỪNG MODALITY
    # ==========================================================

    def _merge_asr(
        self, asr_segments: list[dict], t_start: float, t_end: float
    ) -> str:
        """
        Gộp text từ các ASR segments overlap với [t_start, t_end].

        Chiến lược: lấy tất cả segments có overlap > 0, nối text theo thứ tự.
        Nếu segment chỉ overlap một phần nhỏ, vẫn lấy toàn bộ text
        (vì tách word theo % overlap phức tạp và không chính xác).
        """
        overlapping = self._find_overlapping(
            asr_segments, t_start, t_end, "start", "end"
        )

        if not overlapping:
            return ""

        # Gộp text, loại bỏ trùng lặp
        texts = []
        seen = set()
        for item in overlapping:
            text = item.get("text", "").strip()
            if text and text not in seen:
                texts.append(text)
                seen.add(text)

        return " ".join(texts)

    def _merge_audio(
        self,
        audio_events: list[dict],
        t_start: float,
        t_end: float,
    ) -> tuple[list[str], list[float]]:
        """
        Gộp audio events + chọn embedding có overlap lớn nhất.

        Returns:
            (event_labels, best_embedding):
            - event_labels: danh sách tên sự kiện âm thanh
            - best_embedding: embedding của event có overlap lớn nhất
        """
        overlapping = self._find_overlapping(
            audio_events, t_start, t_end, "start", "end"
        )

        if not overlapping:
            return [], []

        # Gộp event labels (unique)
        events = []
        seen = set()
        for item in overlapping:
            for event in item.get("event_labels", []):
                event = event.strip()
                if event and event not in seen:
                    events.append(event)
                    seen.add(event)

        # Chọn embedding từ event có overlap lớn nhất
        best_embedding = []
        best_overlap = 0.0

        for item in overlapping:
            emb = item.get("embedding", [])
            if not emb:
                continue

            # Tính overlap duration
            item_start = item.get("start", 0.0)
            item_end = item.get("end", 0.0)
            overlap = self._calc_overlap(t_start, t_end, item_start, item_end)

            if overlap > best_overlap:
                best_overlap = overlap
                best_embedding = emb

        return events, best_embedding

    def _merge_ocr(
        self,
        ocr_data: list[dict],
        t_start: float,
        t_end: float,
    ) -> list[str]:
        """
        Gộp OCR texts từ các frames trong khoảng thời gian.

        OCR data thường là point-in-time (timestamp), không phải range.
        Lấy tất cả texts từ các timestamps nằm trong [t_start, t_end].
        """
        if not ocr_data:
            return []

        texts = []
        seen = set()

        for item in ocr_data:
            timestamp = item.get("timestamp", -1.0)

            # Kiểm tra xem timestamp có nằm trong segment không
            if t_start <= timestamp < t_end:
                for text in item.get("texts", []):
                    text = text.strip()
                    if text and text not in seen:
                        texts.append(text)
                        seen.add(text)

            # Hỗ trợ cả format có start/end
            elif "start" in item and "end" in item:
                overlap = self._calc_overlap(
                    t_start, t_end,
                    item.get("start", 0.0), item.get("end", 0.0),
                )
                if overlap > 0:
                    for text in item.get("texts", []):
                        text = text.strip()
                        if text and text not in seen:
                            texts.append(text)
                            seen.add(text)

        return texts

    def _merge_video_clips(
        self,
        video_embeddings: list[list[float]],
        video_times: list[tuple[float, float]],
        t_start: float,
        t_end: float,
    ) -> tuple[list[float], list[str]]:
        """
        Chọn video embedding + action labels từ clip có overlap lớn nhất.

        Returns:
            (best_embedding, action_labels)
        """
        if not video_embeddings or not video_times:
            return [], []

        best_embedding = []
        best_overlap = 0.0

        for emb, (clip_start, clip_end) in zip(video_embeddings, video_times):
            overlap = self._calc_overlap(t_start, t_end, clip_start, clip_end)
            if overlap > best_overlap and emb:
                best_overlap = overlap
                best_embedding = emb

        # Action labels — hiện chưa có action classifier riêng,
        # labels sẽ được thêm ở bước sau (nếu có)
        action_labels = []

        return best_embedding, action_labels

    def _find_closest_frame(
        self,
        frame_embeddings: list[list[float]],
        frame_timestamps: list[float],
        t_start: float,
        t_end: float,
    ) -> list[float]:
        """
        Tìm frame embedding gần nhất với điểm giữa segment.

        Chiến lược: chọn frame có timestamp gần midpoint nhất.
        Ưu tiên frame TRONG segment, nhưng nếu không có thì lấy frame gần nhất.
        """
        if not frame_embeddings or not frame_timestamps:
            return []

        midpoint = (t_start + t_end) / 2.0
        best_idx = -1
        best_dist = float("inf")

        for idx, ts in enumerate(frame_timestamps):
            dist = abs(ts - midpoint)

            # Ưu tiên frame nằm trong segment
            if t_start <= ts < t_end:
                # Frame trong segment → giảm distance để ưu tiên
                dist *= 0.5

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0 and best_idx < len(frame_embeddings):
            return frame_embeddings[best_idx]

        return []

    # ==========================================================
    # TIỆN ÍCH CHUNG
    # ==========================================================

    @staticmethod
    def _find_overlapping(
        items: list[dict],
        start: float,
        end: float,
        time_key_start: str = "start",
        time_key_end: str = "end",
    ) -> list[dict]:
        """
        Tìm tất cả items có overlap với khoảng [start, end].

        Hai khoảng [a, b] và [c, d] overlap khi: a < d AND c < b.

        Args:
            items: Danh sách dict có keys thời gian
            start, end: Khoảng thời gian cần tìm overlap
            time_key_start: Key cho thời điểm bắt đầu trong dict
            time_key_end: Key cho thời điểm kết thúc trong dict

        Returns:
            Danh sách các items có overlap, giữ nguyên thứ tự
        """
        if not items:
            return []

        result = []
        for item in items:
            item_start = item.get(time_key_start, 0.0)
            item_end = item.get(time_key_end, 0.0)

            # Xử lý trường hợp item_end chưa được set (point event)
            if item_end <= item_start:
                item_end = item_start + 0.001  # Coi như instant event

            # Điều kiện overlap: [start, end] ∩ [item_start, item_end] ≠ ∅
            if item_start < end and item_end > start:
                result.append(item)

        return result

    @staticmethod
    def _calc_overlap(
        start1: float, end1: float,
        start2: float, end2: float,
    ) -> float:
        """
        Tính thời lượng overlap giữa hai khoảng thời gian.

        Returns:
            Số giây overlap (>= 0)
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)
