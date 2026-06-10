"""
asr_engine.py — PhoWhisper ASR Engine cho tiếng Việt.

Chức năng:
- Trích xuất audio từ video (ffmpeg subprocess)
- Phiên âm tiếng Việt bằng PhoWhisper-medium (VinAI)
- Chia audio thành chunks 30 giây, inference từng chunk
- Fallback sang openai-whisper nếu PhoWhisper không khả dụng
- Lazy loading model để tiết kiệm VRAM (RTX 4060 8GB)
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SegmentSchema, AnalyzedQuery, SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, free_vram, log_vram, format_time,
    extract_frame, extract_frames_batch, extract_video_clips,
    get_video_duration, tokenize_vietnamese,
)

# Lazy imports — chỉ import khi cần
import torch


class ASREngine:
    """
    PhoWhisper ASR Engine — Phiên âm tiếng Việt từ video.
    
    Model: vinai/PhoWhisper-medium (Whisper fine-tuned cho tiếng Việt)
    Fallback: openai-whisper (nếu PhoWhisper gặp lỗi tải/import)
    
    VRAM: ~2GB khi load, giải phóng sau khi xong.
    """

    # Cấu hình cố định
    CHUNK_DURATION = 30.0       # Độ dài mỗi chunk audio (giây)
    SAMPLE_RATE = 16000         # PhoWhisper yêu cầu 16kHz
    OVERLAP = 1.0               # Overlap giữa các chunk để tránh cắt giữa từ

    def __init__(self):
        """Khởi tạo — KHÔNG load model ở đây (lazy loading)."""
        self.model = None
        self.processor = None
        self.pipe = None            # transformers pipeline (nếu dùng)
        self._use_fallback = False  # Flag dùng openai-whisper
        self._fallback_model = None
        log_info("ASREngine khởi tạo (lazy — chưa load model)")

    def _load_model(self):
        """
        Lazy load PhoWhisper model.
        Thử PhoWhisper trước, fallback sang openai-whisper nếu lỗi.
        """
        if self.model is not None or self._use_fallback:
            return  # Đã load rồi

        log_step("ASR", f"Đang load model {ASR_MODEL}...")
        log_vram("trước khi load ASR")

        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )

            # Xác định dtype và device
            dtype = torch.float16 if USE_FP16 and CUDA_AVAILABLE else torch.float32

            # Log cảnh báo tải model lần đầu
            log_info(f"Nếu đây là lần chạy đầu tiên, hệ thống đang tải model {ASR_MODEL} (~3GB). Vui lòng chờ...")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(ASR_MODEL)

            # Load model với fp16 để tiết kiệm VRAM
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                ASR_MODEL,
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            self.model.to(DEVICE)
            self.model.eval()

            # Tạo pipeline cho inference dễ dàng hơn
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                dtype=dtype,
                device=DEVICE,
            )

            log_success(f"PhoWhisper loaded thành công trên {DEVICE}")
            log_vram("sau khi load ASR")

        except Exception as e:
            log_warning(f"Không thể load PhoWhisper: {e}")
            log_info("Đang thử fallback sang openai-whisper...")
            self._load_fallback()

    def _load_fallback(self):
        """Load openai-whisper như fallback."""
        try:
            import whisper

            self._fallback_model = whisper.load_model(
                "medium",
                device=DEVICE,
            )
            self._use_fallback = True
            log_success("Fallback openai-whisper loaded thành công")

        except ImportError:
            log_error(
                "Cả PhoWhisper và openai-whisper đều không khả dụng! "
                "Cài đặt: pip install openai-whisper"
            )
            self._use_fallback = True  # Đánh dấu để không thử load lại
        except Exception as e:
            log_error(f"Không thể load openai-whisper: {e}")
            self._use_fallback = True

    def _extract_audio(self, video_path: str) -> str:
        """
        Trích xuất audio từ video bằng ffmpeg.
        
        Returns:
            Đường dẫn file WAV tạm (16kHz mono).
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")

        # Tạo file tạm cho audio output
        tmp_dir = PROCESSED_DIR / "tmp_audio"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        audio_path = str(tmp_dir / f"{video_name}_audio.wav")

        # Nếu đã extract trước đó, dùng lại
        if os.path.exists(audio_path):
            if self._is_valid_audio_file(audio_path):
                log_info(f"Dùng lại audio đã extract: {audio_path}")
                return audio_path
            log_warning(f"Audio cache không hợp lệ, tạo lại: {audio_path}")
            try:
                os.remove(audio_path)
            except OSError:
                pass

        log_info(f"Đang extract audio từ {Path(video_path).name}...")

        try:
            cmd = [
                "ffmpeg", "-y",             # Ghi đè nếu tồn tại
                "-i", video_path,            # Input video
                "-vn",                       # Bỏ video stream
                "-acodec", "pcm_s16le",      # PCM 16-bit
                "-ar", str(self.SAMPLE_RATE), # 16kHz
                "-ac", "1",                  # Mono
                "-loglevel", "error",        # Chỉ hiện lỗi
                audio_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # Timeout 2 phút
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr}")

            log_success(f"Audio extracted: {audio_path}")
            return audio_path

        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg không được cài đặt hoặc không có trong PATH! "
                "Tải tại: https://ffmpeg.org/download.html"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timeout khi extract audio từ {video_path}")

    @staticmethod
    def _is_valid_audio_file(audio_path: str) -> bool:
        """Check whether a cached WAV file can be decoded."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.frames > 0 and info.samplerate > 0
        except Exception:
            return False

    def _load_audio_chunks(self, audio_path: str) -> list[tuple[np.ndarray, float, float]]:
        """
        Đọc file audio và chia thành chunks 30 giây.
        
        Returns:
            List of (audio_array, start_time, end_time)
        """
        try:
            import torchaudio

            waveform, sr = torchaudio.load(audio_path)

            # Resample nếu cần
            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.SAMPLE_RATE,
                )
                waveform = resampler(waveform)
                sr = self.SAMPLE_RATE

            # Chuyển sang mono nếu stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Chuyển sang numpy
            audio_np = waveform.squeeze().numpy()

        except Exception as torchaudio_error:
            log_warning(f"torchaudio không đọc được audio, dùng soundfile: {torchaudio_error}")
            try:
                import soundfile as sf
                audio_np, sr = sf.read(audio_path, dtype="float32")
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=1)
                # Resample đơn giản nếu cần
                if sr != self.SAMPLE_RATE:
                    from scipy.signal import resample_poly
                    from math import gcd
                    divisor = gcd(sr, self.SAMPLE_RATE)
                    audio_np = resample_poly(
                        audio_np,
                        self.SAMPLE_RATE // divisor,
                        sr // divisor,
                    ).astype(np.float32)
            except Exception as soundfile_error:
                raise RuntimeError(
                    "Không thể đọc audio bằng torchaudio hoặc soundfile: "
                    f"{soundfile_error}"
                )

        # Chia thành chunks
        chunk_samples = int(self.CHUNK_DURATION * self.SAMPLE_RATE)
        overlap_samples = int(self.OVERLAP * self.SAMPLE_RATE)
        stride_samples = chunk_samples - overlap_samples
        total_samples = len(audio_np)

        chunks = []
        start_idx = 0

        while start_idx < total_samples:
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = audio_np[start_idx:end_idx]

            # Bỏ qua chunk quá ngắn (< 0.5 giây)
            if len(chunk) < self.SAMPLE_RATE * 0.5:
                break

            start_time = start_idx / self.SAMPLE_RATE
            end_time = end_idx / self.SAMPLE_RATE

            # Padding nếu chunk ngắn hơn CHUNK_DURATION
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

            chunks.append((chunk.astype(np.float32), start_time, end_time))
            start_idx += stride_samples

        log_info(
            f"Audio: {total_samples / self.SAMPLE_RATE:.1f}s → "
            f"{len(chunks)} chunks × {self.CHUNK_DURATION}s"
        )
        return chunks

    @timer
    def transcribe(self, video_path: str) -> list[dict]:
        """
        Phiên âm toàn bộ video thành danh sách đoạn transcript.
        
        Args:
            video_path: Đường dẫn video cần phiên âm
            
        Returns:
            List[dict] với mỗi dict: {start: float, end: float, text: str}
        """
        log_step("ASR", f"Bắt đầu phiên âm: {Path(video_path).name}")
        results = []

        try:
            # Bước 1: Extract audio từ video
            audio_path = self._extract_audio(video_path)

            # Bước 2: Load model nếu chưa load
            self._load_model()

            # Bước 3: Chọn phương pháp phiên âm
            if self._use_fallback and self._fallback_model is not None:
                results = self._transcribe_fallback(audio_path)
            elif self.pipe is not None:
                results = self._transcribe_phowhisper(audio_path)
            else:
                log_error("Không có model ASR nào khả dụng!")
                return []

            # Lọc kết quả trống
            results = [r for r in results if r["text"].strip()]

            # Gộp các đoạn quá ngắn liền kề
            results = self._merge_short_segments(results)

            log_success(
                f"Phiên âm xong: {len(results)} đoạn, "
                f"tổng {sum(len(r['text']) for r in results)} ký tự"
            )

        except Exception as e:
            log_error(f"Lỗi phiên âm {Path(video_path).name}: {e}")

        finally:
            # Bước 4: Giải phóng VRAM
            free_vram()

        return results

    def _transcribe_phowhisper(self, audio_path: str) -> list[dict]:
        """Phiên âm bằng PhoWhisper pipeline."""
        chunks = self._load_audio_chunks(audio_path)
        results = []

        for i, (chunk_audio, chunk_start, chunk_end) in enumerate(chunks):
            try:
                # Dùng pipeline — tự xử lý tokenization
                output = self.pipe(
                    chunk_audio,
                    chunk_length_s=30,
                    batch_size=1,
                    return_timestamps=True,
                    generate_kwargs={
                        "language": "vi",
                        "task": "transcribe",
                    },
                )

                # Xử lý output có timestamps
                if "chunks" in output and output["chunks"]:
                    for seg in output["chunks"]:
                        seg_text = seg.get("text", "").strip()
                        if not seg_text:
                            continue

                        # Timestamps từ pipeline là relative trong chunk
                        ts = seg.get("timestamp", (None, None))
                        seg_start = chunk_start + (ts[0] if ts[0] is not None else 0.0)
                        seg_end = chunk_start + (ts[1] if ts[1] is not None else (chunk_end - chunk_start))

                        results.append({
                            "start": round(seg_start, 2),
                            "end": round(seg_end, 2),
                            "text": seg_text,
                        })
                elif output.get("text", "").strip():
                    # Không có timestamps chi tiết → dùng chunk boundaries
                    results.append({
                        "start": round(chunk_start, 2),
                        "end": round(chunk_end, 2),
                        "text": output["text"].strip(),
                    })

                if (i + 1) % 10 == 0:
                    log_info(f"  Đã phiên âm {i + 1}/{len(chunks)} chunks")

            except Exception as e:
                log_warning(f"Lỗi chunk {i} [{chunk_start:.1f}s-{chunk_end:.1f}s]: {e}")
                continue

        return results

    def _transcribe_fallback(self, audio_path: str) -> list[dict]:
        """Phiên âm bằng openai-whisper (fallback)."""
        log_info("Sử dụng openai-whisper fallback...")

        try:
            result = self._fallback_model.transcribe(
                audio_path,
                language="vi",
                task="transcribe",
                verbose=False,
            )

            segments = []
            for seg in result.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    segments.append({
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "text": text,
                    })

            return segments

        except Exception as e:
            log_error(f"openai-whisper fallback lỗi: {e}")
            return []

    @staticmethod
    def _merge_short_segments(
        segments: list[dict],
        min_duration: float = 0.5,
        max_gap: float = 0.3,
    ) -> list[dict]:
        """
        Gộp các đoạn transcript ngắn liền kề thành đoạn dài hơn.
        
        Args:
            segments: Danh sách segments cần gộp
            min_duration: Thời lượng tối thiểu (giây) — đoạn ngắn hơn sẽ bị gộp
            max_gap: Khoảng cách tối đa giữa 2 đoạn để gộp (giây)
        """
        if not segments:
            return []

        merged = [segments[0].copy()]

        for seg in segments[1:]:
            prev = merged[-1]
            duration = seg["end"] - seg["start"]
            gap = seg["start"] - prev["end"]

            # Gộp nếu đoạn quá ngắn hoặc khoảng cách nhỏ
            if duration < min_duration or gap <= max_gap:
                prev["end"] = seg["end"]
                prev["text"] = prev["text"] + " " + seg["text"]
            else:
                merged.append(seg.copy())

        return merged

    def unload(self):
        """Giải phóng toàn bộ model khỏi bộ nhớ."""
        log_info("ASREngine: Đang giải phóng model...")

        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if self._fallback_model is not None:
            del self._fallback_model
            self._fallback_model = None

        free_vram()
        log_success("ASREngine: Đã giải phóng VRAM")
