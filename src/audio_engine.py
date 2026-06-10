"""
audio_engine.py — CLAP Audio Event Detection Engine.

Chức năng:
- Phát hiện sự kiện âm thanh trong video bằng CLAP (laion/larger_clap_music_and_speech)
- Trích xuất audio embeddings (512d) cho mỗi cửa sổ thời gian
- Zero-shot classification với danh sách sự kiện predefined
- Sliding window: 5s windows, 2.5s overlap
- Lazy loading model (RTX 4060 8GB)
"""

import sys
import os
import subprocess
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

import torch


class AudioEventEngine:
    """
    CLAP Audio Event Engine — Phát hiện sự kiện âm thanh trong video.
    
    Model: laion/larger_clap_music_and_speech (Hugging Face transformers)
    Output: 512-dimensional audio embeddings + zero-shot event labels
    
    Window: 5 giây, overlap 2.5 giây
    VRAM: ~800MB khi load
    """

    # Cấu hình cửa sổ phân tích
    WINDOW_DURATION = 5.0       # Độ dài mỗi cửa sổ (giây)
    WINDOW_OVERLAP = 2.5        # Overlap giữa các cửa sổ (giây)
    SAMPLE_RATE = 48000         # CLAP yêu cầu 48kHz

    # Danh sách sự kiện predefined cho zero-shot classification
    EVENT_LABELS = [
        "gunshot",
        "siren",
        "crowd",
        "music",
        "speech",
        "explosion",
        "car horn",
        "rain",
        "applause",
        "silence",
    ]

    # Ngưỡng confidence cho zero-shot — chỉ giữ labels vượt ngưỡng
    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self):
        """Khởi tạo — KHÔNG load model ở đây (lazy loading)."""
        self.model = None
        self.processor = None
        self._text_embeds_cache = None  # Cache text embeddings cho event labels
        log_info("AudioEventEngine khởi tạo (lazy — chưa load model)")

    def _load_model(self):
        """Lazy load CLAP model và processor."""
        if self.model is not None:
            return  # Đã load rồi

        log_step("AudioEvent", f"Đang load model {AUDIO_MODEL}...")
        log_vram("trước khi load AudioEvent")

        try:
            from transformers import ClapModel, ClapProcessor

            # CLAP has BatchNorm layers that are fragile in fp16 on CUDA.
            # Keep this model in fp32 to avoid input/weight dtype mismatches.
            torch_dtype = torch.float32

            # Load processor
            self.processor = ClapProcessor.from_pretrained(AUDIO_MODEL)

            # Load model
            self.model = ClapModel.from_pretrained(
                AUDIO_MODEL,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(DEVICE)
            self.model.eval()

            # Pre-compute text embeddings cho event labels (cache lại)
            self._precompute_event_embeddings()

            log_success(
                f"CLAP loaded trên {DEVICE} "
                f"(dtype=fp32, embedding dim={AUDIO_EMBEDDING_DIM})"
            )
            log_vram("sau khi load AudioEvent")

        except Exception as e:
            log_error(f"Không thể load CLAP: {e}")
            raise RuntimeError(f"AudioEventEngine load thất bại: {e}")

    @torch.no_grad()
    def _precompute_event_embeddings(self):
        """Pre-compute text embeddings cho các event labels (dùng lại nhiều lần)."""
        try:
            text_inputs = self.processor(
                text=self.EVENT_LABELS,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {
                k: self._move_tensor_to_model(v) if isinstance(v, torch.Tensor) else v
                for k, v in text_inputs.items()
            }

            self._text_embeds_cache = self.model.get_text_features(**text_inputs)
            # Normalize
            self._text_embeds_cache = self._text_embeds_cache / (
                self._text_embeds_cache.norm(dim=-1, keepdim=True) + 1e-8
            )

            log_info(f"Đã cache {len(self.EVENT_LABELS)} event label embeddings")

        except Exception as e:
            log_warning(f"Không thể pre-compute event embeddings: {e}")
            self._text_embeds_cache = None

    def _move_tensor_to_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device and match floating dtype with model weights."""
        if self.model is None:
            return tensor.to(DEVICE)

        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

        tensor = tensor.to(DEVICE)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=model_dtype)
        return tensor

    def _extract_audio(self, video_path: str) -> str:
        """
        Trích xuất audio từ video bằng ffmpeg (48kHz mono cho CLAP).
        
        Returns:
            Đường dẫn file WAV tạm.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")

        tmp_dir = PROCESSED_DIR / "tmp_audio"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        audio_path = str(tmp_dir / f"{video_name}_audio_48k.wav")

        # Dùng lại nếu đã extract
        if os.path.exists(audio_path):
            if self._is_valid_audio_file(audio_path):
                log_info(f"Dùng lại audio 48kHz: {audio_path}")
                return audio_path
            log_warning(f"Audio cache không hợp lệ, tạo lại: {audio_path}")
            try:
                os.remove(audio_path)
            except OSError:
                pass

        log_info(f"Đang extract audio 48kHz từ {Path(video_path).name}...")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.SAMPLE_RATE),
                "-ac", "1",
                "-loglevel", "error",
                audio_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr}")

            log_success(f"Audio 48kHz extracted: {audio_path}")
            return audio_path

        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg không được cài đặt hoặc không có trong PATH! "
                "Tải tại: https://ffmpeg.org/download.html"
            )

    @staticmethod
    def _is_valid_audio_file(audio_path: str) -> bool:
        """Check whether a cached WAV file can be decoded."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.frames > 0 and info.samplerate > 0
        except Exception:
            return False

    def _load_audio_windows(self, audio_path: str) -> list[tuple[np.ndarray, float, float]]:
        """
        Đọc audio và chia thành sliding windows.
        
        Returns:
            List of (audio_array, start_time, end_time)
        """
        # Đọc audio. Trên một số môi trường Windows, torchaudio import được
        # nhưng không có backend phù hợp để đọc WAV, nên fallback bằng soundfile
        # cho cả ImportError lẫn RuntimeError.
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

            # Chuyển sang mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            audio_np = waveform.squeeze().numpy()

        except Exception as torchaudio_error:
            log_warning(f"torchaudio không đọc được audio, dùng soundfile: {torchaudio_error}")
            try:
                import soundfile as sf
                audio_np, sr = sf.read(audio_path, dtype="float32")
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=1)
                # Resample nếu cần
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

        # Chia thành sliding windows
        window_samples = int(self.WINDOW_DURATION * self.SAMPLE_RATE)
        stride_samples = int((self.WINDOW_DURATION - self.WINDOW_OVERLAP) * self.SAMPLE_RATE)
        total_samples = len(audio_np)

        windows = []
        start_idx = 0

        while start_idx < total_samples:
            end_idx = min(start_idx + window_samples, total_samples)
            window = audio_np[start_idx:end_idx]

            # Bỏ qua window quá ngắn (< 1 giây)
            if len(window) < self.SAMPLE_RATE:
                break

            start_time = start_idx / self.SAMPLE_RATE
            end_time = end_idx / self.SAMPLE_RATE

            # Padding nếu window ngắn hơn WINDOW_DURATION
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)), mode='constant')

            windows.append((window.astype(np.float32), start_time, end_time))
            start_idx += stride_samples

        log_info(
            f"Audio: {total_samples / self.SAMPLE_RATE:.1f}s → "
            f"{len(windows)} windows × {self.WINDOW_DURATION}s "
            f"(overlap {self.WINDOW_OVERLAP}s)"
        )
        return windows

    @timer
    @torch.no_grad()
    def extract_audio_events(self, video_path: str) -> list[dict]:
        """
        Trích xuất sự kiện âm thanh từ video.
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            List[dict] với mỗi dict chứa:
            - start: float — thời điểm bắt đầu
            - end: float — thời điểm kết thúc
            - event_labels: List[str] — sự kiện phát hiện được
            - embedding: List[float] — CLAP audio embedding (512d)
        """
        log_step("AudioEvent", f"Phân tích audio: {Path(video_path).name}")
        results = []

        try:
            # Bước 1: Extract audio
            audio_path = self._extract_audio(video_path)

            # Bước 2: Load model
            self._load_model()

            # Bước 3: Chia thành windows
            windows = self._load_audio_windows(audio_path)

            if not windows:
                log_warning("Không có audio windows nào")
                return []

            # Bước 4: Process từng window
            for i, (window_audio, w_start, w_end) in enumerate(windows):
                try:
                    result = self._process_window(window_audio, w_start, w_end)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    log_warning(f"Lỗi window {i} [{w_start:.1f}s-{w_end:.1f}s]: {e}")
                    continue

                # Log tiến trình mỗi 20 windows
                if (i + 1) % 20 == 0:
                    log_info(f"  Đã xử lý {i + 1}/{len(windows)} windows")

            # Thống kê
            total_events = sum(len(r["event_labels"]) for r in results)
            log_success(
                f"Audio analysis hoàn thành: {len(results)} windows, "
                f"{total_events} events phát hiện"
            )

        except Exception as e:
            log_error(f"Lỗi phân tích audio {Path(video_path).name}: {e}")

        finally:
            free_vram()

        return results

    def _process_window(
        self,
        audio_array: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> dict | None:
        """
        Xử lý một audio window: tạo embedding + zero-shot classification.
        
        Returns:
            Dict hoặc None nếu lỗi
        """
        # Tạo audio inputs cho CLAP
        audio_inputs = self.processor(
            audio=[audio_array],
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        audio_inputs = {
            k: self._move_tensor_to_model(v) if isinstance(v, torch.Tensor) else v
            for k, v in audio_inputs.items()
        }

        # Lấy audio embedding
        audio_features = self.model.get_audio_features(**audio_inputs)

        # Normalize embedding
        audio_emb = audio_features.detach().cpu().float().numpy().squeeze()
        audio_emb_norm = audio_emb / max(np.linalg.norm(audio_emb), 1e-8)

        # Zero-shot classification
        event_labels = []
        if self._text_embeds_cache is not None:
            # Normalize audio features on GPU
            audio_feat_norm = audio_features / (
                audio_features.norm(dim=-1, keepdim=True) + 1e-8
            )

            # Cosine similarity với event labels
            similarities = (audio_feat_norm @ self._text_embeds_cache.T).squeeze()

            # Softmax để chuyển thành xác suất
            probs = torch.softmax(similarities, dim=-1).detach().cpu().numpy()

            # Lấy labels vượt ngưỡng confidence
            for idx, (label, prob) in enumerate(zip(self.EVENT_LABELS, probs)):
                if prob >= self.CONFIDENCE_THRESHOLD:
                    event_labels.append(label)

        return {
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "event_labels": event_labels,
            "embedding": audio_emb_norm.tolist(),
        }

    @torch.no_grad()
    def encode_text(self, text: str) -> list[float]:
        """
        Encode text query thành embedding trong không gian CLAP.
        Dùng cho audio-text search.
        
        Args:
            text: Text query mô tả âm thanh (vd: "tiếng súng nổ")
            
        Returns:
            List[float] — CLAP text embedding (512d), L2-normalized
        """
        if not text or not text.strip():
            log_warning("Audio text query rỗng — trả về zero vector")
            return [0.0] * AUDIO_EMBEDDING_DIM

        # Load model nếu chưa
        self._load_model()

        try:
            text_inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {
                k: self._move_tensor_to_model(v) if isinstance(v, torch.Tensor) else v
                for k, v in text_inputs.items()
            }

            text_features = self.model.get_text_features(**text_inputs)

            # Normalize và chuyển về list
            emb = text_features.detach().cpu().float().numpy().squeeze()
            emb = emb / max(np.linalg.norm(emb), 1e-8)

            return emb.tolist()

        except Exception as e:
            log_error(f"Lỗi encode audio text: {e}")
            return [0.0] * AUDIO_EMBEDDING_DIM

    @torch.no_grad()
    def encode_audio_from_file(self, audio_path: str) -> list[float]:
        """
        Encode toàn bộ file audio thành một embedding.
        Dùng cho audio-to-audio search (query by audio example).
        
        Args:
            audio_path: Đường dẫn file audio
            
        Returns:
            List[float] — CLAP audio embedding (512d), L2-normalized
        """
        self._load_model()

        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio_np = waveform.squeeze().numpy().astype(np.float32)
        except ImportError:
            import soundfile as sf
            audio_np, sr = sf.read(audio_path)
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
            audio_np = audio_np.astype(np.float32)

        audio_inputs = self.processor(
            audio=[audio_np],
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        audio_inputs = {
            k: self._move_tensor_to_model(v) if isinstance(v, torch.Tensor) else v
            for k, v in audio_inputs.items()
        }

        audio_features = self.model.get_audio_features(**audio_inputs)
        emb = audio_features.detach().cpu().float().numpy().squeeze()
        emb = emb / max(np.linalg.norm(emb), 1e-8)

        return emb.tolist()

    def unload(self):
        """Giải phóng model khỏi VRAM."""
        log_info("AudioEventEngine: Đang giải phóng model...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self._text_embeds_cache = None

        free_vram()
        log_success("AudioEventEngine: Đã giải phóng VRAM")
