"""
config.py — Cấu hình tập trung cho Pipeline Lai Tạo VER (AI Challenge 2026).

Quản lý: đường dẫn, models, hyperparameters, VRAM budget, API keys.
GPU: RTX 4060 8GB → lazy loading bắt buộc, fp16 where possible.
RAM: 40GB → HNSW OK.
"""

import os
import torch
from pathlib import Path

# ============================================================
# 1. ĐƯỜNG DẪN DỰ ÁN
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSCRIPT_DIR = PROCESSED_DIR / "transcripts"
INDEXES_DIR = PROCESSED_DIR / "indexes"
KEYFRAMES_DIR = PROCESSED_DIR / "keyframes"

# Tạo thư mục nếu chưa tồn tại
for d in [RAW_VIDEO_DIR, PROCESSED_DIR, TRANSCRIPT_DIR, INDEXES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Index paths — 4 loại index riêng biệt
TEXT_SPARSE_INDEX_DIR = str(INDEXES_DIR / "text_sparse")
TEXT_DENSE_INDEX_PATH = str(INDEXES_DIR / "text_dense.index")
TEXT_DENSE_META_PATH = str(INDEXES_DIR / "text_dense_meta.json")
VISUAL_INDEX_PATH = str(INDEXES_DIR / "visual.index")
VISUAL_META_PATH = str(INDEXES_DIR / "visual_meta.json")
TEMPORAL_INDEX_PATH = str(INDEXES_DIR / "temporal.index")
TEMPORAL_META_PATH = str(INDEXES_DIR / "temporal_meta.json")
AUDIO_INDEX_PATH = str(INDEXES_DIR / "audio.index")
AUDIO_META_PATH = str(INDEXES_DIR / "audio_meta.json")
SEGMENTS_PATH = str(PROCESSED_DIR / "unified_segments.json")

# ============================================================
# 2. THIẾT BỊ TÍNH TOÁN
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_VRAM_GB = 8      # RTX 4060
SYSTEM_RAM_GB = 40
USE_FP16 = True       # Giảm VRAM, tăng tốc inference

# ============================================================
# 3. API KEYS
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def validate_api_key():
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY chưa được cấu hình!\n"
            "Windows: set GEMINI_API_KEY=your_key\n"
            "Linux:   export GEMINI_API_KEY=your_key"
        )

# ============================================================
# 4. OFFLINE MODELS — "5 Giác quan"
# ============================================================

# --- ASR: PhoWhisper (tiếng Việt native, VinAI) ---
ASR_MODEL = "vinai/PhoWhisper-medium"
ASR_FALLBACK = "openai/whisper-medium"  # Fallback nếu PhoWhisper lỗi

# --- Frame Encoder: SigLIP2 (Google, thay CLIP) ---
FRAME_ENCODER_MODEL = "google/siglip2-base-patch16-224"
FRAME_EMBEDDING_DIM = 768

# --- Video Encoder: InternVideo2 (action recognition) ---
VIDEO_ENCODER_MODEL = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
VIDEO_EMBEDDING_DIM = 768

# --- Audio: CLAP (âm thanh môi trường) ---
AUDIO_MODEL = "laion/larger_clap_music_and_speech"
AUDIO_EMBEDDING_DIM = 512

# --- OCR: VietOCR (text trên hình) ---
OCR_DETECTOR = "paddleocr"  # PaddleOCR cho detection
OCR_RECOGNIZER = "vgg_transformer"  # VietOCR config cho recognition

# ============================================================
# 5. ONLINE MODELS — Retrieval & Reranking
# ============================================================

# --- Text Embedding: BGE-M3 (SOTA multilingual) ---
TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
TEXT_EMBEDDING_DIM = 1024

# --- Reranker Auto Track: bge-reranker (nhanh) ---
AUTO_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --- LLM: Gemini Flash ---
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_VISION_MODEL = "gemini-2.5-flash"

# ============================================================
# 6. HYPERPARAMETERS
# ============================================================

# --- Retrieval ---
TOP_K_RETRIEVAL = 100    # Số candidates sau fusion (theo advice: Top 100)
TOP_K_RERANK = 20        # Số candidates gửi cho reranker
TOP_K_FINAL = 5          # Số kết quả cuối cùng

# --- RRF ---
RRF_K = 60
# Trọng số cross-modal (text chiếm ưu thế vì ASR tiếng Việt)
WEIGHT_TEXT_DENSE = 0.35
WEIGHT_TEXT_SPARSE = 0.20
WEIGHT_VISUAL = 0.25
WEIGHT_AUDIO = 0.10
WEIGHT_TEMPORAL = 0.10

# --- BM25 ---
BM25_K1 = 1.5
BM25_B = 0.75

# --- HyDE ---
HYDE_ENABLED = True

# --- CRAG ---
CRAG_THRESHOLD_CORRECT = 0.7
CRAG_THRESHOLD_AMBIGUOUS = 0.4
CRAG_MAX_RETRIES = 2   # Giảm xuống 2 cho auto track (speed)

# --- Video Processing ---
FRAME_SAMPLE_RATE = 1      # 1 frame/giây
VIDEO_CLIP_LENGTH = 4      # 4 giây mỗi clip cho InternVideo2
FRAME_BATCH_SIZE = 16      # Giảm batch size cho 8GB VRAM
TEMPORAL_WINDOW_SEC = 5.0

# --- Latency Budgets ---
AUTO_TRACK_BUDGET_SEC = 2.0    # Track tự động: <2 giây
SEMI_TRACK_BUDGET_SEC = 15.0   # Track bán tự động: <15 giây

# --- API Rate Limiting ---
API_RETRY_MAX = 3
API_RETRY_DELAY = 2.0
API_CALL_DELAY = 0.3

# --- Web Server ---
WEB_HOST = "localhost"
WEB_PORT = 8000
