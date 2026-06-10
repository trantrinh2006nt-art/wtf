# 🎯 VER — Video Evidence Retrieval System
### Pipeline Lai Tạo cho AI Challenge 2026

Hệ thống Trợ lý ảo thông minh hỗ trợ phân tích và truy xuất thông tin chuyên sâu trong dữ liệu lớn multimedia. Thể thức tương tự LSC/VBS.

## Kiến trúc

```
┌─────────────────── OFFLINE ───────────────────┐
│ Raw Video → PhoWhisper (ASR)                  │
│           → CLAP (Audio Events)               │
│           → VietOCR (Text on Screen)          │
│           → InternVideo2 (Actions)            │
│           → SigLIP2 (Frame Visuals)           │
│           → Temporal Alignment                │
│           → FAISS/HNSW Indexing               │
└───────────────────────────────────────────────┘

┌─────────────────── ONLINE ────────────────────┐
│ Query → [Tầng 1] LLM Query Analyzer          │
│       → [Tầng 2] Multi-Index Retrieval + RRF │
│       → [Tầng 3] ┬─ Auto: bge-reranker (<2s) │
│                   └─ Semi: Gemini + Feedback  │
└───────────────────────────────────────────────┘
```

## Cài đặt & Sử dụng

```bash
# 1. Cài dependencies
uv sync

# 2. Cấu hình API key
set GEMINI_API_KEY=your_key_here

# 3. Tiền xử lý video
uv run python src/main.py preprocess --dir data/raw/

# 4. Tìm kiếm (Auto Track — <2 giây)
uv run python src/main.py search --query "tai nạn giao thông" --track auto

# 5. Tìm kiếm (Semi Track — có feedback)
uv run python src/main.py search --query "cháy rừng miền Trung" --track semi

# 6. Web Dashboard
uv run python src/main.py server

# 7. Interactive REPL
uv run python src/main.py interactive
```

## Công nghệ

| Thành phần | Model |
|-----------|-------|
| ASR tiếng Việt | PhoWhisper (VinAI) |
| Frame Visual | SigLIP2 (Google) |
| Video Action | InternVideo2 |
| Audio Events | CLAP |
| OCR | VietOCR + PaddleOCR |
| Text Embedding | BGE-M3 (BAAI) |
| Auto Reranker | bge-reranker-v2-m3 |
| LLM/Vision | Gemini 2.5 Flash |
| Vector DB | FAISS |
