import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import log_step, log_info, log_error
from pipeline import VERPipeline

# =====================================================================
# Khởi tạo ứng dụng & Pipeline
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pipeline()
    yield

app = FastAPI(title="VER Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo Pipeline ở mode auto ban đầu
pipeline = None

def init_pipeline():
    global pipeline
    if pipeline is None:
        log_info("Khởi tạo VERPipeline cho Web Backend...")
        pipeline = VERPipeline(track="auto")

# =====================================================================
# API Models
# =====================================================================

class SearchRequest(BaseModel):
    query: str
    track: str = "auto"
    top_k: int = 20

class PreprocessRequest(BaseModel):
    video_dir: Optional[str] = None
    video_path: Optional[str] = None

class FeedbackRequest(BaseModel):
    segment_id: str
    relevant: bool

# =====================================================================
# HTML Dashboard (STUNNING MODERN GLASSMORPHISM)
# =====================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VER — Video Evidence Retrieval</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f0f23;
            --panel-bg: rgba(255, 255, 255, 0.05);
            --panel-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #00d4ff;
            --accent-purple: #7c3aed;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --glass-blur: blur(12px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(124, 58, 237, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(0, 212, 255, 0.1) 0%, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Tùy chỉnh thanh cuộn */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        header {
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--panel-border);
            background: rgba(15, 15, 35, 0.8);
            backdrop-filter: var(--glass-blur);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .layout {
            display: grid;
            grid-template-columns: 300px 1fr 280px;
            gap: 1.5rem;
            padding: 1.5rem;
            flex-grow: 1;
            height: calc(100vh - 80px);
        }

        .glass-panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: var(--glass-blur);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }

        /* --- Left Panel (Search) --- */
        .search-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            display: block;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text-main);
            font-size: 1rem;
            resize: none;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }

        .track-selector {
            display: flex;
            gap: 1rem;
        }

        .radio-btn {
            flex: 1;
        }

        .radio-btn input { display: none; }
        
        .radio-btn label {
            display: block;
            text-align: center;
            padding: 0.8rem;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            margin: 0;
            font-weight: 500;
            color: var(--text-main);
        }

        .radio-btn input:checked + label {
            background: rgba(0, 212, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-cyan) 0%, #0284c7 100%);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        }

        .btn-primary:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        /* --- Spinner --- */
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* --- Main Area (Results) --- */
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .results-header h2 {
            font-size: 1.2rem;
            font-weight: 600;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .result-card {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            position: relative;
        }

        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.5);
            border-color: rgba(255,255,255,0.2);
        }

        .thumbnail {
            width: 100%;
            height: 160px;
            background: #000;
            position: relative;
        }
        
        .thumbnail img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .badge-source {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(4px);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-source.text { color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-source.visual { color: var(--accent-purple); border: 1px solid var(--accent-purple); }
        .badge-source.fused { color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
        .badge-source.audio { color: #f59e0b; border: 1px solid #f59e0b; }

        .time-badge {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .card-body {
            padding: 1rem;
        }

        .card-title {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .score-bar-bg {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-bottom: 0.8rem;
            overflow: hidden;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
        }

        .card-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .feedback-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: auto;
        }

        .btn-icon {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--panel-border);
            color: var(--text-main);
            padding: 0.5rem;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 1rem;
        }

        .btn-icon:hover { background: rgba(255,255,255,0.1); }
        .btn-icon.active-up { background: rgba(16, 185, 129, 0.2); border-color: var(--accent-green); }
        .btn-icon.active-down { background: rgba(239, 68, 68, 0.2); border-color: var(--accent-red); }

        /* --- Right Panel (Status) --- */
        .status-item {
            margin-bottom: 1.2rem;
            padding-bottom: 1.2rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .status-item:last-child { border-bottom: none; }

        .status-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }

        .status-val {
            font-size: 1rem;
            font-weight: 500;
        }

        .status-val.good { color: var(--accent-green); }
        .status-val.warn { color: #f59e0b; }
        
        .engine-list {
            list-style: none;
            margin-top: 0.5rem;
        }
        
        .engine-list li {
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .engine-list li.active::before {
            content: "●";
            color: var(--accent-green);
        }
        .engine-list li.inactive::before {
            content: "●";
            color: var(--text-muted);
        }
        
        /* Empty state */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            text-align: center;
            grid-column: 1 / -1;
        }
        
        .empty-state svg {
            width: 64px;
            height: 64px;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        @media (max-width: 1024px) {
            .layout {
                grid-template-columns: 1fr;
                height: auto;
            }
            .glass-panel { max-height: none; }
        }
    </style>
</head>
<body>

    <header>
        <h1>VER Dashboard</h1>
        <div style="font-size: 0.9rem; color: var(--text-muted);">AI Challenge 2026</div>
    </header>

    <div class="layout">
        <!-- Bảng điều khiển Trái -->
        <div class="glass-panel">
            <div class="search-container">
                <div>
                    <label>TRUY VẤN TÌM KIẾM</label>
                    <textarea id="queryInput" placeholder="Ví dụ: Người đàn ông mặc áo đỏ đang chạy trên phố..."></textarea>
                </div>
                
                <div>
                    <label>CHẾ ĐỘ (TRACK)</label>
                    <div class="track-selector">
                        <div class="radio-btn">
                            <input type="radio" id="trackAuto" name="track" value="auto" checked>
                            <label for="trackAuto">Auto (<2s)</label>
                        </div>
                        <div class="radio-btn">
                            <input type="radio" id="trackSemi" name="track" value="semi">
                            <label for="trackSemi">Semi (<15s)</label>
                        </div>
                    </div>
                </div>

                <button class="btn-primary" id="searchBtn" onclick="performSearch()">
                    <span id="btnText">TÌM KIẾM</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </div>
            
            <div style="margin-top: 2rem;">
                <label>THỐNG KÊ TÌM KIẾM</label>
                <div id="searchStats" style="font-size: 0.85rem; color: var(--text-muted);">
                    Chưa có truy vấn nào được thực hiện.
                </div>
            </div>
        </div>

        <!-- Bảng điều khiển Chính (Kết quả) -->
        <div class="glass-panel" style="overflow-y: auto;">
            <div class="results-header">
                <h2>Kết quả tìm kiếm</h2>
                <span id="resultCount" style="color: var(--text-muted); font-size: 0.9rem;">0 kết quả</span>
            </div>
            
            <div class="results-grid" id="resultsGrid">
                <div class="empty-state">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                    <p>Nhập truy vấn và nhấn Tìm kiếm để xem kết quả.</p>
                </div>
            </div>
        </div>

        <!-- Bảng điều khiển Phải (Trạng thái) -->
        <div class="glass-panel">
            <h3 style="margin-bottom: 1.5rem; font-size: 1rem;">Trạng thái Hệ thống</h3>
            
            <div class="status-item">
                <div class="status-label">Thiết bị Compute</div>
                <div class="status-val" id="sysDevice">Đang tải...</div>
            </div>
            
            <div class="status-item">
                <div class="status-label">Chế độ hiện tại</div>
                <div class="status-val" id="sysTrack">Đang tải...</div>
            </div>
            
            <div class="status-item">
                <div class="status-label">Tổng số Segments</div>
                <div class="status-val good" id="sysSegments">Đang tải...</div>
            </div>
            
            <div class="status-item">
                <div class="status-label">Modules Hoạt động</div>
                <ul class="engine-list" id="engineList">
                    <li class="inactive">Đang tải...</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Fetch System Status
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('sysDevice').innerText = data.device;
                document.getElementById('sysDevice').className = data.cuda_available ? 'status-val good' : 'status-val warn';
                
                document.getElementById('sysTrack').innerText = data.track.toUpperCase();
                document.getElementById('sysSegments').innerText = data.segment_count.toLocaleString();
                
                const engineList = document.getElementById('engineList');
                engineList.innerHTML = '';
                for (const [name, active] of Object.entries(data.engines)) {
                    const li = document.createElement('li');
                    li.className = active ? 'active' : 'inactive';
                    li.innerText = name;
                    engineList.appendChild(li);
                }
            } catch (e) {
                console.error("Lỗi lấy trạng thái:", e);
            }
        }

        // Thực hiện tìm kiếm
        async function performSearch() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const track = document.querySelector('input[name="track"]:checked').value;
            
            // UI Loading
            const btn = document.getElementById('searchBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('btnSpinner');
            const grid = document.getElementById('resultsGrid');
            
            btn.disabled = true;
            btnText.innerText = "ĐANG TÌM...";
            spinner.style.display = "block";
            
            grid.innerHTML = `
                <div class="empty-state">
                    <div class="spinner" style="display:block; width:40px; height:40px; border-width:4px; margin-bottom:1rem; border-top-color: var(--accent-cyan);"></div>
                    <p>Hệ thống đang phân tích đa phương thức...</p>
                </div>
            `;

            try {
                const startTime = performance.now();
                
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, track, top_k: 20 })
                });
                
                const results = await res.json();
                const endTime = performance.now();
                const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
                
                document.getElementById('searchStats').innerHTML = `
                    <p>Track: <b>${track.toUpperCase()}</b></p>
                    <p>Thời gian: <b>${timeTaken}s</b></p>
                    <p>Kết quả: <b>${results.length}</b></p>
                `;
                
                renderResults(results, track);
                
            } catch (e) {
                grid.innerHTML = `
                    <div class="empty-state" style="color: var(--accent-red);">
                        <p>Lỗi kết nối tới máy chủ.</p>
                        <p style="font-size: 0.8rem; margin-top: 0.5rem;">${e.message}</p>
                    </div>
                `;
            } finally {
                btn.disabled = false;
                btnText.innerText = "TÌM KIẾM";
                spinner.style.display = "none";
            }
        }

        function renderResults(results, track) {
            const grid = document.getElementById('resultsGrid');
            document.getElementById('resultCount').innerText = `${results.length} kết quả`;
            
            if (results.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <p>Không tìm thấy kết quả phù hợp.</p>
                    </div>
                `;
                return;
            }

            let html = '';
            results.forEach((res, index) => {
                const scorePct = Math.round(res.score * 100);
                let scoreColorClass = 'text'; // Default
                if (res.source.includes('visual')) scoreColorClass = 'visual';
                else if (res.source.includes('audio')) scoreColorClass = 'audio';
                else if (res.source === 'fused') scoreColorClass = 'fused';

                // Nút feedback chỉ hiện ở track semi
                const feedbackHtml = track === 'semi' ? `
                    <div class="feedback-actions">
                        <button class="btn-icon" onclick="sendFeedback('${res.segment_id}', true, this)">👍</button>
                        <button class="btn-icon" onclick="sendFeedback('${res.segment_id}', false, this)">👎</button>
                    </div>
                ` : '';
                
                const timeStr = res.time_display || '00:00';
                const text = res.text ? (res.text.length > 100 ? res.text.substring(0, 100) + '...' : res.text) : '<i>Không có text (visual only)</i>';

                html += `
                    <div class="result-card">
                        <div class="thumbnail">
                            <img src="/api/keyframe/${res.segment_id}" alt="Keyframe" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMzMyMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZmlsbD0iIzc3NyIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Tm8gSW1hZ2U8L3RleHQ+PC9zdmc+'">
                            <div class="badge-source ${scoreColorClass}">${res.source}</div>
                            <div class="time-badge">${timeStr}</div>
                        </div>
                        <div class="card-body">
                            <div class="card-title">${res.video_file || 'Unknown Video'}</div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
                            </div>
                            <div style="font-size: 0.75rem; color: var(--accent-cyan); margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                                <span>Score: ${res.score.toFixed(4)}</span>
                                <span>Rank #${index + 1}</span>
                            </div>
                            <div class="card-text">${text}</div>
                            ${feedbackHtml}
                        </div>
                    </div>
                `;
            });
            grid.innerHTML = html;
        }

        async function sendFeedback(segmentId, isRelevant, btn) {
            // UI Update
            const parent = btn.parentElement;
            const btns = parent.querySelectorAll('.btn-icon');
            btns.forEach(b => b.className = 'btn-icon');
            btn.className = `btn-icon ${isRelevant ? 'active-up' : 'active-down'}`;
            
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ segment_id: segmentId, relevant: isRelevant })
                });
            } catch (e) {
                console.error("Lỗi gửi feedback:", e);
            }
        }

        // Init
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                performSearch();
            }
        });

        // Load status ngay khi mở trang
        fetchStatus();
        // Refresh status mỗi 30s
        setInterval(fetchStatus, 30000);
    </script>
</body>
</html>
"""

# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_TEMPLATE

@app.get("/api/status")
async def get_status():
    if pipeline:
        return pipeline.get_status()
    return {"status": "uninitialized"}

@app.post("/api/search")
async def search(req: SearchRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline is initializing")
        
    # Chuyển track nếu cần
    if pipeline.track != req.track:
        pipeline.set_track(req.track)
        
    results = pipeline.search(req.query, top_k=req.top_k)
    return [r.to_dict() for r in results]

@app.post("/api/preprocess")
async def preprocess(req: PreprocessRequest, background_tasks: BackgroundTasks):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline is initializing")
        
    def _run():
        pipeline.preprocess(video_path=req.video_path, video_dir=req.video_dir)
        
    background_tasks.add_task(_run)
    return {"message": "Preprocessing started in background"}

# @app.get("/api/keyframe/{segment_id}")
# async def get_keyframe(segment_id: str):
#     # Tìm segment_id trong database (hoặc generate lại frame nhanh)
#     # Vì để đơn giản, ta tìm kiếm file ảnh trong thư mục
#     # Thường được lưu ở data/processed/keyframes/{segment_id}.jpg
#     kf_path = Path(PROCESSED_DIR) / "keyframes" / f"{segment_id}.jpg"
#     if kf_path.exists():
#         return FileResponse(str(kf_path))
#     else:
#         # Nếu không có ảnh sẵn, ta thử fallback một ảnh 1px
#         raise HTTPException(status_code=404, detail="Keyframe not found")
    
@app.get("/api/keyframe/{segment_id}")
async def get_keyframe(segment_id: str):
    # Đường dẫn dự kiến
    kf_path = Path(PROCESSED_DIR) / "keyframes" / f"{segment_id}.jpg"
    
    # --- THÊM LOG DEBUG ---
    log_info(f"Đang tìm ảnh tại: {kf_path.absolute()}")
    log_info(f"Kiểm tra tồn tại: {kf_path.exists()}")
    
    if kf_path.exists():
        return FileResponse(str(kf_path))
    else:
        # Ép API trả về đường dẫn tìm kiếm để dễ dàng debug trên giao diện
        raise HTTPException(
            status_code=404, 
            detail=f"Lỗi 404: Không thể đọc file tại {kf_path.absolute()}"
        )

@app.post("/api/feedback")
async def process_feedback(req: FeedbackRequest):
    # Trong thực tế, ghi feedback vào cơ sở dữ liệu để finetune model sau này
    log_info(f"Nhận feedback: Segment {req.segment_id} -> Relevant: {req.relevant}")
    return {"status": "ok"}

# =====================================================================
# Khởi chạy Server
# =====================================================================

def run_server(host="localhost", port=8000):
    log_step("SERVER", f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_server()
