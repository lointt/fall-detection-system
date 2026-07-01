"""
main.py — FastAPI Backend Server
==================================
Khởi động:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /                          → redirect về frontend/index.html
  GET  /video_feed                → MJPEG stream từ webcam
  WS   /ws/status                 → WebSocket push trạng thái detection
  GET  /api/history?date=YYYY-MM-DD → danh sách video theo ngày
  GET  /api/video/{filename}      → stream video file lịch sử
  GET  /api/fall_times/{filename} → danh sách timestamp té ngã trong video
  GET  /api/logs                  → 100 dòng log gần nhất
"""

import os
import sys
import json
import asyncio
import threading
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    StreamingResponse, HTMLResponse, FileResponse, JSONResponse, RedirectResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Đường dẫn ────────────────────────────────────────────────────────────────
# BACKEND_DIR = thư mục chứa main.py (backend/)
# ROOT_DIR    = thư mục gốc dự án (chứa frontend/, backend/, logs/, ...)
BACKEND_DIR         = Path(__file__).parent
ROOT_DIR            = BACKEND_DIR.parent
FRONTEND_DIR        = ROOT_DIR / "frontend"
# predicted_videos và fall_snapshots nằm trong backend/ (detector.py dùng BASE_DIR=backend/)
PREDICTED_VIDEO_DIR = BACKEND_DIR / "predicted_videos"
SNAPSHOT_DIR        = BACKEND_DIR / "fall_snapshots"
LOG_PATH            = BACKEND_DIR / "logs" / "info.log"

# Thêm backend vào sys.path để import detector, classifier, sender
sys.path.insert(0, str(BACKEND_DIR))

# ── Import detector ───────────────────────────────────────────────────────────
import detector as det

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("fall_detection")

# ── WebSocket manager ─────────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self.active:
                self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, ensure_ascii=False)
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

ws_manager = WSManager()

# ── Bridge: detector callback → asyncio broadcast ─────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None

def _on_status_update(data: dict):
    """Được gọi từ detector thread, schedule broadcast sang event loop."""
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(data), _loop)

det.register_status_callback(_on_status_update)

# ── Lifespan: thay thế on_event (không còn deprecated) ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────
    global _loop
    _loop = asyncio.get_running_loop()

    PREDICTED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("FastAPI server khởi động")

    t = threading.Thread(target=det.run_detection, daemon=True, name="detector")
    t.start()
    logger.info("Detector thread đã khởi động")

    yield   # ← ứng dụng đang chạy

    # ── SHUTDOWN ─────────────────────────────────
    det.stop_detection()
    logger.info("FastAPI server dừng")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Fall Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─────────────────────────────────────────────
# ROUTES — FRONTEND
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h2>Frontend không tìm thấy. Kiểm tra thư mục frontend/</h2>")

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    history_path = FRONTEND_DIR / "history.html"
    if history_path.exists():
        return FileResponse(str(history_path))
    return HTMLResponse("<h2>history.html không tìm thấy.</h2>")

# ─────────────────────────────────────────────
# ROUTES — WEBCAM STREAM
# ─────────────────────────────────────────────

@app.get("/video_feed")
async def video_feed():
    """MJPEG stream từ webcam (dùng cho <img src='/video_feed'>)."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )

async def _mjpeg_generator():
    """Async generator lấy frames từ det.frame_queue."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Chạy blocking queue.get trong executor để không block event loop
            jpeg_bytes = await loop.run_in_executor(
                None, lambda: det.frame_queue.get(timeout=2.0)
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
        except Exception:
            await asyncio.sleep(0.05)
            continue

# ─────────────────────────────────────────────
# ROUTES — WEBSOCKET STATUS
# ─────────────────────────────────────────────

@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Push trạng thái detection (label, prob, is_fall) real-time."""
    await ws_manager.connect(websocket)
    try:
        # Gửi trạng thái hiện tại ngay khi connect
        with det.status_lock:
            await websocket.send_text(
                json.dumps(det.current_status, ensure_ascii=False)
            )
        # Giữ kết nối mở — ping keepalive
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_text(json.dumps({"ping": True}))
            except Exception:
                # Client đã đóng kết nối (đổi trang, đóng tab, mất mạng...)
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await ws_manager.disconnect(websocket)

# ─────────────────────────────────────────────
# ROUTES — HISTORY API
# ─────────────────────────────────────────────

@app.get("/api/history")
async def get_history(date: str = ""):
    """
    Trả về danh sách video đã ghi trong ngày.
    Query param: date=YYYY-MM-DD (mặc định = hôm nay)
    Response: [ { filename, timestamp, fall_count, duration } ]
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return JSONResponse({"error": "date format phải là YYYY-MM-DD"}, status_code=400)

    result = []
    if not PREDICTED_VIDEO_DIR.exists():
        return JSONResponse(result)

    # Video được chia thành các segment ~5 phút (SEGMENT_DURATION_SEC).
    # Segment ĐANG ghi (current_session_path) chưa flush xong header nên
    # KHÔNG thể phát được — phải loại khỏi danh sách. Các segment đã đóng
    # (writer.release() đã chạy) thì xem lại được ngay lập tức.
    # Extension thực tế phụ thuộc codec mà detector.py dò được trên máy
    # (mp4 nếu có H.264, avi nếu fallback Xvid/MJPEG).
    candidates = list(PREDICTED_VIDEO_DIR.glob("*.mp4")) + list(PREDICTED_VIDEO_DIR.glob("*.avi"))

    for vid in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        # Bỏ qua segment đang ghi dở
        if str(vid) == det.current_session_path:
            continue

        stem = vid.stem
        try:
            dt = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        if dt.date() != target_date:
            continue

        # Bỏ qua file rỗng/quá nhỏ (vừa tạo, chưa kịp ghi gì)
        try:
            if vid.stat().st_size < 1024:
                continue
        except OSError:
            continue

        fall_count   = _count_falls_in_video(vid)
        duration_str = _get_video_duration(vid)

        result.append({
            "filename":   vid.name,
            "timestamp":  dt.isoformat(),
            "fall_count": fall_count,
            "duration":   duration_str,
        })

    return JSONResponse(result)


def _count_falls_in_video(video_path: Path) -> int:
    """Đếm số sự kiện té ngã được log trong khoảng thời gian của video."""
    if not LOG_PATH.exists():
        return 0
    try:
        stem = video_path.stem
        dt   = datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return 0

    count = 0
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if "DETECTOR | TÉ NGÃ" in line:
                    # Parse timestamp từ log line: "YYYY-MM-DD HH:MM:SS | ..."
                    try:
                        log_ts_str = line.split("|")[0].strip()
                        log_ts     = datetime.strptime(log_ts_str, "%Y-%m-%d %H:%M:%S")
                        # Kiểm tra log thuộc về video session này (trong vòng 2 giờ)
                        if abs((log_ts - dt).total_seconds()) < 7200:
                            count += 1
                    except Exception:
                        continue
    except Exception:
        pass
    return count


def _get_video_duration(video_path: Path) -> str:
    """Lấy thời lượng video dạng mm:ss."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        secs = int(fc / fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"
    except Exception:
        return "--:--"


# Media type theo extension thực tế (detector.py có thể fallback .avi
# nếu máy không có encoder H.264)
_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
}


@app.get("/api/video/{filename}")
async def get_video(filename: str, request: Request):
    """
    Stream video với HTTP Range Request đầy đủ.
    Cần thiết để browser trên Windows có thể seek/tua không bị WinError 10054.
    """
    safe_name  = Path(filename).name
    video_path = PREDICTED_VIDEO_DIR / safe_name

    if not video_path.exists():
        return JSONResponse({"error": "Video không tìm thấy"}, status_code=404)

    media_type = _VIDEO_MEDIA_TYPES.get(video_path.suffix.lower(), "application/octet-stream")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range", None)

    CHUNK = 1024 * 512  # 512 KB mỗi chunk

    if range_header is None:
        # Không có Range → trả toàn bộ file (chunk từng phần)
        async def full_gen():
            with open(video_path, "rb") as f:
                while True:
                    data = f.read(CHUNK)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            full_gen(),
            status_code=200,
            media_type=media_type,
            headers={
                "Content-Length":      str(file_size),
                "Accept-Ranges":       "bytes",
                "Cache-Control":       "no-cache",
            },
        )

    # Parse Range header: "bytes=start-end"
    try:
        range_val  = range_header.replace("bytes=", "")
        parts      = range_val.split("-")
        start      = int(parts[0]) if parts[0] else 0
        end        = int(parts[1]) if parts[1] else file_size - 1
    except Exception:
        return JSONResponse({"error": "Invalid Range"}, status_code=416)

    end        = min(end, file_size - 1)
    chunk_size = end - start + 1

    async def range_gen():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        range_gen(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk_size),
            "Accept-Ranges":  "bytes",
            "Cache-Control":  "no-cache",
        },
    )


@app.get("/api/fall_times/{filename}")
async def get_fall_times(filename: str):
    """
    Trả về danh sách timestamp (giây) có sự kiện té ngã trong video.
    Dùng để vẽ markers trên timeline ở history.html.
    Response: [ 12.4, 45.0, ... ]
    """
    safe_name  = Path(filename).name
    video_path = PREDICTED_VIDEO_DIR / safe_name

    if not video_path.exists():
        return JSONResponse([])

    # Parse tên file → datetime bắt đầu session
    stem = Path(safe_name).stem
    try:
        session_start = datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return JSONResponse([])

    fall_seconds: List[float] = []

    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if "DETECTOR | TÉ NGÃ" not in line:
                        continue
                    try:
                        log_ts_str = line.split("|")[0].strip()
                        log_ts     = datetime.strptime(log_ts_str, "%Y-%m-%d %H:%M:%S")
                        delta      = (log_ts - session_start).total_seconds()
                        if 0 <= delta < 7200:   # trong session
                            fall_seconds.append(round(delta, 1))
                    except Exception:
                        continue
        except Exception:
            pass

    return JSONResponse(fall_seconds)


# ─────────────────────────────────────────────
# ROUTES — LOGS API
# ─────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Trả về N dòng log gần nhất từ info.log."""
    if not LOG_PATH.exists():
        return JSONResponse({"logs": []})
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        return JSONResponse({"logs": all_lines[-lines:]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────
# CATCH-ALL — Serve frontend static files
# (phải đứng SAU tất cả /api routes)
# ─────────────────────────────────────────────

@app.get("/{filename:path}")
async def serve_frontend_file(filename: str):
    """Serve CSS/JS/static assets từ frontend/."""
    safe      = filename.lstrip("/").replace("..", "")
    file_path = FRONTEND_DIR / safe
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    return JSONResponse({"error": "Not found"}, status_code=404)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # reload=True nếu muốn hot-reload lúc dev
        log_level="info",
    )
