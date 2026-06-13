"""
app.py — Web API & Demo UI (FastAPI)

Endpoints:
  GET  /                 → Demo web UI (HTML)
  GET  /video_feed       → MJPEG stream
  GET  /api/status       → System status
  GET  /api/objects      → Danh sách objects frame hiện tại
  GET  /api/events       → Event log (filter theo level, limit)
  GET  /api/stats        → Thống kê tổng hợp
  GET  /api/zones        → Danh sách zones hiện tại
  POST /api/zones        → Cập nhật zones
  POST /api/source       → Đổi nguồn video

Chạy: python app.py
"""
import os

# PHẢI set trước khi import bất kỳ module nào dùng OpenCV/FFmpeg.
# extra_hw_frames BỊ CẤM — kích hoạt async hw decoder → crash assertion fctx->async_lock.
# thread_type;slice: ngăn H.265/HEVC dùng frame-level async threading.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|threads;1"
    "|thread_type;slice"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|max_delay;500000"
    "|reorder_queue_size;0"
)

import io
import time
import asyncio
import logging
import threading
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import (
    HTMLResponse, StreamingResponse, JSONResponse
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import API_CONFIG, VIDEO_CONFIG, STATIC_DIR, TEMPLATES_DIR, ZONES_PERSIST_FILE, TELEGRAM_CONFIG
from pipeline import CameraPipeline

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title       = API_CONFIG["title"],
    description = API_CONFIG["description"],
    version     = API_CONFIG["version"],
)

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global pipeline instance
pipeline: Optional[CameraPipeline] = None

# Thư mục lưu video upload
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# Pydantic models cho request body
# ============================================================
class ZonesUpdate(BaseModel):
    zones: list[dict]


class SourceUpdate(BaseModel):
    source: str | int


class TelegramConfigUpdate(BaseModel):
    bot_token    : str   = ""
    chat_id      : str   = ""
    enabled      : bool  = False
    min_level    : str   = "alert"
    cooldown_sec : float = 30.0
    send_photo   : bool  = True


# ============================================================
# Startup / Shutdown
# ============================================================
@app.on_event("startup")
async def startup():
    global pipeline
    # ✅ Dùng VIDEO_CONFIG (đã đọc CAMERA_SOURCE từ .env) — không phải API_CONFIG
    cam_source = VIDEO_CONFIG["default_source"]
    logger.info(f"[Startup] Camera source: {cam_source!r}")

    pipeline = CameraPipeline(source=cam_source)
    try:
        pipeline.setup()
        pipeline.start_background()
        # Log zone persistence status (per-camera)
        n_zones    = len(pipeline.get_zones())
        zones_file = pipeline.zone_det.get_persist_file()
        if zones_file.exists():
            logger.info(
                f"Pipeline started. Restored {n_zones} zones "
                f"from {zones_file.name}"
            )
        else:
            logger.info(
                f"Pipeline started. {n_zones} default zones loaded "
                f"(no zones file yet for {zones_file.name})."
            )
    except Exception as e:
        logger.error(f"Pipeline setup failed: {e}")

    # Auto-load Telegram config từ config.py (chạy luôn, kể cả khi pipeline lỗi)
    _apply_telegram_config_from_file()


@app.on_event("shutdown")
async def shutdown():
    if pipeline:
        pipeline.stop()
    logger.info("Shutdown complete.")


# ============================================================
# Helper
# ============================================================
def _ensure_pipeline():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return pipeline


def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Encode frame thành JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _apply_telegram_config_from_file():
    """
    Đọc TELEGRAM_CONFIG từ config.py và áp dụng lên pipeline đang chạy.
    Được gọi tự động khi startup và khi gọi /api/telegram/reload.
    """
    global pipeline
    if pipeline is None:
        return {"status": "skipped", "reason": "pipeline not ready"}

    tg = TELEGRAM_CONFIG
    if tg.get("bot_token") and tg.get("chat_id"):
        pipeline.telegram.configure(
            bot_token    = tg["bot_token"],
            chat_id      = tg["chat_id"],
            enabled      = tg.get("enabled", True),
            min_level    = tg.get("min_level", "alert"),
            cooldown_sec = tg.get("cooldown_sec", 30.0),
            send_photo   = tg.get("send_photo", True),
        )
        logger.info(
            f"[Telegram] Auto-configured from config.py: "
            f"chat_id={tg['chat_id']}, enabled={tg.get('enabled', True)}, "
            f"min_level={tg.get('min_level', 'alert')}"
        )
        return {"status": "ok", "enabled": tg.get("enabled", True), "chat_id": tg["chat_id"]}
    else:
        logger.warning(
            "[Telegram] Not configured: bot_token or chat_id missing in TELEGRAM_CONFIG"
        )
        return {"status": "skipped", "reason": "bot_token or chat_id missing in config.py"}


# ============================================================
# Routes — UI
# ============================================================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """Demo Web UI."""
    return HTMLResponse(content=_build_ui_html())


# ============================================================
# Routes — Video
# ============================================================
@app.get("/video_feed", summary="MJPEG live stream")
async def video_feed():
    """Phát MJPEG stream từ camera pipeline."""

    async def generate():
        while True:
            pl = _ensure_pipeline()
            frame = pl.get_latest_frame()
            if frame is not None:
                jpg = _frame_to_jpeg(frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/snapshot", summary="Single JPEG frame (for Zone Editor)")
async def get_snapshot():
    """
    Trả về 1 frame JPEG tĩnh (không có annotation) để dùng trong Zone Editor.
    Dùng raw frame thay vì annotated frame để không bị che bởi bounding box / zone lines.
    """
    pl = _ensure_pipeline()

    # Thử lấy raw frame từ video processor trước
    raw_frame = pl.video.get_latest_raw_frame() if hasattr(pl.video, "get_latest_raw_frame") else None

    # Fallback: dùng annotated frame nếu không có raw
    if raw_frame is None:
        raw_frame = pl.get_latest_frame()

    if raw_frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    jpg_bytes = _frame_to_jpeg(raw_frame)
    return StreamingResponse(
        io.BytesIO(jpg_bytes),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ============================================================
# Routes — API
# ============================================================
@app.get("/api/status")
async def get_status():
    pl = _ensure_pipeline()
    result = pl.get_latest_result()
    action_status = pl.action_rec.get_status() if hasattr(pl, "action_rec") else {}
    return {
        "status"      : "running",
        "fps"         : round(pl.get_fps(), 1),
        "zones"       : len(pl.get_zones()),
        "objects"     : len(result.objects) if result else 0,
        "frame_id"    : result.frame_id if result else 0,
        "action_mode" : action_status.get("mode", "unknown"),
        "gru_available": action_status.get("gru_available", False),
        "mp_available" : action_status.get("mp_available", False),
    }


@app.get("/api/objects")
async def get_objects():
    pl     = _ensure_pipeline()
    result = pl.get_latest_result()
    if result is None:
        return {"objects": [], "frame_id": 0}
    return result.to_dict()


@app.get("/api/events")
async def get_events(
    level : Optional[str] = Query(None, description="Filter: warning/alert/critical"),
    limit : int           = Query(50,   ge=1, le=500),
    since : Optional[float] = Query(None, description="Unix timestamp"),
):
    pl = _ensure_pipeline()
    events = pl.evt_log.get_recent(n=limit, level=level, since=since)
    return {"total": len(events), "events": events}


@app.get("/api/stats")
async def get_stats():
    pl = _ensure_pipeline()
    return pl.evt_log.get_stats()


@app.get("/api/zones")
async def get_zones():
    pl = _ensure_pipeline()
    return {
        "zones"     : pl.get_zones(),
        "zones_file": pl.zone_det.get_persist_file().name,  # file zones của camera này
    }


@app.post("/api/zones")
async def update_zones(body: ZonesUpdate):
    """Cập nhật zones runtime và lưu persistent theo camera."""
    pl = _ensure_pipeline()
    pl.update_zones(body.zones)
    return {
        "status"      : "ok",
        "zones_count" : len(body.zones),
        "persisted_to": str(pl.zone_det.get_persist_file()),
    }


@app.delete("/api/zones")
async def clear_zones():
    """Xóa toàn bộ zones của camera hiện tại (runtime + file persist)."""
    pl = _ensure_pipeline()
    zones_file = pl.zone_det.get_persist_file().name
    pl.zone_det.clear_zones()
    return {
        "status"     : "ok",
        "message"    : f"All zones cleared for current camera ({zones_file}).",
    }


@app.post("/api/source")
async def change_source(body: SourceUpdate):
    """Đổi nguồn video (dừng pipeline cũ, khởi động mới)."""
    global pipeline
    old = pipeline
    if old:
        old.stop()
        time.sleep(0.5)

    try:
        src = body.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        pipeline = CameraPipeline(source=src)
        pipeline.setup()
        pipeline.start_background()
        return {"status": "ok", "source": str(src)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload file video và chuyển pipeline sang dùng file đó."""
    global pipeline

    # Kiểm tra định dạng file
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {suffix}. Dùng: {', '.join(allowed_exts)}"
        )

    # Lưu file vào thư mục uploads/
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {e}")
    finally:
        await file.close()

    # Dừng pipeline cũ
    old = pipeline
    if old:
        old.stop()
        time.sleep(0.5)

    # Khởi động pipeline với file mới
    try:
        pipeline = CameraPipeline(source=str(save_path))
        pipeline.setup()
        pipeline.start_background()
        logger.info(f"Pipeline switched to uploaded file: {save_path}")
        return {
            "status" : "ok",
            "source" : str(save_path),
            "filename": file.filename,
            "size_kb" : round(save_path.stat().st_size / 1024, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/source")
async def get_current_source():
    """Trả về thông tin nguồn video hiện tại."""
    pl = _ensure_pipeline()
    src = pl._source
    if src is None or src == 0 or src == "0":
        return {"type": "webcam", "label": "Webcam (index 0)"}
    elif isinstance(src, int):
        return {"type": "webcam", "label": f"Webcam (index {src})"}
    elif isinstance(src, str) and src.lower().startswith("rtsp"):
        # Che mật khẩu trong URL khi trả về
        from modules.video_processor import VideoProcessor
        masked = VideoProcessor._mask_rtsp_url(src)
        return {"type": "rtsp", "label": masked, "path": masked}
    else:
        p = Path(str(src))
        return {"type": "file", "label": p.name, "path": str(src)}


@app.get("/api/rtsp/status", summary="Trạng thái kết nối RTSP camera")
async def get_rtsp_status():
    """
    Trả về trạng thái kết nối RTSP:
    - connected: đang nhận được frame
    - reconnect_count: số lần đã phải reconnect
    - source_masked: URL camera (mật khẩu được che)
    """
    pl = _ensure_pipeline()
    if hasattr(pl.video, "get_connection_status"):
        status = pl.video.get_connection_status()
    else:
        status = {"connected": True, "is_rtsp": False}
    return status


# ============================================================
# Routes — Telegram Notification
# ============================================================
@app.get("/api/telegram/config", summary="Lấy cấu hình Telegram hiện tại")
async def get_telegram_config():
    """Trả về cấu hình Telegram (bot token được che)."""
    pl = _ensure_pipeline()
    return pl.telegram.get_config()


@app.post("/api/telegram/config", summary="Cập nhật cấu hình Telegram")
async def set_telegram_config(body: TelegramConfigUpdate):
    """
    Cấu hình Telegram Bot cho push notification.

    Hướng dẫn:
    1. Tạo bot tại @BotFather trên Telegram → lấy BOT_TOKEN
    2. Gửi tin nhắn cho bot, sau đó truy cập:
       https://api.telegram.org/bot<TOKEN>/getUpdates → lấy chat_id
    3. Điền thông tin vào form bên dướdi và gửi request này.
    """
    pl = _ensure_pipeline()
    pl.telegram.configure(
        bot_token    = body.bot_token,
        chat_id      = body.chat_id,
        enabled      = body.enabled,
        min_level    = body.min_level,
        cooldown_sec = body.cooldown_sec,
        send_photo   = body.send_photo,
    )
    return {
        "status" : "ok",
        "config" : pl.telegram.get_config(),
    }


@app.post("/api/telegram/test", summary="Gửi tin nhắn test đến Telegram")
async def test_telegram():
    """Test kết nối Telegram bằng cách gửi 1 tin nhắn thử."""
    pl = _ensure_pipeline()
    success, message = pl.telegram.test_connection()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@app.post("/api/telegram/reload", summary="Reload Telegram config từ config.py (không cần restart)")
async def reload_telegram_config():
    """
    Áp dụng lại TELEGRAM_CONFIG từ config.py lên pipeline đang chạy.
    Dùng khi bạn đã sửa config.py nhưng không muốn restart server.
    """
    result = _apply_telegram_config_from_file()
    if result.get("status") == "ok":
        pl = _ensure_pipeline()
        return {
            "status"  : "ok",
            "message" : "Telegram config reloaded from config.py",
            "config"  : pl.telegram.get_config(),
        }
    else:
        raise HTTPException(status_code=400, detail=result.get("reason", "Unknown error"))


# ============================================================
# Routes — Alert Video Recorder
# ============================================================
class RecorderConfigUpdate(BaseModel):
    enabled         : bool  = True
    min_level       : str   = "alert"
    pre_buffer_sec  : float = 5.0
    post_buffer_sec : float = 8.0
    cooldown_sec    : float = 20.0
    max_clips       : int   = 50


@app.get("/api/recorder/config", summary="Lấy cấu hình Alert Recorder")
async def get_recorder_config():
    pl = _ensure_pipeline()
    return pl.recorder.get_config()


@app.post("/api/recorder/config", summary="Cập nhật cấu hình Alert Recorder")
async def set_recorder_config(body: RecorderConfigUpdate):
    pl = _ensure_pipeline()
    pl.recorder.configure(
        enabled         = body.enabled,
        min_level       = body.min_level,
        pre_buffer_sec  = body.pre_buffer_sec,
        post_buffer_sec = body.post_buffer_sec,
        cooldown_sec    = body.cooldown_sec,
        max_clips       = body.max_clips,
    )
    return {"status": "ok", "config": pl.recorder.get_config()}


@app.get("/api/recorder/clips", summary="Danh sách video clips đã ghi")
async def list_clips():
    pl = _ensure_pipeline()
    clips = pl.recorder.list_clips()
    return {"total": len(clips), "clips": clips}


@app.delete("/api/recorder/clips/{filename}", summary="Xóa một clip")
async def delete_clip(filename: str):
    from config import RECORDINGS_DIR
    safe_name = Path(filename).name  # prevent path traversal
    clip_path = RECORDINGS_DIR / safe_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{filename}' not found")
    clip_path.unlink()
    return {"status": "ok", "deleted": filename}


@app.get("/api/recorder/clips/{filename}", summary="Download một clip")
async def download_clip(filename: str):
    from fastapi.responses import FileResponse
    from config import RECORDINGS_DIR
    safe_name = Path(filename).name
    clip_path = RECORDINGS_DIR / safe_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{filename}' not found")
    return FileResponse(
        path        = str(clip_path),
        media_type  = "video/mp4",
        filename    = safe_name,
    )


@app.get("/recordings/{filename}", summary="Stream video clip (hỗ trợ HTTP Range cho trình duyệt)")
async def stream_clip(filename: str, request: Request):
    """
    Stream video clip với HTTP Range support để trình duyệt có thể
    tua/seek video trực tiếp qua thẻ <video>.
    """
    from config import RECORDINGS_DIR

    safe_name = Path(filename).name
    clip_path = RECORDINGS_DIR / safe_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{filename}' not found")

    file_size = clip_path.stat().st_size
    range_header = request.headers.get("range")

    def _iter_file(path: Path, start: int, end: int, chunk: int = 1 << 20):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        # Parse "bytes=start-end"
        try:
            range_val = range_header.replace("bytes=", "")
            range_start, range_end = range_val.split("-")
            start = int(range_start)
            end   = int(range_end) if range_end else file_size - 1
        except Exception:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        end = min(end, file_size - 1)
        content_length = end - start + 1

        return StreamingResponse(
            _iter_file(clip_path, start, end),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range"  : f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges"  : "bytes",
                "Content-Length" : str(content_length),
                "Cache-Control"  : "no-cache",
            },
        )
    else:
        return StreamingResponse(
            _iter_file(clip_path, 0, file_size - 1),
            status_code=200,
            media_type="video/mp4",
            headers={
                "Accept-Ranges"  : "bytes",
                "Content-Length" : str(file_size),
                "Cache-Control"  : "no-cache",
            },
        )


# ============================================================
# Routes — Historical Analytics
# ============================================================
@app.get("/api/analytics/summary", summary="Thống kê tổng hợp lịch sử")
async def get_analytics_summary():
    """Trả về analytics chi tiết dựa trên toàn bộ event log."""
    pl = _ensure_pipeline()
    stats = pl.evt_log.get_stats()

    # Thêm thông tin recorder
    clips = pl.recorder.list_clips()
    stats["recordings"] = {
        "total_clips" : len(clips),
        "latest_clip" : clips[0]["filename"] if clips else None,
    }
    return stats


@app.get("/api/analytics/timeline", summary="Timeline events theo giờ (24h gần nhất)")
async def get_analytics_timeline(
    hours: int = Query(24, ge=1, le=168, description="Số giờ nhìn lại")
):
    """
    Trả về số lượng events theo từng giờ trong N giờ gần nhất.
    Dùng để vẽ biểu đồ timeline trên dashboard.
    """
    pl   = _ensure_pipeline()
    now  = time.time()
    since = now - hours * 3600

    events = pl.evt_log.get_recent(n=10000, since=since)

    # Bucket theo giờ
    buckets: dict[str, dict] = {}
    for evt in events:
        ts  = evt.get("timestamp", 0)
        lvl = evt.get("level", "normal")
        # Key: "YYYY-MM-DD HH:00"
        bucket_key = time.strftime("%Y-%m-%d %H:00", time.localtime(ts))
        if bucket_key not in buckets:
            buckets[bucket_key] = {"time": bucket_key, "total": 0, "warning": 0, "alert": 0, "critical": 0}
        buckets[bucket_key]["total"] += 1
        if lvl in ("warning", "alert", "critical"):
            buckets[bucket_key][lvl] = buckets[bucket_key].get(lvl, 0) + 1

    timeline = sorted(buckets.values(), key=lambda x: x["time"])
    return {"hours": hours, "buckets": timeline, "total_events": len(events)}


@app.get("/api/analytics/export", summary="Export toàn bộ events dưới dạng JSON")
async def export_analytics():
    """Export toàn bộ event log dưới dạng JSON để phân tích offline."""
    pl = _ensure_pipeline()
    events = pl.evt_log.export_json()
    return JSONResponse(
        content={"total": len(events), "events": events},
        headers={
            "Content-Disposition": f'attachment; filename="events_{int(time.time())}.json"'
        }
    )


# ============================================================
# Routes — NLG Engine (Gemini AI)
# ============================================================

class NLGConfigUpdate(BaseModel):
    api_key : str  = ""
    enabled : bool = True


@app.get("/api/nlg/status", summary="Trạng thái NLG Engine (Gemini AI)")
async def get_nlg_status():
    """
    Trả về trạng thái chi tiết của NLG Engine:
    - initialized: đã kết nối Gemini chưa
    - available: có thể sinh câu không (không trong cooldown)
    - success_rate: tỷ lệ thành công / tổng số lần gọi
    """
    from modules.nlg_engine import get_nlg_engine
    return get_nlg_engine().get_status()


@app.post("/api/nlg/config", summary="Cấu hình NLG Engine (API key, bật/tắt)")
async def set_nlg_config(body: NLGConfigUpdate):
    """
    Cập nhật cấu hình NLG Engine tại runtime — không cần restart server.

    Hướng dẫn lấy Gemini API Key:
    1. Truy cập https://aistudio.google.com/app/apikey
    2. Đăng nhập Google → Tạo API key mới
    3. Dán key vào trường api_key bên dưới
    """
    from modules.nlg_engine import get_nlg_engine
    nlg = get_nlg_engine()

    # Bật/tắt
    nlg.set_enabled(body.enabled)

    # Cập nhật API key nếu có
    init_ok = True
    if body.api_key.strip():
        init_ok = nlg.set_api_key(body.api_key)

    status = nlg.get_status()
    return {
        "status"     : "ok",
        "init_ok"    : init_ok,
        "nlg_status" : status,
    }


@app.post("/api/nlg/test", summary="Test kết nối Gemini — sinh câu mẫu")
async def test_nlg():
    """
    Gọi Gemini để sinh 1 câu thông báo mẫu (tình huống: người lạ lảng vảng khu cấm).
    Dùng để xác nhận API key hợp lệ và kết nối mạng OK.
    """
    from modules.nlg_engine import get_nlg_engine
    nlg    = get_nlg_engine()
    result = nlg.test_generate()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ============================================================
# Routes — SHAP Explanation (ContextNet XGBoost)
# ============================================================

@app.get("/api/shap/feature_importance", summary="Global feature importance của ContextNet XGBoost")
async def get_shap_feature_importance():
    """
    Trả về global feature importance từ model XGBoost (model.feature_importances_).

    Hoạt động ngay cả khi pipeline chưa nhận được camera — chỉ cần model đã load.
    Dùng để vẽ biểu đồ global importance trên dashboard.

    Response:
    - ml_ready: True nếu đang dùng ML, False nếu dùng Rule Engine
    - feature_names: danh sách tên feature (đã sort theo importance desc)
    - importances: giá trị importance tương ứng (sum ≈ 1.0)
    - classes: danh sách 6 alert classes
    """
    pl = _ensure_pipeline()
    data = pl.get_shap_feature_importance()
    if data is None:
        return {
            "ml_ready": False,
            "message": "ContextNet đang dùng Rule Engine — không có SHAP data. "
                       "Train model trước: python scripts/export_training_data.py --retrain",
            "feature_names": [],
            "importances"  : [],
        }
    return data


@app.get("/api/shap/explain", summary="SHAP explanation cho object cụ thể theo track_id")
async def get_shap_explain(
    track_id: int = Query(..., description="track_id của object cần giải thích"),
):
    """
    Tính SHAP values cho 1 object đang được theo dõi trong frame hiện tại.

    SHAP (SHapley Additive exPlanations) giải thích tại sao ContextNet XGBoost
    đưa ra mức cảnh báo này cho object này, dựa trên đóng góp của từng feature.

    - track_id: ID của object từ /api/objects
    - Trả về SHAP values cho predicted class
    - Feature có SHAP > 0: đẩy về phía mức nguy hiểm hơn
    - Feature có SHAP < 0: đẩy về phía an toàn hơn

    Lưu ý: Lần đầu gọi có thể mất 0.5–2s để khởi tạo SHAP TreeExplainer.
    """
    pl = _ensure_pipeline()

    # Kiểm tra ML mode
    if hasattr(pl.ctx_eng, "using_ml") and not pl.ctx_eng.using_ml:
        return {
            "ml_ready": False,
            "track_id": track_id,
            "message" : "ContextNet đang dùng Rule Engine — không có SHAP data.",
        }

    data = pl.get_shap_for_object(track_id)
    if data is None:
        # Phân biệt: object không tồn tại vs ML chưa sẵn sàng
        result = pl.get_latest_result()
        tracked_ids = [o.track_id for o in result.objects] if result else []
        if track_id not in tracked_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Object track_id={track_id} không tồn tại trong frame hiện tại. "
                       f"IDs đang active: {tracked_ids}",
            )
        return {
            "ml_ready": False,
            "track_id": track_id,
            "message" : "Không thể tính SHAP cho object này (ML chưa sẵn sàng hoặc lỗi nội bộ).",
        }

    data["track_id"] = track_id
    return data



# ============================================================
# Routes — GPU Status & Benchmark
# ============================================================

@app.get("/api/gpu/status", summary="Trạng thái GPU và inference backend")
async def get_gpu_status():
    """
    Trả về thông tin GPU và backend inference đang dùng:
    - active_provider: DML (GPU) / CPU
    - gpu_name: tên GPU phát hiện được
    - dml_available: DirectML (AMD/Intel/NVIDIA Windows)
    - cuda_available: NVIDIA CUDA
    """
    from modules.gpu_manager import gpu_info
    info = gpu_info()
    pl = _ensure_pipeline()
    detector_info = {
        "using_onnx" : pl.detector._using_onnx,
        "device"     : str(pl.detector._device),
    }
    role_info = {
        "using_onnx"    : pl.role_clf.using_onnx,
        "model_version" : pl.role_clf.model_version or "rule_based",
    }
    return {
        **info,
        "detector" : detector_info,
        "role_clf" : role_info,
    }


@app.post("/api/gpu/benchmark", summary="Chạy benchmark GPU vs CPU")
async def run_benchmark(quick: bool = True):
    """
    Chạy benchmark nhanh so sánh tốc độ GPU (DirectML) vs CPU.
    quick=True: 5 lần mỗi model (~5s)
    quick=False: 30 lần mỗi model (~30s)
    """
    from modules.gpu_manager import run_gpu_benchmark
    import asyncio
    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, run_gpu_benchmark, quick)
    return {
        "status"  : "ok",
        "results" : results,
        "summary" : {
            name: f"{row.get('dml_ms', '?')}ms GPU vs {row.get('cpu_ms', '?')}ms CPU "
                  f"({row.get('speedup', '?')}x speedup)"
            for name, row in results.items()
        }
    }


def _build_ui_html() -> str:
    tpl = TEMPLATES_DIR / "dashboard.html"
    if tpl.exists():
        return tpl.read_text(encoding="utf-8")
    return """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Security Camera — Smart System</title>
<style>
  :root {
    --bg: #0a0e1a;
    --panel: #111827;
    --border: #1e2d45;
    --accent: #00d4ff;
    --warning: #f59e0b;
    --alert: #ef4444;
    --normal: #22c55e;
    --watch: #a78bfa;
    --text: #e2e8f0;
    --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  header {
    background: linear-gradient(135deg, #0a0e1a, #0f172a);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { font-size: 1.2rem; color: var(--accent); letter-spacing: 1px; }
  #fps-badge {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 20px; padding: 4px 14px; font-size: 0.82rem;
  }

  .main { display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; height: calc(100vh - 56px); }

  .video-panel {
    background: #000; border-radius: 12px; border: 1px solid var(--border);
    overflow: hidden; display: flex; align-items: center; justify-content: center;
    position: relative;
  }
  .video-panel img { width: 100%; height: 100%; object-fit: contain; }
  .video-label {
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,.7); border: 1px solid var(--accent);
    border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: var(--accent);
  }

  .sidebar { display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }

  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px;
  }
  .card h3 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 10px; }

  /* Stats grid */
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-box {
    background: #0f172a; border-radius: 8px; padding: 8px 12px; text-align: center;
    border: 1px solid var(--border);
  }
  .stat-box .val { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
  .stat-box .lbl { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

  /* Objects list */
  #objects-list { max-height: 200px; overflow-y: auto; }
  .obj-item {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 8px; border-radius: 6px; margin-bottom: 4px;
    border-left: 3px solid var(--muted); background: #0f172a;
    font-size: 0.80rem;
  }
  .obj-item.normal  { border-color: var(--normal); }
  .obj-item.watch   { border-color: var(--watch); }
  .obj-item.warning { border-color: var(--warning); }
  .obj-item.alert, .obj-item.critical { border-color: var(--alert); }
  .obj-level { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
  .level-normal   { background:#14532d; color:#4ade80; }
  .level-watch    { background:#2e1065; color:#c4b5fd; }
  .level-warning  { background:#451a03; color:#fbbf24; }
  .level-alert, .level-critical { background:#450a0a; color:#f87171; }

  /* Events */
  #events-list { max-height: 160px; overflow-y: auto; font-size: 0.75rem; }
  .event-item { padding: 5px 8px; border-bottom: 1px solid var(--border); color: var(--text); }
  .event-item .evt-time { color: var(--muted); font-size: 0.68rem; }

  /* Source Control */
  .src-tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .src-tab {
    flex: 1; padding: 6px 0; font-size: 0.75rem; text-align: center;
    border-radius: 6px; border: 1px solid var(--border);
    background: transparent; color: var(--muted); cursor: pointer; transition: all .2s;
  }
  .src-tab.active { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
  .src-tab:hover:not(.active) { border-color: var(--accent); color: var(--accent); }

  .src-panel { display: none; flex-direction: column; gap: 8px; }
  .src-panel.active { display: flex; }

  .src-input {
    background: #0f172a; border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); padding: 7px 10px; font-size: 0.8rem; width: 100%;
    outline: none; transition: border-color .2s;
  }
  .src-input:focus { border-color: var(--accent); }

  .src-btn {
    padding: 8px 14px; border-radius: 6px; border: none; cursor: pointer;
    font-size: 0.8rem; font-weight: 600; transition: all .2s; width: 100%;
    background: var(--accent); color: #000;
  }
  .src-btn:hover { opacity: .85; transform: translateY(-1px); }
  .src-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  /* Drop zone */
  .drop-zone {
    border: 2px dashed var(--border); border-radius: 8px;
    padding: 20px; text-align: center; cursor: pointer;
    transition: all .2s; position: relative;
  }
  .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); background: rgba(0,212,255,.05); }
  .drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .drop-zone .dz-icon { font-size: 2rem; margin-bottom: 6px; }
  .drop-zone .dz-text { font-size: 0.78rem; color: var(--muted); }
  .drop-zone .dz-name { font-size: 0.8rem; color: var(--accent); margin-top: 4px; font-weight: 600; }

  /* Progress bar */
  .upload-progress { display: none; }
  .upload-progress.show { display: block; }
  .progress-bar {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 6px;
  }
  .progress-fill {
    height: 100%; background: var(--accent); border-radius: 2px;
    transition: width .2s; width: 0%;
  }

  /* Source badge */
  .src-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #0f172a; border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px; font-size: 0.72rem; color: var(--muted);
  }
  .src-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--normal); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
  .alert-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--alert); animation: blink 1s infinite; }

  .status-msg {
    font-size: 0.75rem; padding: 5px 8px; border-radius: 5px;
    display: none; margin-top: 4px;
  }
  .status-msg.ok   { display: block; background: #14532d; color: #4ade80; }
  .status-msg.err  { display: block; background: #450a0a; color: #f87171; }
  .status-msg.info { display: block; background: #1e2d45; color: var(--accent); }

  /* Zone edit button on video */
  .zone-edit-btn {
    position: absolute; bottom: 12px; right: 12px;
    background: rgba(0,0,0,.75); border: 1px solid var(--accent);
    color: var(--accent); padding: 6px 14px; border-radius: 6px;
    font-size: 0.78rem; cursor: pointer; transition: all .2s;
    backdrop-filter: blur(4px);
  }
  .zone-edit-btn:hover { background: var(--accent); color: #000; }

  /* Zone Editor Modal */
  .zone-modal {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.85); z-index: 1000;
    align-items: center; justify-content: center;
    backdrop-filter: blur(6px);
  }
  .zone-modal.open { display: flex; }
  .zone-modal-inner {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; width: min(96vw, 1100px);
    max-height: 92vh; display: flex; flex-direction: column;
    overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,.6);
  }
  .zone-modal-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    color: var(--accent); font-size: 0.9rem; font-weight: 600;
  }
  .zone-close-btn {
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); width: 28px; height: 28px;
    border-radius: 50%; cursor: pointer; font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
    transition: all .2s;
  }
  .zone-close-btn:hover { border-color: var(--alert); color: var(--alert); }
  .zone-modal-body {
    display: flex; gap: 0; overflow: hidden; flex: 1;
  }
  .zone-canvas-wrap {
    flex: 1; position: relative; background: #000;
    display: flex; align-items: center; justify-content: center;
    min-height: 0; overflow: hidden;
  }
  #zone-snapshot {
    max-width: 100%; max-height: 100%; display: block;
  }
  #zone-canvas {
    position: absolute; inset: 0; cursor: crosshair;
    width: 100%; height: 100%;
  }
  .zone-canvas-hint {
    position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,.7); color: var(--muted); font-size: 0.7rem;
    padding: 4px 10px; border-radius: 4px; white-space: nowrap; pointer-events: none;
  }
  .zone-controls {
    width: 280px; display: flex; flex-direction: column;
    border-left: 1px solid var(--border); overflow-y: auto;
  }
  .zone-form {
    padding: 16px; border-bottom: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 6px;
  }
  .zone-label {
    font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .5px;
  }
  .zone-input {
    background: #0f172a; border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); padding: 7px 10px;
    font-size: 0.8rem; outline: none; transition: border-color .2s;
  }
  .zone-input:focus { border-color: var(--accent); }
  .zone-type-btns { display: flex; gap: 5px; }
  .zone-type-btn {
    flex: 1; padding: 5px 0; font-size: 0.72rem; border-radius: 6px;
    border: 1px solid var(--border); background: transparent;
    color: var(--muted); cursor: pointer; transition: all .2s; text-align: center;
  }
  .zone-type-btn[data-type=allowed].active  { background:#14532d; color:#4ade80; border-color:#22c55e; }
  .zone-type-btn[data-type=restricted].active { background:#450a0a; color:#f87171; border-color:#ef4444; }
  .zone-type-btn[data-type=monitored].active  { background:#2e1065; color:#c4b5fd; border-color:#a78bfa; }
  .zone-type-btn:hover { border-color: var(--accent); color: var(--accent); }
  .zone-point-info {
    font-size: 0.72rem; padding: 5px 8px; border-radius: 5px;
    background: #0f172a; border: 1px solid var(--border); color: var(--muted);
  }
  .zone-action-btn {
    flex: 1; padding: 5px 8px; font-size: 0.72rem; border-radius: 5px;
    border: 1px solid var(--border); background: transparent; color: var(--muted);
    cursor: pointer; transition: all .2s;
  }
  .zone-action-btn:hover { border-color: var(--accent); color: var(--accent); }
  .zone-save-btn {
    padding: 8px; border-radius: 6px; border: none; cursor: pointer;
    background: var(--accent); color: #000; font-weight: 700; font-size: 0.82rem;
    margin-top: 4px; transition: all .2s;
  }
  .zone-save-btn:hover { opacity: .85; }
  /* Zone list */
  .zone-list-panel { padding: 14px; flex: 1; overflow-y: auto; }
  .zone-list-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 8px; border-radius: 7px; margin-bottom: 5px;
    background: #0f172a; border: 1px solid var(--border);
    font-size: 0.78rem;
  }
  .zone-list-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .zone-list-name { flex: 1; }
  .zone-type-tag {
    font-size: .65rem; padding: 1px 5px; border-radius: 3px;
    background: var(--border); color: var(--muted); margin-left: 4px;
  }
  .zone-pts { font-size: .65rem; color: var(--muted); margin-left: 4px; }
  .zone-del-btn {
    background: transparent; border: none; cursor: pointer;
    color: var(--muted); font-size: 0.82rem; padding: 2px 5px;
    border-radius: 4px; transition: all .2s;
  }
  .zone-del-btn:hover { color: var(--alert); }

  /* ===================== SHAP Explanation Panel ===================== */
  .shap-feature-row {
    display: flex; align-items: center; gap: 6px;
    font-size: .72rem; margin-bottom: 6px;
  }
  .shap-feat-name {
    width: 108px; color: var(--muted); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; flex-shrink: 0;
    font-size: .68rem;
  }
  .shap-feat-val {
    width: 38px; color: var(--text); text-align: right;
    font-size: .65rem; flex-shrink: 0; font-family: monospace;
  }
  .shap-bar-wrap {
    flex: 1; position: relative; height: 9px;
    background: #0f172a; border-radius: 5px; overflow: hidden;
  }
  .shap-bar {
    height: 100%; border-radius: 5px;
    transition: width .35s ease; min-width: 2px;
  }
  .shap-pos { background: linear-gradient(90deg, #ef4444, #fb923c); }
  .shap-neg { background: linear-gradient(90deg, #86efac, #22c55e); }
  .shap-zero { background: var(--border); }
  .shap-shap-val {
    width: 44px; color: var(--muted); text-align: right;
    font-size: .62rem; flex-shrink: 0; font-family: monospace;
  }
  .shap-proba-row {
    display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px;
  }
  .shap-proba-badge {
    font-size: .62rem; padding: 2px 6px; border-radius: 4px;
    background: #0f172a; border: 1px solid var(--border); color: var(--muted);
    transition: all .2s;
  }
  .shap-proba-badge.active {
    border-color: var(--accent); color: var(--accent); font-weight: 700;
  }
  .shap-pred-banner {
    padding: 6px 10px; border-radius: 6px; font-size: .75rem;
    font-weight: 600; margin-bottom: 8px; text-align: center;
    border: 1px solid currentColor;
  }
  .shap-loading {
    color: var(--muted); font-size: .73rem; text-align: center;
    padding: 12px 0; animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:.5} 50%{opacity:1} }
  .shap-empty {
    color: var(--muted); font-size: .73rem; text-align: center; padding: 10px 0;
  }
  .shap-section-label {
    font-size: .65rem; text-transform: uppercase; letter-spacing: .8px;
    color: var(--muted); margin: 8px 0 4px; border-bottom: 1px solid var(--border); padding-bottom: 3px;
  }
</style>
</head>
<body>
<header>
  <h1>&#127909; AI Security Camera System</h1>
  <div style="display:flex;align-items:center;gap:12px;">
    <span class="src-badge"><span class="dot"></span><span id="src-label">...</span></span>
    <div id="fps-badge">FPS: <span id="fps">--</span></div>
  </div>
</header>

<div class="main">
  <!-- Video -->
  <div class="video-panel" id="video-panel">
    <span class="video-label" id="live-label">&#128225; LIVE</span>
    <img id="stream" src="/video_feed" alt="Camera Stream">
    <button class="zone-edit-btn" id="zone-edit-btn" onclick="openZoneEditor()">&#9874; Ch&#7881;nh Zones</button>
  </div>

  <!-- Zone Editor Modal -->
  <div class="zone-modal" id="zone-modal">
    <div class="zone-modal-inner">
      <div class="zone-modal-header">
        <span>&#9874; Zone Editor &mdash; Click l&#234;n video &#273;&#7875; v&#7869; polygon</span>
        <button class="zone-close-btn" onclick="closeZoneEditor()">&#10005;</button>
      </div>
      <div class="zone-modal-body">
        <!-- Canvas editor -->
        <div class="zone-canvas-wrap" id="zone-canvas-wrap">
          <img id="zone-snapshot" alt="frame" style="max-width:100%;max-height:100%;display:block;object-fit:contain;">
          <canvas id="zone-canvas"></canvas>
          <div class="zone-canvas-hint" id="zone-canvas-hint">Click &#273;&#7875; th&#234;m &#273;i&#7875;m &bull; Double-click ho&#7863;c Enter &#273;&#7875; ho&#224;n t&#7845;t &bull; Ctrl+Z ho&#224;n t&#225;c &bull; Esc x&#243;a h&#236;nh</div>
        </div>
        <!-- Controls -->
        <div class="zone-controls">
          <div class="zone-form">
            <label class="zone-label">T&#234;n zone</label>
            <input class="zone-input" type="text" id="zone-name-input" placeholder="VD: entrance, restricted_area, ...">
            <label class="zone-label" style="margin-top:8px">Lo&#7841;i zone</label>
            <div class="zone-type-btns">
              <button class="zone-type-btn active" data-type="allowed" onclick="selectZoneType('allowed',this)">&#10003; Cho Ph&#233;p</button>
              <button class="zone-type-btn" data-type="restricted" onclick="selectZoneType('restricted',this)">&#9888; C&#7845;m</button>
              <button class="zone-type-btn" data-type="monitored" onclick="selectZoneType('monitored',this)">&#128065; Gi&#225;m S&#225;t</button>
            </div>
            <div class="zone-point-info" id="zone-point-info">Ch&#432;a c&#243; &#273;i&#7875;m n&#224;o</div>
            <div style="display:flex;gap:6px;margin-top:8px">
              <button class="zone-action-btn" onclick="undoLastPoint()">&#8592; Ho&#224;n t&#225;c &#273;i&#7875;m</button>
              <button class="zone-action-btn" onclick="clearDrawing()">&#128465; X&#243;a h&#236;nh</button>
            </div>
            <button class="zone-save-btn" onclick="saveCurrentZone()">&#10003; L&#432;u Zone N&#224;y</button>
            <div class="status-msg" id="zone-save-msg"></div>
          </div>
          <div class="zone-list-panel">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
              <span style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Danh s&#225;ch zones</span>
              <div style="display:flex;gap:4px">
                <button class="zone-action-btn" title="Tai lai zones tu server" onclick="reloadZonesFromServer()" style="padding:3px 7px;font-size:.68rem">&#128260; T&#7843;i l&#7841;i</button>
                <button class="zone-action-btn" title="Xoa tat ca zones" onclick="clearAllZonesFromServer()" style="padding:3px 7px;font-size:.68rem;color:#f87171;border-color:#f87171">&#128465; X&#243;a c&#7843;</button>
              </div>
            </div>
            <div id="zone-list"></div>
            <button class="zone-save-btn" onclick="applyAllZones()" style="margin-top:10px;width:100%">&#9654; &#193;p d&#7909;ng t&#7845;t c&#7843; v&#224;o pipeline</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Source Control -->
    <div class="card">
      <h3>&#128229; Nguon Video</h3>

      <div class="src-tabs">
        <button class="src-tab active" onclick="switchTab('webcam',this)">&#128247; Webcam</button>
        <button class="src-tab" onclick="switchTab('upload',this)">&#128193; Upload</button>
        <button class="src-tab" onclick="switchTab('path',this)">&#128279; Path/URL</button>
      </div>

      <!-- Tab: Webcam -->
      <div class="src-panel active" id="tab-webcam">
        <select class="src-input" id="cam-index">
          <option value="0">Camera 0 (mac dinh)</option>
          <option value="1">Camera 1</option>
          <option value="2">Camera 2</option>
        </select>
        <button class="src-btn" onclick="switchToWebcam()">&#9654; Dung Webcam</button>
        <div class="status-msg" id="cam-status"></div>
      </div>

      <!-- Tab: Upload -->
      <div class="src-panel" id="tab-upload">
        <div class="drop-zone" id="drop-zone">
          <input type="file" id="video-file" accept="video/*" onchange="onFileSelected(event)">
          <div class="dz-icon">&#127910;</div>
          <div class="dz-text">Keo tha hoac click de chon file</div>
          <div class="dz-text" style="margin-top:3px;font-size:.68rem;">MP4, AVI, MOV, MKV, WEBM</div>
          <div class="dz-name" id="dz-name"></div>
        </div>
        <div class="upload-progress" id="upload-progress">
          <div style="font-size:.72rem;color:var(--muted)" id="upload-info">Dang upload...</div>
          <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        </div>
        <button class="src-btn" id="upload-btn" onclick="uploadVideo()" disabled>&#11014; Upload &amp; Chay</button>
        <div class="status-msg" id="upload-status"></div>
      </div>

      <!-- Tab: Path -->
      <div class="src-panel" id="tab-path">
        <input class="src-input" type="text" id="path-input"
          placeholder="VD: data/sample_videos/test.mp4">
        <button class="src-btn" onclick="switchToPath()">&#9654; Dung File/URL</button>
        <div class="status-msg" id="path-status"></div>
      </div>
    </div>

    <!-- Stats -->
    <div class="card">
      <h3>&#128202; Live Stats</h3>
      <div class="stats-grid">
        <div class="stat-box"><div class="val" id="s-people">0</div><div class="lbl">People</div></div>
        <div class="stat-box"><div class="val" id="s-vehicles">0</div><div class="lbl">Vehicles</div></div>
        <div class="stat-box"><div class="val" id="s-alerts">0</div><div class="lbl">Alerts</div></div>
        <div class="stat-box"><div class="val" id="s-total">0</div><div class="lbl">Total Events</div></div>
      </div>
    </div>

    <!-- Objects -->
    <div class="card" style="flex:1; min-height:0;">
      <h3>&#128101; Detected Objects</h3>
      <div id="objects-list"></div>
    </div>

    <!-- Events -->
    <div class="card" style="flex:1; min-height:0;">
      <h3>&#128680; Recent Alerts</h3>
      <div id="events-list"><div style="color:var(--muted);font-size:.78rem">No alerts yet...</div></div>
    </div>

    <!-- Telegram Notification -->
    <div class="card" id="tg-card">
      <h3 style="display:flex;align-items:center;justify-content:space-between">
        <span>&#128338; Telegram Alert</span>
        <span id="tg-status-dot" style="width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block"></span>
      </h3>
      <div style="display:flex;flex-direction:column;gap:6px">
        <input class="src-input" type="text" id="tg-token" placeholder="Bot Token (tu @BotFather)" autocomplete="off" style="font-size:.72rem">
        <input class="src-input" type="text" id="tg-chat"  placeholder="Chat ID (tu getUpdates)" autocomplete="off" style="font-size:.72rem">
        <div style="display:flex;gap:4px">
          <select class="src-input" id="tg-level" style="flex:1;font-size:.72rem">
            <option value="warning">Warning+</option>
            <option value="alert" selected>Alert+</option>
            <option value="critical">Critical only</option>
          </select>
          <input class="src-input" type="number" id="tg-cooldown" value="30" min="5" max="300" title="Cooldown giay" style="width:64px;font-size:.72rem">
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:.75rem;color:var(--muted)">
          <input type="checkbox" id="tg-photo" checked style="accent-color:var(--accent)">
          <label for="tg-photo">Dinh kem anh snapshot</label>
        </div>
        <div style="display:flex;gap:5px">
          <button class="src-btn" onclick="saveTgConfig()" style="flex:2">Luu & Bat</button>
          <button class="src-btn" onclick="testTelegram()" style="flex:1;background:#1e3a5f;color:var(--accent)">Test</button>
        </div>
        <button class="src-btn" onclick="disableTelegram()" style="background:transparent;border:1px solid var(--border);color:var(--muted);font-weight:400">Tat thong bao</button>
        <div class="status-msg" id="tg-msg"></div>
      </div>
    </div>

    <!-- NLG Engine (Gemini AI) -->
    <div class="card" id="nlg-card">
      <h3 style="display:flex;align-items:center;justify-content:space-between">
        <span>&#129504; AI Assistant (NLG)</span>
        <span id="nlg-status-dot" style="width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block" title="Trang thai Gemini"></span>
      </h3>
      <div id="nlg-status-badge" style="font-size:.7rem;padding:3px 8px;border-radius:4px;background:#1e2d45;color:var(--muted);margin-bottom:8px;text-align:center">
        Dang kiem tra...
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <input class="src-input" type="password" id="nlg-apikey"
          placeholder="Gemini API Key (tu aistudio.google.com)"
          autocomplete="off" style="font-size:.72rem">
        <div style="font-size:.68rem;color:var(--muted)">
          Lay key tai: <a href="https://aistudio.google.com/app/apikey" target="_blank"
          style="color:var(--accent)">aistudio.google.com/app/apikey</a>
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:.75rem;color:var(--muted)">
          <input type="checkbox" id="nlg-enabled" checked style="accent-color:var(--accent)">
          <label for="nlg-enabled">Bat NLG Engine (sinh cau tu nhien)</label>
        </div>
        <div style="display:flex;gap:5px">
          <button class="src-btn" onclick="saveNlgConfig()" style="flex:2">Luu &amp; Ket noi</button>
          <button class="src-btn" onclick="testNlg()" style="flex:1;background:#1a3a2a;color:#4ade80">Test AI</button>
        </div>
        <div class="status-msg" id="nlg-msg"></div>
        <div id="nlg-test-result" style="display:none;margin-top:4px;padding:8px;background:#0f172a;border:1px solid var(--border);border-radius:6px;font-size:.75rem;line-height:1.5;color:var(--text)"></div>
        <div id="nlg-stats" style="display:none;font-size:.68rem;color:var(--muted);margin-top:2px"></div>
      </div>
    </div>

    <!-- SHAP Explanation Card -->
    <div class="card" id="shap-card">
      <h3 style="display:flex;align-items:center;justify-content:space-between">
        <span>&#129302; ContextNet SHAP</span>
        <span id="shap-mode-dot" style="width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block" title="Trang thai SHAP"></span>
      </h3>
      <!-- Mode badge -->
      <div id="shap-mode-badge" style="font-size:.7rem;padding:3px 8px;border-radius:4px;background:#1e2d45;color:var(--muted);margin-bottom:8px;text-align:center">
        Dang kiem tra...
      </div>
      <!-- Tabs -->
      <div class="src-tabs" style="margin-bottom:8px">
        <button class="src-tab active" id="shap-tab-global" onclick="switchShapTab('global',this)">&#127758; Tong quan</button>
        <button class="src-tab" id="shap-tab-object" onclick="switchShapTab('object',this)">&#127919; Theo Object</button>
      </div>

      <!-- Global Feature Importance Tab -->
      <div id="shap-global-panel">
        <div class="shap-section-label">Feature Importance (XGBoost)</div>
        <div id="shap-global-chart"><div class="shap-loading">&#8987; Dang tai...</div></div>
        <button class="src-btn" onclick="loadShapGlobal()" style="margin-top:8px;background:transparent;border:1px solid var(--border);color:var(--muted);font-weight:400;font-size:.72rem">
          &#8635; Lam moi
        </button>
      </div>

      <!-- Per-Object SHAP Tab -->
      <div id="shap-object-panel" style="display:none">
        <div class="shap-section-label">Chon object de giai thich</div>
        <select class="src-input" id="shap-object-select" style="font-size:.75rem;margin-bottom:8px" onchange="loadShapForObject(this.value)">
          <option value="">-- Chon Object --</option>
        </select>
        <div id="shap-pred-banner" class="shap-pred-banner" style="display:none"></div>
        <div id="shap-object-chart"><div class="shap-empty">&#128270; Chon object o tren de xem giai thich</div></div>
        <div id="shap-proba-row" class="shap-proba-row" style="display:none"></div>
      </div>
    </div>

  </div>
</div>

<script>
// --- Source switcher ---
function switchTab(name, btn) {
  document.querySelectorAll('.src-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.src-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function showMsg(id, msg, type) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = 'status-msg ' + type;
  setTimeout(() => { el.className = 'status-msg'; }, 5000);
}

async function switchToWebcam() {
  const idx = parseInt(document.getElementById('cam-index').value);
  showMsg('cam-status', 'Dang ket noi...', 'info');
  try {
    const r = await fetch('/api/source', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: idx})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    showMsg('cam-status', 'OK Da chuyen sang Webcam ' + idx, 'ok');
    reloadStream();
  } catch(e) {
    showMsg('cam-status', 'LOI: ' + e.message, 'err');
  }
}

let selectedFile = null;
function onFileSelected(e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    document.getElementById('dz-name').textContent = selectedFile.name;
    document.getElementById('upload-btn').disabled = false;
  }
}

async function uploadVideo() {
  if (!selectedFile) return;
  const btn = document.getElementById('upload-btn');
  btn.disabled = true;
  btn.textContent = 'Dang upload...';

  const progress = document.getElementById('upload-progress');
  const fill     = document.getElementById('progress-fill');
  const info     = document.getElementById('upload-info');
  progress.classList.add('show');

  const form = new FormData();
  form.append('file', selectedFile);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round(e.loaded / e.total * 100);
      fill.style.width = pct + '%';
      info.textContent = 'Uploading... ' + pct + '% (' + (e.loaded/1024).toFixed(0) + ' KB / ' + (e.total/1024).toFixed(0) + ' KB)';
    }
  };

  xhr.onload = () => {
    progress.classList.remove('show');
    btn.textContent = 'Upload & Chay';
    btn.disabled = false;
    fill.style.width = '0%';
    if (xhr.status === 200) {
      const d = JSON.parse(xhr.responseText);
      showMsg('upload-status', 'OK ' + d.filename + ' (' + d.size_kb + ' KB) - Pipeline da chay!', 'ok');
      reloadStream();
    } else {
      const d = JSON.parse(xhr.responseText);
      showMsg('upload-status', 'LOI: ' + (d.detail || 'Loi khong xac dinh'), 'err');
    }
  };

  xhr.onerror = () => {
    progress.classList.remove('show');
    btn.textContent = 'Upload & Chay';
    btn.disabled = false;
    showMsg('upload-status', 'LOI ket noi mang', 'err');
  };

  xhr.send(form);
}

async function switchToPath() {
  const path = document.getElementById('path-input').value.trim();
  if (!path) { showMsg('path-status', 'Nhap duong dan truoc', 'err'); return; }
  showMsg('path-status', 'Dang ket noi...', 'info');
  try {
    const r = await fetch('/api/source', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: path})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    showMsg('path-status', 'OK Da chuyen sang: ' + path, 'ok');
    reloadStream();
  } catch(e) {
    showMsg('path-status', 'LOI: ' + e.message, 'err');
  }
}

function reloadStream() {
  const img = document.getElementById('stream');
  img.src = '/video_feed?' + Date.now();
}

// Drag and drop
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) {
    selectedFile = f;
    document.getElementById('dz-name').textContent = f.name;
    document.getElementById('upload-btn').disabled = false;
  }
});

// --- Data polling ---
async function fetchData() {
  try {
    // Source info
    const rs = await fetch('/api/source');
    if (rs.ok) {
      const ds = await rs.json();
      document.getElementById('src-label').textContent = ds.label || ds.type;
    }

    // Objects
    const r1 = await fetch('/api/objects');
    const d1 = await r1.json();

    document.getElementById('fps').textContent = (d1.fps || 0).toFixed(1);

    const stats = d1.stats || {};
    document.getElementById('s-people').textContent   = stats.total_persons  || 0;
    document.getElementById('s-vehicles').textContent = stats.total_vehicles || 0;
    document.getElementById('s-alerts').textContent   = stats.active_warnings || 0;

    // Object list
    const objs  = d1.objects || [];
    const list  = objs.map(o => {
      const lvl      = o.alert_level || 'normal';
      const role     = o.role || o.class_name || '';
      const action   = o.action || '';
      const zoneName = o.zone_name || '';
      const zoneStatus = o.zone_status || '';
      const identity = o.identity || '';

      // Action icon mapping
      const actionIcons = {
        'standing':'\ud83e\uddd0','walking':'\ud83d\udeb6','running':'\ud83c\udfc3',
        'falling':'\ud83d\udea8','climbing':'\ud83e\uddd7','fighting':'\u26a0\ufe0f',
        'raising_hand':'\ud83d\ude4b','gathering':'\ud83d\udc65','unknown':''
      };
      const actionIcon = actionIcons[action.toLowerCase()] || '';

      const actionBadge = action && action !== 'unknown' && action !== 'standing'
        ? `<span style="font-size:.65rem;background:#1e2d45;color:var(--accent);padding:1px 5px;border-radius:3px;margin-left:4px">${actionIcon} ${action}</span>`
        : '';
      const idBadge = identity && identity !== 'unknown'
        ? `<span style="font-size:.65rem;background:#2d1a45;color:#c4b5fd;padding:1px 5px;border-radius:3px;margin-left:4px">\ud83d\udc64 ${identity}</span>`
        : '';

      // Zone badge with status
      let zoneBadge = '';
      if (zoneName) {
        const zoneStatusMap = { entering: '>>',  inside: '\u25cf', leaving: '<<', outside: '' };
        const zoneTypeColors = { restricted: '#f87171', allowed: '#4ade80', monitored: '#c4b5fd' };
        const statusIcon  = zoneStatusMap[zoneStatus] || '';
        const zoneType    = o.zone_type || 'monitored';
        const zoneColor   = zoneTypeColors[zoneType] || '#94a3b8';
        zoneBadge = `<span style="font-size:.65rem;background:#0f172a;border:1px solid ${zoneColor};color:${zoneColor};padding:1px 5px;border-radius:3px;margin-left:4px">${statusIcon} ${zoneName}</span>`;
      }

      return '<div class="obj-item ' + lvl + '">'
        + '<span class="obj-level level-' + lvl + '">' + lvl.toUpperCase() + '</span>'
        + '<span style="flex:1;min-width:0;display:flex;flex-wrap:wrap;align-items:center;gap:2px">'
          + '<span>#' + o.track_id + ' ' + role + '</span>'
          + actionBadge + idBadge + zoneBadge
        + '</span>'
        + '</div>';
    });
    document.getElementById('objects-list').innerHTML = list.join('') || '<div style="color:var(--muted);font-size:.78rem">No objects detected</div>';

    // Update SHAP Object dropdown
    if (typeof updateShapObjectDropdown === 'function') {
      updateShapObjectDropdown(objs);
    }
    // Auto refresh SHAP for selected object
    const selectedTrackId = document.getElementById('shap-object-select')?.value;
    if (typeof _shapCurrentTab !== 'undefined' && _shapCurrentTab === 'object' && selectedTrackId && typeof loadShapForObject === 'function') {
      loadShapForObject(selectedTrackId);
    }

    // Stats
    const r2 = await fetch('/api/stats');
    const d2 = await r2.json();
    document.getElementById('s-total').textContent = d2.total || 0;

    // Events
    const r3 = await fetch('/api/events?limit=15');
    const d3  = await r3.json();
    const evts = d3.events || [];
    if (evts.length === 0) {
      document.getElementById('events-list').innerHTML = '<div style="color:var(--muted);font-size:.78rem">No alerts yet...</div>';
    } else {
      const lvlColors = {
        critical: '#f87171', alert: '#fb923c',
warning: '#fbbf24', watch: '#c4b5fd',
        normal: '#4ade80', ignore: 'var(--muted)'
      };
      const evtHtml = evts.slice().reverse().map(e => {
        const lvl   = e.level || 'normal';
        const color = lvlColors[lvl] || 'var(--muted)';
        const action= e.action || '';
        const role  = e.object_role || '';
        const zone  = e.zone_name ? ` @ ${e.zone_name}` : '';
        return '<div class="event-item">'
          + `<span class="evt-time">${e.datetime || ''}</span>`
          + `<span style="float:right;font-size:.65rem;font-weight:700;color:${color}">${lvl.toUpperCase()}</span>`
          + `<div style="margin-top:2px">`
            + (role ? `<b>${role}</b> ` : '')
            + (action && action !== 'unknown' ? `\u2022 ${action} ` : '')
            + `<span style="color:var(--muted)">${zone}</span>`
          + `</div>`
          + `<div style="font-size:.72rem;color:var(--muted);margin-top:1px">${e.reason || ''}</div>`
          + '</div>';
      }).join('');
      document.getElementById('events-list').innerHTML = evtHtml;
    }
  } catch(err) {
    console.warn('Fetch error:', err);
  }
}

setInterval(fetchData, 1500);
fetchData();

// =====================================================
// ZONE EDITOR
// =====================================================
const FRAME_W = 640, FRAME_H = 480;
let zonePoints   = [];    // [{x,y}] pixel coords relative to 640x480
let pendingZones = [];    // zones chua ap dung
let zoneType     = 'allowed';
let _mousePos    = null;  // vi tri chuot hien tai tren canvas

const ZONE_COLORS = {
  allowed    : { stroke:'#22c55e', fill:'rgba(34,197,94,0.18)' },
  restricted : { stroke:'#ef4444', fill:'rgba(239,68,68,0.18)' },
  monitored  : { stroke:'#a78bfa', fill:'rgba(167,139,250,0.18)' },
};
const ZONE_BGR = {
  allowed    : [0, 200, 0],
  restricted : [0, 0, 220],
  monitored  : [220, 180, 0],
};

async function openZoneEditor() {
  // Dung /api/snapshot (raw frame, khong co annotation) thay vi /video_feed
  const snap = document.getElementById('zone-snapshot');
  showMsg('zone-save-msg', 'Dang tai anh...', 'info');

  // Decode an toan ca khi anh da duoc cache
  try {
    const resp = await fetch('/api/snapshot?' + Date.now());
    if (!resp.ok) throw new Error('Snapshot unavailable');
    const blob = await resp.blob();
    const objUrl = URL.createObjectURL(blob);
    snap.onload = () => { resizeCanvas(); URL.revokeObjectURL(objUrl); };
    snap.src = objUrl;
  } catch(e) {
    showMsg('zone-save-msg', 'Khong lay duoc anh (pipeline chua chay?)', 'err');
  }

  // Load zones hien tai tu server
  await _reloadZonesFromServer();

  zonePoints = [];
  renderZoneList();
  document.getElementById('zone-modal').classList.add('open');
  updatePointInfo();
}

async function _reloadZonesFromServer() {
  try {
    const r = await fetch('/api/zones');
    const d = await r.json();
    pendingZones = d.zones || [];
  } catch(e) {
    pendingZones = [];
  }
}

async function reloadZonesFromServer() {
  await _reloadZonesFromServer();
  renderZoneList();
  redrawCanvas();
  showMsg('zone-save-msg', 'Da tai lai zones tu server!', 'ok');
}

async function clearAllZonesFromServer() {
  if (!confirm('Xoa toan bo zones cua camera nay? Khong the hoan tac.')) return;
  try {
    const r = await fetch('/api/zones', { method: 'DELETE' });
    if (!r.ok) throw new Error('Xoa that bai');
    pendingZones = [];
    renderZoneList();
    redrawCanvas();
    showMsg('zone-save-msg', 'Da xoa toan bo zones!', 'ok');
  } catch(e) {
    showMsg('zone-save-msg', 'Loi: ' + e.message, 'err');
  }
}

function closeZoneEditor() {
  document.getElementById('zone-modal').classList.remove('open');
  zonePoints = [];
  _mousePos  = null;
}

function resizeCanvas() {
  const snap   = document.getElementById('zone-snapshot');
  const canvas = document.getElementById('zone-canvas');
  // Su dung natural size de scale chuan xac
  canvas.width  = snap.naturalWidth  || FRAME_W;
  canvas.height = snap.naturalHeight || FRAME_H;
  // Hien thi khop voi kich thuoc thuc te cua image element
  const rect = snap.getBoundingClientRect();
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';
  redrawCanvas();
}

function selectZoneType(type, btn) {
  zoneType = type;
  document.querySelectorAll('.zone-type-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  redrawCanvas();
}

// ── Canvas events ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('zone-canvas');

  // Helper: convert event coords -> canvas pixel coords (640x480 space)
  function toCanvasCoords(e) {
    const rect   = canvas.getBoundingClientRect();
    const scaleX = (canvas.width  || FRAME_W) / rect.width;
    const scaleY = (canvas.height || FRAME_H) / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top)  * scaleY),
    };
  }

  // Click: them diem
  canvas.addEventListener('click', (e) => {
    if (!document.getElementById('zone-modal').classList.contains('open')) return;
    const pt = toCanvasCoords(e);
    zonePoints.push(pt);
    updatePointInfo();
    redrawCanvas();
  });

  // Double-click: hoan tat zone
  canvas.addEventListener('dblclick', () => {
    if (zonePoints.length >= 3) finalizeZone();
  });

  // Mouse move: ve preview line tu diem cuoi den chuot
  canvas.addEventListener('mousemove', (e) => {
    if (!document.getElementById('zone-modal').classList.contains('open')) return;
    if (zonePoints.length === 0) return;
    _mousePos = toCanvasCoords(e);
    redrawCanvas();
  });

  canvas.addEventListener('mouseleave', () => {
    _mousePos = null;
    redrawCanvas();
  });

  // Resize modal -> resize canvas
  const resizeObs = new ResizeObserver(() => {
    if (document.getElementById('zone-modal').classList.contains('open')) resizeCanvas();
  });
  const wrap = document.getElementById('zone-canvas-wrap');
  if (wrap) resizeObs.observe(wrap);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (!document.getElementById('zone-modal').classList.contains('open')) return;
  if (e.key === 'Enter' && zonePoints.length >= 3) finalizeZone();
  if (e.key === 'Escape') { clearDrawing(); }
  if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undoLastPoint(); }
});

function undoLastPoint() {
  zonePoints.pop();
  updatePointInfo();
  redrawCanvas();
}

function clearDrawing() {
  zonePoints = [];
  _mousePos  = null;
  updatePointInfo();
  redrawCanvas();
}

function updatePointInfo() {
  const el = document.getElementById('zone-point-info');
  const n = zonePoints.length;
  if (n === 0) {
    el.textContent = 'Chua co diem nao — click vao video de bat dau';
    el.style.color = 'var(--muted)';
  } else if (n < 3) {
    el.textContent = n + ' diem — can them ' + (3-n) + ' diem nua';
    el.style.color = 'var(--warning)';
  } else {
    el.textContent = n + ' diem — Enter/double-click de hoan tat | Ctrl+Z hoan tac';
    el.style.color = 'var(--normal)';
  }
}

function redrawCanvas() {
  const canvas = document.getElementById('zone-canvas');
  const ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Ve tat ca pending zones (da hoan tat)
  pendingZones.forEach(z => {
    const col = ZONE_COLORS[z.type] || ZONE_COLORS.allowed;
    drawPolygon(ctx, z.polygon.map(p => ({x:p[0], y:p[1]})), col.stroke, col.fill, z.name, z.type);
  });

  // Ve zone dang ve + preview line
  if (zonePoints.length > 0) {
    const col = ZONE_COLORS[zoneType];

    // Fill preview (chi khi >= 3 diem)
    if (zonePoints.length >= 3) {
      ctx.fillStyle = col.fill;
      ctx.beginPath();
      ctx.moveTo(zonePoints[0].x, zonePoints[0].y);
      zonePoints.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fill();
    }

    // Cac doan da ve (solid)
    ctx.strokeStyle = col.stroke;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(zonePoints[0].x, zonePoints[0].y);
    zonePoints.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.stroke();

    // Preview line tu diem cuoi den chuot (dash)
    if (_mousePos) {
      const last = zonePoints[zonePoints.length - 1];
      ctx.strokeStyle = col.stroke;
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(_mousePos.x, _mousePos.y);
      ctx.stroke();
      // Preview dong cua tu chuot ve diem dau
      if (zonePoints.length >= 2) {
        ctx.setLineDash([3, 6]);
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.moveTo(_mousePos.x, _mousePos.y);
        ctx.lineTo(zonePoints[0].x, zonePoints[0].y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    // Ve cac diem (circles)
    zonePoints.forEach((p, i) => {
      // Outline trang
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI*2);
      ctx.fill();
      // Filled voi mau zone
      ctx.fillStyle = col.stroke;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI*2);
      ctx.fill();
      // So thu tu
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(i+1, p.x, p.y + 3);
      ctx.textAlign = 'left';
    });

    // Highlight diem dau (diem dong zone)
    if (zonePoints.length >= 3) {
      const first = zonePoints[0];
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(first.x, first.y, 8, 0, Math.PI*2);
      ctx.stroke();
    }
  }
}

function drawPolygon(ctx, points, stroke, fill, name, type) {
  if (points.length < 2) return;
  const typeIcons = {allowed:'\u2713', restricted:'\u26a0', monitored:'\u25ce'};

  ctx.fillStyle   = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth   = 2.5;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Vertex dots
  points.forEach(p => {
    ctx.fillStyle = stroke;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI*2);
    ctx.fill();
  });

  // Label at centroid
  const cx = points.reduce((s,p) => s+p.x, 0)/points.length;
  const cy = points.reduce((s,p) => s+p.y, 0)/points.length;
  const label = (typeIcons[type]||'') + ' ' + name;

  ctx.font      = 'bold 13px sans-serif';
  ctx.textAlign = 'center';
  // Shadow/outline
  ctx.strokeStyle = 'rgba(0,0,0,0.8)';
  ctx.lineWidth   = 3;
  ctx.strokeText(label, cx, cy);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, cx, cy);
  ctx.textAlign = 'left';
}

function finalizeZone() {
  if (zonePoints.length < 3) return;
  const nameInput = document.getElementById('zone-name-input');
  const zoneName = (nameInput.value.trim() || 'zone_' + (pendingZones.length + 1)).replace(/\s+/g, '_');
  pendingZones.push({
    name   : zoneName,
    type   : zoneType,
    polygon: zonePoints.map(p => [p.x, p.y]),
    color  : ZONE_BGR[zoneType] || [0,200,0],
  });
  zonePoints   = [];
  _mousePos    = null;
  nameInput.value = '';
  renderZoneList();
  updatePointInfo();
  redrawCanvas();
  showMsg('zone-save-msg', '&#10003; Zone "' + zoneName + '" da them!', 'ok');
}

function saveCurrentZone() {
  if (zonePoints.length >= 3) { finalizeZone(); return; }
  showMsg('zone-save-msg', '&#9888; Can ve it nhat 3 diem de tao zone', 'err');
}

async function applyAllZones() {
  if (pendingZones.length === 0) {
    showMsg('zone-save-msg', '&#9888; Chua co zone nao de luu', 'err');
    return;
  }
  try {
    const r = await fetch('/api/zones', {
      method : 'POST',
      headers: {'Content-Type': 'application/json'},
      body   : JSON.stringify({zones: pendingZones}),
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    showMsg('zone-save-msg', '&#10003; Da luu ' + pendingZones.length + ' zones vao pipeline!', 'ok');
    setTimeout(closeZoneEditor, 1500);
  } catch(e) {
    showMsg('zone-save-msg', '&#10060; Loi: ' + e.message, 'err');
  }
}

function renderZoneList() {
  const el = document.getElementById('zone-list');
  if (pendingZones.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:.75rem;text-align:center;padding:12px">Chua co zone nao</div>';
    return;
  }
  const typeLabel = {allowed:'Cho phep', restricted:'Cam', monitored:'Giam sat'};
  const typeIcons = {allowed:'&#10003;', restricted:'&#9888;', monitored:'&#128065;'};
  el.innerHTML = pendingZones.map((z, i) => {
    const col = ZONE_COLORS[z.type] || ZONE_COLORS.allowed;
    return '<div class="zone-list-item">'
      + '<span class="zone-list-dot" style="background:' + col.stroke + '"></span>'
      + '<span class="zone-list-name">'
        + (icons[z.type]||'') + ' <b>' + z.name + '</b>'
        + ' <span class="zone-type-tag">' + (typeLabel[z.type]||z.type) + '</span>'
        + '<span class="zone-pts">' + z.polygon.length + ' pts</span>'
      + '</span>'
      + '<button class="zone-del-btn" onclick="deleteZone(' + i + ')">&#128465;</button>'
      + '</div>';
  }).join('');
}

function deleteZone(idx) {
  pendingZones.splice(idx, 1);
  renderZoneList();
  redrawCanvas();
}

// =====================================================
// TELEGRAM NOTIFICATION
// =====================================================
async function loadTgConfig() {
  try {
    const r = await fetch('/api/telegram/config');
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('tg-chat').value       = d.chat_id || '';
    document.getElementById('tg-level').value      = d.min_level || 'alert';
    document.getElementById('tg-cooldown').value   = d.cooldown_sec || 30;
    document.getElementById('tg-photo').checked    = d.send_photo !== false;
    // Update status dot
    const dot = document.getElementById('tg-status-dot');
    dot.style.background = d.enabled ? 'var(--normal)' : 'var(--muted)';
    dot.title = d.enabled ? 'Dang hoat dong' : 'Chua bat';
  } catch(e) { /* silent */ }
}

async function saveTgConfig() {
  const token    = document.getElementById('tg-token').value.trim();
  const chat_id  = document.getElementById('tg-chat').value.trim();
  const min_level= document.getElementById('tg-level').value;
  const cooldown = parseFloat(document.getElementById('tg-cooldown').value) || 30;
  const photo    = document.getElementById('tg-photo').checked;

  if (!token || !chat_id) {
    showMsg('tg-msg', 'Nhap Bot Token va Chat ID truoc', 'err');
    return;
  }
  showMsg('tg-msg', 'Dang luu...', 'info');
  try {
    const r = await fetch('/api/telegram/config', {
      method : 'POST',
      headers: {'Content-Type': 'application/json'},
      body   : JSON.stringify({
        bot_token   : token,
        chat_id     : chat_id,
        enabled     : true,
        min_level   : min_level,
        cooldown_sec: cooldown,
        send_photo  : photo,
      }),
    });
    const d = await r.json();
    if (r.ok && d.status === 'ok') {
      showMsg('tg-msg', '✓ Da luu! Thong bao da bat.', 'ok');
      document.getElementById('tg-token').value = '';
      loadTgConfig();
    } else {
      showMsg('tg-msg', 'Loi: ' + (d.detail || JSON.stringify(d)), 'err');
    }
  } catch(e) {
    showMsg('tg-msg', 'Loi ket noi: ' + e.message, 'err');
  }
}

async function testTelegram() {
  showMsg('tg-msg', 'Dang gui tin nhan test...', 'info');
  try {
    const r = await fetch('/api/telegram/test', { method: 'POST' });
    const d = await r.json();
    if (r.ok) {
      showMsg('tg-msg', '✓ ' + (d.message || 'Ket noi thanh cong!'), 'ok');
    } else {
      showMsg('tg-msg', '✗ ' + (d.detail || 'Loi'), 'err');
    }
  } catch(e) {
    showMsg('tg-msg', 'Loi: ' + e.message, 'err');
  }
}

async function disableTelegram() {
  try {
    await fetch('/api/telegram/config', {
      method : 'POST',
      headers: {'Content-Type': 'application/json'},
      body   : JSON.stringify({ bot_token:'', chat_id:'', enabled: false }),
    });
    showMsg('tg-msg', 'Da tat thong bao Telegram.', 'info');
    loadTgConfig();
  } catch(e) { /* silent */ }
}

// Load Telegram config on startup
loadTgConfig();

// =====================================================
// NLG ENGINE (GEMINI AI)
// =====================================================
async function loadNlgStatus() {
  try {
    const r = await fetch('/api/nlg/status');
    if (!r.ok) return;
    const d = await r.json();

    const dot   = document.getElementById('nlg-status-dot');
    const badge = document.getElementById('nlg-status-badge');
    const stats = document.getElementById('nlg-stats');

    if (d.available) {
      dot.style.background   = 'var(--normal)';
      dot.title              = 'Gemini dang hoat dong';
      badge.style.background = '#14532d';
      badge.style.color      = '#4ade80';
      badge.textContent      = '🤖 Gemini NLG — Dang hoat dong | ' + d.model;
    } else if (d.initialized) {
      dot.style.background   = 'var(--warning)';
      dot.title              = 'Gemini co loi tam thoi (cooldown)';
      badge.style.background = '#451a03';
      badge.style.color      = '#fbbf24';
      badge.textContent      = '⚠️ Gemini loi (cooldown ' + (d.error_count) + ' lan) — dung template';
    } else if (d.api_key_set) {
      dot.style.background   = '#ef4444';
      dot.title              = 'Chua ket noi duoc Gemini';
      badge.style.background = '#450a0a';
      badge.style.color      = '#f87171';
      badge.textContent      = '❌ Gemini chua ket noi — dung template fallback';
    } else {
      dot.style.background   = 'var(--muted)';
      dot.title              = 'Chua cau hinh API key';
      badge.style.background = '#1e2d45';
      badge.style.color      = 'var(--muted)';
      badge.textContent      = '🔑 Chua cau hinh API Key — dung template trang Vietnamese';
    }

    // Thống kê
    if (d.total_calls > 0) {
      stats.style.display  = 'block';
      stats.textContent    = `Tong: ${d.total_calls} | Thanh cong: ${d.success_calls} | Ty le: ${d.success_rate}`;
    }

    // Sync checkbox
    document.getElementById('nlg-enabled').checked = d.enabled;

  } catch(e) { /* silent */ }
}

async function saveNlgConfig() {
  const apiKey  = document.getElementById('nlg-apikey').value.trim();
  const enabled = document.getElementById('nlg-enabled').checked;

  if (!apiKey && enabled) {
    showMsg('nlg-msg', '⚠ Nhap Gemini API Key truoc khi bat NLG', 'err');
    return;
  }

  showMsg('nlg-msg', 'Dang luu va ket noi Gemini...', 'info');
  try {
    const r = await fetch('/api/nlg/config', {
      method : 'POST',
      headers: {'Content-Type': 'application/json'},
      body   : JSON.stringify({ api_key: apiKey, enabled: enabled }),
    });
    const d = await r.json();
    if (r.ok && d.status === 'ok') {
      if (d.nlg_status.initialized) {
        showMsg('nlg-msg', '✓ Ket noi Gemini thanh cong! NLG Engine san sang.', 'ok');
      } else if (!enabled) {
        showMsg('nlg-msg', '✓ Da tat NLG Engine. Dung template fallback.', 'ok');
      } else {
        showMsg('nlg-msg', '⚠ Da luu nhung chua ket noi duoc — kiem tra API key.', 'err');
      }
      document.getElementById('nlg-apikey').value = '';
      loadNlgStatus();
    } else {
      showMsg('nlg-msg', 'Loi: ' + (d.detail || JSON.stringify(d)), 'err');
    }
  } catch(e) {
    showMsg('nlg-msg', 'Loi ket noi: ' + e.message, 'err');
  }
}

async function testNlg() {
  const resultEl = document.getElementById('nlg-test-result');
  resultEl.style.display = 'none';
  showMsg('nlg-msg', '🤖 Dang goi Gemini sinh cau mau...', 'info');

  try {
    const r = await fetch('/api/nlg/test', { method: 'POST' });
    const d = await r.json();

    if (r.ok && d.success) {
      showMsg('nlg-msg', '✓ Gemini phan hoi thanh cong!', 'ok');
      resultEl.style.display = 'block';
      resultEl.innerHTML = '<span style="color:var(--muted);font-size:.65rem">📨 Cau Gemini sinh ra:</span><br>'
        + '<span style="color:#4ade80">' + escapeHtml(d.message) + '</span>';
    } else {
      showMsg('nlg-msg', '✗ ' + (d.detail || d.message || 'That bai'), 'err');
      resultEl.style.display = 'none';
    }
  } catch(e) {
    showMsg('nlg-msg', 'Loi: ' + e.message, 'err');
  }
}

function escapeHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
             .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

// Load NLG status on startup and refresh every 5 seconds
loadNlgStatus();
setInterval(loadNlgStatus, 5000);

// =====================================================
// SHAP EXPLANATION ENGINE
// =====================================================

// Mapping alert level → màu sắc
const LEVEL_COLORS = {
  ignore  : { bg:'#1e2d45', color:'#64748b', border:'#334155' },
  normal  : { bg:'#14532d', color:'#4ade80', border:'#22c55e' },
  watch   : { bg:'#2e1065', color:'#c4b5fd', border:'#7c3aed' },
  warning : { bg:'#451a03', color:'#fbbf24', border:'#d97706' },
  alert   : { bg:'#450a0a', color:'#f87171', border:'#dc2626' },
  critical: { bg:'#3b0a0a', color:'#ff4444', border:'#b91c1c' },
};

// Tên hiển thị thân thiện cho từng feature
const FEAT_LABELS = {
  role_id         : 'Vai trò XH',
  identity_id     : 'Nhận dạng',
  zone_type_id    : 'Loại vùng',
  zone_status_id  : 'Trạng thái vùng',
  loitering       : 'Lang thang',
  time_in_zone    : 'Thời gian vùng',
  visit_count     : 'Số lần ghé',
  direction_id    : 'Hướng di chuyển',
  hour            : 'Giờ trong ngày',
  role_confidence : 'Độ tin cậy role',
  category_id     : 'Danh mục',
  frames_tracked  : 'Số frame theo dõi',
  is_night        : 'Ban đêm',
  is_business_hour: 'Giờ hành chính',
  action_id       : 'Hành động',
  action_confidence:'Độ tin cậy HĐ',
};

let _shapCurrentTab = 'global';
let _shapAutoRefreshTimer = null;

function switchShapTab(tab, btn) {
  _shapCurrentTab = tab;
  document.querySelectorAll('#shap-card .src-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('shap-global-panel').style.display = (tab === 'global') ? '' : 'none';
  document.getElementById('shap-object-panel').style.display = (tab === 'object') ? '' : 'none';
  if (tab === 'global') loadShapGlobal();
}

// ── Global Feature Importance ──────────────────────────────
async function loadShapGlobal() {
  const chartEl  = document.getElementById('shap-global-chart');
  const dot      = document.getElementById('shap-mode-dot');
  const badge    = document.getElementById('shap-mode-badge');
  chartEl.innerHTML = '<div class="shap-loading">⏳ Đang tải feature importance...</div>';
  try {
    const r = await fetch('/api/shap/feature_importance');
    const d = await r.json();

    if (!d.ml_ready) {
      dot.style.background   = 'var(--muted)';
      badge.style.background = '#1e2d45';
      badge.style.color      = 'var(--muted)';
      badge.textContent      = '⚙️ Rule Engine — SHAP không khả dụng';
      chartEl.innerHTML      = '<div class="shap-empty">⚙️ ContextNet đang dùng Rule Engine<br><span style="font-size:.65rem">SHAP chỉ hoạt động khi model XGBoost đã được train.</span></div>';
      return;
    }

    // Update badge
    dot.style.background   = 'var(--accent)';
    badge.style.background = '#0f2d45';
    badge.style.color      = 'var(--accent)';
    badge.textContent      = '🧠 XGBoost ML — SHAP sẵn sàng';

    renderGlobalBars(chartEl, d.feature_names, d.importances);

  } catch(e) {
    chartEl.innerHTML = '<div class="shap-empty">❌ Lỗi tải data: ' + e.message + '</div>';
  }
}

function renderGlobalBars(container, names, importances) {
  const maxVal = Math.max(...importances, 0.001);
  const html = names.map((name, i) => {
    const val   = importances[i];
    const pct   = Math.round((val / maxVal) * 100);
    const label = FEAT_LABELS[name] || name;
    const colorClass = val > 0.05 ? 'shap-pos' : (val > 0.02 ? 'shap-shap-val' : 'shap-zero');
    const barColor = val > 0.05
      ? 'background:linear-gradient(90deg,#00d4ff,#0ea5e9)'
      : (val > 0.02 ? 'background:linear-gradient(90deg,#a78bfa,#6d28d9)' : 'background:var(--border)');
    return `<div class="shap-feature-row">
      <span class="shap-feat-name" title="${name}">${label}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar" style="width:${pct}%;${barColor}"></div>
      </div>
      <span class="shap-shap-val">${(val * 100).toFixed(1)}%</span>
    </div>`;
  }).join('');
  container.innerHTML = html || '<div class="shap-empty">Không có dữ liệu</div>';
}

// ── Per-Object SHAP ────────────────────────────────────────
async function loadShapForObject(trackIdStr) {
  if (!trackIdStr) return;
  const trackId   = parseInt(trackIdStr);
  const chartEl   = document.getElementById('shap-object-chart');
  const banner    = document.getElementById('shap-pred-banner');
  const probaRow  = document.getElementById('shap-proba-row');

  chartEl.innerHTML = '<div class="shap-loading">⏳ Đang tính SHAP values...</div>';
  banner.style.display = 'none';
  probaRow.style.display = 'none';

  try {
    const r = await fetch('/api/shap/explain?track_id=' + trackId);
    const d = await r.json();

    if (!d.ml_ready) {
      chartEl.innerHTML = '<div class="shap-empty">⚙️ ' + (d.message || 'Không có SHAP data') + '</div>';
      return;
    }

    // Banner predicted class
    const lvlCfg = LEVEL_COLORS[d.predicted_class] || LEVEL_COLORS.normal;
    banner.style.display     = 'block';
    banner.style.background  = lvlCfg.bg;
    banner.style.color       = lvlCfg.color;
    banner.style.borderColor = lvlCfg.border;
    banner.textContent       = `🎯 Dự đoán: ${d.predicted_class.toUpperCase()} | Base: ${d.base_value.toFixed(3)}`;

    // SHAP bar chart
    renderShapBars(chartEl, d.feature_names, d.shap_values, d.feature_values);

    // Probabilities row
    probaRow.style.display = 'flex';
    const probaHtml = Object.entries(d.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([cls, prob]) => {
        const isActive = cls === d.predicted_class;
        const pct = (prob * 100).toFixed(1);
        return `<span class="shap-proba-badge ${isActive ? 'active' : ''}" title="${cls}: ${pct}%">${cls}: ${pct}%</span>`;
      }).join('');
    probaRow.innerHTML = probaHtml;

  } catch(e) {
    if (e.message && e.message.includes('404')) {
      chartEl.innerHTML = '<div class="shap-empty">⚠️ Object không còn trong frame. Chọn lại.</div>';
    } else {
      chartEl.innerHTML = '<div class="shap-empty">❌ Lỗi: ' + e.message + '</div>';
    }
    banner.style.display = 'none';
    probaRow.style.display = 'none';
  }
}

function renderShapBars(container, names, shapValues, featValues) {
  // Sort by |shap| descending
  const indices = names.map((_, i) => i).sort((a, b) => Math.abs(shapValues[b]) - Math.abs(shapValues[a]));
  const maxAbs  = Math.max(...shapValues.map(Math.abs), 0.001);

  const html = indices.map(i => {
    const name    = names[i];
    const sv      = shapValues[i];
    const fv      = featValues[i];
    const label   = FEAT_LABELS[name] || name;
    const pct     = Math.round((Math.abs(sv) / maxAbs) * 100);
    const isPos   = sv >= 0;
    const barCls  = isPos ? 'shap-pos' : 'shap-neg';
    const svStr   = (sv >= 0 ? '+' : '') + sv.toFixed(3);
    const fvStr   = Number.isInteger(fv) ? fv : fv.toFixed(2);
    const tooltip = `${label}: SHAP=${svStr}, value=${fvStr}`;
    return `<div class="shap-feature-row" title="${tooltip}">
      <span class="shap-feat-name">${label}</span>
      <span class="shap-feat-val">${fvStr}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar ${barCls}" style="width:${pct}%"></div>
      </div>
      <span class="shap-shap-val" style="color:${isPos ? '#fb923c' : '#4ade80'}">${svStr}</span>
    </div>`;
  }).join('');

  container.innerHTML = html || '<div class="shap-empty">Không có dữ liệu SHAP</div>';
}

// ── Đồng bộ dropdown Object với fetchData ─────────────────
function updateShapObjectDropdown(objects) {
  if (_shapCurrentTab !== 'object') return;
  const sel = document.getElementById('shap-object-select');
  const currentVal = sel.value;
  sel.innerHTML = '<option value="">-- Chon Object --</option>'
    + objects.map(o => {
        const role  = o.role || o.class_name || 'unknown';
        const lvl   = o.alert_level || 'normal';
        return `<option value="${o.track_id}">#${o.track_id} ${role} [${lvl}]</option>`;
      }).join('');
  // Giữ lại lựa chọn cũ nếu còn tồn tại
  if (currentVal && objects.some(o => String(o.track_id) === currentVal)) {
    sel.value = currentVal;
  }
}

// Khởi động SHAP panel
loadShapGlobal();

</script>
</body>
</html>"""


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host   = API_CONFIG["host"],
        port   = API_CONFIG["port"],
        reload = False,
        log_level = "info",
    )
