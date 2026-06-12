"""
telegram_notifier.py — Tầng 5: Output (Telegram Push Notification)

Gửi thông báo Telegram khi có alert/critical event.
Features:
  - Gửi text message với thông tin đầy đủ (role, action, zone, level)
  - Đính kèm ảnh snapshot của frame khi phát hiện sự kiện
  - Cooldown per-object để tránh spam
  - Background queue bất đồng bộ (không block pipeline)
  - Tự động retry khi lỗi mạng
  - Enable/disable runtime qua API

Cấu hình:
  - Đặt TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID trong config.py
    hoặc override qua POST /api/telegram/config
"""
import io
import time
import queue
import logging
import threading
import urllib.request
import urllib.parse
import urllib.error
import json
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass
class TelegramConfig:
    """Runtime config cho Telegram notifier."""
    bot_token    : str  = ""
    chat_id      : str  = ""
    enabled      : bool = False
    min_level    : str  = "alert"       # Chỉ gửi từ level này trở lên: warning/alert/critical
    cooldown_sec : float = 30.0         # Thời gian tối thiểu giữa 2 thông báo cho cùng 1 object
    send_photo   : bool = True          # Đính kèm ảnh snapshot hay không
    max_retries  : int  = 2             # Số lần thử lại khi lỗi mạng


@dataclass
class NotifyTask:
    """Một tác vụ gửi thông báo."""
    message    : str
    frame      : Optional[np.ndarray] = field(default=None, repr=False)
    track_id   : int = -1
    level      : str = "alert"
    created_at : float = field(default_factory=time.time)


# ============================================================
# Alert level order
# ============================================================
_LEVEL_ORDER = {"ignore": 0, "normal": 1, "watch": 2, "warning": 3, "alert": 4, "critical": 5}


def _level_gte(a: str, b: str) -> bool:
    """Trả True nếu level a >= level b."""
    return _LEVEL_ORDER.get(a, 0) >= _LEVEL_ORDER.get(b, 0)


# ============================================================
# Telegram API helpers
# ============================================================

def _api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(url: str, data: dict, timeout: int = 10) -> dict:
    """POST JSON tới Telegram API, trả về response dict."""
    body = json.dumps(data).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_photo(url: str, chat_id: str, photo_bytes: bytes,
                caption: str, timeout: int = 15) -> dict:
    """Upload ảnh lên Telegram (multipart/form-data)."""
    import email.mime.multipart
    import email.mime.base
    import email.encoders

    boundary = b"----TelegramBoundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="chat_id"\r\n\r\n' +
        chat_id.encode() + b"\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="caption"\r\n\r\n' +
        caption.encode("utf-8") + b"\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="photo"; filename="alert.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n" +
        photo_bytes + b"\r\n"
        b"--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ============================================================
# Notifier class
# ============================================================

class TelegramNotifier:
    """
    Gửi push notification lên Telegram khi có alert/critical event.

    Usage:
        notifier = TelegramNotifier()
        notifier.configure(bot_token="...", chat_id="...", enabled=True)
        notifier.notify(event_dict, frame=current_frame)
    """

    def __init__(self):
        self._cfg      = TelegramConfig()
        self._queue    : queue.Queue[NotifyTask] = queue.Queue(maxsize=50)
        self._cooldown : dict[int, float] = {}   # track_id → last_sent_time
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()
        self._worker   = threading.Thread(
            target=self._worker_loop,
            name="TelegramWorker",
            daemon=True,
        )
        self._worker.start()
        logger.info("TelegramNotifier started (disabled by default)")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def configure(
        self,
        bot_token    : str   = "",
        chat_id      : str   = "",
        enabled      : bool  = False,
        min_level    : str   = "alert",
        cooldown_sec : float = 30.0,
        send_photo   : bool  = True,
    ):
        """Cập nhật cấu hình Telegram runtime."""
        with self._lock:
            self._cfg.bot_token    = bot_token.strip()
            self._cfg.chat_id      = chat_id.strip()
            self._cfg.enabled      = enabled and bool(bot_token) and bool(chat_id)
            self._cfg.min_level    = min_level
            self._cfg.cooldown_sec = max(5.0, cooldown_sec)
            self._cfg.send_photo   = send_photo
        status = "ENABLED" if self._cfg.enabled else "DISABLED"
        logger.info(f"TelegramNotifier {status} | min_level={min_level} | cooldown={cooldown_sec}s")

    def get_config(self) -> dict:
        """Trả về config hiện tại (che token)."""
        with self._lock:
            cfg = self._cfg
            token_masked = ("***" + cfg.bot_token[-4:]) if len(cfg.bot_token) > 4 else ("***" if cfg.bot_token else "")
            return {
                "enabled"      : cfg.enabled,
                "bot_token"    : token_masked,
                "chat_id"      : cfg.chat_id,
                "min_level"    : cfg.min_level,
                "cooldown_sec" : cfg.cooldown_sec,
                "send_photo"   : cfg.send_photo,
            }

    def notify(self, event: dict, frame: Optional[np.ndarray] = None):
        """
        Kiểm tra event và đẩy vào queue nếu đủ điều kiện.

        Args:
            event: AlertEvent dict (từ EventLogger.log / pipeline result)
            frame: Frame hiện tại (numpy array BGR) để đính kèm ảnh
        """
        with self._lock:
            if not self._cfg.enabled:
                return
            level     = event.get("level", "normal")
            min_level = self._cfg.min_level
            if not _level_gte(level, min_level):
                return

            track_id     = event.get("track_id", -1)
            now          = time.time()
            last_sent    = self._cooldown.get(track_id, 0.0)
            cooldown_sec = self._cfg.cooldown_sec

        if now - last_sent < cooldown_sec:
            logger.debug(
                f"[Telegram] Cooldown active for track #{track_id} "
                f"({now - last_sent:.1f}s < {cooldown_sec}s)"
            )
            return

        # Build message
        message = self._build_message(event)

        task = NotifyTask(
            message  = message,
            frame    = frame.copy() if (frame is not None and self._cfg.send_photo) else None,
            track_id = track_id,
            level    = level,
        )

        try:
            self._queue.put_nowait(task)
            # Update cooldown ngay khi đã enqueue (tránh duplicate đẩy vào queue)
            with self._lock:
                self._cooldown[track_id] = now
            logger.debug(f"[Telegram] Task enqueued for track #{track_id} level={level}")
        except queue.Full:
            logger.warning("[Telegram] Notification queue full, dropping task")

    def test_connection(self) -> tuple[bool, str]:
        """
        Test kết nối Telegram API.
        Trả về (success: bool, message: str).
        """
        with self._lock:
            token   = self._cfg.bot_token
            chat_id = self._cfg.chat_id

        if not token or not chat_id:
            return False, "Bot token hoặc Chat ID chưa được cấu hình"

        try:
            url  = _api_url(token, "sendMessage")
            data = {
                "chat_id"    : chat_id,
                "text"       : "🔔 *AI Security Camera* — Kết nối thành công! Thông báo Telegram đã được kích hoạt.",
                "parse_mode" : "Markdown",
            }
            resp = _post_json(url, data, timeout=10)
            if resp.get("ok"):
                return True, "Kết nối thành công! Tin nhắn test đã được gửi."
            else:
                return False, f"Telegram API lỗi: {resp.get('description', 'Unknown error')}"
        except Exception as e:
            return False, f"Lỗi kết nối: {str(e)}"

    def stop(self):
        """Dừng worker thread."""
        self._stop_evt.set()
        logger.info("TelegramNotifier stopped.")

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _build_message(self, event: dict) -> str:
        """Tạo tin nhắn cảnh báo thân thiện, tự nhiên như chat Zalo/Messenger."""
        level      = event.get("level", "normal")
        reason     = event.get("reason", "")
        datetime_s = event.get("datetime", "")

        # Nội dung chính: dùng câu NLG đã sinh sẵn từ context engine
        # (Gemini hoặc fallback template — đều đã được cải thiện)
        if reason:
            main_text = reason
        else:
            # Fallback nếu không có reason (hiếm gặp)
            role = event.get("object_role", "")
            zone = event.get("zone_name") or "khu vực giám sát"
            import re
            if not zone or re.match(r'^(zone_\d+|rtsp_|webcam_)', zone, re.IGNORECASE):
                zone = "khu vực giám sát"
            role_vi = {
                "unknown": "người lạ", "normal": "người", "shipper": "Shipper",
                "security": "bảo vệ", "police": "cảnh sát",
            }.get(role, "người") if role else "người"

            if level == "critical":
                main_text = f"🚨 Khẩn! Camera phát hiện tình huống nguy hiểm tại {zone}. Kiểm tra ngay bạn ơi!"
            elif level == "alert":
                main_text = f"🔴 Bạn ơi, có {role_vi} cần chú ý tại {zone}!"
            elif level == "warning":
                main_text = f"⚠️ Camera thấy có điều bất thường tại {zone}, để ý nhé."
            else:
                main_text = f"📷 Camera ghi nhận có {role_vi} tại {zone}."

        # Timestamp nhẹ nhàng ở cuối
        time_part = f"\n_🕐 {datetime_s}_" if datetime_s else ""

        return f"{main_text}{time_part}"

    def _send_task(self, task: NotifyTask):
        """Thực sự gửi task qua Telegram API (chạy trong worker thread)."""
        with self._lock:
            token      = self._cfg.bot_token
            chat_id    = self._cfg.chat_id
            send_photo = self._cfg.send_photo
            max_retry  = self._cfg.max_retries

        if not token or not chat_id:
            return

        for attempt in range(max_retry + 1):
            try:
                if send_photo and task.frame is not None:
                    # Encode frame thành JPEG
                    ok, buf = cv2.imencode(".jpg", task.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        url = _api_url(token, "sendPhoto")
                        _post_photo(url, chat_id, buf.tobytes(), task.message)
                        logger.info(
                            f"[Telegram] Photo sent for track #{task.track_id} level={task.level}"
                        )
                        return

                # Fallback: gửi text-only
                url  = _api_url(token, "sendMessage")
                data = {
                    "chat_id"    : chat_id,
                    "text"       : task.message,
                    "parse_mode" : "Markdown",
                }
                _post_json(url, data)
                logger.info(
                    f"[Telegram] Message sent for track #{task.track_id} level={task.level}"
                )
                return

            except urllib.error.URLError as e:
                logger.warning(
                    f"[Telegram] Network error (attempt {attempt+1}/{max_retry+1}): {e}"
                )
                if attempt < max_retry:
                    time.sleep(2.0 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                logger.error(f"[Telegram] Unexpected error: {e}")
                return

    def _worker_loop(self):
        """Background thread: xử lý queue gửi thông báo."""
        logger.info("[Telegram] Worker thread started")
        while not self._stop_evt.is_set():
            try:
                task = self._queue.get(timeout=1.0)
                self._send_task(task)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Telegram] Worker error: {e}")
        logger.info("[Telegram] Worker thread stopped")
