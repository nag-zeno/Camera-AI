"""
alert_recorder.py — Tầng 5: Output (Video Alert Recorder)

Tự động ghi clip video khi phát hiện sự kiện alert/critical.
Features:
  - Pre-buffer: giữ N giây TRƯỚC khi alert để clip có context
  - Post-buffer: tiếp tục ghi N giây SAU sự kiện
  - Per-event cooldown: tránh ghi quá nhiều clip
  - Tự động dọn file cũ (giữ tối đa N clips)
  - Clip được đặt tên theo timestamp + level + role
  - Thread-safe, không block pipeline
"""
import cv2
import time
import queue
import logging
import threading
import collections
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Config defaults
# ============================================================
DEFAULT_PRE_BUFFER_SEC  = 5.0     # giây trước event được giữ trong buffer vòng
DEFAULT_POST_BUFFER_SEC = 8.0     # tiếp tục ghi sau event
DEFAULT_FPS             = 15.0    # FPS ghi video
DEFAULT_MAX_CLIPS       = 50      # số clip tối đa giữ lại
DEFAULT_COOLDOWN_SEC    = 20.0    # cooldown giữa 2 clip cùng track_id
DEFAULT_MIN_LEVEL       = "alert" # mức tối thiểu để trigger ghi


_LEVEL_ORDER = {"ignore": 0, "normal": 1, "watch": 2, "warning": 3, "alert": 4, "critical": 5}

def _level_gte(a: str, b: str) -> bool:
    return _LEVEL_ORDER.get(a, 0) >= _LEVEL_ORDER.get(b, 0)


# ============================================================
# AlertRecorder
# ============================================================
class AlertRecorder:
    """
    Ghi clip video khi có alert event.

    Usage:
        recorder = AlertRecorder(output_dir="recordings/")
        # Mỗi frame: cập nhật pre-buffer
        recorder.push_frame(frame)
        # Khi có event:
        recorder.on_event(event_dict, level="alert")
        # Dừng:
        recorder.stop()
    """

    def __init__(
        self,
        output_dir             : str | Path = "recordings",
        pre_buffer_sec         : float = DEFAULT_PRE_BUFFER_SEC,
        post_buffer_sec        : float = DEFAULT_POST_BUFFER_SEC,
        fps                    : float = DEFAULT_FPS,
        max_clips              : int   = DEFAULT_MAX_CLIPS,
        cooldown_sec           : float = DEFAULT_COOLDOWN_SEC,
        min_level              : str   = DEFAULT_MIN_LEVEL,
        enabled                : bool  = True,
    ):
        self._output_dir     = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._pre_buffer_sec  = pre_buffer_sec
        self._post_buffer_sec = post_buffer_sec
        self._fps             = fps
        self._max_clips       = max_clips
        self._cooldown_sec    = cooldown_sec
        self._min_level       = min_level
        self._enabled         = enabled

        # Pre-buffer: ring buffer của (timestamp, frame)
        max_pre_frames = int(pre_buffer_sec * fps) + 10
        self._pre_buffer: collections.deque = collections.deque(maxlen=max_pre_frames)

        # Active recording state
        self._recording        = False
        self._current_writer  : Optional[cv2.VideoWriter] = None
        self._current_path    : Optional[Path] = None
        self._post_end_time   : float = 0.0
        self._post_frames_left: int   = 0

        # Cooldown tracker per track_id
        self._cooldowns: dict[int, float] = {}
        self._lock = threading.Lock()

        # Worker thread
        self._task_queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop_evt   = threading.Event()
        self._worker     = threading.Thread(
            target=self._worker_loop,
            name="AlertRecorderWorker",
            daemon=True,
        )
        self._worker.start()
        logger.info(
            f"AlertRecorder started → {self._output_dir} "
            f"| pre={pre_buffer_sec}s post={post_buffer_sec}s "
            f"| min_level={min_level} | enabled={enabled}"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def push_frame(self, frame):
        """
        Gọi MỖI FRAME từ pipeline để cập nhật pre-buffer và ghi nếu đang recording.
        Phải gọi từ pipeline thread.
        """
        if not self._enabled:
            return

        ts = time.time()
        # Thêm vào pre-buffer (vòng)
        self._pre_buffer.append((ts, frame))

        # Nếu đang ghi post-buffer
        with self._lock:
            recording = self._recording
            writer    = self._current_writer

        if recording and writer is not None:
            try:
                writer.write(frame)
            except Exception:
                pass
            with self._lock:
                self._post_frames_left -= 1
                if self._post_frames_left <= 0:
                    self._stop_recording()

    def on_event(self, event: dict):
        """
        Gọi khi có AlertEvent mới từ pipeline.
        Tự động kiểm tra level + cooldown và trigger recording.
        """
        if not self._enabled:
            return

        level    = event.get("level", "normal")
        track_id = event.get("track_id", -1)
        role     = event.get("object_role", "unknown")
        reason   = event.get("reason", "")

        # Kiểm tra level tối thiểu
        if not _level_gte(level, self._min_level):
            return

        now = time.time()
        with self._lock:
            last = self._cooldowns.get(track_id, 0.0)
            if now - last < self._cooldown_sec:
                return
            self._cooldowns[track_id] = now

            # Không bắt đầu recording mới nếu đang ghi
            if self._recording:
                # Extend post-buffer thay vì bắt đầu clip mới
                self._post_frames_left = int(self._post_buffer_sec * self._fps)
                logger.debug(f"[AlertRecorder] Extended post-buffer for track #{track_id}")
                return

        # Enqueue task bắt đầu recording
        try:
            # Snapshot pre-buffer ngay bây giờ
            pre_frames = list(self._pre_buffer)
            self._task_queue.put_nowait({
                "action"     : "start",
                "pre_frames" : pre_frames,
                "level"      : level,
                "track_id"   : track_id,
                "role"       : role,
                "reason"     : reason,
                "ts"         : now,
            })
            logger.info(f"[AlertRecorder] Triggered clip for track #{track_id} level={level}")
        except queue.Full:
            logger.warning("[AlertRecorder] Task queue full, dropping recording trigger")

    def configure(
        self,
        enabled         : bool  = True,
        min_level       : str   = "alert",
        pre_buffer_sec  : float = DEFAULT_PRE_BUFFER_SEC,
        post_buffer_sec : float = DEFAULT_POST_BUFFER_SEC,
        cooldown_sec    : float = DEFAULT_COOLDOWN_SEC,
        max_clips       : int   = DEFAULT_MAX_CLIPS,
    ):
        """Cập nhật cấu hình runtime."""
        with self._lock:
            self._enabled         = enabled
            self._min_level       = min_level
            self._pre_buffer_sec  = pre_buffer_sec
            self._post_buffer_sec = post_buffer_sec
            self._cooldown_sec    = cooldown_sec
            self._max_clips       = max_clips
        logger.info(
            f"[AlertRecorder] Configured: enabled={enabled} min_level={min_level} "
            f"pre={pre_buffer_sec}s post={post_buffer_sec}s cooldown={cooldown_sec}s"
        )

    def get_config(self) -> dict:
        with self._lock:
            return {
                "enabled"         : self._enabled,
                "min_level"       : self._min_level,
                "pre_buffer_sec"  : self._pre_buffer_sec,
                "post_buffer_sec" : self._post_buffer_sec,
                "cooldown_sec"    : self._cooldown_sec,
                "max_clips"       : self._max_clips,
                "output_dir"      : str(self._output_dir),
                "is_recording"    : self._recording,
            }

    def list_clips(self) -> list[dict]:
        """Trả về danh sách các clip đã ghi, mới nhất trước."""
        clips = []
        for p in sorted(self._output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                stat = p.stat()
                clips.append({
                    "filename"    : p.name,
                    "path"        : str(p),
                    "size_kb"     : round(stat.st_size / 1024, 1),
                    "created_at"  : time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(stat.st_mtime)
                    ),
                })
            except OSError:
                pass
        return clips

    def stop(self):
        """Dừng worker và đóng file ghi nếu đang mở."""
        self._stop_evt.set()
        with self._lock:
            if self._recording:
                self._stop_recording()
        logger.info("[AlertRecorder] Stopped.")

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _stop_recording(self):
        """Đóng VideoWriter (PHẢI giữ self._lock khi gọi)."""
        if self._current_writer is not None:
            try:
                self._current_writer.release()
            except Exception:
                pass
            self._current_writer = None
        self._recording = False
        if self._current_path:
            logger.info(f"[AlertRecorder] Clip saved: {self._current_path.name}")
            self._current_path = None

    def _cleanup_old_clips(self):
        """Xóa clip cũ nhất nếu vượt max_clips."""
        with self._lock:
            max_c = self._max_clips
        clips = sorted(self._output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime)
        while len(clips) > max_c:
            try:
                clips[0].unlink()
                logger.debug(f"[AlertRecorder] Deleted old clip: {clips[0].name}")
                clips.pop(0)
            except OSError:
                break

    def _start_recording(self, task: dict):
        """
        Bắt đầu một clip mới:
        1. Tạo VideoWriter
        2. Ghi các pre-buffer frames
        3. Set trạng thái recording để push_frame() tiếp tục ghi
        """
        pre_frames  = task["pre_frames"]
        level       = task["level"]
        role        = task.get("role", "unknown")
        ts          = task["ts"]

        # Xác định kích thước frame từ pre-buffer
        frame_size = (640, 480)  # default
        if pre_frames:
            _, sample = pre_frames[-1]
            if sample is not None:
                h, w = sample.shape[:2]
                frame_size = (w, h)

        # Tên file: alert_20260505_213000_alert_shipper.mp4
        ts_str   = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        filename = f"alert_{ts_str}_{level}_{role}.mp4"
        out_path = self._output_dir / filename

        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, frame_size)
        if not writer.isOpened():
            logger.error(f"[AlertRecorder] Failed to open VideoWriter for {out_path}")
            return

        # Ghi pre-buffer frames
        for _, frm in pre_frames:
            if frm is not None:
                try:
                    f_resized = cv2.resize(frm, frame_size) if frm.shape[:2] != (frame_size[1], frame_size[0]) else frm
                    writer.write(f_resized)
                except Exception:
                    pass

        post_frames = int(self._post_buffer_sec * self._fps)

        with self._lock:
            self._current_writer  = writer
            self._current_path    = out_path
            self._recording       = True
            self._post_frames_left = post_frames

        logger.info(
            f"[AlertRecorder] Recording started → {filename} "
            f"(pre={len(pre_frames)} frames, post={post_frames} frames)"
        )

        # Dọn file cũ
        self._cleanup_old_clips()

    def _worker_loop(self):
        """Background thread xử lý task queue."""
        logger.info("[AlertRecorder] Worker thread started.")
        while not self._stop_evt.is_set():
            try:
                task = self._task_queue.get(timeout=1.0)
                if task.get("action") == "start":
                    self._start_recording(task)
                self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[AlertRecorder] Worker error: {e}")
        logger.info("[AlertRecorder] Worker thread stopped.")
