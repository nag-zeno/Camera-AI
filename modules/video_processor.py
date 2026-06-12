"""
video_processor.py — Tầng 1: Video Input

Nhận vào: camera index / RTSP URL / file path
Trả ra  : iterator của (frame_id, numpy frame) đã normalize

Kiến trúc:
  - RTSP stream  : FFmpeg subprocess pipe (imageio-ffmpeg) → hoàn toàn tránh
                   bug async_lock của OpenCV 4.x / FFmpeg H.265 decoder.
  - File / Webcam: cv2.VideoCapture trong reader thread riêng biệt.
  - Pipeline thread KHÔNG BAO GIỜ gọi cap.read() hay đọc subprocess trực tiếp.
  - Mọi nguồn đều lưu frame vào buffer thread-safe (_last_frame).
"""
import os
import cv2
import time
import subprocess
import threading
import logging
import shutil
from typing import Generator, Optional

import numpy as np

from config import VIDEO_CONFIG

logger = logging.getLogger(__name__)


def _get_ffmpeg_exe() -> Optional[str]:
    """Tìm ffmpeg binary: PATH → imageio_ffmpeg → None."""
    # 1. Thử PATH
    ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ff:
        return ff
    # 2. Thử imageio_ffmpeg (đi kèm binary riêng)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return None


# Cache đường dẫn ffmpeg để không phải tìm lại mỗi lần
_FFMPEG_EXE: Optional[str] = _get_ffmpeg_exe()


class VideoProcessor:
    """
    Hỗ trợ 3 nguồn đầu vào:
      - Webcam (int index)
      - File video (str path)
      - RTSP stream (str starting rtsp://)

    RTSP: dùng FFmpeg subprocess pipe để hoàn toàn tránh bug async_lock
    của OpenCV FFmpeg backend với H.265/HEVC stream.

    File/Webcam: dùng cv2.VideoCapture trong reader thread riêng.

    Pipeline thread KHÔNG BAO GIỜ gọi cap.read() hay đọc pipe trực tiếp.
    """

    # Timeout chờ frame đầu tiên (giây)
    RTSP_FIRST_FRAME_TIMEOUT = 20
    LOCAL_OPEN_TIMEOUT       = 10
    # Delay reconnect RTSP (giây)
    RTSP_RECONNECT_DELAY     = 3
    # Số lần thất bại liên tiếp trước khi reconnect
    RTSP_MAX_FAILS           = 150   # ~5s at 30fps

    def __init__(self, source=None):
        self.source     = source if source is not None else VIDEO_CONFIG["default_source"]
        if isinstance(self.source, str):
            self.source = self.source.strip().strip("'").strip('"')
        self.target_fps = VIDEO_CONFIG["target_fps"]
        self.width      = VIDEO_CONFIG["frame_width"]
        self.height     = VIDEO_CONFIG["frame_height"]
        self.buf_size   = VIDEO_CONFIG["buffer_size"]
        self.loop_video = VIDEO_CONFIG["loop_video"]

        # Detect source type
        self._is_rtsp = (
            isinstance(self.source, str)
            and self.source.lower().startswith("rtsp")
        )

        # Shared state
        self._lock             = threading.Lock()
        self._last_frame       = None
        self._latest_raw_frame = None
        self._running          = False
        self._thread           : Optional[threading.Thread] = None
        self._frame_id         = 0

        # Sync: chờ frame đầu tiên từ reader thread
        self._first_frame_event = threading.Event()
        self._open_error        = False
        self._source_exhausted  = False

        # RTSP state
        self._connected        = False
        self._reconnect_count  = 0

        # OpenCV cap — chỉ dùng cho non-RTSP, được quản lý trong reader thread
        self._cap : Optional[cv2.VideoCapture] = None

        logger.info(
            f"[VideoProcessor] source={self._mask_rtsp_url(str(self.source))!r} "
            f"is_rtsp={self._is_rtsp} "
            f"ffmpeg={'available' if _FFMPEG_EXE else 'NOT FOUND'}"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def open(self) -> bool:
        """Mở nguồn video, khởi động reader thread, chờ frame đầu tiên."""
        self._open_error       = False
        self._source_exhausted = False
        self._first_frame_event.clear()
        self._running          = True

        if self._is_rtsp:
            if _FFMPEG_EXE:
                target  = self._rtsp_ffmpeg_loop
                name    = "rtsp-ffmpeg"
                timeout = self.RTSP_FIRST_FRAME_TIMEOUT
                logger.info(
                    f"[VideoProcessor] Mở RTSP qua FFmpeg subprocess: "
                    f"{self._mask_rtsp_url(self.source)}"
                )
            else:
                # Fallback: OpenCV (có thể gặp assertion nhưng không còn lựa chọn)
                target  = self._rtsp_opencv_loop
                name    = "rtsp-opencv"
                timeout = self.RTSP_FIRST_FRAME_TIMEOUT
                logger.warning(
                    "[VideoProcessor] FFmpeg không có trong PATH. "
                    "Dùng OpenCV RTSP fallback (có thể gặp assertion lỗi)."
                )
        else:
            target  = self._local_reader_loop
            name    = "local-reader"
            timeout = self.LOCAL_OPEN_TIMEOUT

        self._thread = threading.Thread(target=target, daemon=True, name=name)
        self._thread.start()

        got = self._first_frame_event.wait(timeout=timeout)
        if not got or self._open_error:
            logger.error(
                f"[VideoProcessor] Không mở được nguồn sau {timeout}s: "
                f"{self._mask_rtsp_url(str(self.source))!r}"
            )
            self._running = False
            if self._thread:
                self._thread.join(timeout=3.0)
            return False

        logger.info(f"[VideoProcessor] ✅ Nguồn sẵn sàng: {self._mask_rtsp_url(str(self.source))!r}")
        return True

    def frames(self) -> Generator[tuple[int, any], None, None]:
        """Iterator trả về (frame_id, bgr_frame). Throttle theo target_fps."""
        if self._thread is None:
            raise RuntimeError("Call open() before frames()")

        interval         = 1.0 / self.target_fps
        _no_signal_frame = self._make_no_signal_frame()

        while True:
            t0 = time.monotonic()

            with self._lock:
                frame = self._last_frame.copy() if self._last_frame is not None else None

            if frame is None:
                if self._source_exhausted:
                    logger.info("[VideoProcessor] Nguồn video đã kết thúc.")
                    break
                if self._is_rtsp:
                    frame = _no_signal_frame.copy()
                else:
                    time.sleep(0.01)
                    continue

            self._frame_id += 1
            with self._lock:
                self._latest_raw_frame = frame.copy()
            yield self._frame_id, frame

            elapsed = time.monotonic() - t0
            sleep   = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def get_latest_raw_frame(self):
        """Frame thô mới nhất (không annotation) — dùng cho Zone Editor snapshot."""
        with self._lock:
            return self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None

    def get_connection_status(self) -> dict:
        """Trạng thái kết nối RTSP."""
        return {
            "is_rtsp"         : self._is_rtsp,
            "connected"       : self._connected,
            "reconnect_count" : self._reconnect_count,
            "source_masked"   : self._mask_rtsp_url(str(self.source)) if self._is_rtsp else str(self.source),
        }

    def release(self):
        """Giải phóng tài nguyên."""
        self._running   = False
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4.0)
        logger.info("[VideoProcessor] Released.")

    # ----------------------------------------------------------
    # Reader loop — RTSP via FFmpeg subprocess (PRIMARY)
    # ----------------------------------------------------------

    def _rtsp_ffmpeg_loop(self):
        """
        Đọc RTSP stream qua FFmpeg subprocess pipe.
        Hoàn toàn tránh bug async_lock của OpenCV FFmpeg backend.
        FFmpeg chạy như 1 process riêng biệt, output raw BGR frames ra stdout.
        """
        frame_size = self.width * self.height * 3
        first_frame_signaled = False

        while self._running:
            cmd = [
                _FFMPEG_EXE,
                "-loglevel",       "error",
                "-rtsp_transport", "tcp",
                "-i",              self.source,
                "-vf",             f"fps={self.target_fps},scale={self.width}:{self.height}",
                "-f",              "rawvideo",
                "-pix_fmt",        "bgr24",
                "-an",             # no audio
                "pipe:1",
            ]

            logger.info(f"[RTSP-FFmpeg] Khởi động FFmpeg process...")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_size * 4,
                )
            except Exception as e:
                logger.error(f"[RTSP-FFmpeg] Không thể khởi động FFmpeg: {e}")
                self._open_error = True
                self._first_frame_event.set()
                return

            consecutive_empty = 0

            try:
                while self._running:
                    raw = proc.stdout.read(frame_size)

                    if len(raw) == frame_size:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                            (self.height, self.width, 3)
                        ).copy()  # .copy() để tách khỏi buffer của subprocess

                        with self._lock:
                            self._last_frame       = frame
                            self._latest_raw_frame = frame.copy()

                        self._connected       = True
                        consecutive_empty     = 0

                        if not first_frame_signaled:
                            first_frame_signaled = True
                            logger.info(
                                f"[RTSP-FFmpeg] ✅ Kết nối thành công: "
                                f"{self._mask_rtsp_url(self.source)} "
                                f"({self.width}x{self.height})"
                            )
                            self._first_frame_event.set()

                    else:
                        # FFmpeg đầu ra rỗng → stream kết thúc hoặc lỗi
                        if len(raw) == 0:
                            consecutive_empty += 1
                            if consecutive_empty >= 3:
                                logger.warning("[RTSP-FFmpeg] FFmpeg process kết thúc bất ngờ.")
                                break
                        time.sleep(0.01)

                        if not first_frame_signaled and proc.poll() is not None:
                            stderr_out = proc.stderr.read(500).decode(errors='replace')
                            logger.error(f"[RTSP-FFmpeg] FFmpeg lỗi: {stderr_out}")
                            self._open_error = True
                            self._first_frame_event.set()
                            return

            finally:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            if not self._running:
                break

            self._connected = False
            self._reconnect_count += 1
            logger.info(
                f"[RTSP-FFmpeg] Thử reconnect lần {self._reconnect_count} "
                f"sau {self.RTSP_RECONNECT_DELAY}s..."
            )
            time.sleep(self.RTSP_RECONNECT_DELAY)

    # ----------------------------------------------------------
    # Reader loop — RTSP via OpenCV (FALLBACK, nếu không có ffmpeg)
    # ----------------------------------------------------------

    def _rtsp_opencv_loop(self):
        """
        Fallback: đọc RTSP qua OpenCV khi không có FFmpeg subprocess.
        Mở VÀ đọc trong cùng 1 thread để giảm thiểu cross-thread issues.
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|threads;1|fflags;nobuffer"
            "|flags;low_delay|max_delay;500000|reorder_queue_size;0"
        )

        first_frame_signaled = False
        consecutive_fails    = 0

        while self._running:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,        self.buf_size)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)

            if not cap.isOpened():
                logger.warning(f"[RTSP-OpenCV] Không mở được kết nối.")
                cap.release()
                if not first_frame_signaled:
                    self._open_error = True
                    self._first_frame_event.set()
                    return
                time.sleep(self.RTSP_RECONNECT_DELAY)
                self._reconnect_count += 1
                continue

            self._cap = cap

            while self._running:
                ret, frame = cap.read()
                if ret and frame is not None:
                    resized = self._resize(frame)
                    with self._lock:
                        self._last_frame       = resized
                        self._latest_raw_frame = resized.copy()
                    self._connected   = True
                    consecutive_fails = 0
                    if not first_frame_signaled:
                        first_frame_signaled = True
                        logger.info(f"[RTSP-OpenCV] ✅ Kết nối: {self._mask_rtsp_url(self.source)}")
                        self._first_frame_event.set()
                else:
                    consecutive_fails += 1
                    time.sleep(0.01)
                    if not first_frame_signaled and consecutive_fails >= 30:
                        self._open_error = True
                        self._first_frame_event.set()
                        cap.release()
                        self._cap = None
                        return
                    if consecutive_fails >= self.RTSP_MAX_FAILS:
                        self._connected = False
                        break

            cap.release()
            self._cap = None

            if not self._running:
                break

            self._reconnect_count += 1
            time.sleep(self.RTSP_RECONNECT_DELAY)

    # ----------------------------------------------------------
    # Reader loop — Local (webcam / file)
    # ----------------------------------------------------------

    def _local_reader_loop(self):
        """
        Đọc webcam hoặc file video.
        Mở VÀ đọc VideoCapture trong cùng 1 thread.

        Với file video: throttle theo FPS gốc của video để tránh hiện tượng
        phát ở tốc độ tua nhanh (do đọc frame không giới hạn khiến _last_frame
        bị ghi đè liên tục và pipeline bỏ qua nhiều frame ở giữa).
        """
        # Cho RTSP URL bị phát hiện muộn (an toàn tuyệt đối)
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            logger.warning("[VideoProcessor] RTSP URL đến _local_reader_loop — chuyển hướng!")
            self._is_rtsp = True
            if _FFMPEG_EXE:
                self._rtsp_ffmpeg_loop()
            else:
                self._rtsp_opencv_loop()
            return

        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            logger.error(f"[VideoProcessor] Không mở được: {self.source!r}")
            self._open_error = True
            self._first_frame_event.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Xác định FPS gốc của video để throttle đúng tốc độ phát.
        # Chỉ áp dụng cho file video (không phải webcam — webcam tự throttle qua phần cứng).
        is_file = isinstance(self.source, str) and not isinstance(self.source, int)
        src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        if src_fps and src_fps > 0 and is_file:
            # Dùng FPS gốc của video (capped tại target_fps để không vượt quá năng lực xử lý)
            read_fps      = min(src_fps, self.target_fps)
            read_interval = 1.0 / read_fps
            logger.info(
                f"Opened source '{self.source}' "
                f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"@ {src_fps:.1f} fps) — throttle reader at {read_fps:.1f} fps"
            )
        else:
            # Webcam hoặc FPS không xác định → không cần throttle (hardware tự giới hạn)
            read_interval = 0.0
            logger.info(
                f"Opened source '{self.source}' "
                f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} fps)"
            )

        self._cap              = cap
        first_frame_signaled   = False

        try:
            while self._running:
                t_read_start = time.monotonic()

                ret, frame = cap.read()
                if ret and frame is not None:
                    resized = self._resize(frame)
                    with self._lock:
                        self._last_frame       = resized
                        self._latest_raw_frame = resized.copy()
                    self._connected = True
                    if not first_frame_signaled:
                        first_frame_signaled = True
                        self._first_frame_event.set()

                    # Throttle: ngủ đủ thời gian để đọc đúng FPS nguồn video.
                    # Chỉ áp dụng cho file video (read_interval > 0).
                    if read_interval > 0:
                        elapsed = time.monotonic() - t_read_start
                        sleep   = read_interval - elapsed
                        if sleep > 0:
                            time.sleep(sleep)
                else:
                    if self.loop_video and not isinstance(self.source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    else:
                        if not first_frame_signaled:
                            self._open_error = True
                            self._first_frame_event.set()
                        else:
                            self._source_exhausted = True
                        break
        finally:
            cap.release()
            self._cap       = None
            self._connected = False

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def _make_no_signal_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype="uint8")
        frame[:] = (20, 20, 30)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        text1 = "DANG KET NOI CAMERA..."
        text2 = self._mask_rtsp_url(str(self.source))[:50]
        cv2.putText(frame, text1, (self.width // 2 - 180, self.height // 2 - 20),
                    font, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, text2, (self.width // 2 - 200, self.height // 2 + 20),
                    font, 0.45, (100, 100, 100), 1)
        return frame

    @staticmethod
    def _mask_rtsp_url(url: str) -> str:
        try:
            if "://" in url and "@" in url:
                proto, rest = url.split("://", 1)
                creds, host = rest.split("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    return f"{proto}://{user}:***@{host}"
        except Exception:
            pass
        return url

    # ----------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
