"""
behavior_analyzer.py — Tầng 3: Understanding (Temporal Analysis)

Nhận vào: TrackedObject (có lịch sử vị trí)
Trả ra  : TrackedObject với:
  - time_in_zone  : số giây ở trong zone
  - loitering     : True nếu đứng lâu
  - direction     : "stationary" / "moving_in" / "moving_out" / "moving"
  - visit_count   : số lần vào zone trong cửa sổ thời gian
"""
import time
import logging
from collections import deque

from config import BEHAVIOR_CONFIG
from models import TrackedObject, ZoneStatus

logger = logging.getLogger(__name__)


class _TrackHistory:
    """Lịch sử của 1 track."""

    def __init__(self, cfg: dict):
        self.cfg             = cfg
        self.positions       : deque = deque(maxlen=cfg["history_max_length"])
        self.zone_enter_time : float | None = None
        self.zone_exit_times : deque = deque()  # timestamps khi rời zone
        self.time_in_zone    : float = 0.0
        self.visit_count     : int   = 0


class BehaviorAnalyzer:
    """
    Phân tích hành vi theo thời gian dựa trên lịch sử vị trí.
    """

    def __init__(self):
        self._cfg          = BEHAVIOR_CONFIG
        self._loiter_thr   = self._cfg["loitering_time_threshold"]
        self._visit_thr    = self._cfg["frequent_visit_threshold"]
        self._visit_window = self._cfg["frequent_visit_window"]
        self._min_move     = self._cfg["movement_min_pixels"]

        self._histories: dict[int, _TrackHistory] = {}   # track_id → history

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def analyze(self, obj: TrackedObject) -> TrackedObject:
        """
        Phân tích hành vi của 1 TrackedObject.

        Args:
            obj: TrackedObject đã có zone info

        Returns:
            obj với time_in_zone, loitering, direction, visit_count đã cập nhật
        """
        track_id = obj.track_id
        if track_id not in self._histories:
            self._histories[track_id] = _TrackHistory(self._cfg)

        hist = self._histories[track_id]
        now  = time.time()

        # 1. Ghi vị trí hiện tại
        cx, cy = obj.bbox.center
        hist.positions.append((cx, cy, now))

        # 2. Tính direction
        obj.direction = self._compute_direction(hist)

        # 3. Cập nhật time_in_zone
        obj.time_in_zone, hist.zone_enter_time = self._update_zone_time(
            obj, hist, now
        )

        # 4. Loitering
        obj.loitering = obj.time_in_zone >= self._loiter_thr

        # 5. Visit count
        obj.visit_count = self._compute_visit_count(obj, hist, now)

        return obj

    def forget_track(self, track_id: int):
        """Xóa lịch sử khi track mất."""
        self._histories.pop(track_id, None)

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _compute_direction(self, hist: _TrackHistory) -> str:
        """Tính hướng di chuyển từ 5 frame gần nhất."""
        pts = list(hist.positions)
        if len(pts) < 5:
            return "stationary"

        recent = pts[-5:]
        dx     = recent[-1][0] - recent[0][0]
        dy     = recent[-1][1] - recent[0][1]
        dist   = (dx**2 + dy**2) ** 0.5

        if dist < self._min_move:
            return "stationary"

        # Hướng theo trục Y: moving_in (đến gần camera) / moving_out
        if abs(dy) > abs(dx):
            return "moving_in" if dy > 0 else "moving_out"
        else:
            return "moving"

    @staticmethod
    def _update_zone_time(
        obj: TrackedObject,
        hist: _TrackHistory,
        now: float,
    ) -> tuple[float, float | None]:
        """
        Cập nhật thời gian trong zone.

        Returns:
            (time_in_zone, updated zone_enter_time)
        """
        zone_enter = hist.zone_enter_time

        if obj.zone_status.value in ("entering", "inside"):
            if zone_enter is None:
                zone_enter = now
            time_in_zone = now - zone_enter
        elif obj.zone_status.value == "leaving":
            # Nhớ tổng thời gian đã ở trong zone này
            time_in_zone     = now - (zone_enter or now)
            hist.zone_exit_times.append(now)
            zone_enter = None
        else:  # outside
            time_in_zone = 0.0
            zone_enter   = None

        hist.time_in_zone = time_in_zone
        return time_in_zone, zone_enter

    def _compute_visit_count(
        self,
        obj: TrackedObject,
        hist: _TrackHistory,
        now: float,
    ) -> int:
        """Đếm số lần vào zone trong cửa sổ thời gian."""
        # Xóa exits quá cũ
        window_start = now - self._visit_window
        while hist.zone_exit_times and hist.zone_exit_times[0] < window_start:
            hist.zone_exit_times.popleft()

        count = len(hist.zone_exit_times)

        # Nếu đang trong zone, tính là đang visit
        if obj.zone_name and obj.zone_status.value in ("entering", "inside"):
            count += 1

        hist.visit_count = count
        return count
