"""
event_logger.py — Tầng 5: Output (Event Logger)

Ghi AlertEvent ra file JSONL (JSON Lines) để phân tích sau.
Thread-safe, hỗ trợ query + export.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from config import API_CONFIG
from models import AlertEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Ghi event ra file JSONL (1 JSON object per line).
    Giữ lịch sử trong memory để query nhanh.
    """

    def __init__(self, log_file: str | Path | None = None):
        self._log_file = Path(log_file or API_CONFIG["log_file"])
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        self._events: list[dict] = []   # In-memory cache
        self._lock   = threading.Lock()

        # Load events cũ nếu file đã tồn tại
        self._load_existing()
        logger.info(f"EventLogger writing to '{self._log_file}'")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def log(self, event: AlertEvent):
        """Ghi 1 event ra file và memory."""
        event_dict = event.to_dict()

        with self._lock:
            self._events.append(event_dict)
            self._append_to_file(event_dict)

    def get_recent(
        self,
        n: int = 50,
        level: str | None = None,
        since: float | None = None,
    ) -> list[dict]:
        """
        Lấy N events gần nhất.

        Args:
            n    : Số events tối đa
            level: Lọc theo alert level (vd: "alert", "critical")
            since: Chỉ lấy events sau timestamp này
        """
        with self._lock:
            events = list(self._events)

        if level:
            events = [e for e in events if e.get("level") == level]
        if since:
            events = [e for e in events if e.get("timestamp", 0) >= since]

        return events[-n:]

    def get_stats(self) -> dict:
        """Trả về thống kê tổng hợp."""
        with self._lock:
            events = list(self._events)

        if not events:
            return {"total": 0}

        level_counts: dict[str, int] = {}
        role_counts : dict[str, int] = {}
        zone_counts : dict[str, int] = {}
        action_counts: dict[str, int] = {}

        for e in events:
            level = e.get("level", "unknown")
            role  = e.get("object_role", "unknown")
            zone  = e.get("zone_name") or "outside"
            action= e.get("action", "unknown")

            level_counts[level]   = level_counts.get(level, 0) + 1
            role_counts[role]     = role_counts.get(role, 0) + 1
            zone_counts[zone]     = zone_counts.get(zone, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1

        first_ts = events[0].get("timestamp", 0)
        last_ts  = events[-1].get("timestamp", 0)

        return {
            "total"          : len(events),
            "by_level"       : level_counts,
            "by_role"        : role_counts,
            "by_zone"        : zone_counts,
            "by_action"      : action_counts,
            "first_event_at" : events[0].get("datetime", ""),
            "last_event_at"  : events[-1].get("datetime", ""),
            "duration_hours" : round((last_ts - first_ts) / 3600, 2) if last_ts > first_ts else 0,
        }

    def export_json(self) -> list[dict]:
        """Export toàn bộ events dưới dạng list dict."""
        with self._lock:
            return list(self._events)

    def clear(self):
        """Xóa memory cache (không xóa file)."""
        with self._lock:
            self._events.clear()

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _append_to_file(self, event_dict: dict):
        """Ghi 1 dòng JSON vào file."""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Failed to write event log: {e}")

    def _load_existing(self):
        """Load events đã có trong file (nếu tồn tại)."""
        if not self._log_file.exists():
            return
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            logger.info(f"Loaded {len(self._events)} existing events from log.")
        except OSError as e:
            logger.warning(f"Could not load existing log: {e}")
