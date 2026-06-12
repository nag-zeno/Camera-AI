"""
zone_detector.py — Tầng 3: Understanding (Spatial Context)

Nhận vào: TrackedObject + zone config
Trả ra  : TrackedObject với zone_name, zone_type, zone_status

Zone được định nghĩa bằng polygon pixel (trong frame 640x480).
Có thể cập nhật zone runtime qua update_zones().

Persistence per-camera:
  - Mỗi camera source có file zones riêng (ví dụ: data/zones/rtsp_192.168.1.200_ch201.json)
  - Khi đổi camera → tự động load zones của camera đó (không bị dùng zones cũ)
  - Khi update_zones() được gọi → lưu xuống file đúng camera
"""
import json
import cv2
import numpy as np
import logging
from pathlib import Path

from config import ZONE_CONFIG, ZONES_PERSIST_FILE
from models import TrackedObject, ZoneType, ZoneStatus

logger = logging.getLogger(__name__)


class Zone:
    """Đại diện cho một khu vực."""

    def __init__(self, cfg: dict):
        self.name    : str               = cfg["name"]
        self.type    : ZoneType          = ZoneType(cfg["type"])
        self.polygon : np.ndarray        = np.array(cfg["polygon"], dtype=np.int32)
        self.color   : tuple[int,int,int]= tuple(cfg["color"])

    def contains_point(self, x: float, y: float) -> bool:
        """Kiểm tra điểm (x,y) có nằm trong polygon không."""
        return cv2.pointPolygonTest(
            self.polygon,
            (float(x), float(y)),
            measureDist=False,
        ) >= 0

    def to_dict(self) -> dict:
        """Chuyển Zone thành dict để serialize JSON."""
        return {
            "name"   : self.name,
            "type"   : self.type.value,
            "polygon": self.polygon.tolist(),
            "color"  : list(self.color),
        }


class ZoneDetector:
    """
    Xác định object đang ở zone nào dựa trên tọa độ bbox center.
    Theo dõi trạng thái entering/inside/leaving.

    Persistence per-camera:
      - Mỗi camera source → 1 file JSON riêng trong data/zones/
      - Khi khởi động: load từ file của camera này,
        fallback về ZONE_CONFIG mặc định trong config.py
      - Khi update_zones() được gọi: lưu xuống file đúng camera
    """

    def __init__(self, persist_file: Path = None):
        self._zones    : list[Zone]              = []
        self._prev_zone: dict[int, str | None]   = {}  # track_id -> zone_name
        self._persist  : Path                    = persist_file or ZONES_PERSIST_FILE

        # Ưu tiên 1: file zones riêng cho camera này
        if self._persist.exists():
            loaded = self._load_from_file(self._persist)
            if loaded:
                logger.info(
                    f"[ZoneDetector] Loaded {len(self._zones)} zones "
                    f"from {self._persist.name}"
                )
                return

        # Ưu tiên 2: migrate từ file zones.json cũ (global) nếu có
        # Chỉ migrate khi file đích khác file legacy để tránh vòng lặp
        if ZONES_PERSIST_FILE.exists() and self._persist != ZONES_PERSIST_FILE:
            logger.info(
                f"[ZoneDetector] Thử migrate zones.json cũ → {self._persist.name}"
            )
            loaded = self._load_from_file(ZONES_PERSIST_FILE)
            if loaded:
                self._save_to_file()  # Lưu sang file của camera mới
                logger.info(
                    f"[ZoneDetector] Migrate xong: {len(self._zones)} zones "
                    f"→ {self._persist.name}"
                )
                return

        # Ưu tiên 3: ZONE_CONFIG mặc định trong config.py
        for z_cfg in ZONE_CONFIG.get("zones", []):
            self._zones.append(Zone(z_cfg))
        logger.info(
            f"[ZoneDetector] Initialized {len(self._zones)} default zones "
            f"(no file for {self._persist.name})."
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_zones(self, zones_config: list[dict]):
        """
        Cập nhật zones tại runtime (gọi từ API).
        Tự động lưu xuống file của camera này.
        """
        self._zones = [Zone(z) for z in zones_config]
        self._save_to_file()
        logger.info(
            f"[ZoneDetector] Zones updated ({len(self._zones)}) → {self._persist.name}"
        )

    def detect(self, obj: TrackedObject) -> TrackedObject:
        """
        Xác định zone cho 1 TrackedObject.
        Cập nhật zone_name, zone_type, zone_status.
        """
        cx, cy    = obj.bbox.center
        prev_zone = self._prev_zone.get(obj.track_id)

        current_zone: Zone | None = None
        for zone in self._zones:
            if zone.contains_point(cx, cy):
                current_zone = zone
                break  # Ưu tiên zone đầu tiên match

        if current_zone is None:
            if prev_zone is not None:
                obj.zone_status = ZoneStatus.LEAVING
            else:
                obj.zone_status = ZoneStatus.OUTSIDE
            obj.zone_name = None
            obj.zone_type = None
        else:
            obj.zone_name = current_zone.name
            obj.zone_type = current_zone.type

            if prev_zone is None or prev_zone != current_zone.name:
                obj.zone_status = ZoneStatus.ENTERING
            else:
                obj.zone_status = ZoneStatus.INSIDE

        self._prev_zone[obj.track_id] = (
            current_zone.name if current_zone else None
        )
        return obj

    def forget_track(self, track_id: int):
        """Xóa cache khi track mất."""
        self._prev_zone.pop(track_id, None)

    def get_zones(self) -> list[dict]:
        """Trả về dữ liệu zones (cho API)."""
        return [z.to_dict() for z in self._zones]

    def get_persist_file(self) -> Path:
        """Trả về đường dẫn file zones của camera này."""
        return self._persist

    def clear_zones(self):
        """Xóa toàn bộ zones và xóa file persist của camera này."""
        self._zones = []
        if self._persist.exists():
            self._persist.unlink()
            logger.info(f"[ZoneDetector] Zones cleared: {self._persist.name}")
        else:
            logger.info("[ZoneDetector] Zones cleared (no persist file existed).")

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _save_to_file(self):
        """Ghi danh sách zones xuống file JSON của camera này."""
        try:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            payload = {"zones": [z.to_dict() for z in self._zones]}
            self._persist.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(f"[ZoneDetector] Failed to save {self._persist}: {exc}")

    def _load_from_file(self, path: Path) -> bool:
        """
        Đọc zones từ file JSON.
        Trả về True nếu thành công, False nếu lỗi.
        """
        try:
            text      = path.read_text(encoding="utf-8")
            data      = json.loads(text)
            zones_cfg = data.get("zones", [])
            self._zones = [Zone(z) for z in zones_cfg]
            return True
        except Exception as exc:
            logger.warning(
                f"[ZoneDetector] Could not load {path.name}: {exc}. "
                "Falling back to default config."
            )
            self._zones = []
            return False
