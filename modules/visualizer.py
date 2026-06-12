"""
visualizer.py — Tầng 5: Output (Visualization)

Nhận vào: frame + List[TrackedObject] + List[Zone]
Trả ra  : frame đã annotate với bounding boxes, labels, zones, tracks
"""
import cv2
import numpy as np
import time
import logging
from typing import Optional

from config import VIS_CONFIG, ACTION_CONFIG
from models import TrackedObject, AlertLevel, ActionLabel

logger = logging.getLogger(__name__)

# Màu bounding box theo alert level (BGR)
_ALERT_COLORS = {
    AlertLevel.IGNORE  : (100, 100, 100),
    AlertLevel.NORMAL  : (0,   200, 0),
    AlertLevel.WATCH   : (220, 200, 0),
    AlertLevel.WARNING : (0,   165, 255),
    AlertLevel.ALERT   : (0,   50,  255),
    AlertLevel.CRITICAL: (0,   0,   200),
}

# Action tags (ASCII fallback — OpenCV không hiển thị emoji tốt trên mọi hệ)
_ACTION_TAGS = {
    "standing"    : "[STAND]",
    "walking"     : "[WALK]",
    "running"     : "[RUN]",
    "falling"     : "[FALL!]",
    "climbing"    : "[CLIMB!]",
    "fighting"    : "[FIGHT!]",
    "raising_hand": "[WAVE]",
    "gathering"   : "[GROUP]",
    "unknown"     : "",
}

# Role icons (ASCII fallback cho OpenCV)
_ROLE_ICONS = {
    "shipper"     : "[SHIP]",
    "postman"     : "[POST]",
    "technician"  : "[TECH]",
    "worker"      : "[WORK]",
    "doctor"      : "[DOC]",
    "nurse"       : "[NRS]",
    "police"      : "[POL]",
    "military"    : "[MIL]",
    "security"    : "[SEC]",
    "civil_guard" : "[CVG]",
    "student"     : "[STU]",
    "construction": "[CON]",
    "chef"        : "[CHF]",
    "janitor"     : "[JAN]",
    "normal"      : "[PPL]",
    "unknown"     : "[?]",
}

# Actions cần highlight đặc biệt (màu đỏ cam)
_DANGER_ACTIONS = {"falling", "climbing", "fighting"}
_WARN_ACTIONS   = {"running", "raising_hand", "gathering"}



class Visualizer:
    """
    Vẽ annotations lên frame:
      - Zone polygons (bán trong suốt)
      - Bounding boxes (màu theo alert level)
      - Labels (role, confidence, identity, alert)
      - Track trails (đường di chuyển)
      - Overlay stats (fps, count)
    """

    def __init__(self):
        self._cfg         = VIS_CONFIG
        self._thickness   = self._cfg["bbox_thickness"]
        self._font_scale  = self._cfg["font_scale"]
        self._show_zones  = self._cfg["show_zones"]
        self._show_tracks = self._cfg["show_tracks"]
        self._show_labels = self._cfg["show_labels"]
        self._zone_alpha  = self._cfg["zone_alpha"]

        # Trail lưu vị trí gần đây của track (track_id → list of (cx, cy))
        self._trails      : dict[int, list[tuple[int, int]]] = {}
        # Màu trail per-track (track_id → BGR color)
        self._trail_colors: dict[int, tuple] = {}
        self._fps_history : list[float] = []
        self._last_time   = time.monotonic()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        zones: list[dict] | None = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Vẽ toàn bộ annotations lên frame.

        Args:
            frame  : BGR frame gốc (sẽ được copy, không mutate)
            objects: List[TrackedObject]
            zones  : List[zone dict] từ ZoneDetector.get_zones()
            fps    : FPS hiện tại để hiển thị

        Returns:
            Annotated frame (BGR numpy)
        """
        out = frame.copy()

        # 0. Xóa trail của track đã BIẾN MẤT khỏi frame này
        current_ids = {obj.track_id for obj in objects}
        lost_ids = set(self._trails.keys()) - current_ids
        for lost_id in lost_ids:
            self._trails.pop(lost_id, None)
            self._trail_colors.pop(lost_id, None)

        # 1. Vẽ zones
        if self._show_zones and zones:
            out = self._draw_zones(out, zones)

        # 2. Cập nhật trails (chỉ cho objects hiện tại)
        for obj in objects:
            self._update_trail(obj)

        # 3. Vẽ trails
        if self._show_tracks:
            self._draw_trails(out)

        # 4. Vẽ từng object
        for obj in objects:
            self._draw_object(out, obj)

        # 5. Stats overlay
        self._draw_stats(out, objects, fps)

        return out

    # ----------------------------------------------------------
    # Internal — Drawing
    # ----------------------------------------------------------

    def _draw_zones(self, frame: np.ndarray, zones: list[dict]) -> np.ndarray:
        """Vẽ zone polygons bán trong suốt với label outline rõ ràng."""
        overlay = frame.copy()

        # Zone type icons (ASCII-safe cho OpenCV)
        _ZONE_ICONS = {
            "restricted": "[CAM]",
            "allowed"   : "[OK]",
            "monitored" : "[MON]",
        }

        for zone in zones:
            poly  = np.array(zone["polygon"], dtype=np.int32)
            color = tuple(int(c) for c in zone.get("color", [100, 100, 100]))

            # Fill polygon (bán trong suốt)
            cv2.fillPoly(overlay, [poly], color)

            # Border dày hơn (3px) + double-border để nổi bật
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 0),   thickness=4)
            cv2.polylines(frame, [poly], isClosed=True, color=color,        thickness=2)

            # Label ở tâm polygon
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            zone_type = zone.get("type", "")
            icon  = _ZONE_ICONS.get(zone_type, "")
            label = f"{icon} {zone['name']}"

            # Outline text (đen + màu) để đọc được trên mọi nền
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(frame, label, (tx, ty), font, 0.5, (0, 0, 0),   2, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Sub-label: loại zone (nhỏ hơn, phía dưới)
            sub = zone_type.upper()
            (sw, sh), _ = cv2.getTextSize(sub, font, 0.38, 1)
            sx = cx - sw // 2
            sy = ty + sh + 4
            cv2.putText(frame, sub, (sx, sy), font, 0.38, (0, 0, 0),  2, cv2.LINE_AA)
            cv2.putText(frame, sub, (sx, sy), font, 0.38, color,       1, cv2.LINE_AA)

        # Blend overlay với alpha
        cv2.addWeighted(overlay, self._zone_alpha, frame, 1 - self._zone_alpha, 0, frame)
        return frame

    def _draw_object(self, frame: np.ndarray, obj: TrackedObject):
        """Vẽ bounding box + label cho 1 object."""
        color = _ALERT_COLORS.get(obj.alert_level, (180, 180, 180))
        x1, y1, x2, y2 = obj.bbox.to_int()

        # --- CRITICAL: nhấp nháy ---
        if obj.alert_level == AlertLevel.CRITICAL:
            if int(time.monotonic() * 4) % 2 == 0:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._thickness)

        if not self._show_labels:
            return

        # --- Label panel ---
        role_icon = _ROLE_ICONS.get(obj.role.value, "[?]")
        id_marker = "✓" if obj.identity.value == "known" else "?"
        conf_pct  = int(obj.role_confidence * 100)

        lines = [
            f"#{obj.track_id} {role_icon} {conf_pct}%",
            f"ID:{id_marker} {obj.alert_level.value.upper()}",
        ]

        # Optional zone info — hiển thị zone_name và zone_status rõ ràng
        if obj.zone_name:
            zone_status = getattr(obj, 'zone_status', None)
            status_str = ""
            if zone_status is not None:
                sv = zone_status.value if hasattr(zone_status, 'value') else str(zone_status)
                status_map = {
                    "entering": ">>ENTER",
                    "inside"  : "INSIDE",
                    "leaving" : "LEAVE<<",
                    "outside" : "",
                }
                status_str = status_map.get(sv, sv.upper())
            zone_display = f"Z:{obj.zone_name[:8]}"
            if status_str:
                zone_display += f" {status_str}"
            lines.append(zone_display)
        if obj.loitering:
            lines.append(f"LOITER:{int(obj.time_in_zone)}s")

        # Action line (hiển thị nếu không phải UNKNOWN/STANDING)
        action_str = obj.action.value if hasattr(obj, "action") else "unknown"
        action_tag = _ACTION_TAGS.get(action_str, "")
        if action_tag and action_str not in ("unknown", "standing"):
            conf_a = int(getattr(obj, "action_confidence", 0) * 100)
            lines.append(f"{action_tag} {conf_a}%")

        # Background panel
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs     = self._font_scale
        pad    = 3
        line_h = int(fs * 22 + pad)
        panel_h= line_h * len(lines) + pad * 2
        panel_w= 150

        px1 = x1
        py1 = max(0, y1 - panel_h)
        px2 = min(frame.shape[1], x1 + panel_w)
        py2 = y1

        # Semi-transparent background
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
        cv2.addWeighted(panel_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

        for i, line in enumerate(lines):
            ty = py1 + pad + (i + 1) * line_h - 4
            # Màu action nguy hiểm khác màu
            if action_tag and action_str in _DANGER_ACTIONS and ("FALL" in line or "CLIMB" in line or "FIGHT" in line):
                line_color = (0, 50, 255)    # Đỏ 
            elif action_tag and action_str in _WARN_ACTIONS and ("RUN" in line or "WAVE" in line or "GROUP" in line):
                line_color = (0, 165, 255)   # Cam
            else:
                line_color = color
            cv2.putText(frame, line, (px1 + pad, ty),
                        font, fs, line_color, 1, cv2.LINE_AA)

    def _update_trail(self, obj: TrackedObject):
        """Cập nhật trail cho track. Lưu màu theo alert level."""
        tid = obj.track_id
        cx, cy = int(obj.bbox.center[0]), int(obj.bbox.center[1])
        if tid not in self._trails:
            self._trails[tid] = []
        self._trails[tid].append((cx, cy))
        # Giữ tối đa 20 điểm (ngắn hơn → gọn hơn)
        if len(self._trails[tid]) > 20:
            self._trails[tid].pop(0)
        # Cập nhật màu theo alert level
        self._trail_colors[tid] = _ALERT_COLORS.get(
            obj.alert_level, (120, 120, 120)
        )

    def _draw_trails(self, frame: np.ndarray):
        """
        Vẽ trail đường di chuyển.
        Chỉ vẽ track có trong self._trails (đã được dọn sạch track mất).
        """
        for tid, pts in list(self._trails.items()):
            if len(pts) < 2:
                continue
            base_color = self._trail_colors.get(tid, (120, 120, 120))
            for i in range(1, len(pts)):
                # Fade: đoạn cũ mờ hơn
                alpha = i / len(pts)
                color = tuple(int(c * alpha * 0.8) for c in base_color)
                cv2.line(frame, pts[i-1], pts[i], color, 1)

    def _draw_stats(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        fps: float,
    ):
        """Vẽ stats overlay góc trên-phải."""
        from models import ObjectCategory, AlertLevel as AL
        persons  = sum(1 for o in objects if o.category == ObjectCategory.PERSON)
        vehicles = sum(1 for o in objects if o.category.value == "vehicle")
        warnings = sum(1 for o in objects
                       if o.alert_level.value in ("warning", "alert", "critical"))

        h, w = frame.shape[:2]
        stats = [
            f"FPS: {fps:.1f}",
            f"People: {persons}",
            f"Vehicles: {vehicles}",
            f"Alerts: {warnings}",
        ]

        # Đếm action nguy hiểm
        danger_actions = sum(
            1 for o in objects
            if hasattr(o, "action") and o.action.value in ("falling", "climbing", "fighting")
        )
        if danger_actions > 0:
            stats.append(f"Danger Act: {danger_actions}")

        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.48
        pad   = 6
        line_h= 20
        panel_w = 130
        panel_h = line_h * len(stats) + pad * 2

        ox = w - panel_w - 8
        oy = 8

        bg = frame.copy()
        cv2.rectangle(bg, (ox, oy), (ox + panel_w, oy + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(bg, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (ox, oy), (ox + panel_w, oy + panel_h), (80, 80, 80), 1)

        alert_color = (0, 50, 255) if warnings > 0 else (0, 200, 0)

        for i, line in enumerate(stats):
            color = alert_color if "Alerts" in line else (220, 220, 220)
            cv2.putText(frame, line,
                        (ox + pad, oy + pad + (i + 1) * line_h - 4),
                        font, fs, color, 1, cv2.LINE_AA)
