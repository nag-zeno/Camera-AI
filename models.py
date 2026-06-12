"""
models.py — Data Models (Hợp đồng dữ liệu giữa các module)

Mọi module đều nhận và trả về các kiểu dữ liệu được định nghĩa ở đây.
Thay đổi ở đây ảnh hưởng toàn bộ hệ thống → cân nhắc kỹ.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time


# ============================================================
# Enums — Các tập giá trị cố định
# ============================================================

class ObjectCategory(str, Enum):
    """Loại đối tượng tổng quát."""
    PERSON    = "person"
    ANIMAL    = "animal"
    VEHICLE   = "vehicle"
    ACCESSORY = "accessory"
    UNKNOWN   = "unknown"


class SocialRole(str, Enum):
    """
    16 vai trò xã hội mà hệ thống có thể nhận diện.

    Nhóm A — Dịch vụ & Giao hàng:
        SHIPPER, POSTMAN, TECHNICIAN, WORKER

    Nhóm B — Y Tế:
        DOCTOR, NURSE

    Nhóm C — An Ninh & Pháp Luật:
        POLICE, MILITARY, SECURITY, CIVIL_GUARD

    Nhóm D — Chuyên Môn Khác:
        STUDENT, CONSTRUCTION, CHEF, JANITOR

    Nhóm E — Mặc Định:
        NORMAL, UNKNOWN
    """
    # Nhóm A
    SHIPPER      = "shipper"
    POSTMAN      = "postman"
    TECHNICIAN   = "technician"
    WORKER       = "worker"
    # Nhóm B
    DOCTOR       = "doctor"
    NURSE        = "nurse"
    # Nhóm C
    POLICE       = "police"
    MILITARY     = "military"
    SECURITY     = "security"
    CIVIL_GUARD  = "civil_guard"
    # Nhóm D
    STUDENT      = "student"
    CONSTRUCTION = "construction"
    CHEF         = "chef"
    JANITOR      = "janitor"
    # Nhóm E
    NORMAL       = "normal"
    UNKNOWN      = "unknown"


class IdentityStatus(str, Enum):
    """Trạng thái danh tính (không lưu tên thật)."""
    KNOWN   = "known"
    UNKNOWN = "unknown"
    PENDING = "pending"   # Chưa kiểm tra


class ZoneType(str, Enum):
    """Loại khu vực không gian."""
    ALLOWED    = "allowed"
    RESTRICTED = "restricted"


class ZoneStatus(str, Enum):
    """Trạng thái đối tượng trong khu vực."""
    OUTSIDE  = "outside"
    ENTERING = "entering"
    INSIDE   = "inside"
    LEAVING  = "leaving"


class AlertLevel(str, Enum):
    """Mức độ cảnh báo — theo thứ tự tăng dần."""
    IGNORE   = "ignore"
    NORMAL   = "normal"
    WATCH    = "watch"
    WARNING  = "warning"
    ALERT    = "alert"
    CRITICAL = "critical"


class ActionLabel(str, Enum):
    """
    8 hành động mà ActionNet có thể nhận diện.

    Nguy hiểm: FALLING, CLIMBING, FIGHTING
    Cảnh báo : RUNNING, RAISING_HAND, GATHERING
    Bình thường: STANDING, WALKING
    """
    STANDING     = "standing"      # Đứng yên — bình thường
    WALKING      = "walking"       # Đi bộ — bình thường
    RUNNING      = "running"       # Chạy nhanh — cảnh báo
    FALLING      = "falling"       # Ngã/Ngất — nguy hiểm 🔴
    CLIMBING     = "climbing"      # Leo trèo — nguy hiểm 🔴
    FIGHTING     = "fighting"      # Đánh nhau — nguy hiểm 🔴
    RAISING_HAND = "raising_hand"  # Giơ tay/Vẫy — chú ý
    GATHERING    = "gathering"     # Tụ tập — chú ý
    UNKNOWN      = "unknown"       # Chưa xác định


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class BoundingBox:
    """
    Bounding box theo format xyxy.
    Tọa độ pixel tuyệt đối trong frame đã normalize.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]

    def to_int(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class Detection:
    """
    Output của Object Detector — kết quả raw từ YOLO.

    Module tạo ra: object_detector.py
    Module tiêu thụ: object_tracker.py
    """
    bbox       : BoundingBox
    class_name : str
    category   : ObjectCategory
    confidence : float
    frame_id   : int = 0

    def to_dict(self) -> dict:
        return {
            "bbox"       : self.bbox.to_xyxy(),
            "class_name" : self.class_name,
            "category"   : self.category.value,
            "confidence" : round(self.confidence, 3),
            "frame_id"   : self.frame_id,
        }


@dataclass
class RoleEvidence:
    """
    Bằng chứng tại sao hệ thống chọn role này.
    Đảm bảo tính explainability.
    """
    color_match        : str   = ""    # Tên màu khớp
    color_ratio        : float = 0.0   # Tỷ lệ diện tích màu
    region             : str   = ""    # Vùng cơ thể (torso/head)
    accessories_found  : list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "color_match"       : self.color_match,
            "color_ratio"       : round(self.color_ratio, 3),
            "region"            : self.region,
            "accessories_found" : self.accessories_found,
        }


@dataclass
class TrackedObject:
    """
    Đối tượng đang được theo dõi — được làm giàu dần qua từng module.

    Luồng dữ liệu:
      object_tracker   → track_id, bbox, class_name, category
      role_classifier  → role, role_confidence, role_evidence
      identity_manager → identity
      zone_detector    → zone_name, zone_type, zone_status
      behavior_analyzer→ time_in_zone, loitering, direction, visit_count
      context_engine   → alert_level, alert_reason
    """
    # --- Từ Tracker ---
    track_id      : int
    bbox          : BoundingBox
    class_name    : str
    category      : ObjectCategory
    confidence    : float
    frames_tracked: int   = 0
    velocity      : tuple = (0.0, 0.0)   # (vx, vy) pixel/frame

    # --- Từ Role Classifier ---
    role           : SocialRole  = SocialRole.UNKNOWN
    role_confidence: float       = 0.0
    role_evidence  : Optional[RoleEvidence] = None

    # --- Từ Identity Manager ---
    identity       : IdentityStatus = IdentityStatus.PENDING

    # --- Từ Zone Detector ---
    zone_name  : Optional[str]      = None
    zone_type  : Optional[ZoneType] = None
    zone_status: ZoneStatus         = ZoneStatus.OUTSIDE

    # --- Từ Behavior Analyzer ---
    time_in_zone : float = 0.0
    loitering    : bool  = False
    direction    : str   = "stationary"
    visit_count  : int   = 0

    # --- Từ Action Recognizer ---
    action            : ActionLabel = ActionLabel.UNKNOWN
    action_confidence : float       = 0.0
    action_top3       : list        = field(default_factory=list)  # [(label, prob), ...]

    # --- Từ Context Engine ---
    alert_level  : AlertLevel = AlertLevel.NORMAL
    alert_reason : str        = ""
    rule_name    : str        = ""

    # --- Timestamps ---
    first_seen   : float = field(default_factory=time.time)
    last_seen    : float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Chuyển sang JSON-serializable dict để gửi qua API."""
        return {
            "track_id"       : self.track_id,
            "bbox"           : self.bbox.to_xyxy(),
            "class_name"     : self.class_name,
            "category"       : self.category.value,
            "confidence"     : round(self.confidence, 3),

            "role"             : self.role.value,
            "role_confidence"  : round(self.role_confidence, 3),
            "role_evidence"    : self.role_evidence.to_dict() if self.role_evidence else {},

            "identity"         : self.identity.value,

            "zone_name"        : self.zone_name,
            "zone_type"        : self.zone_type.value if self.zone_type else None,
            "zone_status"      : self.zone_status.value,
            "time_in_zone"     : round(self.time_in_zone, 1),

            "loitering"        : self.loitering,
            "direction"        : self.direction,
            "visit_count"      : self.visit_count,

            "action"           : self.action.value,
            "action_confidence": round(self.action_confidence, 3),
            "action_top3"      : self.action_top3,

            "alert_level"      : self.alert_level.value,
            "alert_reason"     : self.alert_reason,
            "rule_name"        : self.rule_name,

            "frames_tracked"   : self.frames_tracked,
            "first_seen"       : round(self.first_seen, 2),
            "last_seen"        : round(self.last_seen, 2),
        }


@dataclass
class AlertEvent:
    """
    Một sự kiện cảnh báo được ghi log.
    Output của Context Engine khi phát hiện tình huống đáng chú ý.

    Bao gồm đầy đủ 16 features cần thiết để retrain ContextNet:
      role_id, identity_id, zone_type_id, zone_status_id, loitering,
      time_in_zone, visit_count, direction_id, hour, role_confidence,
      category_id, frames_tracked, is_night, is_business_hour,
      action_id, action_confidence.
    """
    import uuid as _uuid

    event_id        : str
    timestamp       : float
    track_id        : int
    level           : AlertLevel
    rule_name       : str
    reason          : str
    object_category : ObjectCategory
    object_role     : SocialRole
    zone_name       : Optional[str]
    position        : tuple[float, float]

    # --- Action info ---
    action           : ActionLabel = ActionLabel.UNKNOWN
    action_confidence: float       = 0.0

    # --- ContextNet ML features (bổ sung để retrain chính xác hơn) ---
    role_confidence : float            = 0.0
    identity        : IdentityStatus   = IdentityStatus.UNKNOWN
    zone_type       : Optional[ZoneType]  = None
    zone_status     : ZoneStatus       = ZoneStatus.OUTSIDE
    time_in_zone    : float            = 0.0
    loitering       : bool             = False
    direction       : str              = "stationary"
    visit_count     : int              = 0
    frames_tracked  : int              = 0

    @classmethod
    def create(cls, obj: TrackedObject) -> "AlertEvent":
        import uuid
        return cls(
            event_id          = str(uuid.uuid4())[:8],
            timestamp         = time.time(),
            track_id          = obj.track_id,
            level             = obj.alert_level,
            rule_name         = obj.rule_name,
            reason            = obj.alert_reason,
            object_category   = obj.category,
            object_role       = obj.role,
            zone_name         = obj.zone_name,
            position          = obj.bbox.center,
            action            = obj.action,
            action_confidence = obj.action_confidence,
            # ContextNet ML features
            role_confidence   = obj.role_confidence,
            identity          = obj.identity,
            zone_type         = obj.zone_type,
            zone_status       = obj.zone_status,
            time_in_zone      = obj.time_in_zone,
            loitering         = obj.loitering,
            direction         = obj.direction,
            visit_count       = obj.visit_count,
            frames_tracked    = obj.frames_tracked,
        )

    def to_dict(self) -> dict:
        from datetime import datetime
        return {
            "event_id"         : self.event_id,
            "timestamp"        : self.timestamp,
            "datetime"         : datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "track_id"         : self.track_id,
            "level"            : self.level.value,
            "rule_name"        : self.rule_name,
            "reason"           : self.reason,
            "object_category"  : self.object_category.value,
            "object_role"      : self.object_role.value,
            "zone_name"        : self.zone_name,
            "position"         : list(self.position),
            "action"           : self.action.value,
            "action_confidence": round(self.action_confidence, 3),
            # ContextNet ML features
            "role_confidence"  : round(self.role_confidence, 3),
            "identity"         : self.identity.value,
            "zone_type"        : self.zone_type.value if self.zone_type else None,
            "zone_status"      : self.zone_status.value,
            "time_in_zone"     : round(self.time_in_zone, 1),
            "loitering"        : self.loitering,
            "direction"        : self.direction,
            "visit_count"      : self.visit_count,
            "frames_tracked"   : self.frames_tracked,
        }


@dataclass
class FrameResult:
    """
    Kết quả phân tích hoàn chỉnh cho 1 frame.
    Đây là output cuối cùng của Pipeline, được gửi ra API và Web UI.
    """
    frame_id   : int
    timestamp  : float
    objects    : list    # List[TrackedObject]
    new_alerts : list    # List[AlertEvent] — chỉ alerts MỚI trong frame này
    fps        : float = 0.0

    def to_dict(self) -> dict:
        persons  = [o for o in self.objects if o.category == ObjectCategory.PERSON]
        vehicles = [o for o in self.objects if o.category == ObjectCategory.VEHICLE]
        animals  = [o for o in self.objects if o.category == ObjectCategory.ANIMAL]

        return {
            "frame_id"  : self.frame_id,
            "timestamp" : round(self.timestamp, 3),
            "fps"       : round(self.fps, 1),
            "objects"   : [o.to_dict() for o in self.objects],
            "new_alerts": [a.to_dict() for a in self.new_alerts],
            "stats"     : {
                "total_objects"  : len(self.objects),
                "total_persons"  : len(persons),
                "total_vehicles" : len(vehicles),
                "total_animals"  : len(animals),
                "active_warnings": sum(1 for o in self.objects
                                       if o.alert_level.value in
                                       ("warning", "alert", "critical")),
                "role_breakdown" : _count_roles(persons),
            },
        }


def _count_roles(persons: list) -> dict:
    """Đếm số người theo từng vai trò."""
    counts: dict[str, int] = {}
    for p in persons:
        role = p.role.value
        counts[role] = counts.get(role, 0) + 1
    return counts
