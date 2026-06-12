"""
context_engine.py — Tầng 4: Reasoning (Context Reasoning Engine)

Nhận vào: TrackedObject với đầy đủ thông tin từ các module trên
Trả ra  : TrackedObject với alert_level, alert_reason, rule_name

Đây là "bộ não" của hệ thống: suy luận ngữ cảnh để quyết định
mức độ cảnh báo DỰA TRÊN vai trò + zone + thời gian + hành vi.

Thiết kế Rule Engine:
  - Mỗi rule có priority (cao hơn → xét trước)
  - Áp dụng rule đầu tiên thỏa điều kiện
  - Hoàn toàn extensible: thêm rule = gọi add_rule()
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime

from config import REASONING_CONFIG
from models import (
    TrackedObject, AlertLevel, AlertEvent,
    SocialRole, ZoneType, ZoneStatus, IdentityStatus, ObjectCategory, ActionLabel,
)

logger = logging.getLogger(__name__)


# ============================================================
# Rule Definition
# ============================================================

@dataclass
class ContextRule:
    """Một luật suy luận."""
    name         : str
    priority     : int                       # Cao hơn → ưu tiên hơn
    condition_fn : Callable[[TrackedObject, dict], bool]  # fn(obj, time_ctx) → bool
    alert_level  : AlertLevel
    reason_tpl   : str                       # Template: dùng {track_id}, {zone}, {role}


def _fmt(tpl: str, obj: TrackedObject) -> str:
    return tpl.format(
        track_id = obj.track_id,
        zone     = obj.zone_name or "unknown",
        role     = obj.role.value,
        direction= obj.direction,
        time_in  = round(obj.time_in_zone, 0),
    )


# ============================================================
# Built-in Rules (với priority từ cao xuống thấp)
# ============================================================

def _build_default_rules() -> list[ContextRule]:
    rules = []

    # ============================================================
    # --- ACTION-BASED RULES (ưu tiên cao nhất) ---
    # Các rules này dựa trên ActionNet output → phải được xét trước
    # tất cả các rule vai trò/zone để đảm bảo hành vi nguy hiểm
    # luôn nhận được mức cảnh báo đúng, bất kể vai trò là gì.
    # ============================================================

    rules.append(ContextRule(
        name         = "action_fighting",
        priority     = 200,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.FIGHTING
            and o.action_confidence >= 0.50
        ),
        alert_level  = AlertLevel.CRITICAL,
        reason_tpl   = "💥 Phát hiện hành vi ĐÁNH NHAU! {role} #{track_id} đang xô xát tại khu vực '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "action_falling",
        priority     = 195,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.FALLING
            and o.action_confidence >= 0.65
            and o.frames_tracked >= 10
        ),
        alert_level  = AlertLevel.ALERT,
        reason_tpl   = "🚨 Phát hiện sự cố TÉ NGÃ/NGẤT XỈU! {role} #{track_id} có thể cần trợ giúp khẩn cấp tại '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "action_climbing",
        priority     = 190,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.CLIMBING
            and o.action_confidence >= 0.50
        ),
        alert_level  = AlertLevel.ALERT,
        reason_tpl   = "🧗 Phát hiện hành vi LEO TRÈO! {role} #{track_id} đang leo trèo tại '{zone}' — nguy cơ xâm nhập trái phép",
    ))

    rules.append(ContextRule(
        name         = "action_running_restricted",
        priority     = 185,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.RUNNING
            and o.zone_type == ZoneType.RESTRICTED
            and o.action_confidence >= 0.55
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "🏃 Cảnh báo: {role} #{track_id} đang chạy nhanh trong khu vực cấm '{zone}'!",
    ))

    rules.append(ContextRule(
        name         = "action_gathering",
        priority     = 180,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.GATHERING
            and o.action_confidence >= 0.55
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "👥 Phát hiện nhóm người TỤ TẬP — {role} #{track_id} cùng nhóm tại '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "action_running",
        priority     = 175,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.action == ActionLabel.RUNNING
            and o.action_confidence >= 0.60
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "🏃 Phát hiện đối tượng đang CHẠY — {role} #{track_id} tại '{zone}'",
    ))

    # --- Mức CRITICAL ---
    rules.append(ContextRule(
        name         = "unknown_loitering_restricted",
        priority     = 100,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role in (SocialRole.UNKNOWN, SocialRole.NORMAL)
            and o.zone_type == ZoneType.RESTRICTED
            and o.loitering
        ),
        alert_level  = AlertLevel.CRITICAL,
        reason_tpl   = "🚨 Đối tượng khả nghi #{track_id} ({role}) đang lảng vảng tại khu vực cấm '{zone}' (đã ở đây {time_in} giây)",
    ))

    # --- Mức ALERT ---
    rules.append(ContextRule(
        name         = "unknown_in_restricted",
        priority     = 90,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role in (SocialRole.UNKNOWN, SocialRole.NORMAL)
            and o.zone_type == ZoneType.RESTRICTED
            and o.identity == IdentityStatus.UNKNOWN
            and o.frames_tracked >= 30
        ),
        alert_level  = AlertLevel.ALERT,
        reason_tpl   = "🔴 Xâm nhập: Người lạ #{track_id} đi vào khu vực cấm '{zone}'",
    ))

    # Unknown vừa vào restricted zone (< 30 frames) → chỉ WATCH trước
    rules.append(ContextRule(
        name         = "unknown_entering_restricted",
        priority     = 89,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role in (SocialRole.UNKNOWN, SocialRole.NORMAL)
            and o.zone_type == ZoneType.RESTRICTED
            and o.identity == IdentityStatus.UNKNOWN
            and o.frames_tracked < 30
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "⚠️ Phát hiện người lạ #{track_id} bắt đầu đi vào khu vực cấm '{zone}' — đang giám sát",
    ))

    rules.append(ContextRule(
        name         = "chef_in_restricted",
        priority     = 85,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.CHEF
            and o.zone_type == ZoneType.RESTRICTED
        ),
        alert_level  = AlertLevel.ALERT,
        reason_tpl   = "🍳 Phát hiện bất thường: Đầu bếp #{track_id} đi vào khu vực cấm '{zone}'!",
    ))

    rules.append(ContextRule(
        name         = "student_night_restricted",
        priority     = 84,
        condition_fn = lambda o, tc: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.STUDENT
            and tc.get("period") == "night"
            and o.zone_type == ZoneType.RESTRICTED
        ),
        alert_level  = AlertLevel.ALERT,
        reason_tpl   = "🎒 Cảnh báo: Học sinh #{track_id} đi vào khu vực cấm '{zone}' vào ban đêm!",
    ))

    # --- Mức WARNING ---
    rules.append(ContextRule(
        name         = "shipper_in_restricted",
        priority     = 70,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.SHIPPER
            and o.zone_type == ZoneType.RESTRICTED
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "🛵 Cảnh báo: Nhân viên giao hàng (Shipper) #{track_id} đi nhầm vào khu vực cấm '{zone}'!",
    ))

    rules.append(ContextRule(
        name         = "construction_in_office",
        priority     = 68,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.CONSTRUCTION
            and o.zone_type == ZoneType.RESTRICTED
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "🏗️ Phát hiện Công nhân xây dựng #{track_id} xuất hiện tại khu vực văn phòng '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "janitor_in_restricted",
        priority     = 65,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.JANITOR
            and o.zone_type == ZoneType.RESTRICTED
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "🧹 Phát hiện Nhân viên vệ sinh #{track_id} xuất hiện tại khu vực hạn chế '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "unknown_loitering",
        priority     = 60,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role in (SocialRole.UNKNOWN, SocialRole.NORMAL)
            and o.loitering
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "⏱️ Đối tượng khả nghi #{track_id} ({role}) đang lảng vảng (đã ở đây {time_in} giây)",
    ))

    rules.append(ContextRule(
        name         = "shipper_night",
        priority     = 55,
        condition_fn = lambda o, tc: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.SHIPPER
            and tc.get("period") == "night"
        ),
        alert_level  = AlertLevel.WARNING,
        reason_tpl   = "🌙 Phát hiện Shipper #{track_id} xuất hiện vào ban đêm — cần xác minh đơn hàng",
    ))

    # --- Mức WATCH ---
    rules.append(ContextRule(
        name         = "security_patrol",
        priority     = 40,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.SECURITY
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "🛡️ Nhân viên Bảo vệ #{track_id} đang tuần tra khu vực '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "police_on_site",
        priority     = 38,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.POLICE
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "👮 Lực lượng Cảnh sát #{track_id} có mặt tại khu vực '{zone}' — đang giám sát",
    ))

    rules.append(ContextRule(
        name         = "military_on_site",
        priority     = 36,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.MILITARY
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "🪖 Lực lượng Quân đội #{track_id} có mặt tại khu vực '{zone}' — đang giám sát",
    ))

    rules.append(ContextRule(
        name         = "frequent_visitor",
        priority     = 30,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.visit_count >= 3
        ),
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "🔁 Đối tượng #{track_id} ({role}) đã ghé thăm khu vực này {time_in} lần gần đây",
    ))

    # --- Mức NORMAL ---
    rules.append(ContextRule(
        name         = "medical_staff",
        priority     = 20,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role in (SocialRole.DOCTOR, SocialRole.NURSE)
        ),
        alert_level  = AlertLevel.NORMAL,
        reason_tpl   = "🩺 Nhân viên Y tế #{track_id} làm nhiệm vụ tại '{zone}' — hợp lệ",
    ))

    rules.append(ContextRule(
        name         = "known_person",
        priority     = 15,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.identity == IdentityStatus.KNOWN
        ),
        alert_level  = AlertLevel.NORMAL,
        reason_tpl   = "✅ Phát hiện người quen #{track_id} tại '{zone}' — hợp lệ",
    ))

    rules.append(ContextRule(
        name         = "shipper_at_entrance",
        priority     = 12,
        condition_fn = lambda o, _: (
            o.category == ObjectCategory.PERSON
            and o.role == SocialRole.SHIPPER
            and o.zone_type == ZoneType.ALLOWED
        ),
        alert_level  = AlertLevel.NORMAL,
        reason_tpl   = "🛵 Có nhân viên giao hàng (Shipper) #{track_id} vừa đến cổng '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "default_person",
        priority     = 1,
        condition_fn = lambda o, _: o.category == ObjectCategory.PERSON,
        alert_level  = AlertLevel.NORMAL,
        reason_tpl   = "🧑 Phát hiện người #{track_id} tại '{zone}'",
    ))

    # Non-person objects
    rules.append(ContextRule(
        name         = "vehicle_detected",
        priority     = 5,
        condition_fn = lambda o, _: o.category.value == "vehicle",
        alert_level  = AlertLevel.WATCH,
        reason_tpl   = "🚗 Phát hiện phương tiện #{track_id} tại '{zone}'",
    ))

    rules.append(ContextRule(
        name         = "default_ignore",
        priority     = 0,
        condition_fn = lambda o, _: True,
        alert_level  = AlertLevel.IGNORE,
        reason_tpl   = "Bỏ qua đối tượng #{track_id} ({role})",
    ))

    # Sắp xếp theo priority giảm dần
    rules.sort(key=lambda r: r.priority, reverse=True)
    return rules


def generate_vietnamese_reason(obj: TrackedObject) -> str:
    """Tạo chuỗi mô tả ngữ cảnh bằng tiếng Việt tự nhiên, không có ID kỹ thuật."""
    from datetime import datetime

    role_map = {
        SocialRole.SHIPPER     : "Shipper (Giao hàng)",
        SocialRole.POSTMAN     : "Nhân viên bưu điện",
        SocialRole.TECHNICIAN  : "Kỹ thuật viên",
        SocialRole.WORKER      : "Nhân viên / Công nhân",
        SocialRole.DOCTOR      : "Bác sĩ",
        SocialRole.NURSE       : "Y tá",
        SocialRole.POLICE      : "Cảnh sát",
        SocialRole.MILITARY    : "Quân nhân",
        SocialRole.SECURITY    : "Nhân viên bảo vệ",
        SocialRole.CIVIL_GUARD : "Dân phòng",
        SocialRole.STUDENT     : "Học sinh / Sinh viên",
        SocialRole.CONSTRUCTION: "Công nhân xây dựng",
        SocialRole.CHEF        : "Đầu bếp",
        SocialRole.JANITOR     : "Nhân viên vệ sinh",
        SocialRole.NORMAL      : "Người dân",
        SocialRole.UNKNOWN     : "Người lạ",
    }

    role_str   = role_map.get(obj.role, "Người")
    zone_str   = f"{obj.zone_name}" if obj.zone_name else "khu vực giám sát"

    # Làm sạch tên zone kỹ thuật
    import re
    if re.match(r'^zone_\d+$', zone_str, re.IGNORECASE) or re.match(r'^(rtsp_|webcam_)', zone_str, re.IGNORECASE):
        zone_str = "khu vực giám sát"

    # Thời điểm trong ngày
    hour = datetime.now().hour
    if 22 <= hour or hour < 5:
        time_ctx = "đêm khuya"
    elif 18 <= hour < 22:
        time_ctx = "buổi tối"
    elif 5 <= hour < 9:
        time_ctx = "sáng sớm"
    else:
        time_ctx = ""
    time_suffix = f" ({time_ctx})".strip() if time_ctx else ""

    time_str = f"đã có mặt {int(obj.time_in_zone)} giây" if obj.time_in_zone > 5 else ""
    time_part = f" — {time_str}" if time_str else ""

    # 1. Ẩu đả / đánh nhau
    if obj.action == ActionLabel.FIGHTING and obj.action_confidence >= 0.5:
        return f"🚨 Khẩn! Camera phát hiện có người đang đánh nhau tại {zone_str}{time_suffix}. Xử lý ngay bạn ơi!"

    # 2. Té ngã / ngất xỉu
    if obj.action == ActionLabel.FALLING and obj.action_confidence >= 0.5:
        return f"🚑 Camera phát hiện có người vừa bị ngã / ngất xỉu tại {zone_str}! Kiểm tra xem họ có ổn không nha!"

    # 3. Leo trèo nghi ngờ
    if obj.action == ActionLabel.CLIMBING and obj.action_confidence >= 0.5:
        return f"🧗 Có người đang leo trèo tại {zone_str}{time_suffix}. Trông có vẻ đột nhập, kiểm tra ngay bạn ơi!"

    # 4. Shipper đến
    if obj.role == SocialRole.SHIPPER:
        if obj.zone_type == ZoneType.RESTRICTED:
            return f"🛵 Cảnh báo: Shipper vừa đi vào {zone_str} là khu vực cấm!"
        return f"🛵 Ơi, có Shipper đến {zone_str} rồi nè! Ra lấy đồ nha~"

    # 5. Lảng vảng dài ở khu vực cấm
    if obj.loitering and obj.zone_type == ZoneType.RESTRICTED:
        t = f"{int(obj.time_in_zone)} giây" if obj.time_in_zone > 5 else ""
        t_part = f", đã nhập mô {t}" if t else ""
        return f"🚨 Bạn ơi, có {role_str.lower()} đang lảng vảng ở {zone_str}{t_part}{time_suffix}. Rất đáng ngờ, kiểm tra ngay!"

    # 6. Người lạ ở khu vực cấm
    if obj.zone_type == ZoneType.RESTRICTED and obj.role in (SocialRole.UNKNOWN, SocialRole.NORMAL):
        urgent = "đêm khuya" in time_suffix
        prefix = "🚨 Khẩn! " if urgent else "🔴 "
        return f"{prefix}Có người lạ xuất hiện tại {zone_str}{time_suffix}{time_part}. Kiểm tra ngay bạn ơi!"

    # 7. Chạy nhanh trong khu vực cấm
    if obj.action == ActionLabel.RUNNING and obj.zone_type == ZoneType.RESTRICTED:
        return f"🏃 Cảnh báo: Có người đang chạy rất nhanh trong {zone_str}{time_suffix}!"

    # 8. Bảo vệ tuần tra
    if obj.role == SocialRole.SECURITY:
        return f"✅ Bảo vệ đang tuần tra {zone_str}, mọi thứ ổn nha."

    # 9. Cảnh sát / Quân đội
    if obj.role in (SocialRole.POLICE, SocialRole.MILITARY):
        role_label = "cảnh sát" if obj.role == SocialRole.POLICE else "lực lượng quân sự"
        return f"👮 Có {role_label} có mặt tại {zone_str}, mọi thứ đang được kiểm soát."

    # 10. Lảng vảng thông thường
    if obj.loitering:
        return f"⚠️ Bạn ơi, có người đang đứng lảng vảng khá lâu ở {zone_str}{time_part}. Để ý một chút nhé!"

    # 11. Tụ tập đông người
    if obj.action == ActionLabel.GATHERING:
        return f"👥 Camera ghi nhận có nhóm người đang tụ tập tại {zone_str}."

    # 12. Mặc định bình thường
    action_map = {
        ActionLabel.STANDING     : "đứng",
        ActionLabel.WALKING      : "đi bộ",
        ActionLabel.RUNNING      : "chạy",
        ActionLabel.RAISING_HAND : "vẫy tay",
    }
    act = action_map.get(obj.action, "")
    act_part = f" đang {act}" if act else ""
    return f"📷 Camera ghi nhận có {role_str.lower()}{act_part} tại {zone_str}."


# ============================================================
# Context Reasoning Engine
# ============================================================

class ContextEngine:
    """
    Áp dụng rule engine để quyết định alert level.
    Phát ra AlertEvent khi cần thiết (có cooldown).
    """

    def __init__(self):
        self._rules       : list[ContextRule] = _build_default_rules()
        self._cfg         = REASONING_CONFIG
        self._cooldown    = self._cfg["alert_cooldown"]
        self._alert_times : dict[str, float] = {}   # rule_name:track_id → last_alert_time
        logger.info(f"ContextEngine initialized with {len(self._rules)} rules.")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def add_rule(self, rule: ContextRule):
        """Thêm rule tùy chỉnh vào engine. Re-sort theo priority."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added rule '{rule.name}' (priority={rule.priority})")

    def evaluate(
        self,
        obj: TrackedObject,
    ) -> tuple[TrackedObject, Optional[AlertEvent]]:
        """
        Đánh giá 1 TrackedObject và quyết định alert level.

        Args:
            obj: TrackedObject đầy đủ thông tin

        Returns:
            (obj đã cập nhật alert_level/reason, AlertEvent hoặc None)
        """
        time_ctx = self._get_time_context()

        # Tìm rule đầu tiên match
        for rule in self._rules:
            if rule.condition_fn(obj, time_ctx):
                obj.alert_level = rule.alert_level
                obj.rule_name   = rule.name

                # ── Sinh câu thông báo: Gemini NLG → Fallback Template ──
                try:
                    from modules.nlg_engine import get_nlg_engine
                    nlg = get_nlg_engine()
                    if nlg.is_available:
                        # Xác định time_period
                        from datetime import datetime
                        _h = datetime.now().hour
                        if 22 <= _h or _h < 5:
                            _tp = "đêm khuya"
                        elif 18 <= _h < 22:
                            _tp = "buổi tối"
                        elif 5 <= _h < 9:
                            _tp = "sáng sớm"
                        elif 12 <= _h < 14:
                            _tp = "buổi trưa"
                        elif 14 <= _h < 18:
                            _tp = "buổi chiều"
                        else:
                            _tp = "buổi sáng"

                        ctx_data = {
                            "role"        : obj.role.value if obj.role else "unknown",
                            "action"      : obj.action.value if obj.action else "unknown",
                            "zone"        : obj.zone_name,
                            "zone_type"   : obj.zone_type.value if obj.zone_type else "allowed",
                            "alert_level" : rule.alert_level.value,
                            "time_in_zone": obj.time_in_zone,
                            "loitering"   : obj.loitering,
                            "rule_name"   : rule.name,
                            "time_period" : _tp,
                        }
                        obj.alert_reason = nlg.generate(
                            ctx_data,
                            fallback_fn=lambda: generate_vietnamese_reason(obj)
                        )
                    else:
                        obj.alert_reason = generate_vietnamese_reason(obj)
                except Exception:
                    obj.alert_reason = generate_vietnamese_reason(obj)

                break

        # Phát AlertEvent (nếu đáng chú ý và chưa hết cooldown)
        alert_event = None
        if obj.alert_level.value in ("warning", "alert", "critical"):
            alert_event = self._maybe_emit_event(obj)

        return obj, alert_event

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    @staticmethod
    def _get_time_context() -> dict:
        """Trả về ngữ cảnh thời gian hiện tại."""
        hour = datetime.now().hour
        if 6 <= hour < 9:
            period = "morning"
        elif 9 <= hour < 18:
            period = "business"
        elif 18 <= hour < 22:
            period = "evening"
        else:
            period = "night"
        return {"hour": hour, "period": period}

    def _maybe_emit_event(self, obj: TrackedObject) -> Optional[AlertEvent]:
        """Emits AlertEvent nếu chưa gửi gần đây (cooldown)."""
        key = f"{obj.rule_name}:{obj.track_id}"
        now = time.time()

        last = self._alert_times.get(key, 0.0)
        if now - last < self._cooldown:
            return None

        self._alert_times[key] = now
        event = AlertEvent.create(obj)
        logger.warning(
            f"[{event.level.value.upper()}] {obj.alert_reason} "
            f"(event_id={event.event_id})"
        )
        return event
