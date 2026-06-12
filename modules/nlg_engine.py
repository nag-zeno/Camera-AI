"""
modules/nlg_engine.py
─────────────────────────────────────────────────────────────
Natural Language Generation (NLG) Engine
Dùng Gemini API để sinh câu thông báo tự nhiên bằng tiếng Việt
từ dữ liệu thô của camera AI.

Thiết kế có Fallback hoàn chỉnh:
  Nếu Gemini không khả dụng (mất mạng, quota hết, key sai...)
  → hệ thống tự động dùng lại bộ template tiếng Việt nội bộ
  → đảm bảo hệ thống không bao giờ bị gián đoạn.

Thread-safe: Singleton toàn cục, dùng ThreadPoolExecutor để
gọi API không đồng bộ (không block camera pipeline).
─────────────────────────────────────────────────────────────
"""

import os
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Lazy-load: chỉ import google.generativeai nếu đã cài đặt.
# Nếu chưa cài, hệ thống vẫn chạy bình thường với template.
# ──────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning(
        "[NLG] google-generativeai chưa được cài đặt. "
        "Chạy: pip install google-generativeai\n"
        "Hệ thống sẽ dùng template tiếng Việt thay thế."
    )

# Chỉ thị hệ thống: dạy Gemini đóng vai trò trợ lý camera an ninh Việt Nam
_SYSTEM_INSTRUCTION = """
Bạn là trợ lý AI của hệ thống Camera An Ninh tại Việt Nam.
Nhiệm vụ: chuyển đổi dữ liệu thô từ camera thành 1-2 câu thông báo NGẮN GỌN,
THÂN THIỆN, tự nhiên bằng tiếng Việt — như người thân nhắn tin Zalo/Messenger
để cảnh báo chủ nhà.

Quy tắc bắt buộc:
- Tối đa 2 câu, KHÔNG dài dòng
- Văn phong đời thường, thân mật ("bạn ơi", "nè", "coi chừng nha", "ơi")
- Có emoji phù hợp, không lạm dụng
- TUYỆT ĐỐI không đề cập track ID, số kỹ thuật, từ chuyên ngành
- Tên khu vực: mô tả vị trí thực tế, không dùng 'zone_1', 'zone_2'... 
- Tình huống nguy hiểm: ngắn gọn, cấp bách, chỉ rõ việc cần làm ngay
- Tình huống bình thường: nhẹ nhàng, thân thiện, thông tin
- Phù hợp ngữ cảnh Việt Nam (shipper, bảo vệ, công an, nhà kho, cửa hàng...)
- Chú ý thời điểm trong ngày: ban đêm nguy hiểm hơn ban ngày

Ví dụ mẫu (few-shot):
• [Người lạ vào khu cấm, tối] → "🚨 Coi chừng nha! Có người lạ vừa lọt vào khu vực sau nhà lúc đêm khuya. Bạn kiểm tra ngay giúp mình với!"
• [Shipper đến cổng, ban ngày] → "🛵 Ơi có Shipper đến cổng rồi nè! Ra lấy đồ nha~"
• [Đánh nhau trong sảnh] → "🚨 Khẩn! Camera thấy có người đang đánh nhau ở khu vực sảnh. Xử lý ngay bạn ơi!"
• [Bảo vệ tuần tra bình thường] → "✅ Bảo vệ đang tuần tra khu vực cổng, mọi thứ ổn nha."
• [Người lảng vảng nghi ngờ] → "⚠️ Bạn ơi, có người đang đứng lảng vảng trước cửa khá lâu rồi. Để ý chút nhé!"
• [Người ngã trong khu vực] → "🚑 Ơi, camera phát hiện có người vừa bị ngã ở khu vực kho. Kiểm tra xem họ có ổn không nha!"
""".strip()


class NLGEngine:
    """
    Engine sinh ngôn ngữ tự nhiên tiếng Việt dùng Gemini API.
    Thread-safe và có cơ chế Fallback hoàn chỉnh.
    """

    def __init__(self):
        self._lock            = threading.Lock()
        self._api_key         : Optional[str]    = None
        self._model           : Optional[object] = None
        self._initialized     : bool  = False
        self._enabled         : bool  = True
        self._last_error_time : float = 0.0
        self._error_count     : int   = 0
        self._call_count      : int   = 0
        self._success_count   : int   = 0
        self._last_error_msg  : str   = ""

        # ThreadPoolExecutor riêng cho Gemini API call (tránh block pipeline)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nlg-gemini")

        # Cấu hình — đọc từ config sau khi khởi tạo
        self._model_name     : str   = "gemini-1.5-flash"
        self._timeout_sec    : float = 5.0
        self._cooldown_sec   : float = 60.0
        self._temperature    : float = 0.7
        self._min_alert_level: str   = "warning"

        self._setup()

    def _setup(self):
        """Khởi tạo từ NLG_CONFIG trong config.py."""
        try:
            from config import NLG_CONFIG
            self._enabled          = NLG_CONFIG.get("enabled", True)
            self._model_name       = NLG_CONFIG.get("model", "gemini-2.5-flash")
            self._timeout_sec      = float(NLG_CONFIG.get("timeout_seconds", 5))
            self._cooldown_sec     = float(NLG_CONFIG.get("cooldown_on_error", 60))
            self._temperature      = float(NLG_CONFIG.get("temperature", 0.7))
            self._min_alert_level  = NLG_CONFIG.get("min_alert_level", "warning")
            api_key                = NLG_CONFIG.get("api_key", "").strip()
        except Exception as e:
            logger.warning(f"[NLG] Không đọc được NLG_CONFIG: {e}. Dùng giá trị mặc định.")
            api_key = ""

        if not self._enabled:
            logger.info("[NLG] NLG Engine bị tắt trong config (enabled=False).")
            return

        self._init_gemini(api_key)

    def _init_gemini(self, api_key: str):
        """Khởi tạo Gemini client với API key đã cho."""
        if not _GENAI_AVAILABLE:
            return

        # Fallback: đọc trực tiếp từ môi trường nếu config không có key
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if not api_key or api_key == "PASTE_YOUR_NEW_KEY_HERE":
            logger.warning(
                "[NLG] GEMINI_API_KEY chưa được cấu hình.\n"
                "  → Mở file .env và điền: GEMINI_API_KEY=your_key_here\n"
                "  → Hoặc cấu hình qua Dashboard → AI Assistant → Gemini API Key\n"
                "  → Hệ thống vẫn chạy bình thường với template tiếng Việt."
            )
            return

        try:
            with self._lock:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(
                    model_name         = self._model_name,
                    system_instruction = _SYSTEM_INSTRUCTION,
                    generation_config  = genai.types.GenerationConfig(
                        temperature       = self._temperature,
                        max_output_tokens = 150,  # Giới hạn output ngắn gọn
                    ),
                )
                self._api_key     = api_key
                self._initialized = True
                self._error_count = 0
            logger.info(
                f"[NLG] ✅ Gemini NLG Engine sẵn sàng "
                f"(model={self._model_name}, timeout={self._timeout_sec}s)"
            )
        except Exception as e:
            logger.error(f"[NLG] ❌ Không thể khởi tạo Gemini: {e}")
            self._last_error_msg = str(e)

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def set_api_key(self, api_key: str) -> bool:
        """
        Cập nhật API key tại runtime (không cần restart server).
        Trả về True nếu khởi tạo thành công.
        """
        api_key = api_key.strip()
        if not api_key:
            return False
        old_state = self._initialized
        with self._lock:
            self._initialized = False
            self._model       = None
        self._init_gemini(api_key)
        return self._initialized

    def set_enabled(self, enabled: bool):
        """Bật/tắt NLG Engine tại runtime."""
        with self._lock:
            self._enabled = enabled
        logger.info(f"[NLG] Engine {'BẬT' if enabled else 'TẮT'} theo yêu cầu.")

    @property
    def is_available(self) -> bool:
        """Kiểm tra NLG Engine có sẵn sàng hoạt động không."""
        if not self._enabled:
            return False
        if not self._initialized:
            return False
        # Cooldown sau lỗi: tạm dừng gọi API
        if self._error_count >= 3:
            if time.time() - self._last_error_time < self._cooldown_sec:
                return False
            else:
                with self._lock:
                    self._error_count = 0  # Reset sau cooldown
        return True

    def generate(self, context_data: dict, fallback_fn: Optional[Callable] = None) -> str:
        """
        Sinh câu thông báo tự nhiên từ dữ liệu ngữ cảnh camera.

        Args:
            context_data: Dict chứa thông tin từ camera
                          (role, action, zone, alert_level, time_in_zone, loitering...)
            fallback_fn:  Hàm fallback, gọi khi Gemini không khả dụng.

        Returns:
            Câu thông báo tiếng Việt tự nhiên (từ Gemini hoặc fallback template).
        """
        with self._lock:
            self._call_count += 1

        # Kiểm tra mức alert — chỉ gọi Gemini cho cảnh báo quan trọng
        alert_level   = context_data.get("alert_level", "normal")
        _LEVEL_ORDER  = {"ignore": 0, "normal": 1, "watch": 2, "warning": 3, "alert": 4, "critical": 5}
        min_idx       = _LEVEL_ORDER.get(self._min_alert_level, 3)
        current_idx   = _LEVEL_ORDER.get(alert_level, 0)
        if current_idx < min_idx:
            # Cấp thấp → trả về fallback ngay, tiết kiệm quota
            return self._run_fallback(context_data, fallback_fn)

        # NLG không khả dụng → dùng fallback
        if not self.is_available:
            return self._run_fallback(context_data, fallback_fn)

        try:
            prompt  = self._build_prompt(context_data)
            future  = self._executor.submit(self._call_gemini, prompt)
            result  = future.result(timeout=self._timeout_sec)

            if result:
                with self._lock:
                    self._success_count += 1
                    self._error_count   = 0
                logger.debug(f"[NLG] ✅ Gemini: {result[:80]}...")
                return result

        except FuturesTimeoutError:
            with self._lock:
                self._error_count     += 1
                self._last_error_time  = time.time()
                self._last_error_msg   = f"Timeout sau {self._timeout_sec}s"
            logger.warning(f"[NLG] ⏱️ Gemini timeout ({self._timeout_sec}s) — dùng template.")
        except Exception as e:
            with self._lock:
                self._error_count     += 1
                self._last_error_time  = time.time()
                self._last_error_msg   = str(e)
            logger.warning(f"[NLG] ⚠️ Gemini lỗi lần {self._error_count}: {e} — dùng template.")

        return self._run_fallback(context_data, fallback_fn)

    def get_status(self) -> dict:
        """Trả về trạng thái đầy đủ của NLG Engine (dùng cho API /api/nlg/status)."""
        with self._lock:
            success_rate = (
                f"{self._success_count / self._call_count:.0%}"
                if self._call_count > 0 else "N/A"
            )
            return {
                "enabled"       : self._enabled,
                "initialized"   : self._initialized,
                "available"     : self.is_available,
                "model"         : self._model_name if self._initialized else "—",
                "api_key_set"   : bool(self._api_key),
                "total_calls"   : self._call_count,
                "success_calls" : self._success_count,
                "error_count"   : self._error_count,
                "success_rate"  : success_rate,
                "last_error"    : self._last_error_msg,
                "cooldown_active": (
                    self._error_count >= 3
                    and (time.time() - self._last_error_time) < self._cooldown_sec
                ),
            }

    def test_generate(self) -> dict:
        """
        Thử sinh câu mẫu để kiểm tra kết nối Gemini.
        Dùng cho endpoint /api/nlg/test.
        """
        sample_ctx = {
            "role"        : "unknown",
            "action"      : "loitering",
            "zone"        : "khu vực cấm",
            "zone_type"   : "restricted",
            "alert_level" : "alert",
            "time_in_zone": 45,
            "loitering"   : True,
        }

        if not self.is_available:
            return {
                "success" : False,
                "source"  : "unavailable",
                "message" : "Gemini Engine chưa sẵn sàng. Kiểm tra API key và kết nối mạng.",
                "status"  : self.get_status(),
            }

        try:
            prompt = self._build_prompt(sample_ctx)
            future = self._executor.submit(self._call_gemini, prompt)
            result = future.result(timeout=self._timeout_sec + 2)  # Thêm 2s cho test

            if result:
                # Đảm bảo result là string UTF-8 hợp lệ (không bị encoding issue)
                if isinstance(result, bytes):
                    result = result.decode("utf-8", errors="replace")
                return {
                    "success" : True,
                    "source"  : "gemini",
                    "message" : result,
                    "status"  : self.get_status(),
                }
        except FuturesTimeoutError:
            return {
                "success" : False,
                "source"  : "timeout",
                "message" : f"Gemini timeout sau {self._timeout_sec + 2}s. Kiểm tra kết nối mạng.",
                "status"  : self.get_status(),
            }
        except Exception as e:
            err_msg = repr(e)  # repr() luôn ASCII-safe, tránh UnicodeEncodeError
            return {
                "success" : False,
                "source"  : "error",
                "message" : f"Loi Gemini API: {err_msg}",
                "status"  : self.get_status(),
            }

        return {
            "success" : False,
            "source"  : "empty",
            "message" : "Gemini khong tra ve ket qua.",
            "status"  : self.get_status(),
        }

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _call_gemini(self, prompt: str) -> str:
        """Gọi Gemini API (chạy trong thread riêng). Thread-safe."""
        with self._lock:
            model = self._model
        if model is None:
            return ""
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else ""

    def _build_prompt(self, ctx: dict) -> str:
        """Xây dựng prompt gửi cho Gemini từ dữ liệu ngữ cảnh camera."""
        from datetime import datetime
        role        = ctx.get("role", "unknown")
        action      = ctx.get("action", "unknown")
        zone        = ctx.get("zone") or "khu vực giám sát"
        alert       = ctx.get("alert_level", "normal")
        time_zone   = ctx.get("time_in_zone", 0)
        loitering   = ctx.get("loitering", False)
        zone_type   = ctx.get("zone_type", "allowed")
        rule_name   = ctx.get("rule_name", "")

        # Làm sạch tên zone kỹ thuật → mô tả tự nhiên
        zone_clean = zone
        import re
        if re.match(r'^zone_\d+$', zone, re.IGNORECASE):
            zone_clean = "khu vực giám sát"
        elif re.match(r'^rtsp_', zone, re.IGNORECASE):
            zone_clean = "khu vực camera"
        elif re.match(r'^webcam_', zone, re.IGNORECASE):
            zone_clean = "khu vực camera"

        # Thời điểm trong ngày
        hour = datetime.now().hour
        if 5 <= hour < 9:
            time_period = "buổi sáng sớm"
        elif 9 <= hour < 12:
            time_period = "buổi sáng"
        elif 12 <= hour < 14:
            time_period = "buổi trưa"
        elif 14 <= hour < 18:
            time_period = "buổi chiều"
        elif 18 <= hour < 22:
            time_period = "buổi tối"
        else:
            time_period = "đêm khuya"

        # Ánh xạ tên role sang tiếng Việt
        role_vi = {
            "shipper"     : "Nhân viên giao hàng (Shipper)",
            "doctor"      : "Bác sĩ",
            "nurse"       : "Y tá",
            "police"      : "Cảnh sát",
            "military"    : "Quân nhân",
            "security"    : "Nhân viên bảo vệ",
            "civil_guard" : "Dân phòng",
            "student"     : "Học sinh/sinh viên",
            "chef"        : "Đầu bếp",
            "janitor"     : "Nhân viên vệ sinh",
            "worker"      : "Nhân công/Công nhân",
            "construction": "Công nhân xây dựng",
            "technician"  : "Kỹ thuật viên",
            "postman"     : "Nhân viên bưu điện",
            "normal"      : "Người dân bình thường",
            "unknown"     : "Người lạ chưa xác định được danh tính",
        }.get(role, role)

        action_vi = {
            "standing"    : "đang đứng yên một chỗ",
            "walking"     : "đang đi bộ",
            "running"     : "đang chạy nhanh",
            "falling"     : "bị té ngã hoặc ngất xỉu",
            "climbing"    : "đang leo trèo",
            "fighting"    : "đang đánh nhau / xô xát",
            "raising_hand": "đang giơ tay vẫy",
            "gathering"   : "đang tụ tập thành nhóm đông người",
            "loitering"   : "đang lảng vảng không rõ mục đích",
        }.get(action, action)

        alert_vi = {
            "warning" : "đáng chú ý",
            "alert"   : "nguy hiểm",
            "critical": "CỰC KỲ NGUY HIỂM — cần xử lý NGAY",
        }.get(alert, "thông thường")

        zone_type_vi = {
            "restricted": "khu vực cấm, không được phép vào",
            "allowed"   : "khu vực cho phép",
            "monitored" : "khu vực được giám sát",
        }.get(zone_type, zone_type)

        # Gợi ý loại tình huống từ rule_name
        situation_hint = ""
        if "fighting" in rule_name:
            situation_hint = "Đây là tình huống ẩu đả, rất khẩn cấp."
        elif "falling" in rule_name:
            situation_hint = "Người này có thể đang cần cấp cứu."
        elif "climbing" in rule_name:
            situation_hint = "Đây là hành vi nghi vấn đột nhập."
        elif "loitering" in rule_name and "restricted" in rule_name:
            situation_hint = "Người này lảng vảng lâu ở nơi cấm, rất đáng ngờ."
        elif "shipper" in rule_name:
            situation_hint = "Có người giao hàng đến."
        elif "security" in rule_name or "police" in rule_name:
            situation_hint = "Lực lượng an ninh đang làm nhiệm vụ."

        parts = [f"Tình huống an ninh [{alert_vi.upper()}] lúc {time_period}:"]
        parts.append(f"- Đối tượng: {role_vi}")
        parts.append(f"- Hành động: {action_vi}")
        parts.append(f"- Vị trí: {zone_clean} ({zone_type_vi})")
        if time_zone > 5:
            parts.append(f"- Thời gian có mặt: khoảng {int(time_zone)} giây")
        if loitering:
            parts.append("- Hành vi lảng vảng đáng ngờ: CÓ")
        if situation_hint:
            parts.append(f"- Ghi chú: {situation_hint}")
        if time_period in ("đêm khuya", "buổi tối"):
            parts.append("- Lưu ý: đây là thời điểm NHẠY CẢM (tối/đêm khuya)")

        parts.append(
            "\nHãy viết đúng 1-2 câu thông báo tự nhiên, thân mật cho chủ nhà. "
            "Không dùng từ kỹ thuật. Không đề cập ID hay số kỹ thuật."
        )
        return "\n".join(parts)

    def _run_fallback(self, context_data: dict, fallback_fn: Optional[Callable]) -> str:
        """Chạy hàm fallback hoặc trả về chuỗi mặc định."""
        if fallback_fn:
            try:
                return fallback_fn()
            except Exception:
                pass
        # Fallback tự sinh câu không kỹ thuật
        alert = context_data.get("alert_level", "normal")
        zone  = context_data.get("zone") or ""
        import re
        if not zone or re.match(r'^(zone_\d+|rtsp_|webcam_)', zone, re.IGNORECASE):
            zone = "khu vực giám sát"
        emoji = {"warning": "⚠️", "alert": "🔴", "critical": "🚨"}.get(alert, "📷")
        role  = context_data.get("role", "unknown")
        role_vi = {
            "shipper": "Shipper", "security": "bảo vệ", "police": "cảnh sát",
            "unknown": "người lạ", "normal": "người", "student": "học sinh",
        }.get(role, "người")
        if alert == "critical":
            return f"{emoji} Khẩn! Camera phát hiện tình huống nguy hiểm tại {zone}. Kiểm tra ngay bạn ơi!"
        elif alert == "alert":
            return f"{emoji} Bạn ơi, có {role_vi} xuất hiện tại {zone} cần chú ý!"
        elif alert == "warning":
            return f"{emoji} Camera thấy có điều bất thường tại {zone}, để ý một chút nhé."
        return f"📷 Camera ghi nhận có {role_vi} tại {zone}."


# ──────────────────────────────────────────────
# Singleton instance — dùng chung toàn dự án
# ──────────────────────────────────────────────
_nlg_engine_instance: Optional[NLGEngine] = None
_nlg_lock = threading.Lock()


def get_nlg_engine() -> NLGEngine:
    """Lấy singleton NLGEngine (lazy init, thread-safe)."""
    global _nlg_engine_instance
    if _nlg_engine_instance is None:
        with _nlg_lock:
            if _nlg_engine_instance is None:
                _nlg_engine_instance = NLGEngine()
    return _nlg_engine_instance
