"""
config.py — Trung tâm cấu hình của toàn hệ thống.
Mọi tham số đều nằm ở đây, KHÔNG hard-code trong từng module.

Để thay đổi cấu hình: sửa file này → khởi động lại server.
"""
import os
from pathlib import Path

# Fix lỗi "Assertion fctx->async_lock failed at libavcodec/pthread_frame.c":
# - threads;1       : tắt multi-thread decoder
# - thread_type;slice: chỉ dùng slice-threading (tránh frame-threading của H.265/HEVC)
# - extra_hw_frames KHÔNG được dùng vì sẽ bật async hw decoder gây assertion
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
    "|threads;1"
    "|thread_type;slice"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|max_delay;500000"
    "|reorder_queue_size;0"
)

# Tự động load biến môi trường từ file .env (nếu có python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv chưa cài — đọc thủ công bên dưới

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT      = Path(__file__).parent
DATA_DIR          = PROJECT_ROOT / "data"
KNOWN_FACES_DIR   = DATA_DIR / "known_faces"
SAMPLE_VIDEO_DIR  = DATA_DIR / "sample_videos"
MODELS_DIR        = PROJECT_ROOT / "models"
LOGS_DIR          = PROJECT_ROOT / "logs"
STATIC_DIR        = PROJECT_ROOT / "static"
TEMPLATES_DIR     = PROJECT_ROOT / "templates"
ZONES_DIR         = DATA_DIR / "zones"          # Thư mục chứa zones của từng camera
ZONES_PERSIST_FILE= DATA_DIR / "zones.json"     # Legacy — chỉ dùng khi migrate
RECORDINGS_DIR    = PROJECT_ROOT / "recordings"  # Thư mục lưu video alert clips

# Tạo thư mục nếu chưa có
for _d in [DATA_DIR, KNOWN_FACES_DIR, SAMPLE_VIDEO_DIR,
           MODELS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR, RECORDINGS_DIR, ZONES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def get_zones_file_for_source(source) -> Path:
    """
    Trả về đường dẫn file zones JSON riêng cho từng camera source.

    Mỗi camera có file zones độc lập — đổi camera không bị dùng zones cũ.

    Ví dụ:
      0                                          → zones/webcam_0.json
      "rtsp://admin:pw@192.168.1.200:554/.../201" → zones/rtsp_192.168.1.200_ch201.json
      "/path/to/video.mp4"                       → zones/file_video.json
    """
    import re
    src = str(source).strip()

    if src.isdigit():
        # Webcam index
        filename = f"webcam_{src}.json"

    elif src.lower().startswith("rtsp://"):
        # Trích xuất host và channel từ RTSP URL (không giữ credentials)
        # rtsp://user:pass@192.168.1.200:554/Streaming/Channels/201
        try:
            # Bỏ schema
            rest = src[7:]
            # Bỏ phần credentials (user:pass@)
            if "@" in rest:
                rest = rest.split("@", 1)[1]
            # Lấy host:port
            host_part = rest.split("/")[0]   # "192.168.1.200:554"
            host      = host_part.split(":")[0]  # "192.168.1.200"
            # Lấy channel từ path (ví dụ: /Streaming/Channels/201 → 201)
            path_part = "/" + "/".join(rest.split("/")[1:])
            numbers   = re.findall(r"\d+", path_part)
            channel   = numbers[-1] if numbers else "0"
            # Tạo tên file an toàn
            safe_host = re.sub(r"[^\w]", "_", host)
            filename  = f"rtsp_{safe_host}_ch{channel}.json"
        except Exception:
            # Fallback: hash đơn giản của URL
            filename = f"rtsp_{abs(hash(src)) % 100000}.json"

    else:
        # File video — lấy stem của tên file
        safe_name = re.sub(r"[^\w]", "_", Path(src).stem)[:40]
        filename  = f"file_{safe_name}.json"

    return ZONES_DIR / filename


# ============================================================
# Video Processing
# ============================================================
# Đọc CAMERA_SOURCE từ biến môi trường hoặc file .env thủ công
# Ưu tiên: .env file → os.environ → mặc định 0
_camera_source_env = os.environ.get("CAMERA_SOURCE", "").strip()
_camera_source     = _camera_source_env  # tạm thời lấy từ environ

# Luôn đọc .env để lấy CAMERA_SOURCE (dotenv có thể không được cài)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line.startswith("CAMERA_SOURCE=") and not _line.startswith("#"):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _val:  # Ưu tiên giá trị .env nếu không rỗng
                    _camera_source = _val
                break
    except Exception:
        pass

# Fallback về 0 nếu vẫn rỗng
if not _camera_source:
    _camera_source = "0"

if isinstance(_camera_source, str) and _camera_source.isdigit():
    _camera_source = int(_camera_source)

VIDEO_CONFIG = {
    "default_source" : _camera_source,  # 0 = webcam | "rtsp://..." | "path/to/video.mp4"
    "target_fps"     : 15,      # FPS xử lý mục tiêu
    "frame_width"    : 640,
    "frame_height"   : 480,
    "buffer_size"    : 10,      # Số frame buffer cho RTSP
    "loop_video"     : True,    # Loop khi đọc file video
}


# ============================================================
# Object Detection — YOLOv8
# ============================================================
DETECTION_CONFIG = {
    "model_name"           : "yolov8n.pt",  # nano: nhanh, CPU-friendly
    "confidence_threshold" : 0.50,   # Tăng 0.45→0.50 — giảm nhầm lẫn person/vehicle
    "iou_threshold"        : 0.45,
    "device"               : "auto",        # "auto" | "cpu" | "cuda"

    # COCO class ID → tên nội bộ
    "target_classes": {
        0  : "person",
        14 : "bird",
        15 : "cat",
        16 : "dog",
        17 : "horse",
        18 : "sheep",
        19 : "cow",
        2  : "car",
        3  : "motorcycle",
        5  : "bus",
        7  : "truck",
        24 : "backpack",
        26 : "handbag",
        28 : "suitcase",
    },

    # class name → category lớn
    "class_categories": {
        "person"     : "person",
        "bird"       : "animal",
        "cat"        : "animal",
        "dog"        : "animal",
        "horse"      : "animal",
        "sheep"      : "animal",
        "cow"        : "animal",
        "car"        : "vehicle",
        "motorcycle" : "vehicle",
        "bus"        : "vehicle",
        "truck"      : "vehicle",
        "backpack"   : "accessory",
        "handbag"    : "accessory",
        "suitcase"   : "accessory",
    },
}


# ============================================================
# Object Tracking
# ============================================================
TRACKING_CONFIG = {
    "track_thresh"  : 0.35,   # Ngưỡng confidence detection bắt đầu track
    "track_buffer"  : 30,     # Tăng 20→30: giữ track lâu hơn khi FPS drop hoặc object tạm thời bị che
    "match_thresh"  : 0.25,   # Hạ 0.30→0.25: match được ngay cả khi IoU thấp (object di chuyển nhanh)
    "min_box_area"  : 400,    # Diện tích bbox tối thiểu (pixel²)
    "min_hits"      : 2,      # Frame liên tiếp tối thiểu để confirm track
    "use_velocity_prediction": True,   # Dùng vận tốc dự đoán vị trí trước khi tính IoU
    "category_lock" : True,   # Khóa category: không cho track thay đổi từ person sang vehicle
}


# ============================================================
# Social Role Classification
# ============================================================
#
# Mỗi role có:
#   color_rules   : danh sách luật màu (HSV) để nhận diện
#   accessories   : phụ kiện liên quan (từ YOLO co-detection)
#   threshold     : confidence tối thiểu để gán role
#   display_name  : tên hiển thị tiếng Việt
#   icon          : emoji icon cho UI
#
ROLE_CONFIG = {
    "confidence_threshold" : 0.50,  # Ngưỡng chung tối thiểu
    "crop_size"            : (128, 256),  # (w, h) resize person crop

    # Dải màu HSV: (lower, upper)
    # Hue: 0-180, Sat: 0-255, Val: 0-255
    "hsv_ranges": {
        "bright_green"  : ([35,  80,  80],  [85,  255, 255]),  # Grab
        "bright_orange" : ([10,  150, 150],  [20,  255, 255]),  # Shopee
        "bright_red"    : ([0,   150, 150],  [10,  255, 255]),  # GHN
        "bright_yellow" : ([22,  100, 150],  [35,  255, 255]),  # Nón vàng
        "white"         : ([0,   0,   180],  [180, 45,  255]),  # Bác sĩ, học sinh
        "light_blue"    : ([90,  40,  150],  [120, 150, 255]),  # Y tá
        "navy_blue"     : ([100, 80,  30],   [130, 255, 120]),  # Công an VN
        "dark_green"    : ([35,  40,  20],   [85,  200, 110]),  # Bộ đội
        "khaki"         : ([15,  30,  100],  [35,  130, 200]),  # Dân phòng
        "black"         : ([0,   0,   0],    [180, 80,  60]),   # Bảo vệ
        "neon_orange"   : ([5,   200, 150],  [18,  255, 255]),  # Áo phản quang
        "neon_yellow"   : ([22,  200, 180],  [35,  255, 255]),  # Áo phản quang
        "dark_blue"     : ([100, 60,  20],   [130, 255, 100]),  # Đồng phục
    },

    # ---- Định nghĩa từng vai trò ----
    "roles": {

        # === Nhóm A: Dịch vụ & Giao hàng ===
        "shipper": {
            "display_name" : "Shipper",
            "icon"         : "🛵",
            "color_rules"  : [
                {"color": "bright_green",  "region": "torso", "min_ratio": 0.20},
                {"color": "bright_orange", "region": "torso", "min_ratio": 0.20},
                {"color": "bright_red",    "region": "torso", "min_ratio": 0.20},
            ],
            "accessories"  : ["backpack", "motorcycle"],
            "threshold"    : 0.50,
        },
        "postman": {
            "display_name" : "Bưu Tá",
            "icon"         : "📮",
            "color_rules"  : [
                {"color": "navy_blue", "region": "torso", "min_ratio": 0.25},
            ],
            "accessories"  : ["handbag", "suitcase", "bicycle"],
            "threshold"    : 0.55,
        },
        "technician": {
            "display_name" : "Kỹ Thuật Viên",
            "icon"         : "🔧",
            "color_rules"  : [
                {"color": "dark_blue", "region": "torso", "min_ratio": 0.25},
            ],
            "accessories"  : ["backpack"],
            "threshold"    : 0.50,
        },
        "worker": {
            "display_name" : "Nhân Công",
            "icon"         : "👷",
            "color_rules"  : [
                {"color": "neon_orange", "region": "torso", "min_ratio": 0.25},
                {"color": "neon_yellow", "region": "torso", "min_ratio": 0.25},
            ],
            "accessories"  : [],
            "threshold"    : 0.50,
        },

        # === Nhóm B: Y Tế ===
        "doctor": {
            "display_name" : "Bác Sĩ",
            "icon"         : "🩺",
            "color_rules"  : [
                {"color": "white", "region": "torso", "min_ratio": 0.40},
            ],
            "accessories"  : [],
            "threshold"    : 0.55,
        },
        "nurse": {
            "display_name" : "Y Tá",
            "icon"         : "💉",
            "color_rules"  : [
                {"color": "white",      "region": "torso", "min_ratio": 0.25},
                {"color": "light_blue", "region": "torso", "min_ratio": 0.25},
            ],
            "accessories"  : [],
            "threshold"    : 0.50,
        },

        # === Nhóm C: An Ninh & Pháp Luật ===
        "police": {
            "display_name" : "Công An",
            "icon"         : "👮",
            "color_rules"  : [
                {"color": "navy_blue", "region": "torso", "min_ratio": 0.30},
            ],
            "accessories"  : [],
            "threshold"    : 0.55,
        },
        "military": {
            "display_name" : "Bộ Đội",
            "icon"         : "🪖",
            "color_rules"  : [
                {"color": "dark_green", "region": "torso", "min_ratio": 0.30},
            ],
            "accessories"  : [],
            "threshold"    : 0.55,
        },
        "security": {
            "display_name" : "Bảo Vệ",
            "icon"         : "🛡️",
            "color_rules"  : [
                {"color": "black", "region": "torso", "min_ratio": 0.35},
            ],
            "accessories"  : [],
            "threshold"    : 0.50,
        },
        "civil_guard": {
            "display_name" : "Dân Phòng",
            "icon"         : "🦺",
            "color_rules"  : [
                {"color": "bright_red",    "region": "torso", "min_ratio": 0.30},
                {"color": "bright_orange", "region": "torso", "min_ratio": 0.30},
            ],
            "accessories"  : [],
            "threshold"    : 0.50,
        },

        # === Nhóm D: Chuyên Môn Khác ===
        "student": {
            "display_name" : "Học Sinh/SV",
            "icon"         : "🎒",
            "color_rules"  : [
                {"color": "white", "region": "torso", "min_ratio": 0.30},
            ],
            "accessories"  : ["backpack"],
            "threshold"    : 0.50,
        },
        "construction": {
            "display_name" : "Công Nhân XD",
            "icon"         : "🏗️",
            "color_rules"  : [
                {"color": "neon_orange", "region": "torso", "min_ratio": 0.30},
                {"color": "neon_yellow", "region": "torso", "min_ratio": 0.30},
            ],
            "accessories"  : [],
            "threshold"    : 0.55,
        },
        "chef": {
            "display_name" : "Đầu Bếp",
            "icon"         : "👨‍🍳",
            "color_rules"  : [
                {"color": "white", "region": "torso", "min_ratio": 0.35},
            ],
            "accessories"  : [],
            "threshold"    : 0.50,
        },
        "janitor": {
            "display_name" : "Vệ Sinh",
            "icon"         : "🧹",
            "color_rules"  : [
                {"color": "bright_orange", "region": "torso", "min_ratio": 0.25},
                {"color": "light_blue",    "region": "torso", "min_ratio": 0.25},
            ],
            "accessories"  : [],
            "threshold"    : 0.45,
        },

        # Mặc định — luôn để cuối
        "normal": {
            "display_name" : "Người Thường",
            "icon"         : "🧑",
            "color_rules"  : [],
            "accessories"  : [],
            "threshold"    : 0.0,
        },
        "unknown": {
            "display_name" : "Không Rõ",
            "icon"         : "❓",
            "color_rules"  : [],
            "accessories"  : [],
            "threshold"    : 0.0,
        },
    },
}


# ============================================================
# Identity Awareness
# ============================================================
IDENTITY_CONFIG = {
    "similarity_threshold" : 0.62,    # Ngưỡng để coi là "người quen"
    "feature_dim"          : 96,      # Chiều histogram (32*3 channels)
    "max_known_persons"    : 200,
    "recheck_interval"     : 25,      # Re-check mỗi N frame
}


# ============================================================
# Zone Detection
# ============================================================
# Zones được định nghĩa theo tọa độ pixel của frame 640x480
# Có thể override qua API
ZONE_CONFIG = {
    "zones": [
        {
            "name"    : "entrance",
            "type"    : "allowed",
            "polygon" : [[30, 180], [280, 180], [280, 460], [30, 460]],
            "color"   : [0, 200, 0],   # BGR green
        },
        {
            "name"    : "restricted_area",
            "type"    : "restricted",
            "polygon" : [[360, 80], [620, 80], [620, 360], [360, 360]],
            "color"   : [0, 0, 220],   # BGR red
        },
    ],
}


# ============================================================
# Behavior Analysis
# ============================================================
BEHAVIOR_CONFIG = {
    "loitering_time_threshold"  : 20.0,   # giây — coi là "đứng lâu"
    "frequent_visit_threshold"  : 3,      # lần trong cửa sổ thời gian
    "frequent_visit_window"     : 300.0,  # giây (5 phút)
    "movement_min_pixels"       : 8.0,    # min pixel để coi là "đang di chuyển"
    "history_max_length"        : 300,    # số điểm tối đa trong lịch sử
}


# ============================================================
# Context Reasoning
# ============================================================
REASONING_CONFIG = {
    "alert_cooldown"  : 30.0,   # Tăng 8→30s: tránh spam alert cùng object trong 30 giây
    "context_window"  : 5.0,    # Cửa sổ ngữ cảnh (giây)

    # Ánh xạ alert level → số thứ tự (để so sánh)
    "alert_priority": {
        "ignore"   : 0,
        "normal"   : 1,
        "watch"    : 2,
        "warning"  : 3,
        "alert"    : 4,
        "critical" : 5,
    },
}


# ============================================================
# Visualization
# ============================================================
VIS_CONFIG = {
    "bbox_thickness"   : 2,
    "font_scale"       : 0.52,
    "show_zones"       : True,
    "show_tracks"      : True,
    "show_labels"      : True,
    "zone_alpha"       : 0.15,   # Độ trong suốt của zone fill

    # Màu bounding box theo alert level (BGR)
    "alert_colors": {
        "ignore"   : (100, 100, 100),
        "normal"   : (0,   200, 0),
        "watch"    : (220, 220, 0),
        "warning"  : (0,   165, 255),
        "alert"    : (0,   0,   255),
        "critical" : (0,   0,   180),
    },
}


# ============================================================
# Web API (FastAPI)
# ============================================================
API_CONFIG = {
    "host"        : "0.0.0.0",
    "port"        : 8000,
    "title"       : "Smart AI Security Camera",
    "description" : "Context-aware AI security camera system với 16 vai trò xã hội",
    "version"     : "1.0.0",
    "log_file"    : str(LOGS_DIR / "events.jsonl"),
}


# ============================================================
# Action Recognition — ActionNet (MediaPipe Pose + GRU)
# ============================================================
ACTION_CONFIG = {
    # Model files
    "model_name"     : "actionnet_gru.pt",       # File model đã train
    "model_meta_name": "actionnet_metadata.json", # Metadata

    # Sliding window
    "window_frames"  : 20,      # Giảm 30→20: GRU predict sau 20 frames (nhanh hơn)
    "step_frames"    : 3,       # Giảm 5→3: update action mỗi 3 frames mới

    # Keypoints (MediaPipe Pose)
    "num_keypoints"  : 33,      # Số keypoint của MediaPipe Pose
    "keypoint_dim"   : 4,       # (x, y, z, visibility) cho mỗi keypoint
    "input_dim"      : 132,     # = 33 × 4

    # Model architecture
    "hidden_size"    : 128,     # GRU hidden size
    "num_layers"     : 2,       # Số GRU layers
    "dropout"        : 0.4,

    # Inference
    "confidence_threshold": 0.45,   # Hạ 0.55→0.45: chấp nhận kết quả ít chắc chắn hơn khi người ở xa
    "device"              : "auto", # "auto" | "cpu" | "cuda"

    # 8 Action classes
    "action_classes" : [
        "standing",
        "walking",
        "running",
        "falling",
        "climbing",
        "fighting",
        "raising_hand",
        "gathering",
    ],

    # Action → Alert level mapping (dùng cho ContextEngine)
    "action_alert_map": {
        "standing"    : "normal",
        "walking"     : "normal",
        "running"     : "watch",
        "falling"     : "alert",
        "climbing"    : "alert",
        "fighting"    : "critical",
        "raising_hand": "watch",
        "gathering"   : "watch",
        "unknown"     : "normal",
    },

    # Action → icon hiển thị
    "action_icons": {
        "standing"    : "🧍",
        "walking"     : "🚶",
        "running"     : "🏃",
        "falling"     : "🚨",
        "climbing"    : "🧗",
        "fighting"    : "⚠️",
        "raising_hand": "🙋",
        "gathering"   : "👥",
        "unknown"     : "❓",
    },

    # Rule-based fallback thresholds (khi chưa có model)
    "rule_run_speed_px"  : 15.0,  # pixel/frame để coi là "running"
    "rule_fall_angle_deg": 45.0,  # độ nghiêng để coi là "falling"
}


# ============================================================
# Telegram Push Notification
# ============================================================
# Bot: @Camera_AI_DATN_bot  (t.me/Camera_AI_DATN_bot)
# Tạo bot tại @BotFather — token đã được cấp
#
# Để lấy CHAT_ID:
#   1. Mở Telegram, tìm @Camera_AI_DATN_bot và gửi bất kỳ tin nhắn nào
#   2. Truy cập: https://api.telegram.org/bot<TOKEN>/getUpdates
#   3. Tìm trường "chat" → "id" trong JSON response
#   4. Điền vào TELEGRAM_CHAT_ID bên dưới, hoặc cấu hình qua Web UI
#
_telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
_telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Đọc thủ công từ .env nếu chưa có trong environ (không cần python-dotenv)
if not _telegram_token or not _telegram_chat_id:
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        try:
            for _line in _env_file.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line.startswith("TELEGRAM_BOT_TOKEN=") and not _line.startswith("#"):
                    _telegram_token = _line.split("=", 1)[1].strip().strip('"').strip("'")
                elif _line.startswith("TELEGRAM_CHAT_ID=") and not _line.startswith("#"):
                    _telegram_chat_id = _line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

TELEGRAM_CONFIG = {
    "bot_token"    : _telegram_token,
    "chat_id"      : _telegram_chat_id,  # Chat ID của admin
    "enabled"      : bool(_telegram_token) and bool(_telegram_chat_id), # Tự động bật nếu cấu hình đầy đủ
    "min_level"    : "alert",       # Gửi từ mức alert trở lên (warning/alert/critical)
    "cooldown_sec" : 30.0,          # Thời gian chờ giữa 2 thông báo cho cùng đối tượng
    "send_photo"   : True,          # Đính kèm ảnh snapshot
}


# ============================================================
# NLG Engine — Sinh ngôn ngữ tự nhiên bằng Gemini API
# ============================================================
# Lấy GEMINI_API_KEY từ biến môi trường hoặc file .env
# Cấu hình tại: https://aistudio.google.com/app/apikey
_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

# Đọc thủ công từ .env nếu chưa có trong environ (không cần python-dotenv)
if not _gemini_key:
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        try:
            for _line in _env_file.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line.startswith("GEMINI_API_KEY=") and not _line.startswith("#"):
                    _gemini_key = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    if _gemini_key not in ("", "PASTE_YOUR_NEW_KEY_HERE"):
                        os.environ["GEMINI_API_KEY"] = _gemini_key
                    break
        except Exception:
            pass

NLG_CONFIG = {
    # Bật / tắt toàn bộ NLG Engine
    "enabled"            : True,

    # Gemini API key (đọc từ .env → biến môi trường → cấu hình qua Web UI)
    "api_key"            : _gemini_key,

    # Model Gemini dùng để sinh ngôn ngữ
    # gemini-2.5-flash: model mới nhất, nhanh, thông minh, miễn phí free tier
    "model"              : "gemini-2.5-flash",

    # Timeout tối đa (giây) chờ Gemini phản hồi — tránh block pipeline
    "timeout_seconds"    : 5,

    # Sau N lỗi liên tiếp → ngưng gọi Gemini trong thời gian này (giây)
    "cooldown_on_error"  : 60,

    # Nhiệt độ sinh văn bản: 0.0 = nhất quán, 1.0 = sáng tạo
    "temperature"        : 0.7,

    # Chỉ sinh NLG cho các mức cảnh báo từ đây trở lên
    # (tránh lãng phí API quota cho các sự kiện normal/watch)
    "min_alert_level"    : "warning",
}
