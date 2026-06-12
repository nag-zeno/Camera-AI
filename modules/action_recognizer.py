"""
action_recognizer.py — Tầng 3: Understanding (Action Recognition)

Nhận vào: frame + TrackedObject (persons) + frame_buffer
Trả ra  : TrackedObject với action, action_confidence, action_top3 đã được điền

Hai chế độ:
  1. [ML]   MediaPipe Pose → 132-dim keypoints → GRU → 8 action classes
  2. [Rule] Heuristic từ velocity + bounding box — không cần model

Ưu tiên: ML → Rule-based

Pipeline:
  Frame → MediaPipe Pose → 33 keypoints (x,y,z,vis)
  Tích lũy 30 frame → GRU predict → ActionLabel + confidence

ActionNet Architecture (GRU):
  Input : (batch, 30, 132)  — 30 frames × 33 keypoints × 4 features
  GRU   : hidden=128, layers=2, dropout=0.4
  FC    : 128 → 64 → 8
  Output: 8 action probabilities
"""
import cv2
import numpy as np
import logging
import time
from collections import deque
from typing import Optional

from config import ACTION_CONFIG, MODELS_DIR
from models import TrackedObject, ActionLabel, ObjectCategory

logger = logging.getLogger(__name__)


# ============================================================
# Hằng số
# ============================================================
ACTION_CLASSES = ACTION_CONFIG["action_classes"]
WINDOW_FRAMES  = ACTION_CONFIG["window_frames"]
STEP_FRAMES    = ACTION_CONFIG["step_frames"]
INPUT_DIM      = ACTION_CONFIG["input_dim"]
CONF_THR       = ACTION_CONFIG["confidence_threshold"]

# Keypoint indices MediaPipe Pose (dùng cho rule-based fallback)
MP_NOSE         = 0
MP_LEFT_SHOULDER  = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP       = 23
MP_RIGHT_HIP      = 24
MP_LEFT_KNEE      = 25
MP_RIGHT_KNEE     = 26
MP_LEFT_ANKLE     = 27
MP_RIGHT_ANKLE    = 28
MP_LEFT_WRIST     = 15
MP_RIGHT_WRIST    = 16


# ============================================================
# Model Loading
# ============================================================

def _try_load_actionnet():
    """
    Thử load ActionNet GRU model từ models/.
    Trả về (model, device) hoặc (None, None).
    """
    try:
        import torch
        import torch.nn as nn

        model_path = MODELS_DIR / ACTION_CONFIG["model_name"]
        if not model_path.exists():
            logger.info(f"[ActionRecognizer] Chưa có model tại {model_path}. Dùng rule-based.")
            return None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(str(model_path), map_location=device, weights_only=False)

        # Tái tạo kiến trúc GRU
        class _ActionGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size  = ACTION_CONFIG["input_dim"],
                    hidden_size = ACTION_CONFIG["hidden_size"],
                    num_layers  = ACTION_CONFIG["num_layers"],
                    batch_first = True,
                    dropout     = ACTION_CONFIG["dropout"] if ACTION_CONFIG["num_layers"] > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(ACTION_CONFIG["hidden_size"]),
                    nn.Dropout(ACTION_CONFIG["dropout"]),
                    nn.Linear(ACTION_CONFIG["hidden_size"], 64),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, len(ACTION_CLASSES)),
                )

            def forward(self, x):
                # x: (batch, seq, input_dim)
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])  # lấy hidden state cuối

        net = _ActionGRU()
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        net.to(device)

        val_acc = ckpt.get("val_acc", 0.0)
        logger.info(
            f"[ActionRecognizer] ✅ ActionNet GRU loaded! "
            f"Device={device}, Val Acc={val_acc*100:.1f}%"
        )
        return net, device

    except Exception as e:
        logger.warning(f"[ActionRecognizer] Không load được ActionNet: {e}. Dùng rule-based.")
        return None, None


def _try_init_mediapipe():
    """
    Khởi tạo MediaPipe Pose (hỗ trợ cả API cũ và mới).
    - MediaPipe < 0.10 : dùng mp.solutions.pose
    - MediaPipe >= 0.10: dùng mp.tasks PoseLandmarker (VIDEO mode)
    Trả về pose_runner object hoặc None.
    """
    try:
        import mediapipe as mp
        logger.info(f"[ActionRecognizer] MediaPipe version: {mp.__version__}")
        version_parts = tuple(int(x) for x in mp.__version__.split(".")[:2])

        # --- API cũ (< 0.10) ---
        if version_parts < (0, 10):
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info(f"[ActionRecognizer] ✅ MediaPipe Pose (legacy API v{mp.__version__}).")
            return ("legacy", pose)

        # --- API mới (>= 0.10): dùng PoseLandmarker VIDEO mode ---
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = MODELS_DIR / "pose_landmarker.task"

        # Tự động download nếu chưa có
        if not model_path.exists():
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")
            logger.info(f"[ActionRecognizer] Downloading MediaPipe model to {model_path}...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
                logger.info(f"[ActionRecognizer] Downloaded pose_landmarker_lite.task")
            except Exception as e:
                logger.warning(f"[ActionRecognizer] Download thất bại: {e}. Dùng rule-based.")
                return None

        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))

        # Dùng VIDEO mode thay vì IMAGE — phù hợp với pipeline liên tục
        # IMAGE mode: mỗi frame độc lập, không track landmarks giữa các frame
        # VIDEO mode: có temporal smoothing giữa frames → chính xác hơn
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,   # IMAGE: đúng cho per-person CROP
            # VIDEO mode yêu cầu 1 timeline liên tục — không phù hợp khi cắt crop/người
            num_poses=1,
            min_pose_detection_confidence=0.3,   # Hạ 0.5→0.3: nhận người nhỏ/xa hơn
            min_pose_presence_confidence=0.3,    # Hạ 0.5→0.3
            min_tracking_confidence=0.3,         # Hạ 0.5→0.3
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        logger.info(f"[ActionRecognizer] ✅ MediaPipe PoseLandmarker (Tasks API v{mp.__version__}, IMAGE mode per-crop).")
        return ("tasks", landmarker)

    except ImportError as e:
        logger.warning(f"[ActionRecognizer] mediapipe import lỗi: {e}. Chạy: pip install mediapipe")
        return None
    except Exception as e:
        logger.warning(f"[ActionRecognizer] MediaPipe init lỗi: {type(e).__name__}: {e}")
        return None


# ============================================================
# Per-Track Buffer
# ============================================================

class _TrackActionBuffer:
    """
    Buffer keypoints cho 1 track.
    Tích lũy đủ WINDOW_FRAMES thì predict.

    Cải tiến v2 (rule-based enhanced):
      - velocity_history  : lịch sử speed 5 frame → làm mượt noise
      - aspect_history    : lịch sử aspect 5 frame → phát hiện thay đổi đột ngột
      - bbox_area_history : lịch sử diện tích bbox → scale ngưỡng theo khoảng cách
      - action_vote_buffer: voting 5 frame → action ổn định hơn, ít nhảy
    """
    _HISTORY_LEN = 5   # Số frame lịch sử dùng để làm mượt
    _VOTE_LEN    = 5   # Số frame dùng để vote action

    def __init__(self):
        self.keypoints: deque = deque(maxlen=WINDOW_FRAMES)
        self.frame_count: int = 0          # Tổng số frame đã thêm
        self.last_action: ActionLabel = ActionLabel.UNKNOWN
        self.last_conf  : float       = 0.0
        self.last_top3  : list        = []
        # Bộ đếm consecutive frames cùng action (dùng để xác nhận falling)
        self.consecutive_count : int = 0
        self.consecutive_action: str = ""

        # ── Lịch sử cho rule-based enhanced ──────────────────────
        # speed (px/frame) của 5 frame gần nhất → trung bình làm mượt
        self.velocity_history  : deque = deque(maxlen=self._HISTORY_LEN)
        # aspect (height/width) của bbox 5 frame gần nhất → phát hiện ngã
        self.aspect_history    : deque = deque(maxlen=self._HISTORY_LEN)
        # diện tích bbox (px²) của 5 frame gần nhất → scale ngưỡng speed
        self.bbox_area_history : deque = deque(maxlen=self._HISTORY_LEN)
        # action string của 5 frame gần nhất → lấy mode (voting)
        self.action_vote_buffer: deque = deque(maxlen=self._VOTE_LEN)

    def push(self, kp_vector: np.ndarray):
        """Thêm 1 frame keypoints (shape: (INPUT_DIM,)) vào buffer."""
        self.keypoints.append(kp_vector.copy())
        self.frame_count += 1

    @property
    def ready(self) -> bool:
        """Đủ frame để predict."""
        return len(self.keypoints) >= WINDOW_FRAMES

    @property
    def should_predict(self) -> bool:
        """Theo chu kỳ STEP_FRAMES mới predict 1 lần."""
        return self.ready and (self.frame_count % STEP_FRAMES == 0)

    def get_window(self) -> np.ndarray:
        """Trả về window (WINDOW_FRAMES, INPUT_DIM) dạng float32."""
        return np.array(list(self.keypoints), dtype=np.float32)

    # ── Helpers lịch sử ───────────────────────────────────────────

    @property
    def smoothed_speed(self) -> float:
        """Tốc độ trung bình 5 frame gần nhất (làm mượt noise)."""
        if not self.velocity_history:
            return 0.0
        return float(np.mean(list(self.velocity_history)))

    @property
    def smoothed_aspect(self) -> float:
        """Aspect trung bình 5 frame gần nhất."""
        if not self.aspect_history:
            return 2.0   # Default: người đứng thẳng
        return float(np.mean(list(self.aspect_history)))

    @property
    def delta_aspect(self) -> float:
        """
        Mức thay đổi aspect trong N frame gần nhất.
        Ngã thật: aspect thay đổi đột ngột (lớn → nhỏ).
        Người ngồi: aspect thay đổi chậm/ổn định.
        """
        if len(self.aspect_history) < 3:
            return 0.0
        hist = list(self.aspect_history)
        return float(abs(hist[-1] - hist[0]))

    @property
    def avg_bbox_area(self) -> float:
        """Diện tích bbox trung bình (px²) → proxy khoảng cách với camera."""
        if not self.bbox_area_history:
            return 640 * 480 * 0.05   # Giả sử người chiếm ~5% frame
        return float(np.mean(list(self.bbox_area_history)))

    def vote_action(self, candidate: str) -> str:
        """
        Đẩy candidate vào vote buffer và trả về action được vote nhiều nhất.
        Ưu tiên: falling / climbing / fighting luôn thắng nếu có trong buffer.
        """
        self.action_vote_buffer.append(candidate)
        votes = list(self.action_vote_buffer)
        # Ưu tiên tuyệt đối cho high-risk actions
        for priority_action in ("falling", "climbing", "fighting"):
            if votes.count(priority_action) >= 2:
                return priority_action
        # Lấy mode (action xuất hiện nhiều nhất)
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]


# ============================================================
# Main Module
# ============================================================

class ActionRecognizer:
    """
    Nhận diện hành động từ chuỗi frame.

    Chế độ ML (khi có MediaPipe + ActionNet model):
      - Trích xuất 33 keypoints × 4 features = 132-dim mỗi frame
      - Tích lũy 30 frame (cửa sổ trượt)
      - GRU predict → ActionLabel + confidence

    Chế độ Rule-based (fallback):
      - Velocity → phân biệt standing/walking/running
      - BBox aspect ratio → phát hiện falling
      - Không cần MediaPipe
    """

    # Singleton model (shared across instances)
    _net           = None
    _net_device    = None
    _pose          = None
    _model_loaded  = False
    _mp_lock       = __import__('threading').Lock()  # MediaPipe không thread-safe → serialize tất cả calls
    _mp_detect_ok  = 0   # Số lần MediaPipe phát hiện được keypoints
    _mp_detect_fail= 0   # Số lần trả về None (không detect được)

    def __init__(self):
        self._cfg = ACTION_CONFIG

        # Load model 1 lần (singleton pattern)
        if not ActionRecognizer._model_loaded:
            ActionRecognizer._net, ActionRecognizer._net_device = _try_load_actionnet()
            ActionRecognizer._pose = _try_init_mediapipe()
            ActionRecognizer._model_loaded = True
        elif ActionRecognizer._pose is None:
            # Retry nếu lần đầu init MediaPipe thất bại
            logger.info("[ActionRecognizer] Retrying MediaPipe init (previous attempt failed)...")
            ActionRecognizer._pose = _try_init_mediapipe()

        # Báo cáo trạng thái khi khởi tạo
        _pose_ok = ActionRecognizer._pose is not None
        _gru_ok  = ActionRecognizer._net  is not None
        logger.info(
            f"[ActionRecognizer] Init: MediaPipe={'OK' if _pose_ok else 'FAIL'}, "
            f"GRU={'OK' if _gru_ok else 'FAIL'}, "
            f"Mode={'ml_actionnet_gru' if (_pose_ok and _gru_ok) else 'rule_based'}"
        )

        # Buffer per track
        self._buffers: dict[int, _TrackActionBuffer] = {}
        # Timestamp cho VIDEO mode (MediaPipe Tasks API yêu cầu timestamp tăng dần)
        self._frame_ts_ms: int = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @property
    def _mp_api(self) -> str:
        """Trả về 'legacy', 'tasks', hoặc ''."""
        p = ActionRecognizer._pose
        if isinstance(p, tuple):
            return p[0]
        return ""

    @property
    def _mp_runner(self):
        """Trả về runner object thực sự (pose hoặc landmarker)."""
        p = ActionRecognizer._pose
        if isinstance(p, tuple):
            return p[1]
        return p

    @property
    def using_ml_model(self) -> bool:
        return ActionRecognizer._net is not None and ActionRecognizer._pose is not None

    @property
    def mode(self) -> str:
        if self.using_ml_model:
            return "ml_actionnet_gru"
        elif ActionRecognizer._pose is not None:
            api = self._mp_api
            return f"mediapipe_{api}" if api else "mediapipe_only"
        else:
            return "rule_based"

    def recognize(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
    ) -> TrackedObject:
        """
        Nhận diện hành động cho 1 TrackedObject.

        Args:
            frame: BGR frame gốc
            obj  : TrackedObject đã có bbox và velocity

        Returns:
            obj với action, action_confidence, action_top3 đã cập nhật
        """
        if obj.category != ObjectCategory.PERSON:
            return obj

        track_id = obj.track_id
        if track_id not in self._buffers:
            self._buffers[track_id] = _TrackActionBuffer()

        buf = self._buffers[track_id]

        # === ML Path: MediaPipe + GRU ===
        if ActionRecognizer._pose is not None:
            kp = self._extract_keypoints(frame, obj)
            if kp is not None:
                buf.push(kp)
                if buf.should_predict and ActionRecognizer._net is not None:
                    self._predict_gru(buf)
                elif buf.ready and ActionRecognizer._net is None:
                    # MediaPipe có nhưng chưa có GRU model → dùng rule từ keypoints
                    self._predict_keypoints_rule(buf, kp)

        # === Rule-based Path: chỉ dùng velocity và bbox ===
        else:
            self._predict_simple_rule(buf, obj)

        # Ghi kết quả vào obj
        obj.action            = buf.last_action
        obj.action_confidence = buf.last_conf
        obj.action_top3       = buf.last_top3
        return obj

    def forget_track(self, track_id: int):
        """Xóa buffer khi track mất."""
        self._buffers.pop(track_id, None)

    def get_status(self) -> dict:
        total_mp = ActionRecognizer._mp_detect_ok + ActionRecognizer._mp_detect_fail
        hit_rate = (
            f"{ActionRecognizer._mp_detect_ok}/{total_mp} "
            f"({ActionRecognizer._mp_detect_ok/total_mp:.0%})"
            if total_mp > 0 else "no calls yet"
        )
        return {
            "mode"          : self.mode,
            "mp_available"  : ActionRecognizer._pose is not None,
            "gru_available" : ActionRecognizer._net  is not None,
            "device"        : ActionRecognizer._net_device or "N/A",
            "model_path"    : str(MODELS_DIR / ACTION_CONFIG["model_name"]),
            "model_exists"  : (MODELS_DIR / ACTION_CONFIG["model_name"]).exists(),
            "active_tracks" : len(self._buffers),
            "mp_detect_rate": hit_rate,  # Tỷ lệ MediaPipe detect được keypoints
        }

    # ----------------------------------------------------------
    # Keypoint Extraction (MediaPipe)
    # ----------------------------------------------------------

    def _extract_keypoints(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
    ) -> Optional[np.ndarray]:
        """
        Trích xuất keypoints từ person crop bằng MediaPipe Pose.
        Hỗ trợ cả legacy API (solutions.pose) và Tasks API (PoseLandmarker).

        Returns:
            numpy array shape (132,) hoặc None.
        """
        try:
            x1, y1, x2, y2 = obj.bbox.to_int()
            h_frame, w_frame = frame.shape[:2]

            # Padding 20% (tăng từ 10%) — MediaPipe cần context xung quanh người
            pad_x = int((x2 - x1) * 0.20)
            pad_y = int((y2 - y1) * 0.20)
            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w_frame, x2 + pad_x)
            y2p = min(h_frame, y2 + pad_y)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                return None

            # Resize tối thiểu 128×128 nếu crop quá nhỏ (người đứng xa)
            ch, cw = crop.shape[:2]
            if cw < 128 or ch < 128:
                scale = max(128 / cw, 128 / ch)
                new_w = max(128, int(cw * scale))
                new_h = max(128, int(ch * scale))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Serialize MediaPipe inference hoàn toàn — không thread-safe
            with ActionRecognizer._mp_lock:
                runner = self._mp_runner
                api    = self._mp_api

                if api == "legacy":
                    results = runner.process(crop_rgb)
                    if not results.pose_landmarks:
                        ActionRecognizer._mp_detect_fail += 1
                        return None
                    lm = results.pose_landmarks.landmark
                    kp = np.array(
                        [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm],
                        dtype=np.float32,
                    ).flatten()
                    ActionRecognizer._mp_detect_ok += 1

                elif api == "tasks":
                    import mediapipe as mp
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=crop_rgb,
                    )
                    detection_result = runner.detect(mp_image)
                    if not detection_result.pose_landmarks:
                        ActionRecognizer._mp_detect_fail += 1
                        return None
                    lm = detection_result.pose_landmarks[0]
                    kp = np.array(
                        [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm],
                        dtype=np.float32,
                    ).flatten()
                    ActionRecognizer._mp_detect_ok += 1

                else:
                    return None

            return kp

        except Exception as e:
            logger.debug(f"[ActionRecognizer] Keypoint extract lỗi: {e}")
            return None

    # ----------------------------------------------------------
    # GRU Prediction
    # ----------------------------------------------------------

    def _predict_gru(self, buf: _TrackActionBuffer):
        """Chạy GRU model trên window của buffer."""
        try:
            import torch
            window = buf.get_window()  # (30, 132)
            x = torch.from_numpy(window).unsqueeze(0)  # (1, 30, 132)
            x = x.to(ActionRecognizer._net_device)

            with torch.no_grad():
                logits = ActionRecognizer._net(x)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top_idx  = int(probs.argmax())
            top_conf = float(probs[top_idx])

            if top_conf < CONF_THR:
                buf.last_action = ActionLabel.UNKNOWN
                buf.last_conf   = top_conf
            else:
                label_str       = ACTION_CLASSES[top_idx]
                buf.last_action = ActionLabel(label_str)
                buf.last_conf   = top_conf

            # Top-3
            top3_idx      = probs.argsort()[::-1][:3]
            buf.last_top3 = [
                (ACTION_CLASSES[i], round(float(probs[i]), 3))
                for i in top3_idx
            ]

        except Exception as e:
            logger.debug(f"[ActionRecognizer] GRU predict lỗi: {e}")

    # ----------------------------------------------------------
    # Rule từ Keypoints (có MediaPipe, chưa có GRU model)
    # ----------------------------------------------------------

    def _predict_keypoints_rule(self, buf: _TrackActionBuffer, kp: np.ndarray):
        """
        Rule-based predict từ keypoints MediaPipe.
        Dùng khi có MediaPipe nhưng chưa train GRU.
        """
        try:
            # Reshape về (33, 4)
            pts = kp.reshape(33, 4)  # (x, y, z, vis)

            # Lấy y-coordinate của các điểm quan trọng
            nose_y         = pts[MP_NOSE, 1]
            shoulder_y     = (pts[MP_LEFT_SHOULDER, 1] + pts[MP_RIGHT_SHOULDER, 1]) / 2
            hip_y          = (pts[MP_LEFT_HIP, 1] + pts[MP_RIGHT_HIP, 1]) / 2
            ankle_y        = (pts[MP_LEFT_ANKLE, 1] + pts[MP_RIGHT_ANKLE, 1]) / 2

            # Visibility của các điểm
            shoulder_vis   = (pts[MP_LEFT_SHOULDER, 3] + pts[MP_RIGHT_SHOULDER, 3]) / 2
            hip_vis        = (pts[MP_LEFT_HIP, 3] + pts[MP_RIGHT_HIP, 3]) / 2

            # 1. Falling: khoảng cách dọc shoulder→ankle nhỏ (người nằm ngang)
            # Ngưỡng chặt hơn: vertical_span < 0.20 (trước là 0.30 — quá nhạy)
            vertical_span  = abs(ankle_y - shoulder_y)
            if shoulder_vis > 0.5 and hip_vis > 0.5 and vertical_span < 0.20:
                buf.last_action = ActionLabel.FALLING
                buf.last_conf   = 0.72
                buf.last_top3   = [("falling", 0.72), ("standing", 0.18), ("walking", 0.10)]
                return

            # 2. Raising hand: wrist cao hơn shoulder
            lwrist_y = pts[MP_LEFT_WRIST, 1]
            rwrist_y = pts[MP_RIGHT_WRIST, 1]
            if (lwrist_y < shoulder_y - 0.15) or (rwrist_y < shoulder_y - 0.15):
                buf.last_action = ActionLabel.RAISING_HAND
                buf.last_conf   = 0.65
                buf.last_top3   = [("raising_hand", 0.65), ("standing", 0.25), ("walking", 0.10)]
                return

            # 3. Dùng velocity từ buffer để phân biệt standing/walking/running
            if len(buf.keypoints) >= 10:
                # Lấy x-center của hip qua các frame cuối
                frames = list(buf.keypoints)[-10:]
                hip_xs = [f.reshape(33, 4)[MP_LEFT_HIP, 0] for f in frames]
                speed  = np.std(hip_xs) * 10   # proxy velocity

                if speed > 0.08:
                    buf.last_action = ActionLabel.RUNNING
                    buf.last_conf   = 0.62
                    buf.last_top3   = [("running", 0.62), ("walking", 0.28), ("standing", 0.10)]
                elif speed > 0.02:
                    buf.last_action = ActionLabel.WALKING
                    buf.last_conf   = 0.70
                    buf.last_top3   = [("walking", 0.70), ("standing", 0.20), ("running", 0.10)]
                else:
                    buf.last_action = ActionLabel.STANDING
                    buf.last_conf   = 0.75
                    buf.last_top3   = [("standing", 0.75), ("walking", 0.20), ("running", 0.05)]
            else:
                buf.last_action = ActionLabel.STANDING
                buf.last_conf   = 0.60
                buf.last_top3   = [("standing", 0.60), ("walking", 0.30), ("unknown", 0.10)]

        except Exception as e:
            logger.debug(f"[ActionRecognizer] Keypoint rule lỗi: {e}")

    # ----------------------------------------------------------
    # Simple Rule-based (không có MediaPipe) — Enhanced v2
    # ----------------------------------------------------------

    def _predict_simple_rule(self, buf: _TrackActionBuffer, obj: TrackedObject):
        """
        Heuristic nâng cao từ velocity + bounding box với:
          1. Velocity smoothing: trung bình 5 frame → loại noise/rung camera
          2. Scale-aware thresholds: ngưỡng speed tỉ lệ với kích thước bbox
          3. delta_aspect falling guard: phân biệt ngồi (aspect chậm) vs ngã (đột ngột)
          4. Action voting: mode của 5 frame → action ổn định, không nhảy
          5. Consecutive guard: falling cần 3 frame liên tiếp (giữ nguyên)
        """
        vx, vy = obj.velocity
        speed_instant = (vx**2 + vy**2) ** 0.5

        bbox   = obj.bbox
        # aspect = height/width
        # Người đứng thẳng: aspect ~ 2.0-4.0
        # Người ngồi: aspect ~ 0.8-1.5
        # Người nằm: aspect < 0.7
        aspect   = bbox.height / (bbox.width + 1e-6)
        bbox_area = bbox.width * bbox.height

        # ── Cập nhật lịch sử ──────────────────────────────────────
        buf.velocity_history.append(speed_instant)
        buf.aspect_history.append(aspect)
        buf.bbox_area_history.append(bbox_area)

        # Dùng giá trị đã làm mượt
        speed        = buf.smoothed_speed
        smooth_aspect = buf.smoothed_aspect
        d_aspect     = buf.delta_aspect
        avg_area     = buf.avg_bbox_area

        # ── Scale-aware speed thresholds ──────────────────────────
        # Người gần camera (bbox lớn): threshold cao hơn (pixel/frame nhiều hơn)
        # Người xa camera (bbox nhỏ): threshold thấp hơn (ít pixel/frame nhưng vẫn đang chạy)
        # Normalize theo cạnh bbox so với frame reference (640×480)
        size_factor = min(2.0, max(0.4, (avg_area ** 0.5) / 80.0))
        run_thresh   = self._cfg["rule_run_speed_px"] * size_factor   # 15.0 × factor
        walk_thresh  = 4.0 * size_factor                               # 4.0 × factor

        # ── Phân loại candidate action ────────────────────────────
        candidate_action = None

        # 1. Falling detection — chặt hơn v1:
        #    a) aspect nằm ngang (< 0.75)
        #    b) bbox đủ rộng (không phải người quá nhỏ)
        #    c) speed thấp (người ngã không di chuyển nhanh)
        #    d) delta_aspect đủ lớn (thay đổi đột ngột → ngã thật, không phải ngồi chậm)
        #       Ngồi chậm: delta_aspect < 0.25 (aspect giảm từ từ)
        #       Ngã thật : delta_aspect >= 0.25 (aspect thay đổi nhanh trong 5 frame)
        if (
            smooth_aspect < 0.75
            and bbox.width > 70
            and speed < max(2.5, walk_thresh * 0.5)
            and d_aspect >= 0.20   # Phải có thay đổi aspect đủ lớn
        ):
            candidate_action = "falling"

        # 2. Running: speed cao
        elif speed >= run_thresh:
            candidate_action = "running"

        # 3. Walking: speed trung bình
        elif speed >= walk_thresh:
            candidate_action = "walking"

        # 4. Standing: đứng yên hoặc di chuyển rất chậm
        else:
            candidate_action = "standing"

        # ── Consecutive guard cho falling (giữ từ v1) ────────────
        if candidate_action == buf.consecutive_action:
            buf.consecutive_count += 1
        else:
            buf.consecutive_count  = 1
            buf.consecutive_action = candidate_action

        FALLING_MIN_FRAMES = 3
        if candidate_action == "falling" and buf.consecutive_count < FALLING_MIN_FRAMES:
            # Chưa đủ frames xác nhận → không thay đổi action (giữ nguyên last_action)
            return

        # ── Action voting — lấy action ổn định nhất qua 5 frame ──
        # High-risk actions (falling/climbing/fighting) bypass voting nếu đã
        # confirmed qua consecutive guard ở trên
        if candidate_action == "falling":
            voted_action = "falling"   # Đã qua consecutive guard → xác nhận
        else:
            voted_action = buf.vote_action(candidate_action)

        # ── Apply action với confidence tương ứng ─────────────────
        _ACTION_CONF = {
            "falling" : (0.68, [("falling",  0.68), ("standing", 0.22), ("walking",  0.10)]),
            "running" : (0.68, [("running",  0.68), ("walking",  0.22), ("standing", 0.10)]),
            "walking" : (0.72, [("walking",  0.72), ("standing", 0.20), ("running",  0.08)]),
            "standing": (0.78, [("standing", 0.78), ("walking",  0.17), ("unknown",  0.05)]),
        }
        conf, top3 = _ACTION_CONF.get(voted_action, (0.55, [(voted_action, 0.55)]))
        try:
            buf.last_action = ActionLabel(voted_action)
        except ValueError:
            buf.last_action = ActionLabel.UNKNOWN
        buf.last_conf  = conf
        buf.last_top3  = top3
