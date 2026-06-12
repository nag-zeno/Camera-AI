"""
pipeline.py — Orchestrator: kết nối tất cả các module thành 1 pipeline xử lý.

Luồng 1 frame:
  VideoProcessor
    → ObjectDetector    (List[Detection])
    → ObjectTracker     (List[TrackedObject])
    → [Parallel] RoleClassifier + ActionRecognizer + IdentityManager  ← TỐI ƯU
    → ZoneDetector      (enrich zone)
    → BehaviorAnalyzer  (enrich behavior)
    → ContextEngine     (enrich alert + emit events)
    → EventLogger       (persist events)
    → Visualizer        (draw frame)
    → FrameResult       (output)

Tối ưu hiệu năng:
  1. Per-track inference cache: Role/Identity skip K frame giữa các lần infer nặng
  2. Parallel processing: Role + Action + Identity chạy song song (ThreadPoolExecutor)
  3. Adaptive skip rate: tự động tăng skip khi FPS thấp hơn target
"""
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

import numpy as np

from config import VIDEO_CONFIG, DETECTION_CONFIG
from models import FrameResult, AlertEvent, TrackedObject, ObjectCategory

from modules.video_processor   import VideoProcessor
from modules.object_detector   import ObjectDetector
from modules.object_tracker    import ObjectTracker
from modules.role_classifier   import RoleClassifier
from modules.action_recognizer import ActionRecognizer
from modules.identity_manager  import IdentityManager
from modules.zone_detector     import ZoneDetector
from modules.behavior_analyzer import BehaviorAnalyzer
from modules.context_engine_ml import ContextEngineML as ContextEngine
from modules.visualizer        import Visualizer
from modules.event_logger      import EventLogger
from modules.telegram_notifier import TelegramNotifier
from modules.alert_recorder    import AlertRecorder
from config import KNOWN_FACES_DIR, LOGS_DIR, RECORDINGS_DIR, get_zones_file_for_source

logger = logging.getLogger(__name__)


# ============================================================
# Cấu hình tối ưu hiệu năng
# ============================================================
PERF_CONFIG = {
    # Số frame bỏ qua giữa 2 lần chạy RoleNet (inference nặng)
    "role_skip_frames"    : 8,    # Tăng 5→8: RoleNet chạy 1 lần / 8 frame

    # Số frame bỏ qua giữa 2 lần chạy MediaPipe Pose
    # Round-robin: mỗi người được gọi action_skip frames mỗi vòng
    # Tổng thực tế = action_skip × N_persons → giảm tải đủ nếu round-robin
    "action_skip_frames"  : 3,    # Hạ 6→3: round-robin đã giảm tải tự nhiên

    # Số frame bỏ qua giữa 2 lần chạy IdentityManager
    "identity_skip_frames": 15,   # Tăng 10→15: Identity chạy 1 lần / 15 frame

    # Số worker thread cho parallel per-person inference
    # Chỉ dùng cho RoleNet + Identity (không có MediaPipe)
    "num_workers"         : 2,    # Giảm số workers — giảm CPU contention

    # FPS target để adaptive skip
    "fps_target"          : VIDEO_CONFIG["target_fps"],
    "adaptive_skip_ratio" : 0.6,
    "max_role_skip"       : 20,
    "max_action_skip"     : 15,
}


class CameraPipeline:
    """
    Pipeline hoàn chỉnh với tối ưu hiệu năng FPS.

    Dùng cho:
      - Chạy trực tiếp (standalone demo)
      - Tích hợp vào FastAPI server (background thread)
    """

    def __init__(self, source=None, log_file=None):
        self._source   = source

        # Khởi tạo các module
        self.video     = VideoProcessor(source)
        self.detector  = ObjectDetector()
        self.tracker   = ObjectTracker()
        self.role_clf  = RoleClassifier()
        self.action_rec= ActionRecognizer()          # ← ActionNet GRU
        self.identity  = IdentityManager(KNOWN_FACES_DIR)
        self.zone_det  = ZoneDetector(persist_file=get_zones_file_for_source(source))
        self.behavior  = BehaviorAnalyzer()
        self.ctx_eng   = ContextEngine()
        self.visualizer= Visualizer()
        self.evt_log   = EventLogger(log_file or str(LOGS_DIR / "events.jsonl"))
        self.telegram  = TelegramNotifier()           # ← Push notification
        self.recorder  = AlertRecorder(output_dir=RECORDINGS_DIR)  # ← Video recording

        # State chia sẻ
        self._latest_frame  : Optional[np.ndarray] = None
        self._latest_result : Optional[FrameResult] = None
        self._running       = False
        self._lock          = threading.Lock()
        self._fps           = 0.0

        # Callbacks khi có alert mới
        self._alert_callbacks: list[Callable[[AlertEvent], None]] = []

        # ── Tối ưu hiệu năng ──────────────────────────────────
        _cfg = PERF_CONFIG
        self._role_skip      = _cfg["role_skip_frames"]
        self._action_skip    = _cfg["action_skip_frames"]
        self._identity_skip  = _cfg["identity_skip_frames"]
        self._fps_target     = _cfg["fps_target"]
        self._adaptive_ratio = _cfg["adaptive_skip_ratio"]
        self._max_role_skip  = _cfg["max_role_skip"]
        self._max_action_skip= _cfg["max_action_skip"]

        # Cache kết quả inference per-track: track_id → (result, last_frame_id)
        self._role_cache    : dict[int, tuple[TrackedObject, int]] = {}
        self._identity_cache: dict[int, tuple[TrackedObject, int]] = {}
        # Frame counter nội bộ (khác frame_id từ VideoProcessor)
        self._frame_count   = 0

        # Thread pool cho parallel inference
        import os
        n_workers = _cfg["num_workers"] or min(4, (os.cpu_count() or 2))
        self._executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="infer")
        logger.info(
            f"[Pipeline] Performance config: "
            f"role_skip={self._role_skip}, action_skip={self._action_skip}, "
            f"identity_skip={self._identity_skip}, workers={n_workers}"
        )

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    def setup(self):
        """Load models và mở nguồn video."""
        logger.info("Setting up pipeline...")
        self.detector.load()
        if not self.video.open():
            raise RuntimeError(f"Cannot open video source: {self._source}")
        logger.info("Pipeline ready.")

    def add_alert_callback(self, fn: Callable[[AlertEvent], None]):
        """Đăng ký callback khi có alert mới."""
        self._alert_callbacks.append(fn)

    # ----------------------------------------------------------
    # Running
    # ----------------------------------------------------------

    def run_blocking(self):
        """Chạy pipeline blocking (dùng cho standalone demo)."""
        self._running = True
        try:
            self._loop()
        finally:
            self._running = False
            self.video.release()

    def start_background(self) -> threading.Thread:
        """Chạy pipeline trong background thread."""
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="camera-pipeline")
        t.start()
        logger.info("Pipeline started in background thread.")
        return t

    def stop(self):
        """Dừng pipeline."""
        self._running = False
        self.recorder.stop()
        self._executor.shutdown(wait=False)
        self.video.release()

    # ----------------------------------------------------------
    # State access (thread-safe)
    # ----------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_result(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest_result

    def get_fps(self) -> float:
        return self._fps

    # ----------------------------------------------------------
    # Zone management
    # ----------------------------------------------------------

    def update_zones(self, zones_config: list[dict]):
        """Cập nhật zones runtime (gọi từ API)."""
        self.zone_det.update_zones(zones_config)

    def get_zones(self) -> list[dict]:
        return self.zone_det.get_zones()

    # ----------------------------------------------------------
    # SHAP Explanation (ContextNet)
    # ----------------------------------------------------------

    def get_shap_for_object(self, track_id: int) -> Optional[dict]:
        """
        Tính SHAP explanation cho object theo track_id hiện tại.
        Trả về dict SHAP hoặc None nếu không tìm thấy object / ML chưa sẵn sàng.
        """
        result = self.get_latest_result()
        if result is None:
            return None
        for obj in result.objects:
            if obj.track_id == track_id:
                if hasattr(self.ctx_eng, "get_shap_explanation"):
                    return self.ctx_eng.get_shap_explanation(obj)
                return None
        return None

    def get_shap_feature_importance(self) -> Optional[dict]:
        """
        Trả về global feature importance từ XGBoost model.
        Hoạt động ngay cả khi không có camera chạy.
        """
        if hasattr(self.ctx_eng, "get_feature_importance"):
            return self.ctx_eng.get_feature_importance()
        return None


    # ----------------------------------------------------------
    # Internal loop
    # ----------------------------------------------------------

    def _loop(self):
        """Main processing loop."""
        fps_alpha = 0.1

        for frame_id, frame in self.video.frames():
            if not self._running:
                break

            self._frame_count += 1

            # Adaptive skip: nếu FPS đang thấp, tăng skip rate
            self._adapt_skip_rates()

            t_start = time.monotonic()
            result  = self._process_frame(frame_id, frame)
            t_end   = time.monotonic()

            # Cập nhật FPS (exponential moving average)
            elapsed    = t_end - t_start
            inst_fps   = 1.0 / elapsed if elapsed > 0 else 0
            self._fps  = fps_alpha * inst_fps + (1 - fps_alpha) * self._fps
            result.fps = self._fps

            # Vẽ frame
            annotated = self.visualizer.draw(
                frame   = frame,
                objects = result.objects,
                zones   = self.get_zones(),
                fps     = self._fps,
            )

            # Cập nhật pre-buffer cho recorder (dùng annotated frame)
            self.recorder.push_frame(annotated)

            with self._lock:
                self._latest_frame  = annotated
                self._latest_result = result

        logger.info("Pipeline loop ended.")
        self.video.release()

    def _adapt_skip_rates(self):
        """
        Adaptive skip rate: tự động tăng số frame bỏ qua khi FPS thấp.
        Chỉ áp dụng sau khi pipeline đã warm-up (>= 30 frame).
        """
        if self._frame_count < 30 or self._fps <= 0:
            return

        threshold = self._fps_target * self._adaptive_ratio
        if self._fps < threshold:
            # Tăng skip thêm 1 (capped)
            self._role_skip    = min(self._role_skip    + 1, self._max_role_skip)
            self._action_skip  = min(self._action_skip  + 1, self._max_action_skip)
            self._identity_skip= min(self._identity_skip+ 2, 25)
            if self._frame_count % 30 == 0:  # Log mỗi ~2 giây
                logger.info(
                    f"[Pipeline] Adaptive skip adjusted: "
                    f"fps={self._fps:.1f} < {threshold:.1f} | "
                    f"role_skip={self._role_skip}, "
                    f"action_skip={self._action_skip}"
                )
        elif self._fps > self._fps_target * 0.9 and self._frame_count % 60 == 0:
            # FPS đã tốt → giảm skip về mức gốc
            cfg = PERF_CONFIG
            self._role_skip    = max(cfg["role_skip_frames"],     self._role_skip    - 1)
            self._action_skip  = max(cfg["action_skip_frames"],   self._action_skip  - 1)
            self._identity_skip= max(cfg["identity_skip_frames"], self._identity_skip- 1)

    def _should_run_role(self, track_id: int) -> bool:
        """True nếu cần chạy RoleNet cho track này ở frame hiện tại."""
        if track_id not in self._role_cache:
            return True  # Lần đầu tiên
        _, last_frame = self._role_cache[track_id]
        return (self._frame_count - last_frame) >= self._role_skip

    def _should_run_action(self) -> bool:
        """True nếu cần chạy ActionRecognizer ở frame này."""
        return (self._frame_count % max(1, self._action_skip)) == 0

    def _should_run_identity(self, track_id: int) -> bool:
        """True nếu cần chạy IdentityManager cho track này."""
        if track_id not in self._identity_cache:
            return True
        _, last_frame = self._identity_cache[track_id]
        return (self._frame_count - last_frame) >= self._identity_skip

    def _process_person_parallel(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        non_persons: list[TrackedObject],
        run_action: bool,
    ) -> TrackedObject:
        """
        Xử lý 1 person với inference song song:
          - RoleNet + ActionRecognizer + Identity chạy đồng thời qua futures
          - Dùng kết quả cache khi chưa đến lúc re-infer
        """
        track_id = obj.track_id

        # ── Quyết định có cần chạy từng bước không ──
        do_role     = self._should_run_role(track_id)
        do_action   = run_action  # Tính 1 lần cho toàn frame
        do_identity = self._should_run_identity(track_id)

        futures = {}

        # Submit parallel tasks (chỉ RoleNet và Identity — không có Action)
        # MediaPipe KHÔNG thread-safe → action phải chạy tuần tự riêng
        if do_role:
            futures["role"] = self._executor.submit(
                self.role_clf.classify, frame, obj, non_persons
            )
        if do_identity:
            futures["identity"] = self._executor.submit(
                self.identity.identify, frame, obj
            )

        # Thu kết quả role (parallel)
        if "role" in futures:
            try:
                result_obj = futures["role"].result(timeout=0.5)
                obj.role            = result_obj.role
                obj.role_confidence = result_obj.role_confidence
                obj.role_evidence   = result_obj.role_evidence
                self._role_cache[track_id] = (obj, self._frame_count)
            except Exception as e:
                logger.debug(f"[Pipeline] role future error: {e}")
        else:
            cached, _ = self._role_cache.get(track_id, (None, 0))
            if cached is not None:
                obj.role            = cached.role
                obj.role_confidence = cached.role_confidence
                obj.role_evidence   = cached.role_evidence

        # Action: chạy TUẦN TỰ (MediaPipe không thread-safe)
        # Chạy trong khi chờ role/identity futures (overlap CPU time)
        if do_action:
            try:
                result_obj = self.action_rec.recognize(frame, obj)
                obj.action            = result_obj.action
                obj.action_confidence = result_obj.action_confidence
                obj.action_top3       = result_obj.action_top3
            except Exception as e:
                logger.debug(f"[Pipeline] action sequential error: {e}")
        # Action không cần cache riêng vì ActionRecognizer đã có internal buffer per-track

        # Thu kết quả identity (parallel)
        if "identity" in futures:
            try:
                result_obj = futures["identity"].result(timeout=0.5)
                obj.identity = result_obj.identity
                self._identity_cache[track_id] = (obj, self._frame_count)
            except Exception as e:
                logger.debug(f"[Pipeline] identity future error: {e}")
        else:
            cached, _ = self._identity_cache.get(track_id, (None, 0))
            if cached is not None:
                obj.identity = cached.identity

        return obj

    def _process_frame(self, frame_id: int, frame: np.ndarray) -> FrameResult:
        """Xử lý 1 frame qua toàn bộ pipeline với tối ưu hiệu năng."""
        now = time.time()

        # 1. Detect
        detections = self.detector.detect(frame, frame_id)

        # 2. Track — truyền frame để tính appearance embedding (color histogram)
        tracked = self.tracker.update(detections, frame)

        # 3. Phân loại persons vs non-persons
        new_alerts: list[AlertEvent] = []
        persons    = [o for o in tracked if o.category == ObjectCategory.PERSON]
        non_persons= [o for o in tracked if o.category != ObjectCategory.PERSON]
        current_ids= {o.track_id for o in persons}

        # Round-robin action: chỉ chạy MediaPipe cho 1 người mỗi frame
        # Thay vì N người × MediaPipe cùng lúc (tắc nghẽ CPU)
        # → luân phiên: person[frame % N] được action frame này
        n_persons  = len(persons)
        action_idx = self._frame_count % n_persons if n_persons > 0 else -1

        # Non-persons: chỉ cần zone + context (nhanh)
        processed = []
        for obj in non_persons:
            obj = self.zone_det.detect(obj)
            obj, event = self.ctx_eng.evaluate(obj)
            if event:
                self._emit_alert(event, frame)
                new_alerts.append(event)
            processed.append(obj)

        # Persons: Role + Action(round-robin) + Identity song song, sau đó Zone + Behavior + Context
        for i, obj in enumerate(persons):
            # Chỉ người được chọn theo round-robin mới chạy action frame này
            run_action_this = (i == action_idx) and self._should_run_action()
            obj = self._process_person_parallel(frame, obj, non_persons, run_action_this)

            obj = self.zone_det.detect(obj)
            obj = self.behavior.analyze(obj)
            obj, event = self.ctx_eng.evaluate(obj)
            if event:
                self._emit_alert(event, frame)
                new_alerts.append(event)
            processed.append(obj)

        # ── Gathering detection: hậu xử lý sau khi tất cả persons đã có zone ──
        # Nếu >= 3 người trong cùng zone VÀ tốc độ trung bình thấp → gathering
        self._detect_gathering(processed)

        # Dọn sạch buffer ActionRecognizer và cache cho track đã mất
        for lost_id in (set(self.action_rec._buffers) - current_ids):
            self.action_rec.forget_track(lost_id)
        for lost_id in (set(self._role_cache) - current_ids):
            self._role_cache.pop(lost_id, None)
        for lost_id in (set(self._identity_cache) - current_ids):
            self._identity_cache.pop(lost_id, None)

        return FrameResult(
            frame_id   = frame_id,
            timestamp  = now,
            objects    = processed,
            new_alerts = new_alerts,
        )

    def _detect_gathering(self, objects: list) -> None:
        """
        Phát hiện tụ tập nhóm sau khi tất cả persons đã biết zone.

        Điều kiện:
          - >= 3 người trong cùng zone (không phải 'no_zone')
          - Tốc độ trung bình của nhóm < 5.0 px/frame (đứng yên hoặc đi chậm)

        Khi phát hiện: ghi đè action = GATHERING cho tất cả thành viên nhóm.
        """
        from models import ActionLabel, ObjectCategory
        from collections import defaultdict

        # Gom persons theo zone name
        zone_groups: dict[str, list] = defaultdict(list)
        for obj in objects:
            if obj.category != ObjectCategory.PERSON:
                continue
            zone_name = getattr(obj, "zone_name", None) or "no_zone"
            if zone_name and zone_name != "no_zone":
                zone_groups[zone_name].append(obj)

        GATHERING_MIN_PERSONS = 3
        GATHERING_MAX_SPEED   = 5.0   # px/frame — nhóm đứng yên / đi chậm

        for zone_name, group in zone_groups.items():
            if len(group) < GATHERING_MIN_PERSONS:
                continue

            # Tính tốc độ trung bình của nhóm
            speeds = []
            for obj in group:
                vx, vy = obj.velocity
                speeds.append((vx**2 + vy**2) ** 0.5)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

            if avg_speed <= GATHERING_MAX_SPEED:
                for obj in group:
                    obj.action            = ActionLabel.GATHERING
                    obj.action_confidence = 0.75
                    obj.action_top3       = [
                        ("gathering", 0.75),
                        ("standing",  0.18),
                        ("walking",   0.07),
                    ]
                logger.debug(
                    f"[Pipeline] Gathering detected: {len(group)} persons in zone '{zone_name}', "
                    f"avg_speed={avg_speed:.1f}"
                )


    def _emit_alert(self, event: AlertEvent, frame: np.ndarray):
        """Gửi alert đến tất cả channels: log, Telegram, recorder, callbacks."""
        self.evt_log.log(event)
        edict = event.to_dict()
        self.telegram.notify(edict, frame=frame)
        self.recorder.on_event(edict)
        for cb in self._alert_callbacks:
            cb(event)


# ----------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    # Cấu hình logging để thấy output khi chạy trực tiếp
    logging.basicConfig(
        level  = logging.INFO,
        format = "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt= "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Chạy CameraPipeline trên file video hoặc webcam."
    )
    parser.add_argument(
        "source",
        nargs   = "?",
        default = 0,
        help    = "Đường dẫn tới file video, hoặc index webcam (mặc định: 0)",
    )
    args = parser.parse_args()

    # Nếu source là số thì chuyển sang int (webcam index)
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # giữ nguyên string (đường dẫn file)

    logger.info(f"Khởi động pipeline với nguồn: {source!r}")
    pipeline = CameraPipeline(source=source)

    try:
        pipeline.setup()
        pipeline.run_blocking()
    except RuntimeError as e:
        logger.error(f"Lỗi khởi động: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Nhận Ctrl+C — dừng pipeline.")
    finally:
        pipeline.stop()
