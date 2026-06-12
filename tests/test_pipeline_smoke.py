"""
Smoke test pipeline — khong can camera, khong can video that.
Tao frame synthetic (mau tinh) va chay qua toan bo pipeline 1 lan.

Muc tieu:
  - Kiem tra tat ca module import thanh cong
  - Kiem tra model load (RoleNet, ContextNet)
  - Kiem tra pipeline xu ly frame khong crash
  - In ra trang thai tung module
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = []
FAIL = []

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):
    print(f"  [PASS] {msg}")
    PASS.append(msg)

def fail(msg, err):
    print(f"  [FAIL] {msg}: {err}")
    FAIL.append(msg)


# ─────────────────────────────────────────────────────────────
# 1. Import modules
# ─────────────────────────────────────────────────────────────
section("1. Import Modules")

try:
    from config import (
        MODELS_DIR, KNOWN_FACES_DIR, LOGS_DIR,
        VIDEO_CONFIG, DETECTION_CONFIG, ROLE_CONFIG
    )
    ok("config.py")
except Exception as e:
    fail("config.py", e)

try:
    from models import (
        TrackedObject, Detection, FrameResult, AlertEvent,
        SocialRole, AlertLevel, ObjectCategory, BoundingBox,
        IdentityStatus, RoleEvidence
    )
    ok("models.py")
except Exception as e:
    fail("models.py", e)

try:
    from modules.object_detector   import ObjectDetector
    ok("object_detector")
except Exception as e:
    fail("object_detector", e)

try:
    from modules.object_tracker    import ObjectTracker
    ok("object_tracker")
except Exception as e:
    fail("object_tracker", e)

try:
    from modules.role_classifier   import RoleClassifier
    ok("role_classifier")
except Exception as e:
    fail("role_classifier", e)

try:
    from modules.identity_manager  import IdentityManager
    ok("identity_manager")
except Exception as e:
    fail("identity_manager", e)

try:
    from modules.zone_detector     import ZoneDetector
    ok("zone_detector")
except Exception as e:
    fail("zone_detector", e)

try:
    from modules.behavior_analyzer import BehaviorAnalyzer
    ok("behavior_analyzer")
except Exception as e:
    fail("behavior_analyzer", e)

try:
    from modules.context_engine_ml import ContextEngineML
    ok("context_engine_ml")
except Exception as e:
    fail("context_engine_ml", e)

try:
    from modules.visualizer        import Visualizer
    ok("visualizer")
except Exception as e:
    fail("visualizer", e)

try:
    from modules.event_logger      import EventLogger
    ok("event_logger")
except Exception as e:
    fail("event_logger", e)


# ─────────────────────────────────────────────────────────────
# 2. Model loading
# ─────────────────────────────────────────────────────────────
section("2. Model Loading")

try:
    detector = ObjectDetector()
    detector.load()
    ok(f"YOLOv8 loaded → device={detector._device if hasattr(detector, '_device') else 'ok'}")
except Exception as e:
    fail("YOLOv8 load", e)

try:
    role_clf = RoleClassifier()
    status = role_clf.get_status()
    mode    = status.get("mode", "unknown")
    ver     = status.get("model_version", "none")
    ok(f"RoleClassifier → mode={mode}")
    if ver == "v2":
        print(f"       >> RoleNet V2 (EfficientNet-B2) ACTIVE | TTA={status.get('tta_enabled')} | device={status.get('device')}")
    elif ver == "v1":
        print(f"       >> RoleNet V1 (MobileNetV3) ACTIVE | device={status.get('device')}")
    else:
        print(f"       >> Rule-based fallback (no ML model loaded)")
except Exception as e:
    fail("RoleClassifier", e)

try:
    ctx_eng = ContextEngineML()
    ok(f"ContextEngineML loaded")
except Exception as e:
    fail("ContextEngineML", e)


# ─────────────────────────────────────────────────────────────
# 3. Synthetic frame test
# ─────────────────────────────────────────────────────────────
section("3. Synthetic Frame Processing")

try:
    # Tạo frame giả: 480x640 màu xanh lá (BGR)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (60, 120, 60)   # Màu nền

    # Giả lập 1 tracked person bbox (100,80)→(220,380)
    bbox = BoundingBox(100, 80, 220, 380)
    person = TrackedObject(
        track_id       = 1,
        bbox           = bbox,
        class_name     = "person",
        category       = ObjectCategory.PERSON,
        confidence     = 0.92,
        role           = SocialRole.NORMAL,
        role_confidence= 0.0,
        identity       = IdentityStatus.UNKNOWN,
        alert_level    = AlertLevel.NORMAL,
        alert_reason   = "",
    )
    ok("Tạo TrackedObject giả thành công")
except Exception as e:
    fail("Tạo TrackedObject", e)

# Test RoleClassifier trên frame giả
try:
    enriched = role_clf.classify(frame, person, nearby_objects=[])
    ok(
        f"RoleClassifier.classify() → role={enriched.role.value}, "
        f"conf={enriched.role_confidence:.2f}"
    )
except Exception as e:
    fail("RoleClassifier.classify()", e)

# Test ZoneDetector
try:
    zone_det = ZoneDetector()
    enriched = zone_det.detect(enriched)
    ok(f"ZoneDetector.detect() → zone={enriched.zone_name}, status={enriched.zone_status.value}")
except Exception as e:
    fail("ZoneDetector.detect()", e)

# Test BehaviorAnalyzer
try:
    behavior = BehaviorAnalyzer()
    enriched = behavior.analyze(enriched)
    ok(f"BehaviorAnalyzer.analyze() → loitering={enriched.loitering}, direction={enriched.direction}")
except Exception as e:
    fail("BehaviorAnalyzer.analyze()", e)

# Test ContextEngineML
try:
    enriched, event = ctx_eng.evaluate(enriched)
    ok(
        f"ContextEngineML.evaluate() → alert={enriched.alert_level.value}, "
        f"event={'YES' if event else 'None'}"
    )
except Exception as e:
    fail("ContextEngineML.evaluate()", e)


# ─────────────────────────────────────────────────────────────
# 4. Visualizer test
# ─────────────────────────────────────────────────────────────
section("4. Visualization")

try:
    vis = Visualizer()
    annotated = vis.draw(
        frame   = frame,
        objects = [enriched],
        zones   = zone_det.get_zones(),
        fps     = 30.0,
    )
    assert annotated.shape == frame.shape, "Shape mismatch"
    ok(f"Visualizer.draw() → output shape={annotated.shape}")
except Exception as e:
    fail("Visualizer.draw()", e)


# ─────────────────────────────────────────────────────────────
# 5. EventLogger test
# ─────────────────────────────────────────────────────────────
section("5. Event Logger")

try:
    log_path = str(LOGS_DIR / "smoke_test.jsonl")
    evt_log  = EventLogger(log_path)
    # Tao event dung AlertEvent.create() tu TrackedObject
    enriched.alert_level  = AlertLevel.WATCH
    enriched.alert_reason = "Smoke test event"
    enriched.rule_name    = "smoke_test"
    test_event = AlertEvent.create(enriched)
    evt_log.log(test_event)
    ok(f"EventLogger.log() → {log_path}")
except Exception as e:
    fail("EventLogger.log()", e)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
section("SUMMARY")
total = len(PASS) + len(FAIL)
print(f"  Passed : {len(PASS)}/{total}")
print(f"  Failed : {len(FAIL)}/{total}")

if FAIL:
    print(f"\n  Cac loi can sua:")
    for f in FAIL:
        print(f"     - {f}")
    sys.exit(1)
else:
    print(f"\n  *** TAT CA PASS - Pipeline san sang chay voi video that! ***")
    sys.exit(0)
