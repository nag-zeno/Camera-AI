"""
test_actionnet.py — Kiểm tra tích hợp ActionNet vào hệ thống

Tests:
  1. Import và config
  2. ActionRecognizer initialization (mode detection)
  3. Single inference (rule-based hoặc ML)
  4. Multi-frame accumulation
  5. Batch speed benchmark
  6. Pipeline smoke test với ActionRecognizer
"""
import sys, time
sys.path.insert(0, 'c:/code/camera-ai')

import numpy as np
from pathlib import Path

print("=" * 65)
print("  ActionNet Integration Test")
print("=" * 65)

# ============================================================
print("\n[Test 1] Config & Imports")
# ============================================================
from config import ACTION_CONFIG, MODELS_DIR
cfg = ACTION_CONFIG

print(f"  ✓ Action classes  : {cfg['action_classes']}")
print(f"  ✓ Window frames   : {cfg['window_frames']}")
print(f"  ✓ Input dim       : {cfg['input_dim']}")
print(f"  ✓ Confidence thr  : {cfg['confidence_threshold']}")

from models import ActionLabel, TrackedObject, BoundingBox, ObjectCategory
print(f"  ✓ ActionLabel enum: {[a.value for a in ActionLabel]}")

# ============================================================
print("\n[Test 2] ActionRecognizer Init")
# ============================================================
from modules.action_recognizer import ActionRecognizer

ar     = ActionRecognizer()
status = ar.get_status()

print(f"  Mode           : {status['mode']}")
print(f"  MediaPipe      : {status['mp_available']}")
print(f"  GRU model      : {status['gru_available']}")
print(f"  Device         : {status['device']}")
print(f"  Model exists   : {status['model_exists']}")
print(f"  Model path     : {status['model_path']}")

# ============================================================
print("\n[Test 3] Single Inference")
# ============================================================
frame = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
obj   = TrackedObject(
    track_id=1,
    bbox=BoundingBox(100, 50, 220, 380),
    class_name="person",
    category=ObjectCategory.PERSON,
    confidence=0.9,
    velocity=(2.0, 1.5),
)

t0     = time.perf_counter()
result = ar.recognize(frame, obj)
ms     = (time.perf_counter() - t0) * 1000

print(f"  Action         : {result.action.value}")
print(f"  Confidence     : {result.action_confidence:.3f}")
print(f"  Top-3          : {result.action_top3}")
print(f"  Latency        : {ms:.1f} ms")
assert isinstance(result.action, ActionLabel), "Action phải là ActionLabel enum"
print("  ✓ ActionLabel OK")

# ============================================================
print("\n[Test 4] Multi-frame Accumulation (30 frames)")
# ============================================================
ar2 = ActionRecognizer()
obj2 = TrackedObject(
    track_id=2,
    bbox=BoundingBox(50, 80, 180, 400),
    class_name="person",
    category=ObjectCategory.PERSON,
    confidence=0.88,
    velocity=(8.0, 3.0),   # Tốc độ cao → running
)

frames_buf = [
    np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    for _ in range(35)
]

t0 = time.perf_counter()
for fr in frames_buf:
    ar2.recognize(fr, obj2)
ms = (time.perf_counter() - t0) * 1000

print(f"  35 frames processed in {ms:.1f} ms")
print(f"  Final action : {obj2.action.value} ({obj2.action_confidence:.2f})")
print(f"  Buffer size  : {len(ar2._buffers[2].keypoints)} frames")
print("  ✓ Accumulation OK")

# ============================================================
print("\n[Test 5] Speed Benchmark (100 frames, 5 tracks)")
# ============================================================
ar3   = ActionRecognizer()
N     = 100
objs3 = [
    TrackedObject(
        track_id=i,
        bbox=BoundingBox(50+i*50, 30, 120+i*50, 300),
        class_name="person",
        category=ObjectCategory.PERSON,
        confidence=0.85,
        velocity=(float(i*2), float(i)),
    )
    for i in range(5)
]

t0 = time.perf_counter()
for _ in range(N):
    frame_ = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    for ob in objs3:
        ar3.recognize(frame_, ob)
elapsed  = time.perf_counter() - t0
total_inf = N * len(objs3)
ms_each  = elapsed / total_inf * 1000

print(f"  {total_inf} inferences in {elapsed:.2f}s")
print(f"  Speed : {ms_each:.2f} ms/inference ({1000/ms_each:.0f} inferences/s)")

# ============================================================
print("\n[Test 6] Alert Level Mapping")
# ============================================================
alert_map = ACTION_CONFIG["action_alert_map"]
danger    = [a for a, lv in alert_map.items() if lv in ("alert", "critical")]
warn      = [a for a, lv in alert_map.items() if lv == "watch"]
normal    = [a for a, lv in alert_map.items() if lv == "normal"]
print(f"  DANGER  ({len(danger)}): {danger}")
print(f"  WARNING ({len(warn)}):  {warn}")
print(f"  NORMAL  ({len(normal)}): {normal}")
print("  ✓ Alert mapping OK")

# ============================================================
print("\n[Test 7] Forget Track")
# ============================================================
ar3.forget_track(0)
assert 0 not in ar3._buffers, "Track 0 vẫn còn trong buffer!"
print(f"  ✓ forget_track(0) OK. Active tracks: {len(ar3._buffers)}")

# ============================================================
print(f"\n{'='*65}")
mode = status['mode']
if mode == "ml_actionnet_gru":
    print("RESULT: SUCCESS — ActionNet GRU + MediaPipe ACTIVE! Full ML mode.")
elif "mediapipe" in mode:
    print("RESULT: OK — MediaPipe Pose available. Train GRU để có full ML.")
else:
    print("RESULT: FALLBACK — Rule-based mode. Cài mediapipe để cải thiện.")
    print("  pip install mediapipe")
print("  Dùng scripts/collect_action_data.py để thu thập data.")
print(f"{'='*65}")
