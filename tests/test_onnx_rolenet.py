"""Test ONNX inference integration trong RoleClassifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from modules.role_classifier import RoleClassifier
from models import TrackedObject, BoundingBox, ObjectCategory, SocialRole, IdentityStatus, AlertLevel

def test_onnx_status():
    rc = RoleClassifier()
    status = rc.get_status()
    print(f"  Mode          : {status['mode']}")
    print(f"  Using ONNX    : {status['using_onnx']}")
    print(f"  ONNX FP32     : {status['onnx_fp32_exists']}")
    print(f"  ONNX INT8     : {status['onnx_int8_exists']}")
    print(f"  Model version : {status['model_version']}")
    print(f"  Device        : {status['device']}")
    assert status['using_onnx'], "Expected ONNX to be active"
    assert status['onnx_fp32_exists'], "ONNX FP32 file missing"
    assert status['onnx_int8_exists'], "ONNX INT8 file missing"
    assert status['mode'] in ('onnx_fp32', 'onnx_int8'), f"Bad mode: {status['mode']}"
    print("  [PASS] Status OK")

def test_onnx_inference():
    rc = RoleClassifier()
    # Tạo frame giả 480x640 BGR
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obj = TrackedObject(
        track_id=1,
        bbox=BoundingBox(100, 50, 250, 400),
        class_name="person",
        category=ObjectCategory.PERSON,
        confidence=0.9,
        role=SocialRole.NORMAL,
        role_confidence=0.5,
        identity=IdentityStatus.UNKNOWN,
        alert_level=AlertLevel.NORMAL,
    )
    result = rc.classify(frame, obj)
    print(f"  Role          : {result.role.value}")
    print(f"  Confidence    : {result.role_confidence:.3f}")
    if result.role_evidence:
        print(f"  Evidence      : {result.role_evidence.color_match[:60]}")
    assert result.role is not None
    assert 0.0 <= result.role_confidence <= 1.0
    print("  [PASS] Inference OK")

def test_inference_speed(n=20):
    import time
    rc = RoleClassifier()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obj = TrackedObject(
        track_id=1, bbox=BoundingBox(100, 50, 250, 400),
        class_name="person", category=ObjectCategory.PERSON,
        confidence=0.9, role=SocialRole.NORMAL, role_confidence=0.5,
        identity=IdentityStatus.UNKNOWN, alert_level=AlertLevel.NORMAL,
    )
    # Warmup
    for _ in range(3):
        rc.classify(frame, obj)
    # Time
    t0 = time.perf_counter()
    for _ in range(n):
        rc.classify(frame, obj)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  Avg inference : {ms:.1f} ms/frame ({n} runs)")
    print(f"  [PASS] Speed test done")

if __name__ == "__main__":
    print("\n=== Test: RoleClassifier ONNX Integration ===\n")
    try:
        print("[1] Status check...")
        test_onnx_status()
        print("\n[2] Inference correctness...")
        test_onnx_inference()
        print("\n[3] Speed test...")
        test_inference_speed()
        print("\n=== ALL PASS ===")
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
