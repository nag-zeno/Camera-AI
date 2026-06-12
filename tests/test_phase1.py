"""
Kiểm tra Phase 1: config.py và models.py import thành công.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

errors = []

# Test config
try:
    from config import (
        VIDEO_CONFIG, DETECTION_CONFIG, ROLE_CONFIG,
        IDENTITY_CONFIG, ZONE_CONFIG, BEHAVIOR_CONFIG,
        REASONING_CONFIG, VIS_CONFIG, API_CONFIG,
        PROJECT_ROOT, KNOWN_FACES_DIR, LOGS_DIR
    )
    print("  [OK] config.py")
    print(f"       Roles defined : {len(ROLE_CONFIG['roles'])} roles")
    print(f"       HSV ranges    : {len(ROLE_CONFIG['hsv_ranges'])} colors")
    print(f"       Zones defined : {len(ZONE_CONFIG['zones'])} zones")
    print(f"       Project root  : {PROJECT_ROOT}")
except Exception as e:
    errors.append(f"config.py: {e}")
    print(f"  [FAIL] config.py — {e}")

# Test models
try:
    from models import (
        Detection, TrackedObject, AlertEvent, FrameResult,
        SocialRole, AlertLevel, ObjectCategory,
        IdentityStatus, ZoneType, ZoneStatus,
        BoundingBox, RoleEvidence
    )
    print("  [OK] models.py")

    roles = [r.value for r in SocialRole]
    print(f"       SocialRole ({len(roles)} total):")
    for r in roles:
        print(f"         - {r}")

    levels = [a.value for a in AlertLevel]
    print(f"       AlertLevel: {levels}")

except Exception as e:
    errors.append(f"models.py: {e}")
    print(f"  [FAIL] models.py — {e}")

# Test data model creation
try:
    import time
    bbox = BoundingBox(100, 80, 200, 300)
    det = Detection(bbox=bbox, class_name="person",
                    category=ObjectCategory.PERSON,
                    confidence=0.92, frame_id=1)
    obj = TrackedObject(
        track_id=1, bbox=bbox, class_name="person",
        category=ObjectCategory.PERSON, confidence=0.92,
        role=SocialRole.SHIPPER, role_confidence=0.87,
        identity=IdentityStatus.UNKNOWN,
        alert_level=AlertLevel.NORMAL,
        alert_reason="Test object"
    )
    d = obj.to_dict()
    assert d["track_id"] == 1
    assert d["role"] == "shipper"
    assert d["identity"] == "unknown"
    print("  [OK] Data model creation & serialization")
except Exception as e:
    errors.append(f"model creation: {e}")
    print(f"  [FAIL] model creation — {e}")

print()
if errors:
    print(f"=== FAILED: {len(errors)} error(s) ===")
    sys.exit(1)
else:
    print("=== Phase 1 HOAN THANH — Foundation san sang ===")
    print("    Buoc tiep theo: Phase 2 — Perception modules")
