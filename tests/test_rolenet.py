"""
test_rolenet.py -- Kiem tra tich hop RoleNet model vao he thong
                   Supports: V2 (EfficientNet-B2) / V1 (MobileNetV3) / Rule-based
"""
import sys, json, time
sys.path.insert(0, 'c:/code/camera-ai')

import numpy as np
from pathlib import Path

print("=" * 60)
print("Test 1: Check model files")
print("=" * 60)
models_dir = Path("c:/code/camera-ai/models")
for fname in [
    "rolenet_v3_best.pt",
    "rolenet_v3_metadata.json",
    "rolenet_v2_best.pt",
    "rolenet_v2_metadata.json",
    "rolenet_best.pt",
    "rolenet_metadata.json",
]:
    fpath   = models_dir / fname
    size_mb = fpath.stat().st_size / 1024 / 1024 if fpath.exists() else 0
    tag     = "[OK]  " if fpath.exists() else "[----]"
    print(f"  {tag} {fname} ({size_mb:.1f} MB)")

print()
print("=" * 60)
print("Test 2: Metadata (active model)")
print("=" * 60)
for meta_name in ["rolenet_v3_metadata.json", "rolenet_v2_metadata.json", "rolenet_metadata.json"]:
    meta_path = models_dir / meta_name
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"  File       : {meta_name}")
        print(f"  Model      : {meta.get('model', 'N/A')}")
        print(f"  Version    : {meta.get('version', 'v1')}")
        print(f"  Num classes: {meta.get('num_classes', 'N/A')}")
        print(f"  Val Acc    : {meta.get('best_val_acc', 'N/A')}%")
        if 'v1_acc' in meta:
            imp = meta.get('improvement', 0)
            print(f"  V1 Acc     : {meta.get('v1_acc')}%  (cai thien: +{imp}%)")
        print(f"  TTA        : {meta.get('tta_enabled', False)}")
        break

print()
print("=" * 60)
print("Test 3: Load RoleClassifier (auto V2 -> V1 -> rule)")
print("=" * 60)
from modules.role_classifier import RoleClassifier
clf    = RoleClassifier()
status = clf.get_status()
print(f"  Mode         : {status['mode']}")
print(f"  Version      : {status['model_version']}")
print(f"  Device       : {status['device']}")
print(f"  TTA enabled  : {status['tta_enabled']}")
print(f"  V3 available : {status['v3_available']}")
print(f"  V1 available : {status['v1_available']}")

print()
print("=" * 60)
print("Test 4: Single inference")
print("=" * 60)
from models import TrackedObject, BoundingBox, ObjectCategory

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
obj   = TrackedObject(
    track_id=1,
    bbox=BoundingBox(100, 50, 200, 300),
    class_name="person",
    category=ObjectCategory.PERSON,
    confidence=0.9,
)

t0     = time.perf_counter()
result = clf.classify(frame, obj)
ms     = (time.perf_counter() - t0) * 1000
ev     = result.role_evidence
print(f"  Role       : {result.role.value}")
print(f"  Confidence : {result.role_confidence:.3f}")
print(f"  Evidence   : {ev.color_match[:70] if ev else 'N/A'}")
print(f"  Latency    : {ms:.1f} ms (TTA={'yes' if status['tta_enabled'] else 'no'})")

print()
print("=" * 60)
print("Test 5: Batch speed benchmark (50 persons)")
print("=" * 60)
N      = 50
frames = [np.random.randint(0, 120, (480, 640, 3), dtype=np.uint8) for _ in range(N)]
objs   = [
    TrackedObject(
        track_id=i,
        bbox=BoundingBox(50, 30, 150, 280),
        class_name="person",
        category=ObjectCategory.PERSON,
        confidence=0.85,
    )
    for i in range(N)
]

t0 = time.perf_counter()
for fr, ob in zip(frames, objs):
    clf.classify(fr, ob)
elapsed    = time.perf_counter() - t0
ms_per     = elapsed / N * 1000
throughput = 1000 / ms_per

print(f"  {N} inferences in {elapsed:.2f}s")
print(f"  Speed : {ms_per:.1f} ms/person  ({throughput:.0f} FPS equivalent)")

print()
print("=" * 60)
ver = status['model_version']
if ver == "v3":
    print("RESULT: SUCCESS -- RoleNet V3 (ConvNeXt-Tiny + TTA) active! Target: 80%+")
elif ver == "v2":
    print("RESULT: SUCCESS -- RoleNet V2 (EfficientNet-B2 + TTA) active! Train V3 de cai thien.")
elif ver == "v1":
    print("RESULT: OK -- RoleNet V1 (MobileNetV3) active. Train V2/V3 de cai thien.")
else:
    print("RESULT: WARNING -- Rule-based fallback. Check model path.")
print("=" * 60)
