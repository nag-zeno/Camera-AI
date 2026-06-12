"""
test_tracker_stability.py
Script debug tracking stability — chạy để kiểm tra tracker có giữ ID ổn định không.

Usage:
    python tests/test_tracker_stability.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from modules.object_tracker import ObjectTracker
from models import Detection, BoundingBox, ObjectCategory


def make_det(x1, y1, x2, y2, conf=0.8, cls="person") -> Detection:
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_name=cls,
        category=ObjectCategory.PERSON,
        confidence=conf,
    )


def simulate_moving_object(
    start_x=100, start_y=100,
    width=60, height=120,
    speed_x=20, speed_y=5,    # pixel/frame
    num_frames=30,
    noise=3,                   # pixel noise mô phỏng detection jitter
):
    """
    Sinh ra danh sách detection mô phỏng 1 người đang đi bộ.
    Mỗi frame box dịch (speed_x, speed_y) pixel + nhiễu ngẫu nhiên.
    """
    np.random.seed(42)
    dets_per_frame = []
    for i in range(num_frames):
        nx = np.random.randint(-noise, noise + 1)
        ny = np.random.randint(-noise, noise + 1)
        x1 = start_x + i * speed_x + nx
        y1 = start_y + i * speed_y + ny
        dets_per_frame.append([make_det(x1, y1, x1 + width, y1 + height)])
    return dets_per_frame


def run_test(label, speed_x, speed_y, num_frames=25):
    tracker = ObjectTracker()
    dets_sequence = simulate_moving_object(
        speed_x=speed_x, speed_y=speed_y, num_frames=num_frames
    )

    id_set = set()
    for frame_i, dets in enumerate(dets_sequence):
        tracked = tracker.update(dets)
        for obj in tracked:
            id_set.add(obj.track_id)

    unique_ids = len(id_set)
    status = "PASS" if unique_ids == 1 else f"FAIL ({unique_ids} IDs instead of 1)"
    print(f"  [{label:20s}] speed=({speed_x:+3d},{speed_y:+3d}) px/frame "
          f"=> IDs assigned: {sorted(id_set)} => {status}")
    return unique_ids == 1


def run_test_two_persons():
    """Kiểm tra 2 người đứng gần nhau không bị merge."""
    tracker = ObjectTracker()
    id_sets = [set(), set()]

    for frame_i in range(20):
        # Người 1: x=50..110, người 2: x=250..310 (cách xa nhau)
        det1 = make_det(50 + frame_i * 3, 100, 110 + frame_i * 3, 220)
        det2 = make_det(250 + frame_i * 3, 100, 310 + frame_i * 3, 220)
        tracked = tracker.update([det1, det2])
        for obj in tracked:
            # Gán theo x-center
            cx = (obj.bbox.x1 + obj.bbox.x2) / 2
            if cx < 200:
                id_sets[0].add(obj.track_id)
            else:
                id_sets[1].add(obj.track_id)

    ids_p1 = len(id_sets[0])
    ids_p2 = len(id_sets[1])
    ok = ids_p1 <= 1 and ids_p2 <= 1
    status = "PASS" if ok else f"FAIL (P1={ids_p1} IDs, P2={ids_p2} IDs)"
    print(f"  [{'2 persons separate':20s}] => P1 IDs={sorted(id_sets[0])}, "
          f"P2 IDs={sorted(id_sets[1])} => {status}")
    return ok


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TRACKER STABILITY TEST")
    print("=" * 65)

    results = []

    print("\n[1] Standing still:")
    results.append(run_test("standing still",   speed_x=0,  speed_y=0))

    print("\n[2] Walking slow:")
    results.append(run_test("walking slow",     speed_x=8,  speed_y=2))

    print("\n[3] Walking normal:")
    results.append(run_test("walking normal",   speed_x=15, speed_y=3))

    print("\n[4] Walking fast:")
    results.append(run_test("walking fast",     speed_x=22, speed_y=5))

    print("\n[5] Running:")
    results.append(run_test("running",          speed_x=35, speed_y=8))

    print("\n[6] Diagonal fast:")
    results.append(run_test("diagonal fast",    speed_x=25, speed_y=20))

    print("\n[7] Two persons - no ID merge:")
    results.append(run_test_two_persons())

    passed = sum(results)
    total  = len(results)

    print("\n" + "=" * 65)
    print(f"  RESULT: {passed}/{total} tests PASSED")
    if passed == total:
        print("  PASS - Tracker is stable, no duplicate IDs when moving!")
    else:
        print("  FAIL - See above for details.")
    print("=" * 65 + "\n")
