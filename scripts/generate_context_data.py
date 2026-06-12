"""
generate_context_data.py — Sinh dữ liệu huấn luyện cho ContextNet

Ý tưởng:
    - Sinh ngẫu nhiên tổ hợp (role, zone, behavior, time)
    - Chạy qua ContextEngine hiện tại (rule-based) để lấy label
    - Lưu (features, label) → CSV → dùng để train XGBoost

Cách chạy:
    python scripts/generate_context_data.py
    python scripts/generate_context_data.py --samples 100000 --output data/context_data.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random
import argparse
from pathlib import Path

import numpy as np

from models import (
    TrackedObject, BoundingBox, ObjectCategory,
    SocialRole, IdentityStatus, ZoneType, ZoneStatus, ActionLabel,
)
from modules.context_engine import ContextEngine

# ============================================================
# Mapping enum → int (để model có thể học)
# ============================================================
ROLE_IDS = {r: i for i, r in enumerate(SocialRole)}
IDENTITY_IDS = {s: i for i, s in enumerate(IdentityStatus)}
ZONE_TYPE_IDS = {None: 0, ZoneType.ALLOWED: 1, ZoneType.RESTRICTED: 2}
ZONE_STATUS_IDS = {s: i for i, s in enumerate(ZoneStatus)}
DIRECTION_IDS = {
    "stationary": 0, "moving_in": 1, "moving_out": 2, "moving": 3
}
CATEGORY_IDS = {c: i for i, c in enumerate(ObjectCategory)}
ACTION_IDS   = {a: i for i, a in enumerate(ActionLabel)}

# Trọng số phân phối action (thực tế: chủ yếu standing/walking, ít sự kiện nguy hiểm)
ACTION_WEIGHTS = {
    ActionLabel.STANDING    : 0.35,
    ActionLabel.WALKING     : 0.30,
    ActionLabel.RUNNING     : 0.12,
    ActionLabel.RAISING_HAND: 0.08,
    ActionLabel.GATHERING   : 0.06,
    ActionLabel.FALLING     : 0.04,
    ActionLabel.CLIMBING    : 0.03,
    ActionLabel.FIGHTING    : 0.02,
}

FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",  # <- NEW
]


# ============================================================
# Sinh object ngẫu nhiên
# ============================================================

def _random_hour() -> int:
    """Sinh giờ ngẫu nhiên với trọng số (nhiều ban đêm hơn để cân bằng data)."""
    # 70% ban ngày (6-22), 30% ban đêm (22-6)
    if random.random() < 0.7:
        return random.randint(6, 21)
    return random.choice(list(range(0, 6)) + [22, 23])


def _make_random_object(track_id: int = 1) -> tuple[TrackedObject, int]:
    """Tạo TrackedObject ngẫu nhiên + giờ tương ứng."""
    # Category với trọng số thực tế
    category = random.choices(
        [ObjectCategory.PERSON, ObjectCategory.VEHICLE, ObjectCategory.ANIMAL],
        weights=[0.75, 0.20, 0.05],
        k=1,
    )[0]

    # Role chỉ có nghĩa với PERSON
    role = (
        random.choice(list(SocialRole))
        if category == ObjectCategory.PERSON
        else SocialRole.UNKNOWN
    )
    role_conf = random.uniform(0.3, 1.0)
    identity  = random.choice(list(IdentityStatus))

    # Zone (30% nằm ngoài zone)
    has_zone    = random.random() > 0.25
    zone_type   = random.choice([ZoneType.ALLOWED, ZoneType.RESTRICTED]) if has_zone else None
    zone_status = (
        random.choice(list(ZoneStatus)) if has_zone else ZoneStatus.OUTSIDE
    )
    # Nếu outside thì không có zone
    if zone_status == ZoneStatus.OUTSIDE:
        zone_type = None
        zone_name = None
    else:
        zone_name = f"zone_{zone_type.value if zone_type else 'none'}"

    # Thời gian trong zone
    if zone_status in (ZoneStatus.INSIDE, ZoneStatus.ENTERING):
        time_in_zone = random.uniform(0, 300)
    else:
        time_in_zone = 0.0

    loitering_thr = 60.0
    loitering     = time_in_zone >= loitering_thr

    direction   = random.choice(["stationary", "moving_in", "moving_out", "moving"])
    visit_count = random.randint(0, 8)
    hour        = _random_hour()

    # Action (chỉ có nghia với PERSON)
    if category == ObjectCategory.PERSON:
        action_labels  = list(ACTION_WEIGHTS.keys())
        action_wts     = list(ACTION_WEIGHTS.values())
        action         = random.choices(action_labels, weights=action_wts, k=1)[0]
        action_conf    = random.uniform(0.50, 0.99)
    else:
        action         = ActionLabel.UNKNOWN
        action_conf    = 0.0

    obj = TrackedObject(
        track_id      = track_id,
        bbox          = BoundingBox(100, 100, 200, 350),
        class_name    = category.value,
        category      = category,
        confidence    = random.uniform(0.5, 1.0),
        role          = role,
        role_confidence = role_conf,
        identity      = identity,
        zone_name     = zone_name,
        zone_type     = zone_type,
        zone_status   = zone_status,
        time_in_zone  = time_in_zone,
        loitering     = loitering,
        direction     = direction,
        visit_count   = visit_count,
        frames_tracked = random.randint(1, 1000),
        action         = action,
        action_confidence = action_conf,
    )
    return obj, hour


def obj_to_features(obj: TrackedObject, hour: int) -> dict:
    """Chuyển TrackedObject + giờ → feature dict (bao gồm action)."""
    is_night    = int(hour < 6 or hour >= 22)
    is_business = int(9 <= hour < 18)

    return {
        "role_id"          : ROLE_IDS.get(obj.role, 0),
        "identity_id"      : IDENTITY_IDS.get(obj.identity, 1),
        "zone_type_id"     : ZONE_TYPE_IDS.get(obj.zone_type, 0),
        "zone_status_id"   : ZONE_STATUS_IDS.get(obj.zone_status, 0),
        "loitering"        : int(obj.loitering),
        "time_in_zone"     : round(obj.time_in_zone, 1),
        "visit_count"      : obj.visit_count,
        "direction_id"     : DIRECTION_IDS.get(obj.direction, 0),
        "hour"             : hour,
        "role_confidence"  : round(obj.role_confidence, 3),
        "category_id"      : CATEGORY_IDS.get(obj.category, 0),
        "frames_tracked"   : obj.frames_tracked,
        "is_night"         : is_night,
        "is_business_hour" : is_business,
        "action_id"        : ACTION_IDS.get(obj.action, 0),        # <- NEW
        "action_confidence": round(obj.action_confidence, 3),      # <- NEW
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sinh data huấn luyện ContextNet")
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Số lượng samples cần sinh (mặc định: 50,000)")
    parser.add_argument("--output", type=str,
                        default="data/context_training_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Tạo thư mục output nếu chưa có
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = ContextEngine()
    # Tắt cooldown để lấy label cho mọi sample
    engine._cooldown = 0.0

    print(f"[*] Dang sinh {args.samples:,} training samples...")
    print(f"    Output: {args.output}\n")

    rows = []
    label_counts: dict[str, int] = {}

    for i in range(args.samples):
        obj, hour = _make_random_object(track_id=i)

        # Chạy qua rule engine để lấy nhãn
        obj_labeled, _ = engine.evaluate(obj)
        label = obj_labeled.alert_level.value

        label_counts[label] = label_counts.get(label, 0) + 1

        features = obj_to_features(obj_labeled, hour)
        features["label"] = label
        rows.append(features)

        if (i + 1) % 10_000 == 0:
            pct = (i + 1) / args.samples * 100
            print(f"  [{pct:5.1f}%] {i+1:,} / {args.samples:,} samples...")

    # Ghi CSV
    fieldnames = FEATURE_NAMES + ["label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Báo cáo
    print(f"\n[OK] Da sinh {len(rows):,} samples -> {args.output}")
    print("\nPhan bo nhan (label distribution):")
    order = ["ignore", "normal", "watch", "warning", "alert", "critical"]
    for label in order:
        if label in label_counts:
            count = label_counts[label]
            pct   = count / len(rows) * 100
            bar   = "#" * int(pct / 2)
            print(f"  {label:10s}: {bar:<25s} {count:6,} ({pct:5.1f}%)")

    print(f"\n[->] Buoc tiep theo: python scripts/train_context_model.py")


if __name__ == "__main__":
    main()
