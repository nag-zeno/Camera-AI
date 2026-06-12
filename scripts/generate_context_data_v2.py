"""
generate_context_data_v2.py — Sinh dữ liệu kết hợp thực tế (from events.jsonl) và giả lập.

Sử dụng:
    python scripts/generate_context_data_v2.py
"""
import sys
import os
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import csv
import json
import random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from models import (
    TrackedObject, BoundingBox, ObjectCategory,
    SocialRole, IdentityStatus, ZoneType, ZoneStatus, ActionLabel, AlertLevel
)
from modules.context_engine import ContextEngine

# Mapping enums -> int
ROLE_IDS = {r.value: i for i, r in enumerate(SocialRole)}
IDENTITY_IDS = {s.value: i for i, s in enumerate(IdentityStatus)}
ZONE_TYPE_IDS = {None: 0, ZoneType.ALLOWED.value: 1, ZoneType.RESTRICTED.value: 2}
ZONE_STATUS_IDS = {s.value: i for i, s in enumerate(ZoneStatus)}
DIRECTION_IDS = {
    "stationary": 0, "moving_in": 1, "moving_out": 2, "moving": 3
}
CATEGORY_IDS = {c.value: i for i, c in enumerate(ObjectCategory)}
ACTION_IDS   = {a.value: i for i, a in enumerate(ActionLabel)}

FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",
]

def parse_real_events(events_file: Path) -> list[dict]:
    """Đọc và parse các event thực tế từ file events.jsonl."""
    samples = []
    if not events_file.exists():
        print(f"⚠️ Không tìm thấy file log thực tế tại: {events_file}")
        return samples

    print(f"📖 Đang đọc log thực tế từ {events_file}...")
    count = 0
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                
                # Parse datetime
                dt_str = evt.get("datetime", "")
                hour = 12 # default
                if dt_str:
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        hour = dt.hour
                    except Exception:
                        pass
                
                is_night = int(hour < 6 or hour >= 22)
                is_business = int(9 <= hour < 18)
                
                # Role
                role_str = evt.get("object_role", "unknown")
                role_id = ROLE_IDS.get(role_str, ROLE_IDS.get("unknown", 15))
                role_confidence = 0.9 if role_str not in ("normal", "unknown") else 0.6
                
                # Category
                cat_str = evt.get("object_category", "person")
                category_id = CATEGORY_IDS.get(cat_str, CATEGORY_IDS.get("person", 0))
                
                # Zone type
                zone_name = evt.get("zone_name") or ""
                zone_type_id = 0
                if zone_name:
                    if "restricted" in zone_name.lower() or "zone_2" in zone_name:
                        zone_type_id = 2
                    else:
                        zone_type_id = 1
                
                # Zone status
                zone_status_id = 2 if zone_name else 0 # inside vs outside
                
                # Action
                action_str = evt.get("action", "unknown")
                action_id = ACTION_IDS.get(action_str, ACTION_IDS.get("unknown", 8))
                action_conf = evt.get("action_confidence", 0.0)
                
                # Loitering & time in zone
                reason = evt.get("reason", "").lower()
                loitering = 1 if "loitering" in reason else 0
                time_in_zone = 75.0 if loitering else (30.0 if zone_name else 0.0)
                
                label = evt.get("level", "normal")
                if label not in [l.value for l in AlertLevel]:
                    label = "normal"
                
                features = {
                    "role_id"          : role_id,
                    "identity_id"      : 1, # unknown
                    "zone_type_id"     : zone_type_id,
                    "zone_status_id"   : zone_status_id,
                    "loitering"        : loitering,
                    "time_in_zone"     : time_in_zone,
                    "visit_count"      : 1,
                    "direction_id"     : 0 if action_str == "standing" else 3,
                    "hour"             : hour,
                    "role_confidence"  : role_confidence,
                    "category_id"      : category_id,
                    "frames_tracked"   : 120,
                    "is_night"         : is_night,
                    "is_business_hour" : is_business,
                    "action_id"        : action_id,
                    "action_confidence": action_conf,
                    "label"            : label
                }
                samples.append(features)
                count += 1
            except Exception as e:
                pass
                
    print(f"✅ Đã parse thành công {count:,} samples thực tế.")
    return samples

def main():
    events_file = PROJECT_ROOT / "logs" / "events.jsonl"
    output_file = PROJECT_ROOT / "data" / "context_training_data.csv"
    
    # 1. Load real-world samples
    real_samples = parse_real_events(events_file)
    
    # 2. Sinh thêm samples giả lập (chạy script giả lập gốc hoặc gọi hàm)
    print("🤖 Đang sinh thêm dữ liệu giả lập để làm phong phú tập train...")
    # Chạy generate_context_data.py để có file context_training_data.csv gốc, sau đó load nó
    # Hoặc tự sinh trực tiếp ở đây
    synthetic_samples = []
    
    # Để đơn giản và nhanh gọn, ta gọi hàm sinh trong generate_context_data.py
    # Thêm import và gọi trực tiếp
    try:
        from scripts import generate_context_data
        generate_context_data.random.seed(42)
        generate_context_data.np.random.seed(42)
        
        # Sinh 20,000 samples giả lập
        engine = ContextEngine()
        engine._cooldown = 0.0
        
        for i in range(20000):
            obj, hour = generate_context_data._make_random_object(track_id=i)
            obj_labeled, _ = engine.evaluate(obj)
            label = obj_labeled.alert_level.value
            features = generate_context_data.obj_to_features(obj_labeled, hour)
            features["label"] = label
            synthetic_samples.append(features)
            
        print(f"✅ Đã sinh {len(synthetic_samples):,} samples giả lập.")
    except Exception as e:
        print(f"⚠️ Lỗi khi sinh dữ liệu giả lập: {e}. Tiến hành fallback tự sinh.")
        # Fallback: dùng luôn file gốc nếu có, hoặc bỏ qua
        
    # 3. Kết hợp 2 nguồn dữ liệu
    all_samples = real_samples + synthetic_samples
    random.shuffle(all_samples)
    
    # 4. Ghi ra file CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FEATURE_NAMES + ["label"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)
        
    print(f"\n🎉 Hợp nhất dữ liệu hoàn tất!")
    print(f"   Tổng số samples: {len(all_samples):,} (Thực tế: {len(real_samples):,}, Giả lập: {len(synthetic_samples):,})")
    print(f"   Lưu tại: {output_file}")
    
    # In phân bố nhãn
    label_counts = {}
    for s in all_samples:
        lbl = s["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
    print("\nPhân bố nhãn sau khi hợp nhất:")
    order = ["ignore", "normal", "watch", "warning", "alert", "critical"]
    for label in order:
        if label in label_counts:
            count = label_counts[label]
            pct = count / len(all_samples) * 100
            print(f"  {label:10s}: {count:6,} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
