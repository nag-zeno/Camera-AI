"""
export_training_data.py — Chuyển đổi log thực tế → Training Data cho ContextNet V2

Luồng xử lý:
  logs/events.jsonl  (log thực từ camera)
       ↓
  [Script này] → chuyển từng event thành feature vector 16 chiều
       ↓
  data/real_context_data.csv  (data từ camera thực)
       ↓
  [Merge] với data/context_training_data.csv (synthetic data cũ)
       ↓
  data/context_training_data_v2.csv (mixed real + synthetic)
       ↓
  [Tự động retrain] → models/context_net.pkl (ContextNet V2)

Cách chạy:
    # Chỉ export, không retrain:
    python scripts/export_training_data.py

    # Export + retrain luôn (khuyến nghị):
    python scripts/export_training_data.py --retrain

    # Export + retrain, không dùng synthetic data cũ:
    python scripts/export_training_data.py --retrain --no-synthetic

Ưu điểm:
    - Data thực từ camera sẽ có weight cao hơn synthetic (x3 lần mặc định)
    - Tự động phân tích chất lượng data trước khi train
    - Hiển thị so sánh accuracy trước/sau retrain
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Thêm project root vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import csv
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

# ============================================================
# Mapping enum → int (phải khớp với generate_context_data.py và context_engine_ml.py)
# ============================================================
from models import SocialRole, IdentityStatus, ZoneType, ZoneStatus, ActionLabel, ObjectCategory

ROLE_IDS = {r.value: i for i, r in enumerate(SocialRole)}
IDENTITY_IDS = {"known": 0, "unknown": 1, "suspicious": 2}
ZONE_TYPE_IDS = {None: 0, "allowed": 1, "restricted": 2}
ZONE_STATUS_IDS = {"outside": 0, "entering": 1, "inside": 2, "leaving": 3}
DIRECTION_IDS = {
    "stationary": 0, "moving_in": 1, "moving_out": 2, "moving": 3
}
CATEGORY_IDS = {"person": 0, "vehicle": 1, "animal": 2, "accessory": 3}
ACTION_IDS = {a.value: i for i, a in enumerate(ActionLabel)}

FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",
]

ALERT_ORDER = ["ignore", "normal", "watch", "warning", "alert", "critical"]

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR     = PROJECT_ROOT / "logs"
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"


# ============================================================
# Đọc và parse event log
# ============================================================

def load_events(log_path: Path) -> list[dict]:
    """Đọc toàn bộ events từ events.jsonl."""
    if not log_path.exists():
        print(f"❌ Không tìm thấy file log: {log_path}")
        print("   Chạy pipeline (python app.py) để thu thập log thực tế trước.")
        sys.exit(1)

    events = []
    skipped = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                events.append(ev)
            except json.JSONDecodeError:
                skipped += 1

    print(f"✅ Đọc được {len(events):,} events từ {log_path.name}")
    if skipped:
        print(f"   ⚠️  Bỏ qua {skipped} dòng lỗi JSON")
    return events


def event_to_features(ev: dict) -> dict | None:
    """
    Chuyển 1 event dict (từ events.jsonl) → feature dict 16 chiều.
    Trả về None nếu event thiếu thông tin quan trọng.
    """
    # Lấy alert level — đây là label thực
    label = ev.get("level", "")
    if label not in ALERT_ORDER:
        return None  # Bỏ qua event không hợp lệ

    # Parse timestamp để lấy giờ
    hour = 12  # Mặc định giữa ngày
    ts = ev.get("timestamp", 0)
    if ts:
        try:
            hour = datetime.fromtimestamp(float(ts)).hour
        except (ValueError, OSError, OverflowError):
            pass

    is_night    = int(hour < 6 or hour >= 22)
    is_business = int(9 <= hour < 18)

    # Role
    role_str = ev.get("object_role", "unknown") or "unknown"
    role_id  = ROLE_IDS.get(role_str, ROLE_IDS.get("unknown", 0))
    role_conf = float(ev.get("role_confidence", 0.5) or 0.5)

    # Identity
    identity_str = ev.get("identity", "unknown") or "unknown"
    identity_id  = IDENTITY_IDS.get(identity_str, 1)

    # Zone
    zone_type_str   = ev.get("zone_type", None) or None
    zone_status_str = ev.get("zone_status", "outside") or "outside"
    zone_type_id    = ZONE_TYPE_IDS.get(zone_type_str, 0)
    zone_status_id  = ZONE_STATUS_IDS.get(zone_status_str, 0)

    # Behavior
    loitering    = int(bool(ev.get("loitering", False)))
    time_in_zone = float(ev.get("time_in_zone", 0) or 0)
    visit_count  = int(ev.get("visit_count", 0) or 0)
    direction    = ev.get("direction", "stationary") or "stationary"
    direction_id = DIRECTION_IDS.get(direction, 0)

    # Object
    category_str  = ev.get("object_category", "person") or "person"
    category_id   = CATEGORY_IDS.get(category_str, 0)
    frames_tracked = int(ev.get("frames_tracked", 30) or 30)

    # Action (ActionNet output trong event log)
    action_str  = ev.get("action", "unknown") or "unknown"
    action_id   = ACTION_IDS.get(action_str, ACTION_IDS.get("unknown", 0))
    action_conf = float(ev.get("action_confidence", 0.0) or 0.0)

    return {
        "role_id"          : role_id,
        "identity_id"      : identity_id,
        "zone_type_id"     : zone_type_id,
        "zone_status_id"   : zone_status_id,
        "loitering"        : loitering,
        "time_in_zone"     : round(time_in_zone, 1),
        "visit_count"      : visit_count,
        "direction_id"     : direction_id,
        "hour"             : hour,
        "role_confidence"  : round(role_conf, 3),
        "category_id"      : category_id,
        "frames_tracked"   : frames_tracked,
        "is_night"         : is_night,
        "is_business_hour" : is_business,
        "action_id"        : action_id,
        "action_confidence": round(action_conf, 3),
        "label"            : label,
    }


def analyze_data_quality(rows: list[dict]) -> dict:
    """Phân tích chất lượng data trước khi train."""
    labels = Counter(r["label"] for r in rows)
    roles  = Counter(r["role_id"] for r in rows)
    actions = Counter(r["action_id"] for r in rows)

    total = len(rows)
    minority_threshold = total * 0.05  # Dưới 5% là thiểu số

    minority_classes = [
        label for label, cnt in labels.items()
        if cnt < minority_threshold
    ]

    return {
        "total"           : total,
        "label_dist"      : dict(labels.most_common()),
        "minority_classes": minority_classes,
        "has_night_data"  : sum(1 for r in rows if r["is_night"]) > 0,
        "loitering_count" : sum(1 for r in rows if r["loitering"]),
    }


# ============================================================
# Export và merge data
# ============================================================

def export_real_data(events: list[dict], output_path: Path, real_weight: int = 3) -> list[dict]:
    """
    Chuyển events → feature rows và lưu CSV.

    Tham số real_weight: mỗi event thực được nhân bản N lần
    để tăng tỷ trọng so với synthetic data.
    """
    rows = []
    skipped = 0

    for ev in events:
        feat = event_to_features(ev)
        if feat is None:
            skipped += 1
            continue
        # Nhân bản để tăng trọng số cho data thực
        for _ in range(real_weight):
            rows.append(feat.copy())

    print(f"\n📊 Kết quả convert events → features:")
    print(f"   Tổng events  : {len(events):,}")
    print(f"   Hợp lệ       : {len(events) - skipped:,} events")
    print(f"   Bị bỏ qua    : {skipped:,} (thiếu label hoặc sai định dạng)")
    print(f"   Sau nhân bản (×{real_weight}): {len(rows):,} rows")

    # Phân tích chất lượng
    quality = analyze_data_quality(rows)
    print(f"\n   Phân bố nhãn (label distribution):")
    for label in ALERT_ORDER:
        cnt = quality["label_dist"].get(label, 0)
        if cnt > 0:
            bar = "█" * min(int(cnt / quality["total"] * 40), 40)
            pct = cnt / quality["total"] * 100
            print(f"   {label:10s}: {bar:<40s} {cnt:5,} ({pct:.1f}%)")

    if quality["minority_classes"]:
        print(f"\n   ⚠️  Lớp thiểu số (< 5%): {quality['minority_classes']}")
        print("       → Model có thể nhận dạng kém với các lớp này")

    # Lưu CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FEATURE_NAMES + ["label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Đã lưu real data: {output_path} ({len(rows):,} rows)")
    return rows


def merge_with_synthetic(real_csv: Path, synthetic_csv: Path, output_csv: Path) -> int:
    """Merge real data với synthetic data, lưu vào output_csv."""
    import csv as csv_mod

    all_rows = []

    # Đọc real data (đã export)
    real_count = 0
    if real_csv.exists():
        with open(real_csv, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                all_rows.append(row)
                real_count += 1

    # Đọc synthetic data (nếu có)
    synth_count = 0
    if synthetic_csv.exists():
        with open(synthetic_csv, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                all_rows.append(row)
                synth_count += 1
        print(f"   + Synthetic data: {synth_count:,} rows ({synthetic_csv.name})")
    else:
        print(f"   ⚠️  Không tìm thấy synthetic data: {synthetic_csv}")
        print("       → Chỉ dùng real data. Chạy generate_context_data.py để tạo thêm.")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_rows)

    # Lưu merged CSV
    if not all_rows:
        print("❌ Không có data để merge!")
        return 0

    fieldnames = list(all_rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    print(f"\n✅ Merged dataset: {total:,} rows")
    print(f"   - Real (×3 nhân bản): {real_count:,} rows")
    print(f"   - Synthetic         : {synth_count:,} rows")
    print(f"   Đã lưu: {output_csv}")
    return total


# ============================================================
# Retrain ContextNet
# ============================================================

def retrain_contextnet(data_csv: Path, model_output: Path):
    """Gọi train_context_model.py để retrain XGBoost."""
    import subprocess
    script = PROJECT_ROOT / "scripts" / "train_context_model.py"

    if not script.exists():
        print(f"❌ Không tìm thấy: {script}")
        return False

    print(f"\n🚀 Bắt đầu retrain ContextNet V2...")
    print(f"   Data: {data_csv}")
    print(f"   Output: {model_output}")
    print("   " + "─" * 50)

    result = subprocess.run(
        [sys.executable, str(script),
         "--data",   str(data_csv),
         "--output", str(model_output),
         "--no-shap"],
        capture_output=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return result.returncode == 0


def load_model_accuracy(model_path: Path) -> str:
    """Đọc accuracy của model hiện tại từ metadata."""
    meta_path = Path(str(model_path).replace(".pkl", "_meta.json"))
    if meta_path.exists():
        try:
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            acc = meta.get("test_accuracy", None)
            n   = meta.get("n_samples", None)
            if acc is not None:
                return f"{acc:.1%} (trained on {n:,} samples)" if n else f"{acc:.1%}"
        except Exception:
            pass
    return "không rõ"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export log thực tế → Training data → Retrain ContextNet V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/export_training_data.py            # Chỉ export, xem báo cáo
  python scripts/export_training_data.py --retrain  # Export + retrain
  python scripts/export_training_data.py --retrain --no-synthetic  # Chỉ dùng real data
  python scripts/export_training_data.py --weight 5  # Nhân bản real data x5
        """
    )
    parser.add_argument(
        "--log", default=str(LOGS_DIR / "events.jsonl"),
        help="Đường dẫn file events.jsonl (mặc định: logs/events.jsonl)"
    )
    parser.add_argument(
        "--real-out", default=str(DATA_DIR / "real_context_data.csv"),
        help="Output CSV cho real data (mặc định: data/real_context_data.csv)"
    )
    parser.add_argument(
        "--synthetic", default=str(DATA_DIR / "context_training_data.csv"),
        help="CSV synthetic data cũ (mặc định: data/context_training_data.csv)"
    )
    parser.add_argument(
        "--merged-out", default=str(DATA_DIR / "context_training_data_v2.csv"),
        help="Output CSV merged (mặc định: data/context_training_data_v2.csv)"
    )
    parser.add_argument(
        "--model-out", default=str(MODELS_DIR / "context_net.pkl"),
        help="Output model path (mặc định: models/context_net.pkl)"
    )
    parser.add_argument(
        "--weight", type=int, default=3,
        help="Số lần nhân bản mỗi real event (mặc định: 3 → tăng tỷ trọng data thực)"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Tự động retrain sau khi export data"
    )
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Chỉ dùng real data, bỏ qua synthetic data"
    )
    parser.add_argument(
        "--min-events", type=int, default=100,
        help="Số events tối thiểu cần có trước khi cho phép retrain (mặc định: 100)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ContextNet V2 — Export Real Data & Retrain")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    log_path    = Path(args.log)
    real_out    = Path(args.real_out)
    synth_path  = Path(args.synthetic)
    merged_out  = Path(args.merged_out)
    model_out   = Path(args.model_out)

    # 1. Kiểm tra model hiện tại
    print("\n📋 Trạng thái hiện tại:")
    if model_out.exists():
        old_acc = load_model_accuracy(model_out)
        print(f"   Model hiện tại: {model_out.name} — accuracy={old_acc}")
    else:
        print(f"   Model hiện tại: CHƯA CÓ (sẽ tạo mới)")

    # 2. Load events
    print(f"\n📂 Đọc log từ: {log_path}")
    events = load_events(log_path)

    if len(events) < args.min_events:
        print(f"\n⚠️  Chỉ có {len(events)} events — quá ít để retrain (cần tối thiểu {args.min_events})")
        print("   Hãy chạy camera thêm và để hệ thống ghi log.")
        if not args.retrain:
            print("   (Thêm --retrain nếu muốn retrain ngay với data ít)")
        sys.exit(0)

    # 3. Convert events → feature CSV
    print(f"\n🔄 Chuyển đổi events → features (weight ×{args.weight})...")
    export_real_data(events, real_out, real_weight=args.weight)

    # 4. Merge với synthetic (nếu cần)
    if not args.no_synthetic:
        print(f"\n🔗 Merge real data + synthetic data...")
        total = merge_with_synthetic(real_out, synth_path, merged_out)
        training_csv = merged_out
    else:
        print("\n⚡ Chỉ dùng real data (--no-synthetic)")
        training_csv = real_out
        total = len(events) * args.weight

    if total == 0:
        print("❌ Không có data để train!")
        sys.exit(1)

    # 5. Retrain (nếu được yêu cầu)
    if args.retrain:
        REPORTS_DIR.mkdir(exist_ok=True)
        success = retrain_contextnet(training_csv, model_out)
        if success:
            print("\n" + "=" * 60)
            print("✅ ContextNet V2 RETRAIN THÀNH CÔNG!")
            if model_out.exists():
                new_acc = load_model_accuracy(model_out)
                print(f"   Accuracy mới: {new_acc}")
            print(f"   Pipeline sẽ tự tải model khi khởi động lại.")
            print("=" * 60)
        else:
            print("\n❌ Retrain thất bại! Kiểm tra lỗi phía trên.")
            sys.exit(1)
    else:
        print(f"\n" + "─" * 60)
        print(f"✅ Export hoàn tất! Data sẵn sàng để retrain.")
        print(f"   Để retrain ContextNet V2, chạy:")
        print(f"   python scripts/export_training_data.py --retrain")
        print(f"   Hoặc:")
        print(f"   python scripts/train_context_model.py --data {training_csv}")
        print("─" * 60)


if __name__ == "__main__":
    main()
