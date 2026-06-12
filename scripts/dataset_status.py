"""
dataset_status.py — Theo dõi tiến độ thu thập dữ liệu RoleNet

Chạy:
    python scripts/dataset_status.py           # In một lần
    python scripts/dataset_status.py --watch   # Cập nhật mỗi 10 giây
"""
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# Fix encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path("data/rolenet_dataset/raw")

CLASS_TARGETS = {
    "shipper": 800, "doctor": 600, "police": 600, "military": 600,
    "security": 600, "student": 600, "chef": 400, "janitor": 400,
    "construction": 400, "nurse": 400, "postman": 300, "technician": 300,
    "worker": 400, "civil_guard": 300, "normal": 1000, "unknown": 500,
}

PHASE_TARGETS = {
    "phase1_min": 150,   # Sprint 1: Minimum viable per class
    "phase2_mid": 300,   # Sprint 2: Medium coverage
    "phase3_full": None, # Sprint 3: Full target (CLASS_TARGETS)
}


def count_images(role_dir: Path) -> int:
    if not role_dir.exists():
        return 0
    return len(list(role_dir.glob("*.jpg")) + list(role_dir.glob("*.png")) + list(role_dir.glob("*.jpeg")) + list(role_dir.glob("*.webp")))


def get_status_emoji(pct: float) -> str:
    if pct >= 100: return "✅"
    if pct >= 75:  return "🟢"
    if pct >= 50:  return "🟡"
    if pct >= 25:  return "🟠"
    if pct > 0:    return "🔴"
    return "⬜"


def print_dashboard():
    print("\n" + "=" * 65)
    print(f"  📊 ROLENET DATASET STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    total_have = 0
    total_need = sum(CLASS_TARGETS.values())
    phase1_done = 0
    phase2_done = 0

    rows = []
    for role, target in CLASS_TARGETS.items():
        n = count_images(BASE_DIR / role)
        pct = min(100.0, n / target * 100)
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        status = get_status_emoji(pct)
        rows.append((role, n, target, pct, bar, status))
        total_have += n
        if n >= PHASE_TARGETS["phase1_min"]: phase1_done += 1
        if n >= PHASE_TARGETS["phase2_mid"]: phase2_done += 1

    # Sắp xếp theo % giảm dần
    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"\n  {'Vai Trò':<18} {'Có':>6} {'Cần':>6}  {'Tiến Độ':<22} {'%':>6}")
    print("  " + "-" * 62)
    for role, n, target, pct, bar, status in rows:
        print(f"  {status} {role:<16} {n:>6,} {target:>6,}  {bar}  {pct:5.1f}%")

    print("  " + "-" * 62)
    total_pct = min(100.0, total_have / total_need * 100)
    bar_len = 20
    filled = int(total_pct / 100 * bar_len)
    total_bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {'TỔNG':<18} {total_have:>6,} {total_need:>6,}  {total_bar}  {total_pct:5.1f}%")

    print(f"\n  📍 Phase 1 (≥150/class): {phase1_done}/16 class hoàn thành")
    print(f"  📍 Phase 2 (≥300/class): {phase2_done}/16 class hoàn thành")

    # Ước tính thời gian (150 ảnh ≈ 3 phút)
    remaining = total_need - total_have
    if remaining > 0:
        minutes_est = (remaining / 150) * 3
        hours = int(minutes_est // 60)
        mins  = int(minutes_est % 60)
        print(f"\n  ⏱️  Còn lại ước tính: ~{hours}h {mins}m (nếu thu thập liên tục)")
    else:
        print(f"\n  🎉 Dataset đã đủ target! Sẵn sàng sang bước preprocessing.")

    print("=" * 65)

    # Tạo JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_have,
        "total_target": total_need,
        "progress_pct": round(total_pct, 2),
        "phase1_complete": phase1_done,
        "phase2_complete": phase2_done,
        "by_class": {
            role: {
                "count": count_images(BASE_DIR / role),
                "target": target,
                "pct": round(min(100.0, count_images(BASE_DIR / role) / target * 100), 1)
            }
            for role, target in CLASS_TARGETS.items()
        }
    }
    status_file = Path("data/rolenet_dataset/status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Lưu status → {status_file}")

    return total_have


def main():
    parser = argparse.ArgumentParser(description="Theo dõi tiến độ thu thập dữ liệu RoleNet")
    parser.add_argument("--watch", action="store_true", help="Cập nhật mỗi 10 giây")
    parser.add_argument("--interval", type=int, default=10, help="Khoảng cách giây giữa các lần cập nhật")
    args = parser.parse_args()

    if args.watch:
        print(f"👀 Đang theo dõi... (cập nhật mỗi {args.interval}s, Ctrl+C để dừng)")
        try:
            while True:
                print_dashboard()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n⏹️  Dừng theo dõi.")
    else:
        print_dashboard()


if __name__ == "__main__":
    main()
