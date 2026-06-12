"""
preprocess_dataset.py — Tiền xử lý & kiểm tra chất lượng dataset

Nhiệm vụ:
    1. Resize tất cả ảnh về 128×256 px
    2. Loại ảnh kém chất lượng (quá tối, mờ, quá nhỏ)
    3. Phát hiện ảnh trùng lặp (hash-based deduplication)
    4. Tạo splits: train.csv / val.csv / test.csv
    5. In báo cáo thống kê

Cách chạy:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --input data/rolenet_dataset/raw --output data/rolenet_dataset/processed
"""
import argparse
import hashlib
import os
import shutil
import sys
import random
import csv
from pathlib import Path

# Fix encoding cho Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

VALID_ROLES = [
    "shipper", "doctor", "police", "military", "security", "student",
    "chef", "janitor", "construction", "nurse", "postman", "technician",
    "worker", "civil_guard", "normal", "unknown",
]

TARGET_SIZE = (128, 256)  # (width, height)


# ============================================================
# Kiểm tra chất lượng ảnh
# ============================================================

def is_blurry(img: np.ndarray, threshold: float = 80.0) -> bool:
    """True nếu ảnh quá mờ (Laplacian variance thấp)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def is_too_dark(img: np.ndarray, threshold: float = 25.0) -> bool:
    """True nếu ảnh quá tối (mean brightness thấp)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def is_too_bright(img: np.ndarray, threshold: float = 230.0) -> bool:
    """True nếu ảnh quá sáng (overexposed)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > threshold


def image_hash(img: np.ndarray) -> str:
    """Tính MD5 hash của ảnh để phát hiện trùng lặp."""
    small = cv2.resize(img, (16, 32))
    return hashlib.md5(small.tobytes()).hexdigest()


# ============================================================
# Xử lý từng ảnh
# ============================================================

def process_image(
    src_path: Path,
    dst_path: Path,
    seen_hashes: set,
    stats: dict,
) -> bool:
    """
    Xử lý 1 ảnh: load → check quality → resize → deduplicate → save.
    Returns True nếu ảnh được lưu.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        stats["error"] += 1
        return False

    h, w = img.shape[:2]

    # Lọc ảnh quá nhỏ
    if w < 32 or h < 32:
        stats["too_small"] += 1
        return False

    # Lọc chất lượng
    if is_blurry(img):
        stats["blurry"] += 1
        return False

    if is_too_dark(img):
        stats["too_dark"] += 1
        return False

    if is_too_bright(img):
        stats["too_bright"] += 1
        return False

    # Dedup
    h_str = image_hash(img)
    if h_str in seen_hashes:
        stats["duplicate"] += 1
        return False
    seen_hashes.add(h_str)

    # Resize
    resized = cv2.resize(img, TARGET_SIZE)

    # Lưu
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    stats["saved"] += 1
    return True


# ============================================================
# Tạo splits
# ============================================================

def create_splits(
    processed_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = [], [], []

    for role in VALID_ROLES:
        role_dir = processed_dir / role
        if not role_dir.exists():
            continue

        imgs = sorted(role_dir.glob("*.jpg"))
        random.shuffle(imgs)

        n       = len(imgs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        for i, img_path in enumerate(imgs):
            rel_path = str(img_path.relative_to(processed_dir))
            row      = {"path": rel_path, "label": role}
            if i < n_train:
                train_rows.append(row)
            elif i < n_train + n_val:
                val_rows.append(row)
            else:
                test_rows.append(row)

    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out = splits_dir / f"{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"   {name:5s}.csv : {len(rows):,} samples")

    return len(train_rows), len(val_rows), len(test_rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý RoleNet dataset"
    )
    parser.add_argument("--input",  default="data/rolenet_dataset/raw",
                        help="Thư mục ảnh raw")
    parser.add_argument("--output", default="data/rolenet_dataset/processed",
                        help="Thư mục ảnh sau xử lý")
    parser.add_argument("--splits", default="data/rolenet_dataset/splits",
                        help="Thư mục chứa CSV splits")
    parser.add_argument("--no-filter", action="store_true",
                        help="Tắt lọc chất lượng (chỉ resize)")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.exists():
        print(f"❌ Không tìm thấy thư mục input: {in_dir}")
        print("   Chạy trước: python scripts/collect_images.py --all")
        sys.exit(1)

    print(f"📂 Input  : {in_dir}")
    print(f"📁 Output : {out_dir}")
    print(f"📐 Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px\n")

    seen_hashes: set = set()
    total_stats = {
        "saved": 0, "error": 0, "too_small": 0,
        "blurry": 0, "too_dark": 0, "too_bright": 0, "duplicate": 0,
    }
    class_stats: dict[str, int] = {}

    for role in VALID_ROLES:
        role_in  = in_dir  / role
        role_out = out_dir / role

        if not role_in.exists():
            continue

        src_imgs = list(role_in.glob("*.jpg")) + list(role_in.glob("*.png"))
        if not src_imgs:
            continue

        role_stats = dict(total_stats)
        role_stats = {k: 0 for k in total_stats}
        role_out.mkdir(parents=True, exist_ok=True)

        for src in src_imgs:
            dst_name = src.stem + ".jpg"
            dst      = role_out / dst_name
            process_image(src, dst, seen_hashes, role_stats)

        saved = role_stats["saved"]
        class_stats[role] = saved
        removed = sum(v for k, v in role_stats.items() if k != "saved")

        for k, v in role_stats.items():
            total_stats[k] += v

        print(
            f"[{role:15s}] ✅ {saved:4d} saved | "
            f"⚠️ {role_stats['blurry']:3d} blur | "
            f"🔵 {role_stats['duplicate']:3d} dup | "
            f"❌ {removed - role_stats['duplicate']:3d} other"
        )

    print(f"\n{'='*55}")
    print(f"TỔNG CỘNG: {total_stats['saved']:,} ảnh hợp lệ")
    print(f"  Đã lọc : {sum(v for k,v in total_stats.items() if k != 'saved'):,} ảnh")
    print(f"  Lỗi    : {total_stats['error']:,}")
    print(f"  Mờ     : {total_stats['blurry']:,}")
    print(f"  Tối    : {total_stats['too_dark']:,}")
    print(f"  Trùng  : {total_stats['duplicate']:,}")
    print(f"  Nhỏ    : {total_stats['too_small']:,}")

    if total_stats["saved"] > 100:
        print(f"\n📋 Đang tạo splits (70/15/15)...")
        n_train, n_val, n_test = create_splits(
            out_dir, Path(args.splits)
        )
        print(f"   Tổng: train={n_train:,} | val={n_val:,} | test={n_test:,}")
        print(f"\n➡️  Bước tiếp theo: python scripts/augment_dataset.py")
    else:
        print("\n⚠️  Cần thêm ảnh trước khi tạo splits.")
        print("   Chạy: python scripts/collect_images.py --all")


if __name__ == "__main__":
    main()
