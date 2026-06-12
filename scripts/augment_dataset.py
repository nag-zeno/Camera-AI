"""
augment_dataset.py — Augmentation pipeline cho RoleNet dataset

Tăng số lượng ảnh ×N bằng các phép biến đổi:
    - Flip ngang
    - Thay đổi độ sáng/tương phản
    - Xoay nhẹ (±10°)
    - Color jitter (thay đổi màu sắc)
    - Gaussian blur nhẹ
    - Random crop 90%
    - Thêm nhiễu Gaussian
    - Biến đổi HSV

Cách chạy:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --factor 8 --input data/rolenet_dataset/processed
"""
import argparse
import random
import sys
from pathlib import Path

# Fix encoding cho Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False


def get_augmentation_pipeline() -> list:
    """Trả về list các phép augmentation. Mỗi ảnh áp dụng 1 phép."""

    if HAS_ALB:
        # Dùng albumentations (mạnh hơn)
        transforms = [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.Rotate(limit=12, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10, 40), p=1.0),
            A.RandomCrop(height=230, width=115, p=1.0),
            A.CLAHE(p=1.0),
        ]
        return transforms
    else:
        # Fallback: dùng OpenCV thuần
        return [
            "flip", "bright+", "bright-", "contrast",
            "rotate+", "rotate-", "blur", "noise",
        ]


def apply_augmentation_alb(img: np.ndarray, transform) -> np.ndarray:
    """Áp dụng albumentations transform."""
    augmented = transform(image=img)["image"]
    return cv2.resize(augmented, (128, 256))  # Đảm bảo kích thước chuẩn


def apply_augmentation_cv(img: np.ndarray, aug_type: str) -> np.ndarray:
    """Áp dụng augmentation bằng OpenCV thuần."""
    result = img.copy()

    if aug_type == "flip":
        result = cv2.flip(img, 1)

    elif aug_type == "bright+":
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1.1, 1.4), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif aug_type == "bright-":
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 0.85), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif aug_type == "contrast":
        alpha  = random.uniform(0.7, 1.5)
        result = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    elif aug_type == "rotate+":
        M = cv2.getRotationMatrix2D((64, 128), random.uniform(5, 12), 1.0)
        result = cv2.warpAffine(img, M, (128, 256))

    elif aug_type == "rotate-":
        M = cv2.getRotationMatrix2D((64, 128), -random.uniform(5, 12), 1.0)
        result = cv2.warpAffine(img, M, (128, 256))

    elif aug_type == "blur":
        k      = random.choice([3, 5])
        result = cv2.GaussianBlur(img, (k, k), 0)

    elif aug_type == "noise":
        noise  = np.random.normal(0, random.uniform(8, 20), img.shape).astype(np.int16)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.resize(result, (128, 256))


def augment_class(
    role: str,
    in_dir: Path,
    out_dir: Path,
    factor: int,
    transforms,
    use_alb: bool,
) -> int:
    """Augment tất cả ảnh của 1 class × factor lần."""
    src_dir = in_dir / role
    dst_dir = out_dir / role
    if not src_dir.exists():
        return 0

    src_imgs = list(src_dir.glob("*.jpg"))
    if not src_imgs:
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy ảnh gốc
    saved = 0
    for img_path in src_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dst_path = dst_dir / img_path.name
        cv2.imwrite(str(dst_path), img)
        saved += 1

    # Sinh augmented
    for i, img_path in enumerate(src_imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Chọn ngẫu nhiên `factor-1` augmentations
        selected = random.sample(transforms, min(factor - 1, len(transforms)))

        for j, t in enumerate(selected):
            try:
                if use_alb:
                    aug_img = apply_augmentation_alb(img, t)
                else:
                    aug_img = apply_augmentation_cv(img, t)

                stem     = img_path.stem
                dst_path = dst_dir / f"{stem}_aug{j}.jpg"
                cv2.imwrite(str(dst_path), aug_img)
                saved += 1

            except Exception as e:
                pass  # Bỏ qua lỗi augmentation

    return saved


VALID_ROLES = [
    "shipper", "doctor", "police", "military", "security", "student",
    "chef", "janitor", "construction", "nurse", "postman", "technician",
    "worker", "civil_guard", "normal", "unknown",
]


def main():
    parser = argparse.ArgumentParser(
        description="Augment RoleNet dataset"
    )
    parser.add_argument("--input",  default="data/rolenet_dataset/processed",
                        help="Thư mục ảnh đã preprocess")
    parser.add_argument("--output", default="data/rolenet_dataset/augmented",
                        help="Thư mục output")
    parser.add_argument("--factor", type=int, default=8,
                        help="Số lượng augmentation per ảnh (mặc định: 8)")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.exists():
        print(f"❌ Không tìm thấy: {in_dir}")
        print("   Chạy trước: python scripts/preprocess_dataset.py")
        sys.exit(1)

    use_alb    = HAS_ALB
    transforms = get_augmentation_pipeline()

    print(f"🎨 Augmentation pipeline: {'Albumentations' if use_alb else 'OpenCV (fallback)'}")
    print(f"📂 Input  : {in_dir}")
    print(f"📁 Output : {out_dir}")
    print(f"✖️  Factor : ×{args.factor}\n")

    if not use_alb:
        print("💡 Tip: pip install albumentations → augmentation mạnh hơn\n")

    total_saved = 0
    for role in VALID_ROLES:
        src = in_dir / role
        if not src.exists():
            continue

        n_orig = len(list(src.glob("*.jpg")))
        if n_orig == 0:
            continue

        n_saved = augment_class(role, in_dir, out_dir, args.factor, transforms, use_alb)
        total_saved += n_saved
        ratio = n_saved / n_orig if n_orig > 0 else 0
        print(f"[{role:15s}] {n_orig:4d} → {n_saved:6,} ảnh (×{ratio:.1f})")

    print(f"\n✅ Augmentation xong! Tổng: {total_saved:,} ảnh")
    print(f"\n➡️  Bước tiếp theo: Train RoleNet trên Google Colab")
    print("   Hoặc: python scripts/train_rolenet.py (nếu có GPU)")


if __name__ == "__main__":
    main()
