"""
extract_crops_from_video.py — Trích xuất person crops từ video

Dùng YOLOv8 để detect người trong video → cắt crop → lưu vào dataset.
Đây là cách thu thập data từ:
  - Video tin tức VN (YouTube)
  - Footage camera an ninh
  - Video quay tại hiện trường

Cách chạy:
    python scripts/extract_crops_from_video.py \\
        --video footage/shipper.mp4 \\
        --role shipper \\
        --every-n 15

    # Tự động detect và hỏi role:
    python scripts/extract_crops_from_video.py --video footage/mixed.mp4 --interactive
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics chưa cài. Chạy: pip install ultralytics")
    sys.exit(1)

VALID_ROLES = [
    "shipper", "doctor", "police", "military", "security", "student",
    "chef", "janitor", "construction", "nurse", "postman", "technician",
    "worker", "civil_guard", "normal", "unknown",
]

BASE_DIR = Path("data/rolenet_dataset/raw")


def extract_crops(
    video_path: str,
    role: str,
    every_n: int = 15,
    min_conf: float = 0.6,
    min_size: int = 64,
    model_path: str = "yolov8n.pt",
    max_crops: int = 2000,
    interactive: bool = False,
):
    out_dir = BASE_DIR / role
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    print(f"\n📹 Video  : {video_path}")
    print(f"🏷️  Role   : {role}")
    print(f"📁 Output : {out_dir}")
    print(f"📸 Đã có  : {existing} ảnh\n")

    model  = YOLO(model_path)
    cap    = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Tổng frames: {total_frames:,} | FPS: {fps:.1f} | Mỗi {every_n} frame lấy 1")

    saved    = 0
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % every_n != 0:
            continue

        if saved >= max_crops:
            print(f"\n⏹  Đã đạt max_crops={max_crops}. Dừng.")
            break

        results = model(frame, classes=[0], conf=min_conf, verbose=False)
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1

            # Lọc crop quá nhỏ
            if w < min_size or h < min_size:
                continue

            # Thêm padding nhỏ
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            H, W  = frame.shape[:2]
            x1c = max(0, x1 - pad_x)
            y1c = max(0, y1 - pad_y)
            x2c = min(W, x2 + pad_x)
            y2c = min(H, y2 + pad_y)

            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            # Resize về 128×256
            crop_r = cv2.resize(crop, (128, 256))

            # Interactive mode: hiện crop và hỏi
            if interactive:
                cv2.imshow("Crop — Nhấn Y=lưu, N=bỏ, Q=thoát", crop_r)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                elif key != ord("y"):
                    continue

            filename = out_dir / f"video_{frame_no:06d}_{saved:04d}.jpg"
            cv2.imwrite(str(filename), crop_r)
            saved += 1

        if frame_no % 300 == 0:
            pct = frame_no / total_frames * 100
            print(f"   Frame {frame_no:,}/{total_frames:,} ({pct:.1f}%) | Saved: {saved}")

    cap.release()
    if interactive:
        cv2.destroyAllWindows()

    total = existing + saved
    print(f"\n✅ Trích xuất xong: +{saved} crops | Tổng {role}: {total} ảnh")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất person crops từ video cho RoleNet dataset"
    )
    parser.add_argument("--video",       required=True, help="Đường dẫn file video")
    parser.add_argument("--role",        default=None,
                        help=f"Vai trò: {', '.join(VALID_ROLES)}")
    parser.add_argument("--every-n",     type=int, default=15,
                        help="Lấy 1 frame mỗi N frame (mặc định: 15)")
    parser.add_argument("--min-conf",    type=float, default=0.60,
                        help="Độ tin cậy tối thiểu của YOLO (mặc định: 0.6)")
    parser.add_argument("--min-size",    type=int, default=64,
                        help="Kích thước crop tối thiểu px (mặc định: 64)")
    parser.add_argument("--max-crops",   type=int, default=2000,
                        help="Số crop tối đa (mặc định: 2000)")
    parser.add_argument("--interactive", action="store_true",
                        help="Duyệt từng crop và chọn Y/N")
    parser.add_argument("--model",       default="yolov8n.pt",
                        help="YOLO model (mặc định: yolov8n.pt)")
    args = parser.parse_args()

    # Kiểm tra file video
    if not Path(args.video).exists():
        print(f"❌ Không tìm thấy file video: {args.video}")
        sys.exit(1)

    # Hỏi role nếu chưa cung cấp
    role = args.role
    if not role:
        print("Các role hợp lệ:")
        for i, r in enumerate(VALID_ROLES):
            print(f"  [{i:2d}] {r}")
        idx = int(input("Nhập số thứ tự role: ").strip())
        role = VALID_ROLES[idx]

    if role not in VALID_ROLES:
        print(f"❌ Role không hợp lệ: {role}")
        sys.exit(1)

    extract_crops(
        video_path  = args.video,
        role        = role,
        every_n     = args.every_n,
        min_conf    = args.min_conf,
        min_size    = args.min_size,
        model_path  = args.model,
        max_crops   = args.max_crops,
        interactive = args.interactive,
    )
    print(f"\n➡️  Bước tiếp theo: python scripts/preprocess_dataset.py")


if __name__ == "__main__":
    main()
