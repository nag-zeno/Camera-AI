"""
add_known_face.py — Tiện ích thêm khuôn mặt quen vào Identity Manager

Cách dùng:
    # Import từ file/thư mục ảnh:
    python scripts/add_known_face.py --name "Nguyen Van A" --images "path/to/photos/"

    # Import 1 ảnh duy nhất:
    python scripts/add_known_face.py --name "Tran Thi B" --images "photo.jpg"

    # Chụp ảnh trực tiếp từ webcam:
    python scripts/add_known_face.py --name "Le Van C" --capture

    # Xem danh sách người đã đăng ký:
    python scripts/add_known_face.py --list

    # Xóa 1 người:
    python scripts/add_known_face.py --delete "Nguyen Van A"

Cấu trúc thư mục sau khi thêm:
    data/known_faces/
    ├── Nguyen_Van_A/
    │   ├── photo_1.jpg
    │   ├── photo_2.jpg
    │   └── photo_3.jpg
    ├── Tran_Thi_B/
    │   └── photo_1.jpg
    └── Le_Van_C/
        ├── capture_001.jpg
        └── capture_002.jpg

Lưu ý:
    - Cần ít nhất 2-3 ảnh/người để nhận dạng chính xác hơn
    - Ảnh cần rõ mặt, đủ ánh sáng, không bị che khuất
    - Độ phân giải tối thiểu khuyến nghị: 100×100 px cho phần mặt
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# Hằng số
# ============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "known_faces"
SUPPORTED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FACES_PER_PERSON = 20  # Tối đa 20 ảnh/người


def _safe_name(name: str) -> str:
    """Chuyển tên người thành tên thư mục an toàn."""
    safe = re.sub(r"[^\w\s-]", "", name.strip())
    safe = re.sub(r"[\s]+", "_", safe)
    return safe[:50]  # Giới hạn độ dài


def _count_all_faces() -> dict[str, int]:
    """Đếm số ảnh của mỗi người đã đăng ký."""
    result = {}
    if not KNOWN_FACES_DIR.exists():
        return result
    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if person_dir.is_dir():
            imgs = [f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
            result[person_dir.name] = len(imgs)
    return result


# ============================================================
# Các chức năng chính
# ============================================================

def list_known_faces():
    """Hiển thị danh sách người đã đăng ký."""
    faces = _count_all_faces()
    if not faces:
        print("\n📂 Chưa có ai được đăng ký trong hệ thống.")
        print(f"   Thư mục: {KNOWN_FACES_DIR}")
        print("\n   Để thêm người, dùng:")
        print("   python scripts/add_known_face.py --name <Tên> --images <ảnh/thư mục>")
        return

    print(f"\n👥 Danh sách người đã đăng ký ({len(faces)} người):")
    print("─" * 50)
    total_imgs = 0
    for folder_name, count in faces.items():
        status = "✅" if count >= 3 else "⚠️ "
        note   = "" if count >= 3 else f"  ← khuyến nghị thêm ảnh (hiện có {count})"
        print(f"  {status} {folder_name:<30s} {count:2d} ảnh{note}")
        total_imgs += count
    print("─" * 50)
    print(f"  Tổng: {total_imgs} ảnh cho {len(faces)} người")
    print(f"\n  ⚠️  Người có < 3 ảnh có thể nhận dạng kém chính xác")


def add_from_files(name: str, source: str, copy_all_from_dir: bool = False):
    """
    Thêm ảnh từ file hoặc thư mục.

    Args:
        name: Tên người cần đăng ký
        source: Đường dẫn file ảnh hoặc thư mục chứa ảnh
        copy_all_from_dir: Nếu True và source là thư mục, copy toàn bộ ảnh
    """
    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ Không tìm thấy: {source}")
        sys.exit(1)

    # Thu thập danh sách file ảnh cần thêm
    image_files = []
    if source_path.is_file():
        if source_path.suffix.lower() in SUPPORTED_EXT:
            image_files = [source_path]
        else:
            print(f"❌ Định dạng không hỗ trợ: {source_path.suffix}")
            print(f"   Hỗ trợ: {', '.join(SUPPORTED_EXT)}")
            sys.exit(1)
    elif source_path.is_dir():
        image_files = [
            f for f in sorted(source_path.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
        ]
        if not image_files:
            print(f"❌ Không tìm thấy ảnh trong thư mục: {source_path}")
            sys.exit(1)
        print(f"   Tìm thấy {len(image_files)} ảnh trong {source_path}")

    # Tạo thư mục người dùng
    safe = _safe_name(name)
    person_dir = KNOWN_FACES_DIR / safe
    person_dir.mkdir(parents=True, exist_ok=True)

    # Đếm số ảnh hiện có để không đặt trùng tên
    existing = [f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
    start_idx = len(existing) + 1

    # Kiểm tra giới hạn
    if start_idx - 1 >= MAX_FACES_PER_PERSON:
        print(f"⚠️  {name} đã có {start_idx-1} ảnh (giới hạn {MAX_FACES_PER_PERSON}).")
        print("   Xóa ảnh cũ trước khi thêm mới.")
        return

    added = 0
    errors = 0
    for i, img_path in enumerate(image_files):
        if start_idx + i > MAX_FACES_PER_PERSON:
            print(f"   ⚠️  Đã đạt giới hạn {MAX_FACES_PER_PERSON} ảnh, dừng lại.")
            break

        dst_name = f"photo_{start_idx + i:03d}{img_path.suffix.lower()}"
        dst_path = person_dir / dst_name

        try:
            # Xác minh ảnh hợp lệ (thử đọc bằng cv2 nếu có)
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ⚠️  Bỏ qua (không đọc được ảnh): {img_path.name}")
                    errors += 1
                    continue
                h, w = img.shape[:2]
                if h < 50 or w < 50:
                    print(f"   ⚠️  Bỏ qua (ảnh quá nhỏ {w}×{h}px): {img_path.name}")
                    errors += 1
                    continue
            except ImportError:
                pass  # Nếu không có cv2, cứ copy

            shutil.copy2(str(img_path), str(dst_path))
            added += 1
            print(f"   ✅ {img_path.name} → {dst_name}")

        except Exception as e:
            print(f"   ❌ Lỗi copy {img_path.name}: {e}")
            errors += 1

    # Báo cáo
    total_now = len([f for f in person_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXT])
    print(f"\n✅ Đã thêm {added} ảnh cho '{name}' (tổng: {total_now} ảnh)")
    if errors:
        print(f"   ⚠️  {errors} ảnh bị bỏ qua do lỗi")

    # Khuyến nghị
    if total_now < 3:
        print(f"\n⚠️  Khuyến nghị: Thêm ít nhất {3 - total_now} ảnh nữa để nhận dạng chính xác hơn.")
    elif total_now >= 3:
        print(f"\n✅ Đủ ảnh để nhận dạng. Pipeline sẽ nhận dạng '{name}' sau khi restart.")
    print(f"   Thư mục: {person_dir}")


def capture_from_webcam(name: str, num_frames: int = 5):
    """
    Chụp ảnh trực tiếp từ webcam để đăng ký.

    Args:
        name: Tên người cần đăng ký
        num_frames: Số ảnh cần chụp (mặc định: 5)
    """
    try:
        import cv2
    except ImportError:
        print("❌ Cần cài opencv-python: pip install opencv-python")
        sys.exit(1)

    safe = _safe_name(name)
    person_dir = KNOWN_FACES_DIR / safe
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
    start_idx = len(existing) + 1

    print(f"\n📷 Chụp ảnh webcam cho '{name}'")
    print(f"   Cần chụp: {num_frames} ảnh")
    print(f"   Hướng dẫn: Nhấn SPACE để chụp, 'q' để thoát")
    print(f"   Đứng trước camera, đảm bảo mặt rõ và đủ ánh sáng\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam! Kiểm tra kết nối.")
        sys.exit(1)

    captured = 0
    frame_count = 0

    while captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lỗi đọc frame từ webcam!")
            break

        frame_count += 1

        # Hiển thị hướng dẫn trên frame
        display = frame.copy()
        cv2.putText(
            display, f"Chup: {captured}/{num_frames} | SPACE=chup | q=thoat",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            display, f"Nguoi: {name}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        cv2.imshow(f"Chup anh - {name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == 32:  # SPACE → chụp ảnh
            img_path = person_dir / f"capture_{start_idx + captured:03d}.jpg"
            cv2.imwrite(str(img_path), frame)
            captured += 1
            print(f"   📸 Đã chụp ảnh {captured}/{num_frames}: {img_path.name}")
            # Flash effect
            cv2.rectangle(display, (0, 0), (frame.shape[1], frame.shape[0]),
                          (255, 255, 255), 10)
            cv2.imshow(f"Chup anh - {name}", display)
            cv2.waitKey(200)

        elif key == ord('q') or key == 27:  # q hoặc ESC → thoát
            print(f"\n⚠️  Đã thoát sớm — chỉ chụp được {captured}/{num_frames} ảnh")
            break

    cap.release()
    cv2.destroyAllWindows()

    total_now = len([f for f in person_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXT])
    if captured > 0:
        print(f"\n✅ Đã thêm {captured} ảnh webcam cho '{name}' (tổng: {total_now} ảnh)")
        if total_now >= 3:
            print(f"   Pipeline sẽ nhận dạng '{name}' sau khi restart.")
        else:
            print(f"   ⚠️  Cần thêm {3 - total_now} ảnh nữa để đạt độ chính xác tốt.")
    else:
        print("❌ Không chụp được ảnh nào!")
        if not existing:
            person_dir.rmdir()  # Xóa thư mục rỗng


def delete_person(name: str):
    """Xóa dữ liệu khuôn mặt của 1 người."""
    safe = _safe_name(name)
    person_dir = KNOWN_FACES_DIR / safe

    if not person_dir.exists():
        # Thử tìm theo tên gốc không safe
        found = None
        for d in KNOWN_FACES_DIR.iterdir():
            if d.is_dir() and d.name.lower() == safe.lower():
                found = d
                break
        if not found:
            print(f"❌ Không tìm thấy '{name}' trong hệ thống.")
            print("   Dùng --list để xem danh sách.")
            return
        person_dir = found

    imgs = [f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
    confirm = input(f"⚠️  Xóa '{person_dir.name}' ({len(imgs)} ảnh)? [y/N]: ").strip().lower()
    if confirm == 'y':
        shutil.rmtree(str(person_dir))
        print(f"✅ Đã xóa '{person_dir.name}' ({len(imgs)} ảnh)")
    else:
        print("   Hủy bỏ.")


def show_summary():
    """Hiển thị tóm tắt hướng dẫn."""
    print("""
╔══════════════════════════════════════════════════════════╗
║          Identity Manager — Quản lý khuôn mặt            ║
╚══════════════════════════════════════════════════════════╝

Các lệnh:
  --list                    Xem danh sách người đã đăng ký
  --name <tên> --images <ảnh/thư mục>  Thêm từ file/thư mục
  --name <tên> --capture    Chụp từ webcam (nhấn SPACE)
  --delete <tên>            Xóa người khỏi hệ thống

Ví dụ:
  python scripts/add_known_face.py --list
  python scripts/add_known_face.py --name "Nguyen Van A" --images "photos/nguyenvana/"
  python scripts/add_known_face.py --name "Le Thi B" --images "lethib.jpg"
  python scripts/add_known_face.py --name "Tran Van C" --capture --num-frames 10

Lưu ý quan trọng:
  ✅ Cần 3+ ảnh/người để nhận dạng chính xác
  ✅ Ảnh phải rõ mặt, không bị che, đủ ánh sáng
  ✅ Sau khi thêm, RESTART server (Ctrl+C → python app.py)
""")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quản lý dữ liệu khuôn mặt cho Identity Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python scripts/add_known_face.py --list
  python scripts/add_known_face.py --name "Nguyen Van A" --images "D:/photos/nguyenvana/"
  python scripts/add_known_face.py --name "Le Van B" --capture
  python scripts/add_known_face.py --delete "Nguyen Van A"
        """
    )

    parser.add_argument("--list",   action="store_true", help="Hiển thị danh sách người đã đăng ký")
    parser.add_argument("--name",   type=str, help="Tên người cần thêm/xóa")
    parser.add_argument("--images", type=str, help="Đường dẫn ảnh hoặc thư mục chứa ảnh")
    parser.add_argument("--capture",action="store_true", help="Chụp ảnh từ webcam")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Số ảnh cần chụp từ webcam (mặc định: 5)")
    parser.add_argument("--delete", type=str, metavar="NAME", help="Xóa người khỏi hệ thống")
    args = parser.parse_args()

    # Tạo thư mục nếu chưa có
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_known_faces()

    elif args.delete:
        delete_person(args.delete)

    elif args.name and args.images:
        print(f"\n👤 Thêm khuôn mặt cho: '{args.name}'")
        print(f"   Nguồn ảnh: {args.images}")
        add_from_files(args.name, args.images)

    elif args.name and args.capture:
        capture_from_webcam(args.name, num_frames=args.num_frames)

    elif args.name:
        print("❌ Cần chỉ định nguồn ảnh: --images <path> hoặc --capture")
        show_summary()

    else:
        show_summary()
        list_known_faces()


if __name__ == "__main__":
    main()
