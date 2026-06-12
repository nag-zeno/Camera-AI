"""
collect_images.py — Thu thập ảnh tự động cho RoleNet Dataset

Dùng icrawler để tải ảnh từ Bing/Google theo từ khóa.

Cài đặt:
    pip install icrawler

Cách chạy:
    # Tải 1 class:
    python scripts/collect_images.py --role shipper --count 500

    # Tải tất cả class cùng lúc:
    python scripts/collect_images.py --all --count 400

    # Dùng engine google:
    python scripts/collect_images.py --role police --count 300 --engine google
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Fix encoding cho Windows terminal (cp1258 không hỗ trợ emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Kiểm tra icrawler
try:
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    HAS_ICRAWLER = True
except ImportError:
    HAS_ICRAWLER = False

# ============================================================
# Từ khóa tìm kiếm cho từng vai trò (Tiếng Việt + Tiếng Anh)
# ============================================================
SEARCH_QUERIES: dict[str, list[str]] = {
    "shipper": [
        "shipper grab việt nam",
        "tài xế giao hàng shopee express",
        "shipper ghn giao hàng nhanh",
        "delivery rider vietnam motorbike",
        "shipper lalamove vietnam",
    ],
    "doctor": [
        "bác sĩ áo blouse trắng việt nam",
        "doctor white coat vietnam hospital",
        "bác sĩ phòng khám bệnh viện",
        "physician stethoscope asian",
    ],
    "police": [
        "công an nhân dân việt nam đồng phục",
        "cảnh sát giao thông việt nam",
        "công an phường xã tuần tra",
        "vietnam police officer uniform",
    ],
    "military": [
        "bộ đội việt nam quân phục",
        "quân nhân nhân dân việt nam",
        "vietnam army soldier uniform",
        "bộ đội biên phòng việt nam",
    ],
    "security": [
        "bảo vệ tòa nhà việt nam đồng phục",
        "nhân viên bảo vệ security guard vietnam",
        "bảo vệ văn phòng siêu thị",
        "security guard asian uniform",
    ],
    "student": [
        "học sinh đồng phục trắng xanh việt nam",
        "sinh viên đại học việt nam",
        "học sinh trung học phổ thông việt nam",
        "vietnam school student uniform",
    ],
    "chef": [
        "đầu bếp mũ trắng nhà hàng việt nam",
        "bếp trưởng chef uniform vietnam restaurant",
        "cook chef white hat apron asian",
        "nhân viên bếp nấu ăn",
    ],
    "janitor": [
        "nhân viên vệ sinh đô thị áo cam vàng",
        "công nhân vệ sinh đường phố việt nam",
        "cleaning staff orange vest vietnam",
        "sanitation worker asia street",
    ],
    "construction": [
        "công nhân xây dựng mũ bảo hộ việt nam",
        "thợ hồ xây dựng hard hat vietnam",
        "construction worker safety helmet asia",
        "nhân công lao động xây dựng",
    ],
    "nurse": [
        "y tá điều dưỡng áo scrubs bệnh viện việt nam",
        "nurse scrubs blue white vietnam hospital",
        "điều dưỡng viên chăm sóc bệnh nhân",
        "nursing staff asian medical",
    ],
    "postman": [
        "bưu tá vnpost áo vàng việt nam",
        "nhân viên bưu điện giao thư",
        "vietnam post office mailman yellow",
        "bưu tá xe máy giao hàng",
    ],
    "technician": [
        "kỹ thuật viên sửa chữa thiết bị việt nam",
        "nhân viên kỹ thuật it sơ mi",
        "technician repair equipment asian",
        "IT support engineer office",
    ],
    "worker": [
        "công nhân nhà máy xưởng việt nam",
        "công nhân đồng phục xưởng sản xuất",
        "factory worker uniform vietnam",
        "nhân công lao động phổ thông",
    ],
    "civil_guard": [
        "dân phòng áo xanh lam băng tay đỏ việt nam",
        "dân quân tự vệ phường xã việt nam",
        "vietnam civil defense local militia",
        "dân phòng tuần tra khu dân cư",
    ],
    "normal": [
        "người đi đường thành phố hà nội saigon",
        "đám đông người việt nam đường phố",
        "vietnam pedestrian casual clothes street",
        "người dân bình thường quần áo thường ngày",
        "vietnamese people walking street",
    ],
    "unknown": [
        "người đeo khẩu trang tối che mặt",
        "người đứng góc tối camera an ninh",
        "suspicious person dark hoodie",
        "person obscured face camera",
        "back of person walking away camera",
    ],
}

# Mục tiêu số lượng ảnh tối thiểu cho mỗi class
CLASS_TARGETS = {
    "shipper": 800, "doctor": 600, "police": 600, "military": 600,
    "security": 600, "student": 600, "chef": 400, "janitor": 400,
    "construction": 400, "nurse": 400, "postman": 300, "technician": 300,
    "worker": 400, "civil_guard": 300, "normal": 1000, "unknown": 500,
}

BASE_DIR = Path("data/rolenet_dataset/raw")


def collect_class(role: str, count: int, engine: str = "bing"):
    """Thu thập ảnh cho 1 class."""
    queries = SEARCH_QUERIES.get(role)
    if not queries:
        print(f"❌ Không tìm thấy queries cho role: {role}")
        return 0

    output_dir = BASE_DIR / role
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đếm ảnh đã có
    existing = len(list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png")))
    needed   = max(0, count - existing)

    if needed == 0:
        print(f"✅ [{role:15s}] Đã đủ {existing} ảnh (target: {count})")
        return existing

    print(f"🔍 [{role:15s}] Đang tải {needed} ảnh (đã có: {existing} / cần: {count})...")

    # Chia đều số query
    per_query = max(1, needed // len(queries)) + 5  # +5 buffer

    total_downloaded = existing
    for i, query in enumerate(queries):
        if total_downloaded >= count:
            break

        try:
            if engine == "google":
                crawler = GoogleImageCrawler(
                    storage={"root_dir": str(output_dir)},
                    log_level=50,  # Tắt log
                )
            else:
                crawler = BingImageCrawler(
                    storage={"root_dir": str(output_dir)},
                    log_level=50,
                )

            crawler.crawl(
                keyword   = query,
                max_num   = per_query,
                file_idx_offset = "auto",
            )

            # Đếm lại
            current = len(list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png")))
            print(f"   Query {i+1}/{len(queries)}: '{query[:50]}' → {current} ảnh")
            total_downloaded = current

            time.sleep(1)  # Tránh bị block

        except Exception as e:
            print(f"   ⚠️ Lỗi query '{query[:40]}': {e}")
            continue

    final_count = len(list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png")))
    print(f"   ✅ [{role:15s}] Tổng cộng: {final_count} ảnh\n")
    return final_count


def print_dataset_stats():
    """In thống kê dataset hiện tại."""
    print("\n📊 Thống kê dataset hiện tại:")
    print(f"{'Vai Trò':20s} {'Có':>8s} {'Cần':>8s} {'Tỷ lệ':>8s}")
    print("-" * 48)
    total_have = 0
    total_need = 0
    for role, target in CLASS_TARGETS.items():
        role_dir = BASE_DIR / role
        n = len(list(role_dir.glob("*.jpg")) + list(role_dir.glob("*.png"))) if role_dir.exists() else 0
        pct = min(100.0, n / target * 100)
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"{role:20s} {n:>8,} {target:>8,} {bar} {pct:5.1f}%")
        total_have += n
        total_need += target
    print("-" * 48)
    print(f"{'TỔNG':20s} {total_have:>8,} {total_need:>8,}")


def main():
    if not HAS_ICRAWLER:
        print("❌ icrawler chưa cài.")
        print("   Chạy: pip install icrawler")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Thu thập ảnh cho RoleNet Dataset")
    parser.add_argument("--role",   type=str, default=None,
                        help="Tên role cần thu thập (ví dụ: shipper)")
    parser.add_argument("--all",    action="store_true",
                        help="Thu thập tất cả 16 roles")
    parser.add_argument("--count",  type=int, default=None,
                        help="Số ảnh cần tải (mặc định: target của class)")
    parser.add_argument("--engine", type=str, default="bing",
                        choices=["bing", "google"],
                        help="Search engine (mặc định: bing)")
    parser.add_argument("--stats",  action="store_true",
                        help="Chỉ in thống kê, không tải ảnh")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats()
        return

    if not args.role and not args.all:
        parser.print_help()
        print("\n💡 Ví dụ:")
        print("   python scripts/collect_images.py --stats")
        print("   python scripts/collect_images.py --role shipper --count 500")
        print("   python scripts/collect_images.py --all --count 300")
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        roles = list(SEARCH_QUERIES.keys())
        print(f"🚀 Thu thập ảnh cho TẤT CẢ {len(roles)} class...\n")
        for role in roles:
            target = args.count or CLASS_TARGETS.get(role, 300)
            collect_class(role, target, args.engine)
    else:
        role   = args.role
        target = args.count or CLASS_TARGETS.get(role, 300)
        collect_class(role, target, args.engine)

    print_dataset_stats()
    print(f"\n➡️  Bước tiếp theo: python scripts/preprocess_dataset.py")


if __name__ == "__main__":
    main()
