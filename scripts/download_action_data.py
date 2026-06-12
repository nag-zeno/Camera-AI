"""
download_action_data.py — Thu thập keypoints từ internet cho ActionNet

Nguồn data:
  1. UCF-101 (ưu tiên): Dataset chuẩn, free, 101 action classes
     URL: https://www.crcv.ucf.edu/data/UCF101.php
     → Chỉ download các nhóm liên quan đến 8 action của ta

  2. YouTube (backup): Dùng yt-dlp để tải video clip từ YouTube

Pipeline:
  Video → cắt thành clip 2 giây (30 frame @15fps) → MediaPipe Pose → CSV

Kết quả:
  data/actionnet_dataset/keypoints/{action}_{idx:04d}.csv
  Mỗi file: 30 dòng × 132 cột (33 keypoints × 4 features)

Dùng:
  python scripts/download_action_data.py --source ucf101
  python scripts/download_action_data.py --source youtube
  python scripts/download_action_data.py --source both
  python scripts/download_action_data.py --source ucf101 --clips 100
"""

import sys
import os
import cv2
import numpy as np
import csv
import time
import logging
import argparse
import urllib.request
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ACTION_CONFIG

# ============================================================
# Config
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_FRAMES = ACTION_CONFIG["window_frames"]   # 30
INPUT_DIM     = ACTION_CONFIG["input_dim"]        # 132
TARGET_FPS    = 15                                # FPS đích
STRIDE_SEC    = 1.0                              # Lấy clip cách nhau 1 giây

OUTPUT_DIR = Path("data/actionnet_dataset/keypoints")
VIDEO_CACHE= Path("data/actionnet_dataset/videos_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CACHE.mkdir(parents=True, exist_ok=True)

# ============================================================
# UCF-101 — Mapping action của ta → category UCF-101
# ============================================================
# UCF-101 có 101 class, ta chọn những class liên quan nhất
UCF101_MAP = {
    "standing"    : ["BalanceBeam"],                        # người đứng cân bằng
    "walking"     : ["WalkingWithDog", "Lunges"],            # đi bộ
    "running"     : ["SoccerJuggling", "Basketball", "Skiing"],  # chạy
    "falling"     : ["FallFloor", "SkateBoarding"],          # ngã
    "climbing"    : ["RockClimbingIndoor", "CliffDiving"],   # leo trèo
    "fighting"    : ["BoxingPunchingBag", "BoxingSpeedBag", "MixedMartialArts"],  # đánh
    "raising_hand": ["HandstandWalking", "HandstandPushups"],  # giơ tay
    "gathering"   : ["Basketball", "SoccerJuggling"],        # tụ tập
}

UCF101_BASE_URL = "https://www.crcv.ucf.edu/data/UCF101"

# ============================================================
# YouTube query cho mỗi action (backup)
# ============================================================
YOUTUBE_QUERIES = {
    "standing"    : ["person standing still crowd surveillance camera"],
    "walking"     : ["people walking street cctv footage"],
    "running"     : ["person running fast outdoor surveillance"],
    "falling"     : ["person falling slip accident safety"],
    "climbing"    : ["person climbing ladder fence"],
    "fighting"    : ["fighting martial arts training full body"],
    "raising_hand": ["person raising hand waving crowd"],
    "gathering"   : ["group people gathering standing outdoor"],
}

# ============================================================
# MediaPipe Init
# ============================================================
def init_mediapipe():
    try:
        import mediapipe as mp
        version_parts = tuple(int(x) for x in mp.__version__.split(".")[:2])

        if version_parts < (0, 10):
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            return ("legacy", pose)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            # Model path
            model_path = Path("models/pose_landmarker.task")
            if not model_path.exists():
                _download_mp_model(model_path)

            base_opts  = mp_python.BaseOptions(model_asset_path=str(model_path))
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
            log.info(f"MediaPipe PoseLandmarker (Tasks API v{mp.__version__}) ready.")
            return ("tasks", landmarker)
    except Exception as e:
        log.error(f"MediaPipe init thất bại: {e}")
        sys.exit(1)


def _download_mp_model(model_path: Path):
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "pose_landmarker/pose_landmarker_lite/float16/latest/"
           "pose_landmarker_lite.task")
    log.info(f"Downloading MediaPipe model...")
    try:
        urllib.request.urlretrieve(url, str(model_path))
        log.info(f"Downloaded: {model_path}")
    except Exception as e:
        log.error(f"Download MediaPipe model thất bại: {e}")
        sys.exit(1)


def extract_kp(frame_rgb, runner, api):
    """Trích xuất keypoint vector (132,) từ 1 frame RGB."""
    try:
        if api == "legacy":
            results = runner.process(frame_rgb)
            if not results.pose_landmarks:
                return None
            lm = results.pose_landmarks.landmark
            return np.array(
                [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm],
                dtype=np.float32,
            ).flatten()
        else:
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = runner.detect(mp_img)
            if not res.pose_landmarks:
                return None
            lm = res.pose_landmarks[0]
            return np.array(
                [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm],
                dtype=np.float32,
            ).flatten()
    except Exception:
        return None


# ============================================================
# CSV Save
# ============================================================
HEADER = []
for i in range(33):
    for feat in ["x", "y", "z", "vis"]:
        HEADER.append(f"kp_{i}_{feat}")
HEADER.append("action")


def save_clip_csv(frames_kp: list, action: str, clip_idx: int) -> Path:
    out_path = OUTPUT_DIR / f"{action}_{clip_idx:04d}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for kp in frames_kp:
            writer.writerow(list(kp) + [action])
    return out_path


# ============================================================
# Extract clips from video
# ============================================================
def extract_clips_from_video(
    video_path: str,
    action: str,
    mp_runner,
    mp_api: str,
    start_clip_idx: int,
    max_clips: int = 20,
    min_detect_ratio: float = 0.6,
) -> int:
    """
    Đọc video, cắt thành clip 30 frame, lấy keypoints.
    Trả về số clip đã lưu thành công.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"  Không mở được: {video_path}")
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stride để lấy frame về ~TARGET_FPS
    frame_stride = max(1, round(orig_fps / TARGET_FPS))

    clip_idx    = start_clip_idx
    saved       = 0
    buffer      = []
    frame_count = 0

    while saved < max_clips:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Chỉ lấy frame theo stride
        if frame_count % frame_stride != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp = extract_kp(frame_rgb, mp_runner, mp_api)

        if kp is not None:
            buffer.append(kp)
        else:
            buffer.append(np.zeros(INPUT_DIM, dtype=np.float32))

        # Đủ 30 frame → lưu (nếu detect đủ)
        if len(buffer) >= WINDOW_FRAMES:
            detect_count = sum(1 for kp in buffer if kp.sum() != 0)
            detect_ratio = detect_count / WINDOW_FRAMES

            if detect_ratio >= min_detect_ratio:
                path = save_clip_csv(buffer, action, clip_idx)
                clip_idx += 1
                saved    += 1
            buffer = []  # Reset (non-overlapping)

    cap.release()
    return saved


# ============================================================
# SOURCE 1: UCF-101
# ============================================================
def count_existing(action: str) -> int:
    return len(list(OUTPUT_DIR.glob(f"{action}_*.csv")))


def process_ucf101_videos(action: str, video_paths: list, mp_runner, mp_api: str,
                           target_clips: int) -> int:
    """Xử lý danh sách video UCF-101, trả về tổng clips đã lưu."""
    existing = count_existing(action)
    start_idx  = existing
    total_saved = 0

    for vpath in video_paths:
        if total_saved + existing >= target_clips:
            break
        remaining = target_clips - existing - total_saved
        n = extract_clips_from_video(
            video_path   = vpath,
            action       = action,
            mp_runner    = mp_runner,
            mp_api       = mp_api,
            start_clip_idx= start_idx + total_saved,
            max_clips    = remaining,
        )
        total_saved += n
        if n > 0:
            log.info(f"    {Path(vpath).name}: {n} clips")

    return total_saved


def download_ucf101_subset(actions: list, target_clips_each: int,
                           mp_runner, mp_api: str):
    """
    Tải và xử lý UCF-101 subset.
    UCF-101 phân phối qua RAR archives theo từng nhóm.
    Ta sẽ tải trực tiếp file danh sách video rồi xử lý.
    """
    log.info("=" * 60)
    log.info("SOURCE 1: UCF-101 Dataset")
    log.info("=" * 60)

    # Tải annotation file (danh sách video)
    annotation_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    ann_zip = VIDEO_CACHE / "ucf101_splits.zip"
    ann_dir = VIDEO_CACHE / "ucf101_splits"

    if not ann_dir.exists():
        log.info("Downloading UCF-101 split annotations (~1MB)...")
        try:
            urllib.request.urlretrieve(annotation_url, str(ann_zip))
            with zipfile.ZipFile(ann_zip, "r") as zf:
                zf.extractall(ann_dir)
            log.info("Annotations extracted.")
        except Exception as e:
            log.warning(f"Không tải được UCF-101 annotations: {e}")
            log.info("Chuyển sang YouTube source...")
            return {}

    results = {}
    for action in actions:
        ucf_categories = UCF101_MAP.get(action, [])
        if not ucf_categories:
            continue

        existing = count_existing(action)
        if existing >= target_clips_each:
            log.info(f"  [{action}] Đã đủ {existing} clips. Skip.")
            results[action] = existing
            continue

        log.info(f"\n[{action}] Target: {target_clips_each} clips (hiện có: {existing})")
        total_saved = 0

        for ucf_cat in ucf_categories:
            if existing + total_saved >= target_clips_each:
                break
            # Download video archive cho category này
            saved = _download_and_process_ucf_category(
                ucf_cat, action, mp_runner, mp_api,
                target_clips_each - existing - total_saved
            )
            total_saved += saved

        results[action] = existing + total_saved
        log.info(f"  [{action}] Tổng: {results[action]} clips")

    return results


def _download_and_process_ucf_category(ucf_cat: str, action: str,
                                        mp_runner, mp_api: str,
                                        needed: int) -> int:
    """Download 1 UCF-101 category và xử lý."""
    # UCF101 video URL pattern
    # Base: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar (13GB - quá lớn)
    # Alternative: download individual class zips (một số mirror cung cấp)

    # Dùng mirror từ Kaggle/academic mirrors
    mirrors = [
        # Mirror 1: Direct UCF categories (một số host có class-level zips)
        f"https://storage.googleapis.com/thumos14_files/UCF_videos/{ucf_cat}.zip",
    ]

    cat_dir = VIDEO_CACHE / ucf_cat
    cat_dir.mkdir(exist_ok=True)

    # Kiểm tra đã có video chưa
    existing_videos = list(cat_dir.glob("*.avi")) + list(cat_dir.glob("*.mp4"))

    if not existing_videos:
        # Thử tải từ mirror
        downloaded = False
        for url in mirrors:
            try:
                zip_path = cat_dir / f"{ucf_cat}.zip"
                log.info(f"  Đang tải {ucf_cat} từ mirror...")
                urllib.request.urlretrieve(url, str(zip_path))
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(cat_dir)
                zip_path.unlink(missing_ok=True)
                existing_videos = list(cat_dir.rglob("*.avi")) + list(cat_dir.rglob("*.mp4"))
                if existing_videos:
                    downloaded = True
                    log.info(f"  Đã tải {len(existing_videos)} videos cho {ucf_cat}")
                    break
            except Exception as e:
                log.debug(f"  Mirror {url} thất bại: {e}")

        if not downloaded:
            log.warning(f"  Không tải được {ucf_cat}. Bỏ qua.")
            return 0

    # Xử lý videos
    start_idx = count_existing(action)
    saved = process_ucf101_videos(
        action, existing_videos[:needed], mp_runner, mp_api,
        start_idx, min(needed, len(existing_videos) * 5)
    )
    return saved


# ============================================================
# SOURCE 2: YouTube (dùng yt-dlp)
# ============================================================
def ensure_ytdlp():
    """Cài yt-dlp nếu chưa có."""
    try:
        import yt_dlp
        return True
    except ImportError:
        log.info("Cài yt-dlp...")
        ret = os.system(f"{sys.executable} -m pip install yt-dlp -q")
        if ret == 0:
            log.info("yt-dlp đã cài.")
            return True
        else:
            log.error("Không cài được yt-dlp!")
            return False


def download_youtube_videos(action: str, query: str, video_dir: Path,
                             max_videos: int = 5) -> list:
    """Tìm kiếm và tải video từ YouTube với yt-dlp."""
    import yt_dlp

    ydl_opts = {
        "format"          : "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl"         : str(video_dir / "%(id)s.%(ext)s"),
        "noplaylist"      : True,
        "quiet"           : True,
        "no_warnings"     : True,
        "max_downloads"   : max_videos,
        "ignoreerrors"    : True,
        "default_search"  : "ytsearch",
        "socket_timeout"  : 30,
        "retries"         : 3,
        "fragment_retries": 3,
        # Giới hạn thời lượng: 10-180 giây
        "match_filter": lambda info_dict: (
            None if (info_dict.get("duration") or 0) < 180
            else "Video quá dài"
        ),
    }

    search_url = f"ytsearch{max_videos}:{query}"
    downloaded = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_url])
    except Exception as e:
        log.debug(f"yt-dlp error: {e}")

    downloaded = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
    return downloaded


def download_youtube_data(actions: list, target_clips_each: int,
                          mp_runner, mp_api: str):
    """Thu thập keypoints từ YouTube videos."""
    if not ensure_ytdlp():
        return {}

    log.info("\n" + "=" * 60)
    log.info("SOURCE 2: YouTube Videos")
    log.info("=" * 60)

    results = {}

    for action in actions:
        existing = count_existing(action)
        if existing >= target_clips_each:
            log.info(f"  [{action}] Đã đủ {existing} clips. Skip.")
            results[action] = existing
            continue

        needed = target_clips_each - existing
        log.info(f"\n[{action}] Cần thêm {needed} clips (hiện có: {existing})")

        queries = YOUTUBE_QUERIES.get(action, [f"person {action} outdoor"])
        yt_dir  = VIDEO_CACHE / f"yt_{action}"
        yt_dir.mkdir(exist_ok=True)

        total_saved = 0
        start_idx   = existing

        for query in queries:
            if total_saved >= needed:
                break

            log.info(f"  Tìm: '{query}'")
            videos = download_youtube_videos(
                action, query, yt_dir,
                max_videos=max(3, (needed - total_saved) // 5 + 2)
            )

            if not videos:
                log.warning(f"  Không tải được video nào.")
                continue

            log.info(f"  Đã tải {len(videos)} videos")
            for vpath in videos:
                if total_saved >= needed:
                    break
                n = extract_clips_from_video(
                    video_path    = str(vpath),
                    action        = action,
                    mp_runner     = mp_runner,
                    mp_api        = mp_api,
                    start_clip_idx= start_idx + total_saved,
                    max_clips     = needed - total_saved,
                )
                total_saved += n
                if n > 0:
                    log.info(f"    {vpath.name}: {n} clips")

        results[action] = existing + total_saved
        log.info(f"  [{action}] Tổng: {results[action]}/{target_clips_each} clips")

    return results


# ============================================================
# SOURCE 3: Synthetic augmentation từ các clips đã có
# ============================================================
def augment_existing_clips(
    action: str,
    target_clips: int,
    mp_runner,
    mp_api: str,
):
    """
    Augment keypoints từ các clip đã có bằng:
    - Gaussian noise nhỏ
    - Time warping (resample temporal axis)
    - Horizontal flip (x = 1 - x)
    Không cần video thật.
    """
    existing_csvs = list(OUTPUT_DIR.glob(f"{action}_*.csv"))
    if not existing_csvs:
        return 0

    current_count = len(existing_csvs)
    needed = target_clips - current_count
    if needed <= 0:
        return 0

    log.info(f"  [{action}] Augment {current_count} clips → {target_clips}")

    import pandas as pd
    saved = 0

    for csv_path in existing_csvs:
        if saved >= needed:
            break

        df = pd.read_csv(csv_path)
        kp_cols = [c for c in df.columns if c.startswith("kp_")]
        kp = df[kp_cols].values.astype(np.float32)  # (30, 132)

        aug_variants = []

        # 1. Gaussian noise
        noise = np.random.normal(0, 0.005, kp.shape).astype(np.float32)
        aug_variants.append(kp + noise)

        # 2. Horizontal flip (x-coord = 1 - x, mỗi kp có 4 features: x,y,z,vis)
        kp_flip = kp.copy()
        kp_matrix = kp_flip.reshape(30, 33, 4)
        kp_matrix[:, :, 0] = 1.0 - kp_matrix[:, :, 0]  # flip x
        aug_variants.append(kp_matrix.reshape(30, 132))

        # 3. Time warp (resample temporal axis với random permutation nhỏ)
        indices = np.linspace(0, 29, 30)
        noise_t = np.random.uniform(-0.5, 0.5, 30)
        indices = np.clip(indices + noise_t, 0, 29).astype(int)
        aug_variants.append(kp[indices])

        for aug_kp in aug_variants:
            if saved >= needed:
                break
            new_idx = current_count + saved
            frames = [aug_kp[i] for i in range(WINDOW_FRAMES)]
            save_clip_csv(frames, action, new_idx)
            saved += 1

    log.info(f"    Augmented: +{saved} clips")
    return saved


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="ActionNet data downloader")
    p.add_argument("--source", choices=["ucf101", "youtube", "both", "augment"],
                   default="youtube",
                   help="Nguồn data: ucf101 | youtube | both | augment")
    p.add_argument("--actions", nargs="+",
                   default=ACTION_CONFIG["action_classes"],
                   help="Danh sách actions (mặc định: tất cả 8)")
    p.add_argument("--clips", type=int, default=80,
                   help="Số clip target mỗi action (default: 80)")
    p.add_argument("--clean-cache", action="store_true",
                   help="Xóa video cache sau khi xử lý")
    return p.parse_args()


def print_summary(results: dict, target: int):
    log.info("\n" + "=" * 60)
    log.info("  KẾT QUẢ THU THẬP DATA")
    log.info("=" * 60)
    total = 0
    for action in ACTION_CONFIG["action_classes"]:
        count = results.get(action, count_existing(action))
        total += count
        bar_len = min(30, int(count / target * 30))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = count / target * 100
        log.info(f"  {action:15s} {bar} {count:4d}/{target} ({pct:.0f}%)")
    log.info(f"\n  TỔNG CỘNG: {total} clips ({len(results)} actions)")
    log.info(f"  Thư mục  : {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)

    if total < len(ACTION_CONFIG["action_classes"]) * 20:
        log.warning("\n  ⚠️  DATA CÒN ÍT. Gợi ý:")
        log.warning("  1. Chạy lại với --source both")
        log.warning("  2. Tăng --clips")
        log.warning("  3. Chạy --source augment để nhân data")
    elif total < len(ACTION_CONFIG["action_classes"]) * 50:
        log.info("\n  Có thể train được nhưng accuracy sẽ chưa cao.")
        log.info("  Gợi ý: chạy --source augment để tăng thêm.")
    else:
        log.info("\n  ✅ Đủ data để train! Chạy ActionNet_Training.ipynb trên Colab.")


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("  ActionNet Data Downloader")
    log.info(f"  Source : {args.source}")
    log.info(f"  Actions: {args.actions}")
    log.info(f"  Target : {args.clips} clips/action")
    log.info("=" * 60)

    # Init MediaPipe
    log.info("\nKhởi tạo MediaPipe Pose...")
    mp_api, mp_runner = init_mediapipe()
    log.info(f"MediaPipe mode: {mp_api}")

    results = {}

    if args.source in ("ucf101", "both"):
        r = download_ucf101_subset(args.actions, args.clips, mp_runner, mp_api)
        results.update(r)

    if args.source in ("youtube", "both"):
        r = download_youtube_data(args.actions, args.clips, mp_runner, mp_api)
        results.update(r)

    if args.source == "augment":
        log.info("\nAugmenting existing clips...")
        for action in args.actions:
            n = augment_existing_clips(action, args.clips, mp_runner, mp_api)
            results[action] = count_existing(action)

    # Augment nếu vẫn thiếu (tự động)
    any_shortage = any(
        results.get(a, count_existing(a)) < args.clips
        for a in args.actions
    )
    if any_shortage and args.source != "augment":
        log.info("\nMột số action vẫn thiếu data. Tự động augment...")
        for action in args.actions:
            cur = results.get(action, count_existing(action))
            if cur < args.clips and cur > 0:
                augment_existing_clips(action, args.clips, mp_runner, mp_api)
                results[action] = count_existing(action)

    # Clean cache
    if args.clean_cache and VIDEO_CACHE.exists():
        shutil.rmtree(VIDEO_CACHE)
        log.info(f"Đã xóa cache: {VIDEO_CACHE}")

    print_summary(results, args.clips)

    # Tạo zip để upload Colab
    _create_upload_zip()


def _create_upload_zip():
    """Tạo zip keypoints để upload lên Colab."""
    zip_path = Path("data/actionnet_keypoints.zip")
    total_csv = list(OUTPUT_DIR.glob("*.csv"))
    if not total_csv:
        return

    log.info(f"\nTạo upload zip ({len(total_csv)} files)...")
    import zipfile as zf
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as zout:
        for csv_f in total_csv:
            zout.write(csv_f, arcname=f"keypoints/{csv_f.name}")

    size_mb = zip_path.stat().st_size / 1024 / 1024
    log.info(f"Đã tạo: {zip_path} ({size_mb:.1f} MB)")
    log.info("Upload file này lên Google Drive để dùng với ActionNet_Training.ipynb")


if __name__ == "__main__":
    main()
