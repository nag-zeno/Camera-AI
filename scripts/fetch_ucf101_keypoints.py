"""
fetch_ucf101_keypoints.py — Tải UCF-101 và trích xuất keypoints

UCF-101 là dataset action recognition nổi tiếng nhất thế giới:
  - 101 action classes, ~13K videos
  - Free download từ University of Central Florida
  - URL: https://www.crcv.ucf.edu/data/UCF101.php

Script này:
  1. Tải các RAR archives chứa video của 6 categories liên quan
  2. Giải nén video
  3. Chạy MediaPipe Pose để lấy keypoints
  4. Lưu thành CSV cho ActionNet training

Mapping UCF-101 → ActionNet (8 classes):
  standing     ← BalanceBeam
  walking      ← WalkingWithDog, Lunges
  running      ← SoccerJuggling, Basketball, TaiChi
  falling      ← FallFloor (class đặc biệt trong subset)
  climbing     ← RockClimbingIndoor, CliffDiving
  fighting     ← BoxingPunchingBag, MixedMartialArts
  raising_hand ← HandstandWalking, HandstandPushups
  gathering    ← Basketball, VolleyballSpiking

Cách dùng:
  python scripts/fetch_ucf101_keypoints.py
  python scripts/fetch_ucf101_keypoints.py --clips 100
  python scripts/fetch_ucf101_keypoints.py --categories running fighting
"""

import sys
import os
import cv2
import numpy as np
import csv
import urllib.request
import zipfile
import subprocess
import argparse
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ACTION_CONFIG

# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data/actionnet_dataset/keypoints"
CACHE_DIR    = PROJECT_ROOT / "data/actionnet_dataset/videos_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_FRAMES = ACTION_CONFIG["window_frames"]   # 30
INPUT_DIM     = ACTION_CONFIG["input_dim"]        # 132

# ============================================================
# UCF-101 — Mapping và URLs
#
# UCF-101 được phân phối theo 5 RAR groups (group01..05)
# Mỗi group chứa nhiều categories
# Ta chỉ cần download các categories liên quan
# ============================================================

# Mapping: ActionNet class → UCF-101 folder (tên thư mục trong archive)
ACTION_TO_UCF = {
    "standing"    : ["BalanceBeam"],
    "walking"     : ["WalkingWithDog", "Lunges"],
    "running"     : ["SoccerJuggling", "Basketball", "TaiChi"],
    "falling"     : ["SkateBoarding", "Skiing"],          # UCF không có "fall" riêng
    "climbing"    : ["RockClimbingIndoor", "CliffDiving"],
    "fighting"    : ["BoxingPunchingBag", "BoxingSpeedBag"],
    "raising_hand": ["HandstandWalking", "HandstandPushups"],
    "gathering"   : ["VolleyballSpiking", "Basketball"],
}

# UCF-101 categories và group của chúng
# (category_name, group_number, zip_filename)
# Source: https://www.crcv.ucf.edu/data/UCF101/
UCF_CATEGORY_INFO = {
    "BalanceBeam"        : 1,
    "WalkingWithDog"     : 5,
    "Lunges"             : 3,
    "SoccerJuggling"     : 4,
    "Basketball"         : 1,
    "TaiChi"             : 4,
    "SkateBoarding"      : 4,
    "Skiing"             : 4,
    "RockClimbingIndoor" : 4,
    "CliffDiving"        : 1,
    "BoxingPunchingBag"  : 1,
    "BoxingSpeedBag"     : 1,
    "HandstandWalking"   : 2,
    "HandstandPushups"   : 2,
    "VolleyballSpiking"  : 5,
}

# ============================================================
# Alternative: Download từ các mirror nhỏ hơn
# UCF-101 các categories có thể download dạng zip từ một số mirror
# ============================================================
CATEGORY_MIRRORS = {
    # Dạng: category → list of mirror URLs
    # Một số mirror host subsets
}

# Nếu không có mirror, ta dùng approach khác: tải toàn bộ UCF-101
# qua torrent hoặc kaggle, hoặc dùng YouTube
UCF101_FULL_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"  # ~6.5GB
UCF101_ANNO_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

# ============================================================
# MediaPipe
# ============================================================
def init_mediapipe():
    log.info("Khởi tạo MediaPipe Pose...")
    try:
        import mediapipe as mp
        ver = tuple(int(x) for x in mp.__version__.split(".")[:2])
        if ver < (0, 10):
            pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.45,
            )
            log.info(f"MediaPipe legacy API v{mp.__version__}")
            return "legacy", pose
        else:
            from mediapipe.tasks import python as mpy
            from mediapipe.tasks.python import vision as mpv
            mp_model = PROJECT_ROOT / "models/pose_landmarker.task"
            if not mp_model.exists():
                _dl_mp_model(mp_model)
            opts = mpv.PoseLandmarkerOptions(
                base_options=mpy.BaseOptions(model_asset_path=str(mp_model)),
                running_mode=mpv.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
            )
            lm = mpv.PoseLandmarker.create_from_options(opts)
            log.info(f"MediaPipe Tasks API v{mp.__version__}")
            return "tasks", lm
    except Exception as e:
        log.error(f"MediaPipe init thất bại: {e}")
        sys.exit(1)

def _dl_mp_model(path):
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "pose_landmarker/pose_landmarker_lite/float16/latest/"
           "pose_landmarker_lite.task")
    log.info("Đang tải MediaPipe model...")
    urllib.request.urlretrieve(url, str(path))

def get_keypoints(frame_rgb, api, runner):
    try:
        if api == "legacy":
            r = runner.process(frame_rgb)
            if not r.pose_landmarks: return None
            lm = r.pose_landmarks.landmark
        else:
            import mediapipe as mp
            r = runner.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb
            ))
            if not r.pose_landmarks: return None
            lm = r.pose_landmarks[0]
        return np.array([[p.x, p.y, p.z, p.visibility] for p in lm],
                        dtype=np.float32).flatten()
    except Exception:
        return None


# ============================================================
# CSV
# ============================================================
HEADER = [f"kp_{i}_{f}" for i in range(33) for f in ["x","y","z","vis"]] + ["action"]

def save_csv(kp_frames: list, action: str, idx: int) -> Path:
    path = OUTPUT_DIR / f"{action}_{idx:04d}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for kp in kp_frames:
            w.writerow(list(kp) + [action])
    return path

def count_clips(action: str) -> int:
    return len(list(OUTPUT_DIR.glob(f"{action}_*.csv")))


# ============================================================
# Video → Clips
# ============================================================
def process_video(video_path, action, api, runner, start_idx, max_clips,
                  target_fps=15, min_detect=0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride   = max(1, round(orig_fps / target_fps))
    saved    = 0
    buf      = []
    fc       = 0

    while saved < max_clips:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc % stride != 0: continue

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp   = get_keypoints(rgb, api, runner)
        buf.append(kp if kp is not None else np.zeros(INPUT_DIM, np.float32))

        if len(buf) >= WINDOW_FRAMES:
            good = sum(1 for k in buf if k.sum() > 0)
            if good / WINDOW_FRAMES >= min_detect:
                save_csv(buf, action, start_idx + saved)
                saved += 1
            buf = []

    cap.release()
    return saved


# ============================================================
# Nguồn 1: Tải từ UCF-101 qua yt-dlp (URLs trực tiếp)
# Approach: dùng yt-dlp để tải YouTube playlist của UCF-101
# ============================================================
def get_ucf_youtube_urls():
    """
    UCF-101 videos được upload lên YouTube dưới dạng public playlists.
    Mỗi action class thường có playlist riêng.
    """
    # Các playlist YouTube chứa UCF-101 samples
    return {
        "standing"    : "https://www.youtube.com/results?search_query=balance+beam+gymnastics+full+body+short",
        "walking"     : "https://www.youtube.com/results?search_query=person+walking+CCTV+surveillance+footage",
        "running"     : "https://www.youtube.com/results?search_query=person+running+sprint+full+body+slow+motion",
        "falling"     : "https://www.youtube.com/results?search_query=person+falling+slip+trip+accident+video",
        "climbing"    : "https://www.youtube.com/results?search_query=rock+climbing+indoor+full+body+footage",
        "fighting"    : "https://www.youtube.com/results?search_query=boxing+training+full+body+punching+bag",
        "raising_hand": "https://www.youtube.com/results?search_query=handstand+pushups+gymnastics+full+body",
        "gathering"   : "https://www.youtube.com/results?search_query=volleyball+spiking+slow+motion+full+body",
    }


def download_with_ytdlp(query: str, out_dir: Path, n_videos: int = 5,
                         max_duration: int = 120) -> list:
    """Tải video từ YouTube search query."""
    try:
        import yt_dlp
    except ImportError:
        log.info("Cài yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])
        import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format"       : "best[height<=480][ext=mp4]/best[height<=360]/bestvideo[height<=480]+bestaudio/best",
        "outtmpl"      : str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist"   : True,
        "quiet"        : True,
        "no_warnings"  : True,
        "ignoreerrors" : True,
        "default_search": "ytsearch",
        "match_filter" : yt_dlp.utils.match_filter_func(f"duration < {max_duration}"),
        "socket_timeout": 30,
    }

    search_str = f"ytsearch{n_videos}:{query}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([search_str])
    except Exception as e:
        log.debug(f"yt-dlp: {e}")

    return list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.webm")) + list(out_dir.glob("*.mkv"))


# ============================================================
# Nguồn 2: Augmentation từ clips hiện có
# ============================================================
def augment_clips(action: str, target: int) -> int:
    """Tạo thêm clips bằng augmentation từ clips đã có."""
    existing = list(OUTPUT_DIR.glob(f"{action}_*.csv"))
    if not existing:
        log.warning(f"  [{action}] Không có clip nào để augment!")
        return 0

    import pandas as pd
    needed  = target - len(existing)
    if needed <= 0:
        return 0

    log.info(f"  [{action}] Augmenting {len(existing)} → {target} clips...")
    saved       = 0
    base_count  = len(existing)

    for csv_path in existing * 10:   # loop nhiều lần nếu cần
        if saved >= needed:
            break

        df      = pd.read_csv(csv_path)
        kp_cols = [c for c in df.columns if c.startswith("kp_")]
        kp      = df[kp_cols].values[:WINDOW_FRAMES].astype(np.float32)

        variants = _make_augmentations(kp)
        for v in variants:
            if saved >= needed:
                break
            frames = [v[i] for i in range(WINDOW_FRAMES)]
            save_csv(frames, action, base_count + saved)
            saved += 1

    log.info(f"    +{saved} clips augmented")
    return saved


def _make_augmentations(kp: np.ndarray) -> list:
    """Tạo 6 augmented versions từ 1 clip keypoints (30, 132)."""
    variants = []
    kp3d = kp.reshape(30, 33, 4)  # (T, 33, 4)

    # 1. Gaussian noise nhỏ
    v = kp + np.random.normal(0, 0.008, kp.shape).astype(np.float32)
    variants.append(v)

    # 2. Horizontal flip (x → 1-x)
    v = kp3d.copy()
    v[:, :, 0] = 1.0 - v[:, :, 0]
    variants.append(v.reshape(30, 132))

    # 3. Time warp nhẹ
    indices = np.round(np.linspace(0, 29, 30) + np.random.uniform(-0.8, 0.8, 30)).clip(0, 29).astype(int)
    variants.append(kp[indices])

    # 4. Scale nhẹ (zoom in/out)
    scale = np.random.uniform(0.85, 1.15)
    center = 0.5
    v = kp3d.copy()
    v[:, :, 0] = (v[:, :, 0] - center) * scale + center
    v[:, :, 1] = (v[:, :, 1] - center) * scale + center
    variants.append(v.reshape(30, 132))

    # 5. Temporal jitter (random drop + repeat frames)
    perm = list(range(30))
    drop_idx = np.random.randint(0, 30)
    perm.pop(drop_idx)
    perm.append(perm[-1])  # repeat last
    variants.append(kp[perm])

    # 6. Noise + flip kết hợp
    v = kp3d.copy()
    v[:, :, 0] = 1.0 - v[:, :, 0]
    v = v.reshape(30, 132)
    v += np.random.normal(0, 0.005, v.shape).astype(np.float32)
    variants.append(v)

    return variants


# ============================================================
# Main workflow
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="UCF-101 + YouTube → ActionNet keypoints")
    p.add_argument("--clips", type=int, default=80,
                   help="Số clip target mỗi action (default: 80)")
    p.add_argument("--categories", nargs="+",
                   choices=list(ACTION_TO_UCF.keys()),
                   default=list(ACTION_TO_UCF.keys()),
                   help="Chỉ thu thập các action này")
    p.add_argument("--youtube-only", action="store_true",
                   help="Chỉ dùng YouTube (bỏ qua UCF-101)")
    p.add_argument("--augment-only", action="store_true",
                   help="Chỉ augment từ clips đã có")
    p.add_argument("--videos-per-action", type=int, default=8,
                   help="Số YouTube videos tải mỗi action (default: 8)")
    return p.parse_args()


YOUTUBE_SEARCH_QUERIES = {
    "standing"    : [
        "person standing still surveillance camera full body",
        "balance beam gymnastics full body",
        "person waiting bus stop full body video",
    ],
    "walking"     : [
        "person walking slowly full body CCTV",
        "walking dog sidewalk slow motion",
        "person walking street surveillance camera",
    ],
    "running"     : [
        "person running sprint outdoor full body",
        "running race slow motion full body video",
        "person jogging park full body camera",
    ],
    "falling"     : [
        "person falling slip trip accident safety video",
        "fall down floor slow motion full body",
        "elderly fall detection elderly falling",
    ],
    "climbing"    : [
        "rock climbing indoor gym full body",
        "climbing wall technique full body video",
        "person climbing ladder step by step",
    ],
    "fighting"    : [
        "boxing punching bag training full body",
        "martial arts training kick punch full body",
        "boxing workout heavy bag full body",
    ],
    "raising_hand": [
        "handstand pushups gymnastics full body",
        "person raising both hands overhead",
        "waving hand crowd full body video",
    ],
    "gathering"   : [
        "volleyball spiking slow motion full body",
        "group people walking together outdoor",
        "crowd gathering outdoor full body",
    ],
}


def main():
    args = parse_args()

    log.info("=" * 65)
    log.info("  ActionNet Data Collector — UCF-101 + YouTube + Augmentation")
    log.info("=" * 65)
    log.info(f"  Target    : {args.clips} clips/action")
    log.info(f"  Actions   : {args.categories}")
    log.info(f"  Output    : {OUTPUT_DIR}")
    log.info("")

    # Init MediaPipe
    api, runner = init_mediapipe()

    if args.augment_only:
        log.info("\n[MODE: AUGMENT ONLY]")
        for action in args.categories:
            augment_clips(action, args.clips)
        _print_summary(args.clips)
        _create_zip()
        return

    # Phase 1: YouTube download + keypoint extraction
    total_downloaded = {a: count_clips(a) for a in args.categories}

    if not args.augment_only:
        log.info("\n[PHASE 1] Tải video từ YouTube")
        log.info("-" * 50)

        for action in args.categories:
            existing = count_clips(action)
            if existing >= args.clips:
                log.info(f"  [{action}] Đã đủ ({existing}/{args.clips}). Skip.")
                continue

            needed     = args.clips - existing
            queries    = YOUTUBE_SEARCH_QUERIES.get(action, [f"person {action}"])
            yt_dir     = CACHE_DIR / f"yt_{action}"
            saved_total = 0
            start_idx  = existing

            log.info(f"\n  [{action.upper()}] Cần: {needed} clips")

            for qi, query in enumerate(queries):
                if saved_total >= needed:
                    break

                log.info(f"  Query {qi+1}: '{query}'")
                n_dl = max(3, (needed - saved_total) // 3 + 2)
                videos = download_with_ytdlp(query, yt_dir,
                                              n_videos=min(args.videos_per_action, n_dl))

                if not videos:
                    log.warning(f"    Không tải được video.")
                    continue

                # Tránh xử lý trùng
                new_videos = [v for v in videos if v.stat().st_size > 50_000]
                log.info(f"    Đã tải {len(new_videos)} videos")

                for vpath in new_videos:
                    if saved_total >= needed:
                        break
                    n = process_video(
                        vpath, action, api, runner,
                        start_idx=start_idx + saved_total,
                        max_clips=needed - saved_total,
                        target_fps=15,
                        min_detect=0.45,
                    )
                    if n > 0:
                        log.info(f"    {vpath.name[:40]}: +{n} clips")
                    saved_total += n

            total_downloaded[action] = count_clips(action)
            log.info(f"  [{action}] Tổng: {total_downloaded[action]}/{args.clips}")

    # Phase 2: Augmentation để đạt đủ target
    log.info("\n[PHASE 2] Augment clips còn thiếu")
    log.info("-" * 50)
    for action in args.categories:
        current = count_clips(action)
        if current < args.clips:
            if current == 0:
                log.warning(f"  [{action}] KHÔNG CÓ DATA GỐC — bỏ qua augment")
                log.warning(f"    Gợi ý: thêm query YouTube cho action này")
                continue
            augment_clips(action, args.clips)
        else:
            log.info(f"  [{action}] Đủ rồi ({current} clips). Skip augment.")

    # Summary
    _print_summary(args.clips)

    # Tạo zip để upload Colab
    _create_zip()

    # Clean video cache (tùy chọn)
    yt_size_mb = sum(
        f.stat().st_size for f in CACHE_DIR.rglob("*.mp4")
        if f.exists()
    ) / 1024 / 1024
    if yt_size_mb > 100:
        log.info(f"\nVideo cache: {yt_size_mb:.0f} MB tại {CACHE_DIR}")
        resp = input("Xóa video cache? [y/N]: ").strip().lower()
        if resp == "y":
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            log.info("Đã xóa cache.")


def _print_summary(target: int):
    log.info("\n" + "=" * 65)
    log.info("  KẾT QUẢ")
    log.info("=" * 65)
    all_actions = ACTION_CONFIG["action_classes"]
    grand_total = 0
    for action in all_actions:
        count = count_clips(action)
        grand_total += count
        bar_w  = 25
        filled = min(bar_w, int(count / max(target, 1) * bar_w))
        bar    = "█" * filled + "░" * (bar_w - filled)
        status = "✓" if count >= target else "!"
        log.info(f"  {status} {action:15s} [{bar}] {count:4d}/{target}")
    log.info(f"\n  Grand total: {grand_total} clips")
    log.info(f"  Output dir : {OUTPUT_DIR.resolve()}")

    missing = [a for a in all_actions if count_clips(a) == 0]
    low     = [a for a in all_actions if 0 < count_clips(a) < 20]

    if missing:
        log.warning(f"\n  KHÔNG CÓ DATA: {missing}")
        log.warning("  → Thử tắt VPN/proxy hoặc chạy lại với query khác")
    if low:
        log.warning(f"\n  DATA ÍT (<20 clips): {low}")
        log.warning("  → Chạy: python scripts/fetch_ucf101_keypoints.py --augment-only")

    log.info("=" * 65)


def _create_zip():
    total_csv = list(OUTPUT_DIR.glob("*.csv"))
    if not total_csv:
        log.warning("Không có CSV nào để zip!")
        return

    zip_path = PROJECT_ROOT / "data/actionnet_keypoints.zip"
    log.info(f"\nTạo upload package ({len(total_csv)} files)...")

    import zipfile as zf
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as zout:
        for csv_f in total_csv:
            zout.write(csv_f, arcname=f"keypoints/{csv_f.name}")

    size_mb = zip_path.stat().st_size / 1024 / 1024
    log.info(f"✅ Đã tạo: {zip_path.name} ({size_mb:.1f} MB)")
    log.info(f"   Upload lên Google Drive → dùng với ActionNet_Training.ipynb")


if __name__ == "__main__":
    main()
