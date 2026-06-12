"""
collect_action_data.py — Thu thập dữ liệu keypoints cho ActionNet

Cách dùng:
  python scripts/collect_action_data.py --action standing --source 0
  python scripts/collect_action_data.py --action walking  --source video.mp4
  python scripts/collect_action_data.py --action falling  --source 0 --clips 30

Điều khiển:
  SPACE   : bắt đầu/dừng ghi 1 clip (30 frame)
  R       : xóa clip cuối
  Q / ESC : thoát

Output:
  data/actionnet_dataset/keypoints/{action}_{NNN}.csv
  Mỗi file: 30 dòng × 132 cột + cột action
"""

import cv2
import numpy as np
import argparse
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ACTION_CONFIG

# ============================================================
# Setup
# ============================================================
OUTPUT_DIR   = Path("data/actionnet_dataset/keypoints")
WINDOW_FRAMES = ACTION_CONFIG["window_frames"]
INPUT_DIM     = ACTION_CONFIG["input_dim"]
ACTION_CLASSES= ACTION_CONFIG["action_classes"]

# MediaPipe keypoint indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11; MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23;      MP_RIGHT_HIP = 24
MP_LEFT_ANKLE = 27;    MP_RIGHT_ANKLE = 28


def parse_args():
    p = argparse.ArgumentParser(description="ActionNet data collector")
    p.add_argument("--action", required=True, choices=ACTION_CLASSES,
                   help="Action label để thu thập")
    p.add_argument("--source", default="0",
                   help="Video source: 0 (webcam), hoặc path video file")
    p.add_argument("--clips",  type=int, default=50,
                   help="Số clip cần thu thập (default: 50)")
    p.add_argument("--show",   action="store_true", default=True,
                   help="Hiển thị skeleton")
    return p.parse_args()


def init_mediapipe():
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        draw = mp.solutions.drawing_utils
        style = mp.solutions.drawing_styles
        return pose, draw, style, mp.solutions.pose
    except ImportError:
        print("[ERROR] mediapipe chưa cài! Chạy: pip install mediapipe")
        sys.exit(1)


def extract_kp_vector(results) -> np.ndarray:
    """Chuyển MediaPipe landmarks thành vector 132-dim."""
    lm = results.pose_landmarks.landmark
    return np.array(
        [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm],
        dtype=np.float32
    ).flatten()


def save_clip(frames_kp: list, action: str, clip_idx: int) -> Path:
    """Lưu 1 clip (30 frame keypoints) ra CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{action}_{clip_idx:04d}.csv"

    # Header: kp_0_x, kp_0_y, kp_0_z, kp_0_vis, ..., kp_32_vis, action
    header = []
    for i in range(33):
        for feat in ["x", "y", "z", "vis"]:
            header.append(f"kp_{i}_{feat}")
    header.append("action")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for kp in frames_kp:
            row = list(kp) + [action]
            writer.writerow(row)

    return out_path


def main():
    args = parse_args()
    action = args.action

    print(f"\n{'='*60}")
    print(f"  ActionNet Data Collector")
    print(f"  Action  : {action.upper()}")
    print(f"  Target  : {args.clips} clips × {WINDOW_FRAMES} frames")
    print(f"  Output  : {OUTPUT_DIR / action}_XXXX.csv")
    print(f"{'='*60}")
    print("  Điều khiển:")
    print("    SPACE   — bắt đầu ghi clip")
    print("    R       — xóa clip vừa ghi")
    print("    Q/ESC   — thoát\n")

    pose, draw_utils, draw_styles, mp_pose = init_mediapipe()

    # Mở video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được source: {args.source}")
        sys.exit(1)

    # Đếm clip đã có
    existing = list(OUTPUT_DIR.glob(f"{action}_*.csv"))
    clip_idx = len(existing)
    saved_clips = 0

    recording   = False
    buffer      = []        # buffer kp vectors của clip đang ghi
    frame_count = 0
    last_paths  = []        # để undo

    print(f"[INFO] Đã có {clip_idx} clips. Bắt đầu từ clip #{clip_idx:04d}")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):  # video file → loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Chạy MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(frame_rgb)

        # Vẽ skeleton
        if args.show and results.pose_landmarks:
            draw_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=draw_utils.DrawingSpec(color=(255, 150, 0), thickness=2),
            )

        # Ghi keypoints vào buffer
        if recording:
            if results.pose_landmarks:
                kp = extract_kp_vector(results)
                buffer.append(kp)
            else:
                # Nếu không detect được → dùng zeros (vẫn ghi)
                buffer.append(np.zeros(INPUT_DIM, dtype=np.float32))

            # Đủ WINDOW_FRAMES → lưu
            if len(buffer) >= WINDOW_FRAMES:
                path = save_clip(buffer, action, clip_idx)
                last_paths.append(path)
                clip_idx    += 1
                saved_clips += 1
                recording    = False
                buffer       = []
                print(f"  ✅ Đã lưu clip #{clip_idx-1:04d} ({saved_clips}/{args.clips})")
                if saved_clips >= args.clips:
                    print(f"\n🎉 Đã thu thập đủ {args.clips} clips!")
                    break

        # HUD overlay
        h, w = frame.shape[:2]
        state_txt = f"{'[REC] ' if recording else ''}Action: {action.upper()}"
        state_col = (0, 50, 255) if recording else (0, 200, 100)
        cv2.putText(frame, state_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2, cv2.LINE_AA)

        prog_txt  = f"Clips: {saved_clips}/{args.clips}  (idx={clip_idx})"
        cv2.putText(frame, prog_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        if recording:
            rec_txt = f"Recording: {len(buffer)}/{WINDOW_FRAMES} frames"
            cv2.putText(frame, rec_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

            # Progress bar
            prog = int((len(buffer) / WINDOW_FRAMES) * (w - 40))
            cv2.rectangle(frame, (20, h-30), (20+prog, h-15), (0, 150, 255), -1)
            cv2.rectangle(frame, (20, h-30), (w-20, h-15), (80, 80, 80), 1)

        if not results.pose_landmarks:
            cv2.putText(frame, "No person detected", (10, h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("ActionNet Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not recording:
                recording = True
                buffer    = []
                print(f"  ⏺ Bắt đầu ghi clip #{clip_idx:04d}...")
            else:
                recording = False
                buffer    = []
                print("  ⏹ Hủy ghi.")
        elif key == ord('r') or key == ord('R'):
            if last_paths:
                p = last_paths.pop()
                if p.exists():
                    p.unlink()
                    clip_idx    -= 1
                    saved_clips -= 1
                    print(f"  🗑 Đã xóa {p.name}")
        elif key in (ord('q'), ord('Q'), 27):  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print(f"\n{'='*60}")
    print(f"  Hoàn tất! Đã lưu {saved_clips} clips cho action '{action}'")
    print(f"  Thư mục: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
