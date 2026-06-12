# -*- coding: utf-8 -*-
"""
demo.py -- Chay thu toan bo pipeline voi webcam / video file.

Su dung:
    python demo.py                    # webcam mac dinh (cam 0)
    python demo.py --source 1         # webcam 1
    python demo.py --source video.mp4 # video file

Phim tat:
    q / ESC  -- Thoat
    p        -- Pause / Resume
    s        -- Luu screenshot
    i        -- Hien thi info chi tiet
"""
import sys
import time
import argparse
import logging
import threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def parse_args():
    p = argparse.ArgumentParser(description="AI Camera Demo -- RoleNet Integrated")
    p.add_argument("--source", default="0",
                   help="0=webcam | 1,2=cam index | path/to/video.mp4")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-display", action="store_true",
                   help="Chay khong hien thi cua so (headless)")
    return p.parse_args()


def source_value(s: str):
    """Chuyen doi source string sang int (webcam) hoac str (file/rtsp)."""
    try:
        return int(s)
    except ValueError:
        return s


def print_banner(mode: str, device: str, val_acc: float = 0.0):
    print()
    print("=" * 60)
    print("   AI CAMERA SYSTEM -- RoleNet Demo")
    print("=" * 60)
    print(f"   Classifier mode : {mode.upper()}")
    print(f"   Device          : {device.upper()}")
    if val_acc > 0:
        print(f"   Val Accuracy    : {val_acc:.1f}%")
    print("-" * 60)
    print("   Phim tat: [q/ESC] Thoat | [p] Pause | [s] Screenshot | [i] Info")
    print("=" * 60)
    print()


def overlay_stats(frame: np.ndarray, result, fps: float, mode: str, paused: bool) -> np.ndarray:
    """Ve thanh thong tin phia tren frame."""
    h, w = frame.shape[:2]

    # Thanh header
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    # FPS
    fps_color = (0, 230, 0) if fps >= 12 else (0, 165, 255) if fps >= 6 else (0, 0, 230)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

    # Mode badge
    badge_color = (0, 200, 100) if "ml" in mode else (0, 165, 255)
    badge_label = "RoleNet ML" if "ml" in mode else "Rule-based"
    cv2.putText(frame, badge_label, (w - 150, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

    # Thong ke
    if result:
        persons = [o for o in result.objects if o.category.value == "person"]
        alerts  = [o for o in result.objects if o.alert_level.value in ("warning", "alert", "critical")]
        info    = f"Persons: {len(persons)}  Alerts: {len(alerts)}"
        cv2.putText(frame, info, (int(w * 0.3), 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # Paused indicator
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

    return frame


def overlay_role_detail(frame: np.ndarray, result) -> np.ndarray:
    """Hien thi chi tiet role + confidence theo dang list ben phai."""
    if not result:
        return frame
    h, w = frame.shape[:2]

    persons = [o for o in result.objects if o.category.value == "person"]
    if not persons:
        return frame

    x0, y0 = w - 210, 50
    panel_h = min(30 + len(persons) * 22, h - 60)

    # Panel nen mo
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (w - 4, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Role Detections", (x0, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    for i, obj in enumerate(persons[:8]):  # Mac dinh toi da 8
        y = y0 + 36 + i * 22
        conf  = obj.role_confidence
        role  = obj.role.value
        tid   = obj.track_id

        # Mau confidence
        if conf >= 0.7:
            col = (80, 255, 80)
        elif conf >= 0.5:
            col = (80, 200, 255)
        else:
            col = (100, 100, 100)

        label = f"#{tid} {role[:12]:12s} {conf*100:4.0f}%"
        cv2.putText(frame, label, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    return frame


def main():
    args = parse_args()
    source = source_value(args.source)

    # -- Import pipeline --
    try:
        from pipeline import CameraPipeline
    except ImportError as e:
        print(f"[ERROR] Khong import duoc pipeline: {e}")
        sys.exit(1)

    # -- Tao pipeline --
    pipeline = CameraPipeline(source=source)

    try:
        pipeline.setup()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("  Kiem tra lai --source. Vi du: --source 0 (webcam) hoac --source video.mp4")
        sys.exit(1)

    # -- In banner --
    mode  = pipeline.role_clf.get_status()["mode"]
    dev   = pipeline.role_clf.get_status()["device"]
    print_banner(mode, dev)

    # -- Alert callback --
    alert_log = []

    def on_alert(event):
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"[ALERT {ts}] {event.level.value.upper():8s} | role={event.object_role.value} zone={event.zone_name}"
        print(msg)
        alert_log.append(msg)

    pipeline.add_alert_callback(on_alert)

    # -- Chay pipeline trong background thread --
    pipeline.start_background()

    # -- Display loop --
    paused       = False
    show_detail  = False
    frame_count  = 0
    t_start      = time.monotonic()

    window_name  = "AI Camera -- RoleNet Demo  [q=quit | p=pause | s=save | i=info]"

    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while pipeline._running:
            if paused:
                key = cv2.waitKey(50) & 0xFF
            else:
                frame  = pipeline.get_latest_frame()
                result = pipeline.get_latest_result()
                fps    = pipeline.get_fps()

                if frame is not None:
                    disp = frame.copy()

                    # Overlay stats
                    disp = overlay_stats(disp, result, fps, mode, paused)

                    # Detail panel
                    if show_detail and result:
                        disp = overlay_role_detail(disp, result)

                    if not args.no_display:
                        cv2.imshow(window_name, disp)

                    frame_count += 1

                key = cv2.waitKey(1) & 0xFF if not args.no_display else ord('q')

            # -- Xu ly phim tat --
            if key in (ord('q'), 27):   # q / ESC
                print("\n[INFO] Thoat...")
                break

            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")

            elif key == ord('s'):
                frame = pipeline.get_latest_frame()
                if frame is not None:
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"screenshot_{ts}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"[INFO] Screenshot saved: {path}")

            elif key == ord('i'):
                show_detail = not show_detail
                print(f"[INFO] Detail overlay: {'ON' if show_detail else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

        # -- Tong ket --
        elapsed = time.monotonic() - t_start
        print()
        print("=" * 60)
        print(f"  Thoi gian chay : {elapsed:.1f}s")
        print(f"  Frames hien thi: {frame_count}")
        avg_display_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"  Avg display FPS: {avg_display_fps:.1f}")
        if alert_log:
            print(f"  Tong so alerts : {len(alert_log)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
