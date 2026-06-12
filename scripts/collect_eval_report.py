"""
collect_eval_report.py — Script thu thập kết quả test thực tế.

Cách dùng:
    # Chạy pipeline bình thường (app.py) khoảng 5-10 phút rồi Ctrl+C
    # Sau đó chạy script này:
    python scripts/collect_eval_report.py

    # Hoặc chạy song song với pipeline đang chạy:
    python scripts/collect_eval_report.py --live

Output:
    reports/eval_report_YYYYMMDD_HHMMSS.txt  ← file này paste cho AI

Nội dung report bao gồm:
  1. Thông tin hệ thống (CPU, RAM, OS)
  2. Model đang dùng (RoleNet version, ActionNet mode)
  3. FPS thực tế (đọc từ API nếu pipeline đang chạy)
  4. Thống kê events từ log (role distribution, action distribution, alert levels)
  5. Mẫu events gần nhất (10 events)
  6. Lỗi/warning từ log file
  7. Nhận xét thủ công (prompt user nhập vào)
"""
import json
import os
import sys
import time
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

# Fix Unicode output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Thêm project root vào path ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR     = PROJECT_ROOT / "logs"
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

EVENTS_FILE  = LOGS_DIR / "events.jsonl"
API_BASE     = "http://localhost:8000"


# ============================================================
# Helpers
# ============================================================

def _section(title: str, char: str = "=") -> str:
    line = char * 60
    return f"\n{line}\n  {title}\n{line}"


def _load_events() -> list[dict]:
    """Đọc toàn bộ events từ JSONL log."""
    if not EVENTS_FILE.exists():
        return []
    events = []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _try_api(path: str) -> dict | None:
    """Gọi API nếu pipeline đang chạy."""
    try:
        import urllib.request
        url = f"{API_BASE}{path}"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _get_system_info() -> dict:
    """Thu thập thông tin hệ thống."""
    info = {
        "os"        : platform.system() + " " + platform.version()[:50],
        "python"    : platform.python_version(),
        "cpu_cores" : os.cpu_count(),
        "hostname"  : platform.node(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1e9, 1)
        info["ram_avail_gb"] = round(mem.available / 1e9, 1)
        info["cpu_usage_pct"] = psutil.cpu_percent(interval=1)
    except ImportError:
        info["ram"] = "psutil not installed"
    return info


def _get_model_info() -> dict:
    """Đọc metadata các model."""
    models_dir = PROJECT_ROOT / "models"
    info = {}

    # RoleNet
    for meta_name, label in [
        ("rolenet_v3_onnx_meta.json", "rolenet_onnx"),
        ("rolenet_v3_metadata.json",  "rolenet_v3"),
        ("rolenet_v2_metadata.json",  "rolenet_v2"),
    ]:
        meta_path = models_dir / meta_name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                info[label] = meta
                break  # Chỉ lấy cái đầu tiên tìm thấy (cao nhất)
            except Exception:
                pass

    # ActionNet
    meta_path = models_dir / "actionnet_metadata.json"
    if meta_path.exists():
        try:
            info["actionnet"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # File sizes
    model_files = {
        "rolenet_v3.onnx"      : "RoleNet ONNX FP32",
        "rolenet_v3_quant.onnx": "RoleNet ONNX INT8",
        "rolenet_v3_best.pt"   : "RoleNet V3 PyTorch",
        "actionnet_gru.pt"     : "ActionNet GRU",
        "context_net.pkl"      : "ContextNet XGBoost",
    }
    present = []
    for fname, label in model_files.items():
        fpath = models_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            present.append(f"  ✅ {label}: {fname} ({size_mb:.1f} MB)")
        else:
            present.append(f"  ❌ {label}: {fname} — MISSING")
    info["files"] = present

    return info


def _analyze_events(events: list[dict]) -> dict:
    """Phân tích phân phối events."""
    if not events:
        return {}

    levels   = Counter(e.get("level", "unknown") for e in events)
    roles    = Counter(e.get("object_role", "unknown") for e in events)
    actions  = Counter(e.get("action", "unknown") for e in events)
    zones    = Counter(e.get("zone_name") or "outside" for e in events)

    first_ts = events[0].get("timestamp", 0)
    last_ts  = events[-1].get("timestamp", 0)
    duration = last_ts - first_ts

    return {
        "total"         : len(events),
        "duration_min"  : round(duration / 60, 1),
        "events_per_min": round(len(events) / (duration / 60), 2) if duration > 0 else 0,
        "by_level"      : dict(levels.most_common()),
        "by_role"       : dict(roles.most_common(10)),
        "by_action"     : dict(actions.most_common()),
        "by_zone"       : dict(zones.most_common()),
        "first_event"   : events[0].get("datetime", ""),
        "last_event"    : events[-1].get("datetime", ""),
    }


def _get_recent_events(events: list[dict], n: int = 15) -> list[dict]:
    """Lấy N events gần nhất, ưu tiên warning/alert/critical."""
    high_priority = [e for e in events if e.get("level") in ("warning", "alert", "critical")]
    recent = (high_priority + events)[-n:]
    return sorted(recent, key=lambda e: e.get("timestamp", 0), reverse=True)[:n]


def _scan_log_errors() -> list[str]:
    """Scan log file tìm ERROR/WARNING gần đây."""
    errors = []
    log_files = list(LOGS_DIR.glob("*.log")) + list(LOGS_DIR.glob("*.txt"))
    for lf in log_files[:3]:
        try:
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[-500:]:  # 500 dòng cuối
                if "ERROR" in line or "WARNING" in line:
                    errors.append(line.strip())
        except Exception:
            pass
    return errors[-30:]  # Tối đa 30 dòng lỗi


# ============================================================
# Nhận xét thủ công từ user
# ============================================================

OBSERVATION_QUESTIONS = [
    "1. FPS quan sát được khi có 1 người trong frame:",
    "2. FPS quan sát được khi có 3+ người trong frame:",
    "3. Role classification có đúng không? (VD: shipper nhận đúng là Shipper) [y/n/một phần]:",
    "4. Action recognition có hoạt động không? (đứng/đi/chạy) [y/n]:",
    "5. Alert có được kích hoạt đúng không? [y/n/chưa test]:",
    "6. Telegram notification có gửi được không? [y/n/chưa cấu hình]:",
    "7. Zone detection có hoạt động không? [y/n/chưa cấu hình]:",
    "8. Vấn đề / lỗi nào gặp phải (mô tả tự do):",
    "9. Môi trường test (VD: webcam laptop, camera IP, file video):",
    "10. Ghi chú thêm:",
]


def _collect_manual_observations(interactive: bool) -> dict:
    """Thu thập nhận xét thủ công từ người dùng."""
    if not interactive:
        return {"note": "Skipped (non-interactive mode). Add observations manually in the report."}

    print("\n" + "="*60)
    print("  📝 NHẬN XÉT THỰC TẾ (nhấn Enter để bỏ qua)")
    print("="*60)
    answers = {}
    for q in OBSERVATION_QUESTIONS:
        try:
            ans = input(f"\n{q} ").strip()
            answers[q] = ans if ans else "(không trả lời)"
        except (EOFError, KeyboardInterrupt):
            break
    return answers


# ============================================================
# Build report
# ============================================================

def build_report(interactive: bool = True) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("=" * 60)
    lines.append("  AI SECURITY CAMERA — BÁO CÁO TEST THỰC TẾ")
    lines.append(f"  Thời điểm: {timestamp}")
    lines.append("=" * 60)

    # 1. System Info
    lines.append(_section("1. THÔNG TIN HỆ THỐNG"))
    sysinfo = _get_system_info()
    for k, v in sysinfo.items():
        lines.append(f"  {k}: {v}")

    # 2. Model Files
    lines.append(_section("2. MODEL FILES"))
    model_info = _get_model_info()
    for line in model_info.get("files", []):
        lines.append(line)

    # 3. API Status (nếu pipeline đang chạy)
    lines.append(_section("3. TRẠNG THÁI PIPELINE (API)"))
    status = _try_api("/api/status")
    if status:
        lines.append("  Pipeline: ĐANG CHẠY ✅")
        lines.append(f"  FPS thực tế: {status.get('fps', 'N/A')}")
        lines.append(f"  Số objects hiện tại: {status.get('objects', 0)}")
        lines.append(f"  Action mode: {status.get('action_mode', 'N/A')}")
        lines.append(f"  GRU available: {status.get('gru_available', False)}")
        lines.append(f"  MediaPipe available: {status.get('mp_available', False)}")
        lines.append(f"  Zones: {status.get('zones', 0)}")
    else:
        lines.append("  Pipeline: KHÔNG CHẠY hoặc không kết nối được ⚠️")
        lines.append("  (Chạy app.py trước nếu muốn thu thập FPS realtime)")

    # Role classifier status
    role_status = _try_api("/api/status")
    obj_data = _try_api("/api/objects")
    if obj_data and obj_data.get("objects"):
        lines.append(f"\n  Objects hiện tại ({len(obj_data['objects'])} tracked):")
        for obj in obj_data["objects"][:5]:
            lines.append(
                f"    Track#{obj.get('track_id','?')} "
                f"| {obj.get('class_name','?')} "
                f"| role={obj.get('role','?')} ({obj.get('role_confidence',0):.0%}) "
                f"| action={obj.get('action','?')} "
                f"| alert={obj.get('alert_level','?')}"
            )

    # 4. Event Log Analysis
    lines.append(_section("4. PHÂN TÍCH EVENT LOG"))
    events = _load_events()
    if events:
        stats = _analyze_events(events)
        lines.append(f"  Tổng events: {stats['total']}")
        lines.append(f"  Thời gian ghi: {stats['duration_min']} phút")
        lines.append(f"  Events/phút: {stats['events_per_min']}")
        lines.append(f"  Khoảng: {stats['first_event']} → {stats['last_event']}")

        lines.append("\n  --- Phân phối theo Alert Level ---")
        for level, cnt in stats["by_level"].items():
            bar = "█" * min(cnt, 30)
            lines.append(f"    {level:12s} : {cnt:4d} {bar}")

        lines.append("\n  --- Phân phối theo Role (top 10) ---")
        for role, cnt in stats["by_role"].items():
            lines.append(f"    {role:20s} : {cnt}")

        lines.append("\n  --- Phân phối theo Action ---")
        for action, cnt in stats["by_action"].items():
            lines.append(f"    {action:20s} : {cnt}")

        lines.append("\n  --- Phân phối theo Zone ---")
        for zone, cnt in stats["by_zone"].items():
            lines.append(f"    {zone:20s} : {cnt}")
    else:
        lines.append("  ⚠️ Chưa có events trong log.")
        lines.append(f"  (File: {EVENTS_FILE})")

    # 5. Sample Events
    lines.append(_section("5. MẪU EVENTS GẦN NHẤT (ưu tiên alert/critical)"))
    recent = _get_recent_events(events, n=15)
    if recent:
        for evt in recent:
            lines.append(
                f"  [{evt.get('datetime','?')}] "
                f"level={evt.get('level','?'):8s} "
                f"role={evt.get('object_role','?'):15s} "
                f"action={evt.get('action','?'):12s} "
                f"zone={evt.get('zone_name') or 'outside':15s} "
                f"| {evt.get('reason','')[:60]}"
            )
    else:
        lines.append("  (không có events)")

    # 6. Log Errors
    lines.append(_section("6. LỖI / WARNING TỪ LOG"))
    errors = _scan_log_errors()
    if errors:
        for err in errors[-20:]:
            lines.append(f"  {err[:120]}")
    else:
        lines.append("  (không tìm thấy lỗi trong log files)")

    # 7. Manual Observations
    observations = _collect_manual_observations(interactive)
    lines.append(_section("7. NHẬN XÉT THỰC TẾ TỪ NGƯỜI DÙNG"))
    if isinstance(observations, dict) and "note" not in observations:
        for q, a in observations.items():
            lines.append(f"\n  {q}")
            lines.append(f"    → {a}")
    else:
        lines.append("  " + str(observations.get("note", observations)))

    # Footer
    lines.append("\n" + "=" * 60)
    lines.append("  CÁCH GỬI REPORT NÀY CHO AI:")
    lines.append("  1. Copy toàn bộ nội dung file này")
    lines.append("  2. Paste vào cửa sổ chat với AI")
    lines.append("  3. Thêm mô tả: 'đây là kết quả test thực tế, phân tích và cải thiện'")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Thu thập kết quả test thực tế thành report văn bản."
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Bỏ qua phần nhập nhận xét thủ công (dùng khi chạy tự động)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Đường dẫn file output (mặc định: reports/eval_report_<timestamp>.txt)"
    )
    args = parser.parse_args()

    print("[*] Dang thu thap ket qua test thuc te...")
    print(f"   Log file: {EVENTS_FILE}")
    print(f"   API: {API_BASE}")

    interactive = not args.no_interactive
    report = build_report(interactive=interactive)

    # Lưu file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else REPORTS_DIR / f"eval_report_{ts}.txt"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n[OK] Report da luu tai: {out_path}")
    print("\n" + "-" * 60)
    print("[HUONG DAN] CACH GUI CHO AI:")
    print("  1. Mo file:", out_path)
    print("  2. Ctrl+A -> Ctrl+C (copy toan bo)")
    print("  3. Paste vao chat va viet:")
    print('     "Day la ket qua test thuc te, hay phan tich va de xuat cai thien"')
    print("-" * 60)

    # Preview
    print("\n[PREVIEW] 50 dong dau:")
    print("-" * 60)
    for line in report.split("\n")[:50]:
        print(line)


if __name__ == "__main__":
    main()
