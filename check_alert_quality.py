# -*- coding: utf-8 -*-
"""check_alert_quality.py — Xem chất lượng alerts và lý do TelegramNotifier không gửi"""
import json
import time
from collections import Counter

print("Dang phan tich event log...")
with open("logs/events.jsonl", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

events = [json.loads(l) for l in lines]

print(f"Tong events: {len(events)}")
print(f"Phan bo level: {dict(Counter(e.get('level') for e in events))}")
print()

# Phan tich events trong 5 phut gan nhat
recent = [e for e in events if e.get("timestamp", 0) > time.time() - 300]
print(f"Events trong 5 phut gan nhat: {len(recent)}")
if recent:
    track_ids = Counter(e.get("track_id") for e in recent)
    print(f"Track IDs xuat hien: {dict(track_ids.most_common(5))}")
    print()

    # Xem cooldown van de: neu 1 track_id gui nhieu event trong 30s
    print("Phan tich cooldown theo track_id (events alert+ trong 5 phut):")
    alert_recent = [e for e in recent if e.get("level") in ("alert","critical")]
    print(f"  Alert/Critical events: {len(alert_recent)}")

    # Tinh khoang cach giua cac event cua cung 1 track
    by_track = {}
    for e in alert_recent:
        tid = e.get("track_id", -1)
        ts  = e.get("timestamp", 0)
        by_track.setdefault(tid, []).append(ts)

    print()
    print("  Khoang cach thoi gian giua events cua moi track (s):")
    for tid, timestamps in sorted(by_track.items())[:5]:
        timestamps.sort()
        if len(timestamps) > 1:
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_gap = sum(gaps) / len(gaps)
            print(f"    track #{tid}: {len(timestamps)} events, avg gap = {avg_gap:.1f}s, min gap = {min(gaps):.1f}s")
            if avg_gap < 30:
                print(f"      => Gap < cooldown 30s -> chi gui TIEN NHAT, bo qua phần còn lại")
        else:
            print(f"    track #{tid}: 1 event")

print()
print("Giai thich van de:")
print("  - TelegramNotifier co cooldown 30s PER track_id")
print("  - Neu events xay ra lien tuc, chi TIN NHAN DAU TIEN duoc gui")
print("  - Neu pipeline khong generate alert trong khi test => khong co gi gui")
print()

# Kiem tra xem app dang chay khong
import socket
try:
    s = socket.create_connection(("127.0.0.1", 8000), timeout=1)
    s.close()
    print("[OK] App dang chay tren port 8000")

    import urllib.request
    r = urllib.request.urlopen("http://127.0.0.1:8000/api/telegram/config", timeout=3)
    cfg = json.loads(r.read().decode())
    print()
    print("CONFIG TELEGRAM RUNTIME (trong pipeline dang chay):")
    print(f"  enabled    : {cfg.get('enabled')}")
    print(f"  chat_id    : {cfg.get('chat_id')}")
    print(f"  min_level  : {cfg.get('min_level')}")
    print(f"  cooldown   : {cfg.get('cooldown_sec')}s")
    if not cfg.get("enabled"):
        print()
        print("  [!] PHAT HIEN VAN DE: Telegram DISABLED trong runtime!")
        print("  => startup() chua goi configure() hoac bi loi")
except Exception as e:
    print(f"[!] App KHONG chay hoac loi: {e}")
    print("  => Phai chay 'python app.py' truoc")
