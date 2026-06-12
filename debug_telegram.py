# -*- coding: utf-8 -*-
"""debug_telegram.py — Chẩn đoán vấn đề Telegram notification"""
import json
import sys
import time
import threading
import urllib.request
import urllib.error
from collections import Counter

sys.path.insert(0, ".")
try:
    from config import TELEGRAM_CONFIG
    TOKEN   = TELEGRAM_CONFIG.get("bot_token", "")
    CHAT_ID = TELEGRAM_CONFIG.get("chat_id", "")
except Exception:
    TOKEN   = ""
    CHAT_ID = ""

# ── 1. Kiểm tra event log ──────────────────────────────────────
print("=" * 60)
print("1. KIEM TRA EVENT LOG")
print("=" * 60)
try:
    with open("logs/events.jsonl", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"   Tong events da log: {len(lines)}")
    levels = Counter(json.loads(l).get("level", "?") for l in lines)
    print(f"   Events by level: {dict(levels)}")

    print("\n   5 events gan nhat:")
    for line in lines[-5:]:
        evt = json.loads(line)
        print(f"     level={evt.get('level'):<10} action={evt.get('action',''):<15} time={str(evt.get('datetime',''))[-8:]}")

    # Kiem tra co event level >= alert khong
    alert_events = [json.loads(l) for l in lines if json.loads(l).get("level") in ("alert","critical")]
    print(f"\n   Events muc alert/critical: {len(alert_events)}")
    if alert_events:
        last = alert_events[-1]
        print(f"   Event cuoi: level={last.get('level')}, action={last.get('action')}, time={str(last.get('datetime',''))[-8:]}")
except FileNotFoundError:
    print("   [!] Chua co file events.jsonl — chua chay pipeline?")
except Exception as e:
    print(f"   [ERR] {e}")

# ── 2. Kiem tra config TELEGRAM_CONFIG ────────────────────────
print()
print("=" * 60)
print("2. KIEM TRA TELEGRAM_CONFIG TRONG CONFIG.PY")
print("=" * 60)
try:
    sys.path.insert(0, ".")
    from config import TELEGRAM_CONFIG
    print(f"   bot_token  : ...{TELEGRAM_CONFIG['bot_token'][-10:]}")
    print(f"   chat_id    : {TELEGRAM_CONFIG['chat_id']}")
    print(f"   enabled    : {TELEGRAM_CONFIG['enabled']}")
    print(f"   min_level  : {TELEGRAM_CONFIG['min_level']}")
    print(f"   cooldown   : {TELEGRAM_CONFIG['cooldown_sec']}s")
    print(f"   send_photo : {TELEGRAM_CONFIG['send_photo']}")

    if not TELEGRAM_CONFIG.get("chat_id"):
        print("   [!] chat_id TRONG CONFIG BI TRONG!")
    if not TELEGRAM_CONFIG.get("enabled"):
        print("   [!] enabled=False — TELEGRAM BI TAT!")
except Exception as e:
    print(f"   [ERR] Khong doc duoc config: {e}")

# ── 3. Test kết nối Telegram API ────────────────────────────────
print()
print("=" * 60)
print("3. TEST KET NOI TELEGRAM API TRUC TIEP")
print("=" * 60)
try:
    url  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": CHAT_ID, "text": "[DEBUG] Test ket noi Telegram OK"}).encode("utf-8")
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json; charset=utf-8"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if result.get("ok"):
        print("   [OK] API ket noi thanh cong! Tin nhan test da gui.")
    else:
        print(f"   [ERR] API tra ve: {result}")
except Exception as e:
    print(f"   [ERR] Khong gui duoc: {e}")

# ── 4. Simulate notify qua TelegramNotifier module ───────────────
print()
print("=" * 60)
print("4. SIMULATE NOTIFY QUA MODULE TELEGRAMNOTIFIER")
print("=" * 60)
try:
    from modules.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.configure(
        bot_token    = TOKEN,
        chat_id      = CHAT_ID,
        enabled      = True,
        min_level    = "warning",   # Ha xuong warning de de test
        cooldown_sec = 5.0,
        send_photo   = False,       # Khong can anh khi test
    )
    print(f"   Notifier enabled: {notifier._cfg.enabled}")
    print(f"   Notifier min_level: {notifier._cfg.min_level}")

    # Tao fake event
    fake_event = {
        "level"       : "alert",
        "object_role" : "unknown",
        "action"      : "climbing",
        "zone_name"   : "restricted_area",
        "reason"      : "[DEBUG TEST] Simulate alert event",
        "track_id"    : 999,
        "confidence"  : 0.92,
        "datetime"    : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"   Gui fake event: level={fake_event['level']}, action={fake_event['action']}")
    notifier.notify(fake_event, frame=None)

    # Cho worker xu ly
    time.sleep(4)
    print("   [OK] Da gui qua queue — kiem tra Telegram!")
    notifier.stop()
except Exception as e:
    print(f"   [ERR] {type(e).__name__}: {e}")

print()
print("=" * 60)
print("XONG. Kiem tra Telegram de xem tin nhan DEBUG.")
print("=" * 60)
