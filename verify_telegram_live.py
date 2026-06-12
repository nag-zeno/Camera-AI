# -*- coding: utf-8 -*-
"""verify_telegram_live.py — Xac nhan Telegram dang hoat dong va gui alert gia lap"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

def api(path, method="GET", body=None):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))

print("=" * 55)
print("XEMTRA TELEGRAM RUNTIME")
print("=" * 55)

# 1. Config hien tai
cfg = api("/api/telegram/config")
print(f"enabled   : {cfg.get('enabled')}")
print(f"chat_id   : {cfg.get('chat_id')}")
print(f"min_level : {cfg.get('min_level')}")
print(f"cooldown  : {cfg.get('cooldown_sec')}s")
print(f"send_photo: {cfg.get('send_photo')}")

import sys
sys.path.insert(0, ".")
try:
    from config import TELEGRAM_CONFIG
    bot_token = TELEGRAM_CONFIG.get("bot_token", "")
    chat_id = TELEGRAM_CONFIG.get("chat_id", "")
except Exception:
    bot_token = ""
    chat_id = ""

if not cfg.get("enabled"):
    print()
    print("[!] TELEGRAM DANG TAT. Bat len...")
    api("/api/telegram/config", "POST", {
        "bot_token"    : bot_token,
        "chat_id"      : chat_id,
        "enabled"      : True,
        "min_level"    : "alert",
        "cooldown_sec" : 30.0,
        "send_photo"   : True,
    })
    print("[OK] Da bat Telegram.")

# 2. Test connection
print()
print("Gui test message...")
try:
    r = api("/api/telegram/test", "POST", {})
    print(f"[OK] {r.get('message', 'Thanh cong')}")
except Exception as e:
    print(f"[ERR] {e}")

# 3. Kiem tra events moi nhat
print()
events_r = api("/api/events?limit=5&level=alert")
events = events_r.get("events", [])
print(f"Events alert gan nhat: {len(events)}")
for e in events[-3:]:
    t = str(e.get("datetime",""))[-8:]
    print(f"  {t} | level={e.get('level'):<8} | action={e.get('action',''):<12} | track=#{e.get('track_id')}")

print()
print("Neu ban thay tin nhan TEST tren Telegram, he thong dang hoat dong!")
print("Canh bao se tu dong gui khi co event level >= alert.")
