# 🎥 AI Security Camera System

Hệ thống camera an ninh thông minh với nhận dạng vai trò xã hội (16 classes), nhận dạng hành động (8 actions), và đánh giá cảnh báo theo ngữ cảnh (ContextNet ML).

---

## 🚀 Chạy Nhanh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server (mặc định webcam index 0)
python app.py

# Truy cập dashboard
http://localhost:8000
```

---

## 🏗 Kiến Trúc Pipeline

```
Camera Frame
     │
     ▼
[YOLOv8n] → Detect objects
     │
     ▼
[ObjectTracker] → ByteTrack ID tracking
     │
     ├──► [RoleClassifier V3]   → ConvNeXt-Tiny, 16 roles, ~99.96% val acc
     ├──► [ActionRecognizer]    → MediaPipe Pose + GRU, 8 actions, 95.31% val acc
     ├──► [IdentityManager]     → Face recognition (known_faces/)
     ├──► [ZoneDetector]        → Polygon zone intrusion detection
     ├──► [BehaviorAnalyzer]    → Loitering, direction analysis
     └──► [ContextEngineML]     → XGBoost alert engine (6 levels)
                │
                ▼
         [EventLogger]       → JSONL log file
         [TelegramNotifier]  → Push alert + photo ← NEW
         [AlertRecorder]     → Auto video clip recording ← NEW
         [Visualizer]        → Frame annotation
         [FastAPI Dashboard] → Web UI
```

---

## ✅ Models Đã Train

| Model | Architecture | Accuracy | Status |
|-------|-------------|----------|--------|
| **RoleNet V3** | ConvNeXt-Tiny | ~99.96% val | ✅ Active |
| **RoleNet V2** | EfficientNet-B2 + TTA | 64.41% val | Fallback |
| **RoleNet V1** | MobileNetV3-Small | 59.05% val | Deprecated |
| **ContextNet** | XGBoost (300 trees) | 100%* synth | ✅ Active |
| **ActionNet** | MediaPipe + GRU | 95.31% val / 90.62% test | ✅ Active |

*100% trên synthetic data — cần retrain với data thực.

---

## 🌐 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/` | Dashboard Web UI |
| `GET`  | `/video_feed` | MJPEG live stream |
| `GET`  | `/api/status` | FPS, ActionNet mode, zone count |
| `GET`  | `/api/objects` | Tracked objects hiện tại |
| `GET`  | `/api/events` | Event log (filter level/limit/since) |
| `GET`  | `/api/stats` | Thống kê tổng hợp |
| `GET`  | `/api/zones` | Danh sách zones |
| `POST` | `/api/zones` | Cập nhật zones (persistent) |
| `DELETE`| `/api/zones` | Xóa tất cả zones |
| `POST` | `/api/source` | Đổi nguồn video |
| `POST` | `/api/upload` | Upload file video |
| `GET`  | `/api/source` | Nguồn video hiện tại |
| `GET`  | `/api/snapshot` | Ảnh tĩnh cho Zone Editor |
| **Telegram** | | |
| `GET`  | `/api/telegram/config` | Lấy cấu hình Telegram |
| `POST` | `/api/telegram/config` | Cập nhật Telegram config |
| `POST` | `/api/telegram/test` | Test kết nối Telegram |
| **Alert Recorder** | | |
| `GET`  | `/api/recorder/config` | Cấu hình video recorder |
| `POST` | `/api/recorder/config` | Cập nhật recorder config |
| `GET`  | `/api/recorder/clips` | Danh sách clip đã ghi |
| `GET`  | `/api/recorder/clips/{filename}` | Download clip |
| `DELETE`| `/api/recorder/clips/{filename}` | Xóa clip |
| **Analytics** | | |
| `GET`  | `/api/analytics/summary` | Thống kê lịch sử + recordings |
| `GET`  | `/api/analytics/timeline` | Timeline events theo giờ |
| `GET`  | `/api/analytics/export` | Export toàn bộ log (JSON) |

---

## 📊 Dashboard Features

- **Live MJPEG stream** — 30fps, annotated
- **Detected Objects list** — với action chip, confidence bar, zone tag
- **Action Distribution chart** — bar chart real-time
- **FPS Sparkline** — 60-point rolling chart
- **Recent Alerts feed** — filter by level, Export CSV
- **Toast notifications** — popup khi có alert mới
- **Zone Editor** — vẽ polygon zone trực tiếp trên video snapshot
- **Source switcher** — Webcam / Upload file / Path/URL
- **📊 Analytics modal** ← NEW
  - Overview: stats tổng hợp + Timeline 24h chart + Top roles/actions
  - Recordings: danh sách clip, download, xóa
  - Telegram config: bot token, chat ID, test kết nối
  - Recorder config: pre/post buffer, cooldown, max clips

---

## 🔔 Telegram Push Notification

Cấu hình tại **Analytics → Telegram** hoặc `POST /api/telegram/config`:

```json
{
  "bot_token": "123456:ABC-...",
  "chat_id": "-1001234567890",
  "enabled": true,
  "min_level": "alert",
  "cooldown_sec": 30,
  "send_photo": true
}
```

**Hướng dẫn:**
1. Tạo bot tại `@BotFather` → lấy `BOT_TOKEN`
2. Gửi tin nhắn cho bot → truy cập `https://api.telegram.org/bot<TOKEN>/getUpdates` → lấy `chat_id`
3. Điền vào Dashboard → Analytics → Telegram → Lưu & Bật

---

## 🎬 Alert Video Recording

Tự động ghi clip MP4 khi có sự kiện alert/critical:
- **Pre-buffer**: 5 giây TRƯỚC sự kiện (vòng buffer)
- **Post-buffer**: 8 giây SAU sự kiện
- Lưu tại `recordings/alert_YYYYMMDD_HHMMSS_level_role.mp4`
- Cấu hình tại **Analytics → Recorder**

---

## ⚙️ Cấu Trúc Thư Mục

```
camera-ai/
├── app.py                    # FastAPI server + Dashboard HTML
├── pipeline.py               # Pipeline orchestrator
├── config.py                 # Toàn bộ cấu hình
├── models.py                 # Pydantic data models
├── templates/
│   └── dashboard.html        # Dashboard UI (50KB+)
├── modules/
│   ├── object_detector.py    # YOLOv8 detection
│   ├── object_tracker.py     # ByteTrack variant
│   ├── role_classifier.py    # RoleNet V3→V2→V1→Rule chain
│   ├── action_recognizer.py  # MediaPipe + GRU → Rule fallback
│   ├── identity_manager.py   # Face recognition
│   ├── zone_detector.py      # Polygon zone detection (persistent)
│   ├── behavior_analyzer.py  # Loitering, direction
│   ├── context_engine_ml.py  # XGBoost ContextNet
│   ├── event_logger.py       # JSONL event logging
│   ├── visualizer.py         # Frame annotation
│   ├── video_processor.py    # Video capture
│   ├── telegram_notifier.py  # Telegram push alerts
│   └── alert_recorder.py     # Video clip recording ← NEW
├── models/
│   ├── rolenet_v3_best.pt    # RoleNet V3 (~107MB)
│   ├── actionnet_gru.pt      # ActionNet GRU (840KB)
│   ├── context_net.pkl       # ContextNet XGBoost (1.9MB)
│   └── pose_landmarker.task  # MediaPipe Pose (5.6MB)
├── data/
│   ├── known_faces/          # Ảnh nhận dạng khuôn mặt
│   ├── zones.json            # Zones persistent config
│   └── sample_videos/
├── recordings/               # Alert video clips ← NEW
├── logs/
│   └── events.jsonl          # Event log
├── tests/
│   ├── test_tracker_stability.py
│   ├── test_pipeline_smoke.py
│   ├── test_rolenet.py
│   └── test_actionnet.py
└── scripts/                  # Training & data collection scripts
```

---

## 🧪 Chạy Tests

```bash
# Tracker stability test
python tests/test_tracker_stability.py

# Pipeline smoke test
python tests/test_pipeline_smoke.py
```

---

## 📋 Trạng Thái Tính Năng

| Tính năng | Trạng thái |
|-----------|-----------|
| YOLOv8 detection | ✅ Hoàn chỉnh |
| ByteTrack tracking | ✅ Ổn định (đã fix ID switching) |
| RoleNet V3 (ConvNeXt) | ✅ ~99.96% val acc |
| ActionNet (GRU) | ✅ 95.31% val acc |
| ContextNet (XGBoost) | ✅ Active (cần data thực) |
| Zone Editor (persistent) | ✅ Hoàn chỉnh |
| Telegram notifications | ✅ Hoàn chỉnh |
| Alert video recording | ✅ Mới thêm |
| Historical analytics | ✅ Mới thêm |
| Analytics dashboard modal | ✅ Mới thêm |
| Multi-camera | ❌ Chưa có |
| Authentication | ❌ Chưa có |
| Face recognition data | ⚠️ Cần thêm ảnh vào known_faces/ |
| ContextNet real data | ⚠️ Cần retrain với log thực |
