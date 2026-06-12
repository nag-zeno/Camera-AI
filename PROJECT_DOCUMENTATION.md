# 📋 TÀI LIỆU KỸ THUẬT DỰ ÁN: HỆ THỐNG CAMERA AI AN NINH

> **Phiên bản tài liệu:** 1.0 — Ngày cập nhật: 12/06/2026  
> **Mục đích:** Tài liệu dành cho người trực tiếp phát triển/tiếp nhận dự án. Đọc xong tài liệu này, bạn sẽ hiểu toàn bộ kiến trúc, luồng xử lý, và cách vận hành hệ thống.

---

## MỤC LỤC

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Kiến Trúc Phân Tầng (5-Layer Architecture)](#2-kiến-trúc-phân-tầng)
3. [Cấu Trúc Thư Mục](#3-cấu-trúc-thư-mục)
4. [Luồng Xử Lý 1 Frame (Data Flow)](#4-luồng-xử-lý-1-frame)
5. [Chi Tiết Từng Module](#5-chi-tiết-từng-module)
6. [Data Models — Hợp Đồng Dữ Liệu](#6-data-models)
7. [Cấu Hình Hệ Thống (config.py)](#7-cấu-hình-hệ-thống)
8. [Concurrency & Threading Model](#8-concurrency--threading-model)
9. [ML Models — Huấn Luyện & Triển Khai](#9-ml-models)
10. [Scripts Tiện Ích](#10-scripts-tiện-ích)
11. [Web API & Frontend](#11-web-api--frontend)
12. [Các Lưu Ý Quan Trọng (Gotchas)](#12-các-lưu-ý-quan-trọng)
13. [Hướng Dẫn Chạy Dự Án](#13-hướng-dẫn-chạy-dự-án)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mô Tả
Hệ thống **Camera AI An Ninh** là ứng dụng giám sát video thời gian thực sử dụng trí tuệ nhân tạo. Hệ thống nhận video từ webcam, file, hoặc camera RTSP, sau đó tự động:

- **Phát hiện** người và vật thể (YOLOv8)
- **Theo dõi** đối tượng liên tục qua các frame (IoU Tracker v2 + Re-ID)
- **Nhận diện vai trò** xã hội: shipper, bảo vệ, cảnh sát, học sinh... (RoleNet — ConvNeXt-Tiny)
- **Nhận diện hành động**: đứng, đi, chạy, ngã, leo trèo, đánh nhau... (ActionNet — GRU + MediaPipe Pose)
- **Suy luận ngữ cảnh**: kết hợp vai trò + zone + hành vi + thời gian → quyết định mức cảnh báo (XGBoost hoặc Rule Engine)
- **Cảnh báo thông minh**: sinh câu tiếng Việt tự nhiên (Gemini API), gửi Telegram, ghi video clip

### 1.2 Tech Stack
| Thành phần | Công nghệ |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Computer Vision | OpenCV, YOLOv8 (ONNX/PyTorch) |
| ML Inference | ONNX Runtime (DirectML/CUDA/CPU), PyTorch |
| Pose Estimation | MediaPipe Pose |
| Role Classification | RoleNet v3 (ConvNeXt-Tiny, ~99.96% accuracy) |
| Action Recognition | ActionNet (GRU, 8 classes) |
| Context Reasoning | XGBoost + Rule Engine |
| NLG (Sinh ngôn ngữ) | Google Gemini API |
| Notification | Telegram Bot API |
| Frontend | HTML/CSS/JS (Vanilla), SSE (Server-Sent Events) |
| GPU Acceleration | DirectML (AMD/Intel/NVIDIA), CUDA (NVIDIA) |

---

## 2. KIẾN TRÚC PHÂN TẦNG

Hệ thống được thiết kế theo kiến trúc **5 tầng (5-Layer Architecture)**, mỗi tầng đảm nhận một nhiệm vụ rõ ràng:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Tầng 5: OUTPUT                                 │
│  Visualizer │ EventLogger │ TelegramNotifier │ AlertRecorder │ NLG  │
├─────────────────────────────────────────────────────────────────────┤
│                      Tầng 4: REASONING                              │
│        ContextEngine (Rule) │ ContextEngineML (XGBoost)             │
├─────────────────────────────────────────────────────────────────────┤
│                      Tầng 3: UNDERSTANDING                          │
│  RoleClassifier │ ActionRecognizer │ ZoneDetector │ IdentityManager │
│                 │ BehaviorAnalyzer                                   │
├─────────────────────────────────────────────────────────────────────┤
│                      Tầng 2: PERCEPTION                             │
│              ObjectDetector │ ObjectTracker                          │
├─────────────────────────────────────────────────────────────────────┤
│                      Tầng 1: INPUT                                  │
│                        VideoProcessor                                │
├─────────────────────────────────────────────────────────────────────┤
│                      ORCHESTRATOR                                    │
│              pipeline.py (CameraPipeline)                            │
├─────────────────────────────────────────────────────────────────────┤
│                      WEB INTERFACE                                   │
│                app.py (FastAPI + API Routes)                         │
├─────────────────────────────────────────────────────────────────────┤
│                      FOUNDATION                                      │
│            config.py │ models.py │ gpu_manager.py                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Nguyên tắc thiết kế:**
- Dữ liệu chảy **từ dưới lên trên** (Input → Output)
- Mỗi module **nhận vào và trả ra** các kiểu dữ liệu trong `models.py`
- Module ở tầng trên **không bao giờ** gọi ngược lại module tầng dưới
- `pipeline.py` là **Orchestrator** duy nhất kết nối tất cả module

---

## 3. CẤU TRÚC THƯ MỤC

```
camera-ai/
│
├── app.py                    ← [ENTRY POINT] FastAPI server, API routes, khởi tạo Pipeline
├── pipeline.py               ← [ORCHESTRATOR] Kết nối tất cả module, xử lý frame-by-frame
├── config.py                 ← [CẤU HÌNH] Mọi tham số hệ thống (KHÔNG hard-code ở module)
├── models.py                 ← [DATA MODELS] Định nghĩa kiểu dữ liệu dùng chung
│
├── modules/                  ← [CÁC MODULE CHỨC NĂNG]
│   ├── __init__.py           ← Bản đồ phân tầng, export tên module
│   ├── gpu_manager.py        ← [FOUNDATION] Quản lý GPU/ONNX/PyTorch backend
│   ├── video_processor.py    ← [Tầng 1] Đọc video: webcam / file / RTSP
│   ├── object_detector.py    ← [Tầng 2] Phát hiện đối tượng (YOLOv8)
│   ├── object_tracker.py     ← [Tầng 2] Theo dõi đối tượng (IoU Tracker v2)
│   ├── role_classifier.py    ← [Tầng 3] Nhận diện vai trò (RoleNet CNN)
│   ├── action_recognizer.py  ← [Tầng 3] Nhận diện hành động (MediaPipe + GRU)
│   ├── zone_detector.py      ← [Tầng 3] Xác định khu vực (Polygon zones)
│   ├── identity_manager.py   ← [Tầng 3] Nhận diện người quen (Color histogram)
│   ├── behavior_analyzer.py  ← [Tầng 3] Phân tích hành vi (Loitering, direction)
│   ├── context_engine.py     ← [Tầng 4] Suy luận ngữ cảnh (Rule Engine)
│   ├── context_engine_ml.py  ← [Tầng 4] Suy luận ML (XGBoost, SHAP)
│   ├── nlg_engine.py         ← [Tầng 5] Sinh ngôn ngữ tự nhiên (Gemini API)
│   ├── visualizer.py         ← [Tầng 5] Vẽ annotations lên frame
│   ├── event_logger.py       ← [Tầng 5] Ghi sự kiện ra JSONL
│   ├── telegram_notifier.py  ← [Tầng 5] Gửi thông báo Telegram
│   └── alert_recorder.py     ← [Tầng 5] Ghi video clip khi có alert
│
├── models/                   ← [ML MODELS] Chứa file model đã train
│   ├── yolov8n.onnx          ← YOLOv8 Nano (Object Detection)
│   ├── rolenet_v3.onnx       ← RoleNet v3 ONNX FP32
│   ├── rolenet_v3_quant.onnx ← RoleNet v3 ONNX INT8 (nhanh hơn)
│   ├── rolenet_v3_best.pt    ← RoleNet v3 PyTorch checkpoint
│   ├── rolenet_v3_metadata.json
│   ├── actionnet_gru_best.pt ← ActionNet GRU checkpoint
│   ├── context_net.pkl       ← XGBoost ContextNet (pickle)
│   └── pose_landmarker.task  ← MediaPipe Pose model (tự download)
│
├── data/                     ← [DỮ LIỆU]
│   ├── known_faces/          ← Ảnh người quen (thư mục con = person_id)
│   ├── sample_videos/        ← Video mẫu để test
│   ├── zones/                ← File zones JSON riêng cho mỗi camera
│   └── zones.json            ← File zones legacy (dùng khi migrate)
│
├── scripts/                  ← [SCRIPTS TIỆN ÍCH]
│   ├── export_training_data.py   ← Xuất dữ liệu train RoleNet từ video
│   ├── export_onnx.py            ← Export PyTorch → ONNX
│   ├── collect_action_data.py    ← Thu thập dữ liệu train ActionNet
│   ├── generate_context_data.py  ← Sinh dữ liệu train ContextNet
│   ├── train_context_model.py    ← Train XGBoost ContextNet
│   ├── add_known_face.py         ← Đăng ký người quen
│   ├── augment_dataset.py        ← Augment dữ liệu train
│   └── ...                       ← Các scripts phụ trợ khác
│
├── tests/                    ← [UNIT TESTS]
│   ├── test_pipeline_smoke.py    ← Smoke test pipeline
│   ├── test_actionnet.py         ← Test ActionNet
│   ├── test_rolenet.py           ← Test RoleNet
│   ├── test_onnx_rolenet.py      ← Test ONNX RoleNet
│   └── test_tracker_stability.py ← Test tracker stability
│
├── static/                   ← [FRONTEND] HTML/CSS/JS cho Dashboard
│   ├── index.html            ← Trang chủ Dashboard
│   ├── style.css             ← CSS
│   └── app.js                ← JavaScript
│
├── templates/                ← [JINJA2 TEMPLATES] (nếu dùng server-side render)
├── recordings/               ← [OUTPUT] Video clips khi có alert
├── logs/                     ← [OUTPUT] File log hệ thống
│   └── events.jsonl          ← AlertEvent log (JSON Lines)
│
├── requirements.txt          ← Dependencies
└── .env                      ← Biến môi trường (GEMINI_API_KEY, etc.)
```

---

## 4. LUỒNG XỬ LÝ 1 FRAME (DATA FLOW)

Đây là luồng xử lý **CHI TIẾT** khi 1 frame video đi qua hệ thống:

```
Frame BGR (640×480)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ [1] ObjectDetector.detect(frame)                             │
│     YOLOv8 → List[Detection]                                 │
│     Mỗi Detection có: bbox, class_name, category, confidence│
│     Ví dụ: [person 0.92, motorcycle 0.87, backpack 0.75]    │
└──────────────────┬──────────────────────────────────────────┘
                   │ List[Detection]
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [2] ObjectTracker.update(detections, frame)                   │
│     IoU Matching + Velocity EMA + Appearance Histogram       │
│     Two-stage: Stage1 (IoU cao) → Stage2 (Appearance)        │
│     Re-ID buffer 30s: khôi phục track đã mất                │
│     Output: List[TrackedObject] (chỉ confirmed tracks)       │
│     Mỗi TrackedObject có: track_id ổn định, velocity, ...   │
└──────────────────┬──────────────────────────────────────────┘
                   │ List[TrackedObject]
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [3] PARALLEL PROCESSING (ThreadPoolExecutor, 2 workers)      │
│                                                               │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │ Thread 1:            │  │ Thread 2 (round-robin):      │   │
│  │ RoleClassifier       │  │ ActionRecognizer             │   │
│  │  .classify_batch()   │  │  .recognize() cho 1 person   │   │
│  │ IdentityManager      │  │  MediaPipe Pose → keypoints  │   │
│  │  .identify()         │  │  GRU → action + confidence   │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                                                               │
│  Skip rate: Role mỗi 8 frame, Identity mỗi 15 frame         │
│  ActionRecognizer: round-robin 1 person/frame (KHÔNG song    │
│  song vì MediaPipe không thread-safe)                        │
└──────────────────┬──────────────────────────────────────────┘
                   │ List[TrackedObject] (enriched: role, action, identity)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [4] ZoneDetector.detect(obj)                                  │
│     Polygon containment test (cv2.pointPolygonTest)          │
│     Gán: zone_name, zone_type (allowed/restricted),          │
│          zone_status (entering/inside/leaving/outside)       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [5] BehaviorAnalyzer.analyze(obj)                             │
│     Tính toán: direction (moving_in/out/stationary),         │
│     loitering (đứng quá lâu), visit_count, time_in_zone     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [6] ContextEngine.evaluate(obj) hoặc ContextEngineML         │
│     Rule Engine: 25+ rules sắp theo priority                 │
│     HOẶC XGBoost: 16 features → predict alert level          │
│     Output: alert_level, alert_reason (tiếng Việt)           │
│     NLG: Gemini API sinh câu tự nhiên (fallback: template)   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ [7] OUTPUT LAYER                                              │
│                                                               │
│  EventLogger.log(event)      → events.jsonl                  │
│  TelegramNotifier.notify()   → Telegram push (background)    │
│  AlertRecorder.on_event()    → Video clip (pre+post buffer)  │
│  Visualizer.draw()           → Annotated frame (BGR)         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
            FrameResult {
                frame_id, timestamp,
                annotated_frame (BGR numpy),
                objects: List[TrackedObject],
                events:  List[AlertEvent],
                fps, detection_count
            }
                   │
                   ▼
           app.py → /api/stream (SSE)
           app.py → /video_feed (MJPEG)
```

### 4.1 Tối Ưu Hiệu Năng Trong Pipeline

| Kỹ thuật | Mô tả | Vị trí |
|---|---|---|
| **Per-track skip rate** | RoleNet chạy 1 lần/8 frame, Identity 1 lần/15 frame | `pipeline.py` PERF_CONFIG |
| **Adaptive skip** | Khi FPS < target, tự động tăng skip rate | `pipeline.py` `_process_frame()` |
| **Round-robin Action** | Mỗi frame chỉ chạy MediaPipe cho 1 person (luân phiên) | `pipeline.py` `_process_persons_optimized()` |
| **Batch ONNX inference** | RoleNet xử lý N person crops cùng 1 GPU call | `role_classifier.py` `classify_batch()` |
| **ThreadPoolExecutor** | Role+Identity chạy song song trong 2 worker threads | `pipeline.py` |
| **ONNX INT8 Quantization** | Model ONNX lượng tử hóa INT8 nhanh hơn FP32 | `rolenet_v3_quant.onnx` |
| **Velocity prediction** | Tracker dự đoán vị trí để cải thiện IoU matching | `object_tracker.py` |

---

## 5. CHI TIẾT TỪNG MODULE

### 5.1 `video_processor.py` — Tầng 1: Video Input

**File:** `modules/video_processor.py` (541 dòng)  
**Class:** `VideoProcessor`  
**Vai trò:** Đọc frame từ nguồn video, normalize về kích thước chuẩn.

**3 chế độ hoạt động:**

| Nguồn | Phương thức | Reader Thread |
|---|---|---|
| **RTSP stream** | FFmpeg subprocess pipe | `_rtsp_ffmpeg_loop()` — Ưu tiên |
| **RTSP (fallback)** | OpenCV VideoCapture | `_rtsp_opencv_loop()` — Khi không có FFmpeg |
| **File/Webcam** | OpenCV VideoCapture | `_local_reader_loop()` |

**Tại sao dùng FFmpeg subprocess cho RTSP?**
- OpenCV 4.x có bug `fctx->async_lock failed` khi decode H.265/HEVC RTSP
- FFmpeg subprocess chạy process riêng biệt, output raw BGR frames qua pipe stdout
- Pipeline thread **KHÔNG BAO GIỜ** gọi `cap.read()` trực tiếp — luôn đọc từ buffer `_last_frame`

**API chính:**
```python
vp = VideoProcessor(source="rtsp://admin:pw@192.168.1.200:554/ch201")
vp.open()                          # Khởi động reader thread, chờ frame đầu tiên
for frame_id, frame in vp.frames(): # Iterator trả về (id, BGR numpy)
    process(frame)
vp.release()                       # Giải phóng
```

**Đặc điểm quan trọng:**
- File video: throttle đọc theo FPS gốc (tránh phát nhanh như tua)
- RTSP: tự động reconnect khi mất kết nối (delay 3s, đếm lần reconnect)
- Webcam: `loop_video = True` → quay lại frame 0 khi hết video
- Mask URL RTSP: ẩn password khi log (`rtsp://user:***@host`)

---

### 5.2 `object_detector.py` — Tầng 2: Object Detection

**File:** `modules/object_detector.py`  
**Class:** `ObjectDetector`  
**Vai trò:** Phát hiện đối tượng trong frame bằng YOLOv8.

**Ưu tiên model loading:**
1. **ONNX** (`yolov8n.onnx`) — ONNX Runtime + DirectML/CUDA → nhanh nhất
2. **PyTorch** (`yolov8n.pt`) — Ultralytics → fallback

**Cách hoạt động:**
```
Frame BGR (640×480)
    │
    ▼
  ONNX/PyTorch → Raw detections
    │
    ▼
  NMS (Non-Maximum Suppression)
    │
    ▼
  Filter: confidence ≥ 0.35, area ≥ min_box_area
    │
    ▼
  Map class → ObjectCategory (person/vehicle/animal/accessory)
    │
    ▼
  List[Detection]
```

**Output:** `List[Detection]` — mỗi Detection có:
- `bbox`: BoundingBox (x1, y1, width, height)
- `class_name`: tên lớp COCO (person, car, motorcycle, backpack...)
- `category`: ObjectCategory enum
- `confidence`: float [0, 1]

---

### 5.3 `object_tracker.py` — Tầng 2: Object Tracking

**File:** `modules/object_tracker.py` (598 dòng)  
**Class:** `ObjectTracker`  
**Vai trò:** Gán `track_id` ổn định cho mỗi đối tượng qua nhiều frame.

**Phiên bản v2 — Cải tiến so với v1:**

| Feature | v1 | v2 |
|---|---|---|
| Matching | IoU only | IoU + Appearance (HSV histogram 96-dim) |
| Velocity | Raw | EMA smoothing (α=0.4) |
| Matching strategy | Greedy | Two-stage + Hungarian algorithm |
| Re-ID | Không có | Buffer 30s, Bhattacharyya similarity ≥ 0.65 |
| Ghost track guard | Không có | Min-hits = 2 frame mới confirmed |

**Two-stage Matching:**
```
Stage 1 — IoU cao (chắc chắn):
  Cost = 0.7 × (1-IoU) + 0.3 × appearance_distance
  Ngưỡng: IoU ≥ 0.25
  Dùng Hungarian algorithm (scipy.optimize.linear_sum_assignment)

Stage 2 — Appearance cho tracks đã missed:
  Chỉ dùng appearance distance
  Ngưỡng: IoU ≥ 0.08 VÀ appearance similarity ≥ 0.70
  Greedy matching
```

**Re-ID Buffer:**
- Khi track mất (missed > buffer), chuyển vào `_reid_buffer` nếu đã confirmed
- Khi detection mới không match track nào → tìm trong Re-ID buffer
- Điều kiện ghép: cùng category + appearance similarity ≥ 0.65
- TTL: 30 giây → xóa track cũ ra khỏi buffer

**Appearance Embedding:**
- Color histogram HSV: 32 bins × 3 channels = **96 chiều**
- Chỉ lấy vùng **torso** (20%-80% chiều cao) — giảm ảnh hưởng nền
- Cập nhật bằng EMA: `histogram = 0.8 × cũ + 0.2 × mới`

---

### 5.4 `role_classifier.py` — Tầng 3: Role Classification

**File:** `modules/role_classifier.py` (815 dòng)  
**Class:** `RoleClassifier`  
**Vai trò:** Phân loại vai trò xã hội từ ảnh crop toàn thân.

**16 vai trò (SocialRole):**
```
shipper, doctor, police, military, security, student,
chef, janitor, construction, nurse, postman, technician,
worker, civil_guard, normal, unknown
```

**6 chế độ hoạt động (theo thứ tự ưu tiên):**
1. **ONNX-GPU** — ONNX Runtime DirectML/CUDA (nhanh nhất)
2. **ONNX-CPU** — ONNX Runtime CPU (tối ưu đa nhân)
3. **ML V3** — RoleNet v3 (ConvNeXt-Tiny, ~99.96%) — PyTorch
4. **ML V2** — RoleNet v2 (EfficientNet-B2, 64%) — fallback
5. **ML V1** — RoleNet v1 (MobileNetV3-Small, 59%) — fallback
6. **Rule** — HSV color + accessory rules — fallback cuối cùng

**Preprocessing:**
```
Person crop BGR
  → Resize (128×256) — (W×H)
  → RGB → Normalize (ImageNet mean/std)
  → NCHW tensor
```

**Batch Inference (ONNX):**
- Khi có >1 person, stack tất cả crops thành 1 batch `(N, 3, 256, 128)`
- Single GPU call cho cả batch → hiệu quả gấp N lần

**TTA (Test-Time Augmentation):**
- Chỉ dùng với PyTorch V2/V3
- Gốc + flip ngang → lấy trung bình softmax

**Confidence threshold:**
- V3/ONNX: `< 0.28` → fallback về `normal`
- V2: `< 0.30`
- V1: `< 0.35`

**Accessories Bonus:**
- Nếu phát hiện phụ kiện liên quan gần đây (motorcycle → shipper), cộng bonus 5%

**Singleton pattern:** Model được load 1 lần duy nhất (class-level), shared giữa tất cả instances.

---

### 5.5 `action_recognizer.py` — Tầng 3: Action Recognition

**File:** `modules/action_recognizer.py` (753 dòng)  
**Class:** `ActionRecognizer`  
**Vai trò:** Nhận diện hành động từ chuỗi frame.

**8 hành động (ActionLabel):**
```
standing, walking, running, falling,
climbing, fighting, raising_hand, gathering, unknown
```

**Hai chế độ hoạt động:**

| Mode | Cần | Cách hoạt động |
|---|---|---|
| **ML** | MediaPipe + ActionNet GRU | Keypoints 132-dim → GRU predict |
| **Rule-based** | Không cần gì | Velocity + BBox aspect ratio |

**ML Pipeline:**
```
Person crop BGR (có padding 20%)
  → MediaPipe Pose → 33 keypoints × (x,y,z,visibility) = 132-dim
  → Push vào _TrackActionBuffer (per-track, deque maxlen=30)
  → Khi đủ 30 frame: GRU predict
  
ActionNet GRU Architecture:
  Input : (batch, 30, 132)
  GRU   : hidden=128, layers=2, dropout=0.4
  FC    : 128 → LayerNorm → 64 → GELU → 8
  Output: 8 action probabilities
```

**⚠️ MediaPipe KHÔNG thread-safe:**
- Tất cả call được serialize qua `threading.Lock()`
- Pipeline dùng **round-robin**: mỗi frame chỉ xử lý 1 person, luân phiên
- Điều này giải thích tại sao ActionRecognizer **KHÔNG** chạy song song với RoleClassifier

**Rule-based Enhanced (v2):**
- **Velocity smoothing**: trung bình 5 frame → loại noise/rung camera
- **Scale-aware thresholds**: ngưỡng speed tỉ lệ với kích thước bbox
- **delta_aspect falling guard**: phân biệt ngồi (aspect chậm) vs ngã (đột ngột)
- **Action voting**: mode 5 frame → action ổn định, không nhảy
- **Consecutive guard**: falling cần 3 frame liên tiếp

---

### 5.6 `zone_detector.py` — Tầng 3: Spatial Context

**File:** `modules/zone_detector.py` (209 dòng)  
**Class:** `ZoneDetector`, `Zone`  
**Vai trò:** Xác định đối tượng đang ở khu vực nào.

**Zone types:**
- `allowed` — Khu vực cho phép (cổng chính, sảnh)
- `restricted` — Khu vực cấm (kho, phòng máy)

**Zone status (per-track):**
- `entering` — Vừa bước vào zone
- `inside` — Đang ở trong zone
- `leaving` — Vừa rời zone
- `outside` — Ngoài tất cả zone

**Persistence per-camera:**
- Mỗi camera source → 1 file JSON riêng trong `data/zones/`
- Ví dụ: `data/zones/rtsp_192.168.1.200_ch201.json`
- Khi đổi camera → tự động load zones đúng camera
- API `update_zones()` → lưu xuống file đúng camera

**Zone matching:** Dùng `cv2.pointPolygonTest()` — kiểm tra center point của bbox.

---

### 5.7 `identity_manager.py` — Tầng 3: Identity Awareness

**File:** `modules/identity_manager.py` (207 dòng)  
**Class:** `IdentityManager`  
**Vai trò:** Phân biệt người quen (KNOWN) vs người lạ (UNKNOWN).

**Cách hoạt động:**
- **KHÔNG dùng face recognition** — chỉ dùng color histogram similarity
- Tính histogram HSV 32 bins × 3 channels = 96-dim cho person crop
- So sánh cosine similarity với danh sách đã đăng ký
- Ngưỡng: `similarity ≥ threshold` → KNOWN

**Đăng ký người quen:**
- Cấu trúc: `data/known_faces/<person_id>/<ảnh1.jpg>` ...
- Script: `scripts/add_known_face.py`
- API runtime: `register_person(person_id, frame, obj)`

**Cache:** Per-track, recheck mỗi `recheck_interval` frame (tiết kiệm tính toán).

---

### 5.8 `behavior_analyzer.py` — Tầng 3: Behavior Analysis

**File:** `modules/behavior_analyzer.py`  
**Class:** `BehaviorAnalyzer`  
**Vai trò:** Phân tích hành vi dài hạn từ lịch sử track.

**Output bổ sung cho TrackedObject:**
- `direction`: `stationary` | `moving_in` | `moving_out` | `moving`
- `loitering`: `True` nếu đứng quá lâu tại 1 vị trí (> ngưỡng config)
- `time_in_zone`: số giây có mặt trong zone hiện tại
- `visit_count`: số lần ghé thăm zone

---

### 5.9 `context_engine.py` — Tầng 4: Rule-based Reasoning

**File:** `modules/context_engine.py` (623 dòng)  
**Class:** `ContextEngine`, `ContextRule`  
**Vai trò:** Suy luận ngữ cảnh bằng hệ thống luật (Rule Engine).

**Cấu trúc 1 Rule:**
```python
ContextRule(
    name         = "unknown_loitering_restricted",   # Tên rule
    priority     = 100,                               # Cao hơn = xét trước
    condition_fn = lambda obj, time_ctx: (             # Điều kiện
        obj.role in (UNKNOWN, NORMAL)
        and obj.zone_type == RESTRICTED
        and obj.loitering
    ),
    alert_level  = AlertLevel.CRITICAL,              # Mức cảnh báo
    reason_tpl   = "🚨 Đối tượng khả nghi...",       # Template lý do
)
```

**25+ built-in rules, nhóm theo priority:**

| Priority | Nhóm | Ví dụ rules |
|---|---|---|
| 200-175 | **Action-based** (cao nhất) | fighting → CRITICAL, falling → ALERT, climbing → ALERT, running_restricted → WARNING |
| 100-84 | **Role + Zone** | unknown_loitering_restricted → CRITICAL, unknown_in_restricted → ALERT |
| 70-55 | **Role anomaly** | shipper_in_restricted → WARNING, shipper_night → WARNING |
| 40-30 | **Monitoring** | security_patrol → WATCH, frequent_visitor → WATCH |
| 20-1 | **Normal** | medical_staff → NORMAL, known_person → NORMAL, default_person → NORMAL |
| 0 | **Catch-all** | default_ignore → IGNORE |

**Alert Levels (từ thấp → cao):**
```
IGNORE < NORMAL < WATCH < WARNING < ALERT < CRITICAL
```

**Mở rộng:** Gọi `engine.add_rule(ContextRule(...))` để thêm rule tùy chỉnh.

---

### 5.10 `context_engine_ml.py` — Tầng 4: ML Reasoning

**File:** `modules/context_engine_ml.py` (397 dòng)  
**Class:** `ContextEngineML`  
**Vai trò:** Suy luận bằng XGBoost model thay vì if-else.

**16 features đầu vào:**
```python
FEATURE_NAMES = [
    "role_id", "identity_id", "zone_type_id", "zone_status_id",
    "loitering", "time_in_zone", "visit_count", "direction_id",
    "hour", "role_confidence", "category_id", "frames_tracked",
    "is_night", "is_business_hour",
    "action_id", "action_confidence",   # ← ActionNet features
]
```

**Hành vi:**
1. Tìm model tại `models/context_net.pkl`
2. Nếu có → dùng XGBoost predict
3. Nếu không → tự động fallback về `ContextEngine` (rule-based)

**SHAP Explainability:**
- `get_shap_explanation(obj)` → giải thích tại sao model predict alert level này
- TreeExplainer (lazy-init khi lần đầu gọi)
- Trả về SHAP values, feature importance, predicted class, probabilities

**Backward-compatible:** Nếu model cũ train với 14 features (chưa có action), tự động trim features mới.

---

### 5.11 `nlg_engine.py` — Tầng 5: Natural Language Generation

**File:** `modules/nlg_engine.py` (513 dòng)  
**Class:** `NLGEngine`  
**Vai trò:** Sinh câu thông báo tự nhiên bằng tiếng Việt qua Gemini API.

**Ví dụ output Gemini:**
```
Input: {role: "unknown", zone_type: "restricted", alert_level: "alert", time_period: "đêm khuya"}
Output: "🚨 Coi chừng nha! Có người lạ vừa lọt vào khu vực sau nhà lúc đêm khuya.
         Bạn kiểm tra ngay giúp mình với!"
```

**Thiết kế có Fallback hoàn chỉnh:**
- **Gemini sẵn sàng** → gọi API sinh câu tự nhiên
- **Gemini không khả dụng** (mất mạng, quota hết, key sai) → tự động dùng template nội bộ
- **Hệ thống KHÔNG BAO GIỜ bị gián đoạn** dù Gemini lỗi

**Tối ưu quota:**
- Chỉ gọi Gemini cho alert level ≥ `min_alert_level` (mặc định: warning)
- Các alert thấp (normal, watch) → dùng template luôn, không tốn API call
- Cooldown 60s sau 3 lỗi liên tiếp

**Thread-safe:** Dùng `ThreadPoolExecutor(max_workers=1)` để gọi API không block pipeline.

**Singleton:** `get_nlg_engine()` trả về instance duy nhất toàn dự án.

---

### 5.12 `telegram_notifier.py` — Tầng 5: Push Notification

**File:** `modules/telegram_notifier.py` (369 dòng)  
**Class:** `TelegramNotifier`  
**Vai trò:** Gửi thông báo lên Telegram khi có alert/critical event.

**Features:**
- Gửi text message + ảnh snapshot (JPEG 75%)
- Cooldown per-object (mặc định 30s) — tránh spam
- Background queue bất đồng bộ (không block pipeline)
- Auto retry (exponential backoff) khi lỗi mạng
- Enable/disable runtime qua API

**Cấu hình:** Đặt `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` trong `config.py` hoặc gọi API:
```
POST /api/telegram/config
{
    "bot_token": "...",
    "chat_id": "...",
    "enabled": true,
    "min_level": "alert",
    "cooldown_sec": 30,
    "send_photo": true
}
```

---

### 5.13 `alert_recorder.py` — Tầng 5: Video Recording

**File:** `modules/alert_recorder.py` (353 dòng)  
**Class:** `AlertRecorder`  
**Vai trò:** Tự động ghi clip video khi phát hiện alert/critical event.

**Cơ chế Pre-buffer + Post-buffer:**
```
                    ←── Pre-buffer (5s) ──→ ← Event → ←── Post-buffer (8s) ──→
Frame stream:  ... [f1][f2][f3][f4][f5][f6]|[ALERT!]|[f7][f8][f9][f10]...
                    └─────── Ring buffer ──┘         └─── Tiếp tục ghi ──┘
                                                        
Clip output:   [f1][f2][f3][f4][f5][f6][ALERT][f7][f8][f9][f10]
               │←─────────── ~13 giây clip video ──────────────→│
```

**Đặc điểm:**
- Tên clip: `alert_20260505_213000_alert_shipper.mp4`
- Tự động dọn clip cũ (giữ tối đa 50 clips)
- Per-event cooldown (20s) — tránh ghi quá nhiều
- Nếu event mới xảy ra khi đang ghi → extend post-buffer (không tạo clip mới)

---

### 5.14 `visualizer.py` — Tầng 5: Visualization

**File:** `modules/visualizer.py` (366 dòng)  
**Class:** `Visualizer`  
**Vai trò:** Vẽ annotations lên frame video.

**Vẽ gì:**
- Zone polygons (bán trong suốt, có border + label)
- Bounding boxes (màu theo alert level: xanh/vàng/cam/đỏ)
- Label panel (track_id, role icon, confidence, identity, alert level)
- Action tag (STAND, WALK, RUN, FALL!, CLIMB!, FIGHT!)
- Zone status (ENTER, INSIDE, LEAVE)
- Trail đường di chuyển (fade effect)
- Stats overlay (FPS, People, Vehicles, Alerts)
- CRITICAL: nhấp nháy đỏ-trắng

---

### 5.15 `event_logger.py` — Tầng 5: Event Logging

**File:** `modules/event_logger.py` (149 dòng)  
**Class:** `EventLogger`  
**Vai trò:** Ghi AlertEvent ra file JSONL.

**Format:** JSON Lines (1 JSON object per line)
```json
{"event_id":"evt_abc123","level":"alert","reason":"🔴 Người lạ...","datetime":"2026-06-12 20:30:00","track_id":5,"object_role":"unknown","zone_name":"khu_cam","action":"walking","timestamp":1718198400}
```

**Query:** `get_recent(n=50, level="alert", since=timestamp)`

---

### 5.16 `gpu_manager.py` — Foundation: GPU Management

**File:** `modules/gpu_manager.py` (285 dòng)  
**Vai trò:** Quản lý GPU/inference backend tập trung.

**Ưu tiên provider:**
1. `DmlExecutionProvider` — AMD/Intel/NVIDIA qua DirectML (Windows DX12)
2. `CUDAExecutionProvider` — NVIDIA GPU qua CUDA
3. `TensorrtExecutionProvider` — NVIDIA TensorRT
4. `CPUExecutionProvider` — Luôn có

**API chính:**
```python
from modules.gpu_manager import create_ort_session, get_torch_device, gpu_info

# Tạo ONNX session với GPU tốt nhất
session, provider = create_ort_session("models/yolov8n.onnx")

# Lấy PyTorch device
device = get_torch_device()  # "cuda:0" hoặc "cpu"

# Thông tin GPU
info = gpu_info()  # {gpu_name, vram_gb, dml_available, cuda_available, ...}
```

**Benchmark:** `run_gpu_benchmark()` so sánh DML vs CPU performance.

---

## 6. DATA MODELS — HỢP ĐỒNG DỮ LIỆU

File `models.py` (437 dòng) định nghĩa **TẤT CẢ** kiểu dữ liệu dùng chung giữa các module.

> ⚠️ **CẢNH BÁO:** Thay đổi ở `models.py` ảnh hưởng TOÀN BỘ hệ thống. Phải kiểm tra tương thích với mọi module trước khi sửa.

### 6.1 Enums

| Enum | Giá trị | Dùng ở |
|---|---|---|
| `ObjectCategory` | person, animal, vehicle, accessory, unknown | ObjectDetector, Tracker |
| `SocialRole` | 16 vai trò (shipper, doctor, police...) | RoleClassifier |
| `ActionLabel` | 9 hành động (standing, walking, falling...) | ActionRecognizer |
| `AlertLevel` | ignore, normal, watch, warning, alert, critical | ContextEngine |
| `ZoneType` | allowed, restricted | ZoneDetector |
| `ZoneStatus` | entering, inside, leaving, outside | ZoneDetector |
| `IdentityStatus` | known, unknown | IdentityManager |

### 6.2 Core Dataclasses

```python
@dataclass
class BoundingBox:
    x1: float; y1: float; width: float; height: float
    # Properties: x2, y2, center, area
    # Methods: to_xyxy(), to_int(), iou(other)

@dataclass
class Detection:
    bbox: BoundingBox
    class_name: str         # "person", "car", "backpack"
    category: ObjectCategory
    confidence: float

@dataclass
class TrackedObject:
    # Từ Tracker
    track_id: int
    bbox: BoundingBox
    class_name: str
    category: ObjectCategory
    confidence: float
    frames_tracked: int = 0
    velocity: tuple = (0.0, 0.0)
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    # Từ RoleClassifier
    role: SocialRole = SocialRole.UNKNOWN
    role_confidence: float = 0.0
    role_evidence: Optional[RoleEvidence] = None
    
    # Từ ActionRecognizer
    action: ActionLabel = ActionLabel.UNKNOWN
    action_confidence: float = 0.0
    action_top3: list = field(default_factory=list)
    
    # Từ ZoneDetector
    zone_name: Optional[str] = None
    zone_type: Optional[ZoneType] = None
    zone_status: ZoneStatus = ZoneStatus.OUTSIDE
    
    # Từ BehaviorAnalyzer
    direction: str = "stationary"
    loitering: bool = False
    time_in_zone: float = 0.0
    visit_count: int = 0
    
    # Từ IdentityManager
    identity: IdentityStatus = IdentityStatus.UNKNOWN
    
    # Từ ContextEngine
    alert_level: AlertLevel = AlertLevel.NORMAL
    alert_reason: str = ""
    rule_name: str = ""

@dataclass
class AlertEvent:
    event_id: str
    timestamp: float
    level: AlertLevel
    reason: str
    track_id: int
    object_role: str
    zone_name: str
    action: str
    # Class method: AlertEvent.create(obj) → AlertEvent

@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    annotated_frame: numpy.ndarray   # BGR frame đã vẽ annotations
    objects: list[TrackedObject]
    events: list[AlertEvent]
    fps: float
    detection_count: int
```

**Luồng dữ liệu qua TrackedObject:**
```
Tracker tạo TrackedObject (cơ bản: track_id, bbox, velocity)
    │
    ├── RoleClassifier bổ sung: role, role_confidence, role_evidence
    ├── ActionRecognizer bổ sung: action, action_confidence, action_top3
    ├── IdentityManager bổ sung: identity
    │
    ├── ZoneDetector bổ sung: zone_name, zone_type, zone_status
    ├── BehaviorAnalyzer bổ sung: direction, loitering, time_in_zone, visit_count
    │
    └── ContextEngine bổ sung: alert_level, alert_reason, rule_name
```

---

## 7. CẤU HÌNH HỆ THỐNG (config.py)

File `config.py` (636 dòng) là **trung tâm cấu hình** của toàn hệ thống. **KHÔNG hard-code** tham số trong module.

### 7.1 Các Nhóm Cấu Hình Chính

| Config | Mô tả | Ví dụ tham số |
|---|---|---|
| `VIDEO_CONFIG` | Nguồn video, FPS, kích thước frame | `default_source=0`, `target_fps=25`, `frame_width=640` |
| `DETECTION_CONFIG` | Object detection | `model_name="yolov8n.onnx"`, `conf_threshold=0.35` |
| `TRACKING_CONFIG` | Tracking | `match_thresh=0.2`, `track_buffer=30`, `min_hits=2` |
| `ROLE_CONFIG` | Role classification | `confidence_threshold=0.25`, `hsv_ranges`, `roles` |
| `ACTION_CONFIG` | Action recognition | `window_frames=30`, `hidden_size=128`, `action_classes` |
| `ZONE_CONFIG` | Zone definitions | `zones: [{name, type, polygon, color}]` |
| `REASONING_CONFIG` | Context reasoning | `alert_cooldown=30` |
| `IDENTITY_CONFIG` | Identity matching | `similarity_threshold=0.65`, `recheck_interval=10` |
| `VIS_CONFIG` | Visualization | `bbox_thickness=2`, `font_scale=0.45`, `show_zones=True` |
| `API_CONFIG` | API server | `host="0.0.0.0"`, `port=8000`, `log_file` |
| `NLG_CONFIG` | Gemini NLG | `enabled=True`, `model="gemini-2.5-flash"`, `api_key` |
| `TELEGRAM_CONFIG` | Telegram | `bot_token`, `chat_id`, `enabled` |
| `ALERT_RECORDER_CONFIG` | Video recording | `pre_buffer_sec=5`, `post_buffer_sec=8` |

### 7.2 Project Paths (quan trọng)

```python
PROJECT_ROOT       = Path(__file__).parent
DATA_DIR           = PROJECT_ROOT / "data"
KNOWN_FACES_DIR    = DATA_DIR / "known_faces"
MODELS_DIR         = PROJECT_ROOT / "models"
LOGS_DIR           = PROJECT_ROOT / "logs"
STATIC_DIR         = PROJECT_ROOT / "static"
ZONES_DIR          = DATA_DIR / "zones"
RECORDINGS_DIR     = PROJECT_ROOT / "recordings"
```

### 7.3 Hàm `get_zones_file_for_source(source)`

Trả về file zones riêng cho mỗi camera:
```python
get_zones_file_for_source(0)
# → data/zones/webcam_0.json

get_zones_file_for_source("rtsp://admin:pw@192.168.1.200:554/ch201")
# → data/zones/rtsp_192.168.1.200_ch201.json
```

### 7.4 OpenCV/FFmpeg Environment (QUAN TRỌNG)

```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"       # Bắt buộc TCP cho RTSP
    "|threads;1"               # Tắt multi-thread decoder
    "|thread_type;slice"       # Chỉ dùng slice-threading
    "|fflags;nobuffer"         # Không buffer
    "|flags;low_delay"         # Low-latency
    "|max_delay;500000"        # Max delay 0.5s
    "|reorder_queue_size;0"    # Không reorder
)
```

> ⚠️ **KHÔNG BAO GIỜ** dùng `extra_hw_frames` — sẽ bật async hw decoder gây crash assertion.

---

## 8. CONCURRENCY & THREADING MODEL

### 8.1 Tổng Quan Threads

```
Main Thread (Uvicorn)
  │
  ├── Pipeline Thread (CameraPipeline._pipeline_loop)
  │   │
  │   ├── Reader Thread (VideoProcessor._rtsp_ffmpeg_loop)
  │   │   └── FFmpeg subprocess (process riêng)
  │   │
  │   ├── ThreadPoolExecutor (2 workers)
  │   │   ├── Worker 1: RoleClassifier + IdentityManager
  │   │   └── Worker 2: (idle hoặc secondary task)
  │   │
  │   └── MediaPipe lock (serial) — ActionRecognizer
  │
  ├── TelegramNotifier Worker Thread
  │   └── Background queue xử lý gửi thông báo
  │
  ├── AlertRecorder Worker Thread
  │   └── Background queue ghi video clips
  │
  └── NLG ThreadPoolExecutor (1 worker)
      └── Gemini API calls (không block pipeline)
```

### 8.2 Thread-Safety Rules

| Component | Thread-safe? | Lý do |
|---|---|---|
| `VideoProcessor._last_frame` | ✅ | Bảo vệ bởi `threading.Lock()` |
| `RoleClassifier` | ✅ | Singleton model, inference stateless |
| `ActionRecognizer` (MediaPipe) | ❌ | MediaPipe không thread-safe → serialize qua `_mp_lock` |
| `IdentityManager` | ✅ | Per-track cache, stateless inference |
| `TelegramNotifier` | ✅ | Background queue + lock |
| `NLGEngine` | ✅ | ThreadPoolExecutor + lock |
| `EventLogger` | ✅ | `threading.Lock()` |

### 8.3 ActionRecognizer Round-Robin

```python
# Trong pipeline.py — chỉ chạy action cho 1 person/frame
if persons_needing_action:
    idx = self._frame_count % len(persons_needing_action)
    person = persons_needing_action[idx]
    self._action_recognizer.recognize(frame, person)
```

Nếu có 5 người: person 0 được infer ở frame 0, person 1 ở frame 1, ... person 4 ở frame 4, quay lại person 0 ở frame 5.

---

## 9. ML MODELS — HUẤN LUYỆN & TRIỂN KHAI

### 9.1 RoleNet v3 (ConvNeXt-Tiny)

| Thuộc tính | Giá trị |
|---|---|
| Backbone | ConvNeXt-Tiny (timm) |
| Head | AdaptiveAvgPool → LayerNorm → Linear(768,256) → GELU → Linear(256,16) |
| Input size | 128×256 (W×H) |
| Classes | 16 vai trò |
| Accuracy | ~99.96% (validation) |
| Export | ONNX FP32 + INT8 Quantized |
| Training data | Crop từ video thực tế + augmentation |

**Cách retrain RoleNet:**
```bash
# 1. Xuất dữ liệu train từ video
python scripts/export_training_data.py --source video.mp4

# 2. Augment dữ liệu
python scripts/augment_dataset.py

# 3. Train (thường trên Google Colab với GPU)
# → Tạo file rolenet_v3_best.pt

# 4. Export ONNX
python scripts/export_onnx.py --model rolenet_v3_best.pt --output rolenet_v3.onnx

# 5. Copy vào models/ và khởi động lại server
```

### 9.2 ActionNet (GRU)

| Thuộc tính | Giá trị |
|---|---|
| Architecture | GRU (2 layers, hidden=128, dropout=0.4) |
| Input | (batch, 30, 132) — 30 frames × 33 keypoints × 4 features |
| Classes | 8 hành động |
| Head | LayerNorm → Linear(128,64) → GELU → Linear(64,8) |
| Pose source | MediaPipe Pose (33 keypoints) |

**Cách retrain ActionNet:**
```bash
# 1. Thu thập keypoints từ video
python scripts/collect_action_data.py --source video.mp4

# 2. Hoặc download dataset sẵn (UCF101 keypoints)
python scripts/download_action_data.py
python scripts/fetch_ucf101_keypoints.py

# 3. Train (Colab)
# → Tạo file actionnet_gru_best.pt

# 4. Copy vào models/
```

### 9.3 ContextNet (XGBoost)

| Thuộc tính | Giá trị |
|---|---|
| Algorithm | XGBoost Classifier |
| Features | 16 (role, identity, zone, action, time...) |
| Classes | 6 alert levels |
| Saved format | Pickle (.pkl) |
| Explainability | SHAP TreeExplainer |

**Cách retrain ContextNet:**
```bash
# 1. Sinh dữ liệu train (synthetic + rule-based labeling)
python scripts/generate_context_data.py

# 2. Train
python scripts/train_context_model.py

# → Output: models/context_net.pkl
```

### 9.4 Model Fallback Chain

```
RoleClassifier:  ONNX INT8 → ONNX FP32 → PyTorch V3 → V2 → V1 → Rule-based
ActionRecognizer: MediaPipe + GRU → MediaPipe + Keypoint Rules → Velocity Rules
ContextEngine:   XGBoost ML → Rule Engine (25+ rules)
NLG Engine:      Gemini API → Template tiếng Việt
```

**Mọi module đều có fallback** — hệ thống vẫn chạy khi thiếu bất kỳ model nào.

---

## 10. SCRIPTS TIỆN ÍCH

| Script | Chức năng |
|---|---|
| `export_training_data.py` | Xuất person crops + labels từ video → dataset train RoleNet |
| `export_onnx.py` | Export RoleNet PyTorch → ONNX (FP32 + INT8 quantized) |
| `export_yolo_onnx.py` | Export YOLOv8 → ONNX |
| `collect_action_data.py` | Thu thập MediaPipe keypoints từ video → dataset train ActionNet |
| `download_action_data.py` | Download action datasets sẵn có |
| `fetch_ucf101_keypoints.py` | Trích xuất keypoints từ UCF101 dataset |
| `generate_context_data.py` | Sinh dữ liệu synthetic cho ContextNet (v1) |
| `generate_context_data_v2.py` | Sinh dữ liệu synthetic cho ContextNet (v2, có action features) |
| `train_context_model.py` | Train XGBoost ContextNet |
| `add_known_face.py` | Đăng ký người quen (chụp ảnh từ webcam/video) |
| `augment_dataset.py` | Augment dữ liệu train (flip, rotate, color jitter) |
| `preprocess_dataset.py` | Tiền xử lý + chuẩn hóa dataset |
| `collect_images.py` | Thu thập ảnh train từ nhiều nguồn |
| `collect_eval_report.py` | Tổng hợp báo cáo đánh giá model |
| `convert_md_to_docx.py` | Chuyển Markdown → DOCX (cho báo cáo tốt nghiệp) |
| `dataset_status.py` | Kiểm tra trạng thái dataset (số ảnh, phân bố classes) |

---

## 11. WEB API & FRONTEND

### 11.1 API Endpoints (FastAPI — `app.py`)

**Video & Streaming:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/video_feed` | MJPEG stream (dùng cho `<img>` tag) |
| GET | `/api/stream` | SSE stream (Server-Sent Events) — realtime data |
| GET | `/api/snapshot` | Ảnh frame hiện tại (JPEG) |

**Pipeline Control:**
| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/source` | Đổi nguồn video (webcam/RTSP/file) |
| GET | `/api/status` | Trạng thái pipeline (fps, mode, modules) |
| GET | `/api/gpu` | Thông tin GPU |

**Zone Management:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/zones` | Lấy danh sách zones hiện tại |
| POST | `/api/zones` | Cập nhật zones (polygon, type, color) |
| DELETE | `/api/zones` | Xóa tất cả zones |
| GET | `/api/zones/snapshot` | Frame thô (không annotation) cho Zone Editor |

**Events & Alerts:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/events` | Lấy events gần nhất |
| GET | `/api/events/stats` | Thống kê events (by level, role, zone) |

**Telegram:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/telegram/config` | Cấu hình hiện tại (token masked) |
| POST | `/api/telegram/config` | Cập nhật cấu hình |
| POST | `/api/telegram/test` | Gửi tin nhắn test |

**NLG (Gemini):**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/nlg/status` | Trạng thái NLG Engine |
| POST | `/api/nlg/test` | Test sinh câu mẫu |
| POST | `/api/nlg/key` | Cập nhật Gemini API key |

**Alert Recorder:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/recorder/config` | Cấu hình recorder |
| POST | `/api/recorder/config` | Cập nhật cấu hình |
| GET | `/api/recorder/clips` | Danh sách clips đã ghi |

**SHAP Explainability:**
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/shap/{track_id}` | SHAP values cho 1 tracked object |
| GET | `/api/shap/importance` | Global feature importance |

### 11.2 Frontend Dashboard

- File: `static/index.html` + `static/style.css` + `static/app.js`
- MJPEG video player
- Zone Editor (vẽ polygon trực tiếp trên frame)
- Event timeline
- Module status panel
- Telegram configuration
- NLG test panel

---

## 12. CÁC LƯU Ý QUAN TRỌNG (GOTCHAS)

### 12.1 OpenCV FFmpeg RTSP Bug
```
Assertion fctx->async_lock failed at libavcodec/pthread_frame.c
```
- **Nguyên nhân:** OpenCV FFmpeg backend dùng async hw decoder cho H.265/HEVC
- **Giải pháp:** Dùng FFmpeg subprocess pipe (mặc định), KHÔNG dùng `extra_hw_frames`
- **Config:** Đặt `threads;1` và `thread_type;slice` trong `OPENCV_FFMPEG_CAPTURE_OPTIONS`

### 12.2 MediaPipe Threading
- MediaPipe Pose **KHÔNG thread-safe**
- Tất cả calls PHẢI serialize qua `ActionRecognizer._mp_lock`
- Pipeline dùng round-robin: 1 person/frame

### 12.3 Model Backward Compatibility
- ContextNet model cũ có 14 features, mới có 16 (thêm action)
- `context_engine_ml.py` tự động trim features nếu model cũ hơn
- Log warning khi phát hiện mismatch

### 12.4 Zone Persistence
- Zones được lưu **per-camera** trong `data/zones/`
- Khi đổi camera → load zones đúng camera, KHÔNG dùng zones cũ
- Migration tự động từ file `zones.json` legacy khi lần đầu chạy camera mới

### 12.5 Singleton Pattern
- `RoleClassifier._net`: Model shared giữa tất cả instances
- `ActionRecognizer._pose`: MediaPipe shared (class-level)
- `NLGEngine`: Singleton qua `get_nlg_engine()`
- **Hậu quả:** Nếu thay model, cần restart server

### 12.6 Memory Management
- `_TrackActionBuffer`: deque maxlen=30 → tự xóa frame cũ
- `ObjectTracker._reid_buffer`: TTL 30s → tự dọn
- `AlertRecorder._pre_buffer`: ring buffer cố định
- `EventLogger._events`: **KHÔNG giới hạn** trong memory → cần clear() định kỳ nếu chạy lâu

---

## 13. HƯỚNG DẪN CHẠY DỰ ÁN

### 13.1 Cài Đặt Dependencies

```bash
pip install -r requirements.txt

# Tùy chọn: GPU acceleration
pip install onnxruntime-directml    # Windows AMD/Intel/NVIDIA
# HOẶC
pip install onnxruntime-gpu         # NVIDIA CUDA

# Tùy chọn: Gemini NLG
pip install google-generativeai

# Tùy chọn: MediaPipe
pip install mediapipe
```

### 13.2 Chuẩn Bị Models

Đặt các file model vào thư mục `models/`:
- `yolov8n.onnx` (bắt buộc)
- `rolenet_v3_quant.onnx` hoặc `rolenet_v3.onnx` (khuyến nghị)
- `actionnet_gru_best.pt` (tùy chọn)
- `context_net.pkl` (tùy chọn)

### 13.3 Cấu Hình

1. Sửa `config.py`:
   - `VIDEO_CONFIG["default_source"]` → nguồn video (0 cho webcam, RTSP URL, hoặc file path)
   - `ZONE_CONFIG["zones"]` → định nghĩa zones mặc định

2. Tạo file `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### 13.4 Khởi Động

```bash
# Chạy server
python app.py

# Hoặc dùng uvicorn trực tiếp
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập Dashboard: `http://localhost:8000`

### 13.5 Thứ Tự Khởi Động Nội Bộ

```
1. config.py:     Load cấu hình, tạo thư mục
2. gpu_manager:   Detect GPU, log summary
3. app.py:        Khởi tạo FastAPI, mount static files
4. CameraPipeline.__init__():
   a. ObjectDetector  → load YOLO model (ONNX/PyTorch)
   b. ObjectTracker   → init
   c. RoleClassifier  → load RoleNet (ONNX → PyTorch → Rule)
   d. ActionRecognizer → load MediaPipe + ActionNet GRU
   e. ZoneDetector    → load zones từ file/config
   f. IdentityManager → load known faces
   g. BehaviorAnalyzer → init
   h. ContextEngine   → load XGBoost / init Rule Engine
   i. Visualizer      → init
   j. EventLogger     → load existing events
   k. TelegramNotifier → init (disabled by default)
   l. AlertRecorder   → init
5. CameraPipeline.start():
   a. VideoProcessor.open() → khởi động reader thread
   b. Bắt đầu pipeline thread (_pipeline_loop)
```

---

> **Tài liệu này được tạo bởi AI Assistant dựa trên review toàn bộ source code dự án.**  
> **Khi code thay đổi, hãy cập nhật tài liệu tương ứng để đảm bảo tính chính xác.**
