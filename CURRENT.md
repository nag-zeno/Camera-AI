# 📝 Tiến độ Dự án AI Security Camera System (CURRENT.md)

*Tệp này ghi lại chi tiết tiến độ công việc, trạng thái các module, lịch sử cập nhật và các bước tiếp theo của dự án. Cần được cập nhật ngay sau mỗi lần chỉnh sửa hoặc nâng cấp.*

---

## 📌 Thông tin chung
- **Ngày cập nhật gần nhất:** 13/06/2026 — 09:15 (GMT+7)
- **Phiên bản hệ thống:** v2.8 (Đang phát triển Tab Recordings & sửa lỗi Video Codec)
- **Trạng thái tổng thể:** Đã thêm giao diện quản lý và xem video Recordings trên Dashboard, viết endpoint stream video hỗ trợ HTTP Range. Đang xử lý lỗi codec video FMP4 chưa tương thích trình duyệt.

---

## 📊 Tóm tắt nhanh Trạng thái Module (1-Line Summary)
1. 🎛️ **Pipeline & CPU Opt:** ✅ Đạt 15+ FPS trên CPU nhờ Round-Robin MediaPipe + Adaptive Skip.
2. 🔄 **ObjectTracker:** ✅ Ổn định, khắc phục ID switching tĩnh ➔ động qua IoU v2.
3. 👤 **RoleClassifier:** ✅ Hoàn thiện RoleNet V3 (ConvNeXt-Tiny, 16 roles). Hỗ trợ export ONNX.
4. 🏃‍♂️ **ActionRecognizer:** ✅ Hoàn thiện ActionNet GRU (8 classes) + Tăng crop padding 20% cho đối tượng ở xa.
5. 🔍 **IdentityManager:** ✅ Đã hoàn thiện logic và có script `add_known_face.py` đăng ký người dùng.
6. 🗺️ **ZoneDetector:** ✅ Hoàn thiện vẽ polygon zone không lệch và **Zone Persistence per-camera**.
7. 🧠 **BehaviorAnalyzer:** ✅ Đo đạc chính xác thời gian lảng vảng, hướng di chuyển, số lượt ghé thăm.
8. ⚡ **ContextEngineML:** ✅ Tích hợp ContextNet V2 (XGBoost) đạt **80.7% accuracy** trên data thực tế.
9. 💬 **NLG Engine:** ✅ Sinh mô tả Tiếng Việt tự nhiên bằng Gemini API + fallback rule-based mượt mà.
10. 📝 **EventLogger:** ✅ Lưu đầy đủ 16 features sự kiện vào `logs/events.jsonl` thời gian thực.
11. 🔔 **TelegramNotifier:** ✅ Push alert + ảnh chụp qua Telegram, cấu hình & test trực quan từ Web UI.
12. 🎬 **AlertRecorder:** ✅ Ghi video clip alert (5s pre-buffer, 8s post-buffer). Đang cập nhật codec sang H.264.
13. 🖥️ **Web Dashboard:** ✅ FastAPI, MJPEG Stream, Analytics Modal (Timeline 24h, quản lý và phát video Recordings, config).
14. 🧠 **SHAP Explanation:** ✅ Giải thích quyết định ContextNet XGBoost (Global Feature Importance & Per-Object Waterfall Chart) trực quan trên dashboard.

---

## 📅 Lịch sử cập nhật gần nhất (Changelog)

### **Phiên bản v2.8 (13/06/2026) — Đang phát triển (In Progress)**
- **Tích hợp Tab xem video Recordings trên Dashboard UI ([templates/dashboard.html](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/templates/dashboard.html))**
  - Thiết kế tab **"Recordings"** bên trong modal Analytics với grid hiển thị các video clip cảnh báo.
  - Tích hợp ô tìm kiếm (theo tên file, role) và các bộ lọc nhanh (Tất cả, 🔴 Critical, 🚨 Alert).
  - Xây dựng Video Player modal tùy chỉnh với đầy đủ thông tin (role, mức độ cảnh báo, ngày giờ, dung lượng), hỗ trợ phím tắt (`←`/`→` để chuyển clip, `Esc` để đóng), tự động đóng và dừng phát khi bấm ra ngoài.
  - Bổ sung nút Tải xuống (Download) và Xóa (Delete) trực tiếp cho mỗi clip.
- **Endpoint Stream Video hỗ trợ HTTP Range ([app.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/app.py))**
  - Viết endpoint `/recordings/{filename}` trả về stream dữ liệu video có hỗ trợ header `Range` (HTTP 206 Partial Content), cho phép trình duyệt tua (seek) video mượt mà.
- **Phát hiện lỗi Codec và lên phương án xử lý (In Progress)**
  - Phát hiện video được ghi bằng codec `FMP4` (MPEG-4 Part 2) không được hỗ trợ mặc định bởi các trình duyệt hiện đại (như Chrome, Firefox), dẫn đến lỗi không load được video ở client.
  - Lên phương án giải quyết:
    1. Sửa `alert_recorder.py` ghi trực tiếp bằng codec H.264.
    2. Sửa endpoint `/recordings/{filename}` để tự động convert on-the-fly từ FMP4 sang H.264 bằng `FFmpeg` (thông qua `imageio_ffmpeg`) đối với các file cũ.

### **Phiên bản v2.6 (06/06/2026)**
- **Tích hợp SHAP Explanation cho ContextNet XGBoost ([app.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/app.py), [pipeline.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/pipeline.py), [context_engine_ml.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/modules/context_engine_ml.py))**
  - Bổ sung hàm tính SHAP values (`get_shap_explanation`) và feature importance (`get_feature_importance`) trong backend.
  - Thêm 2 endpoints: `GET /api/shap/feature_importance` và `GET /api/shap/explain?track_id=X`.
  - Thiết kế UI Card **"ContextNet SHAP"** ở sidebar dashboard với 2 tab:
    - **Tổng quan (Global):** Vẽ biểu đồ độ quan trọng (Feature Importance) của tất cả 16 ML features dựa trên mô hình XGBoost hiện tại.
    - **Theo Object (Local):** Chọn đối tượng đang tracking trong thời gian thực để hiển thị biểu đồ waterfall giải thích chi tiết mức độ đóng góp (tích cực hay tiêu cực) của từng feature tới quyết định cảnh báo (alert level).
  - Tự động đồng bộ hóa dropdown list các object đang được tracking và tự động cập nhật biểu đồ SHAP sau mỗi chu kỳ `fetchData()`.

### **Phiên bản v2.5 (06/06/2026)**
- **Bổ sung 9 trường ML cho `AlertEvent` ([models.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/models.py))**
  - Khắc phục lỗi thiếu trường khi chuyển đổi sang dictionary của `AlertEvent`.
  - Các trường được thêm đầy đủ: `role_confidence`, `identity`, `zone_type`, `zone_status`, `time_in_zone`, `loitering`, `direction`, `visit_count`, `frames_tracked`.
  - Đảm bảo ghi đầy đủ 16 trường dữ liệu phục vụ huấn luyện ContextNet ML vào `logs/events.jsonl`.
- **Tạo script xuất dữ liệu & Retrain ContextNet ([scripts/export_training_data.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/scripts/export_training_data.py))**
  - Chuyển đổi logs sự kiện thực tế thành vector đặc trưng 16 chiều.
  - Hỗ trợ trộn dữ liệu thực tế với dữ liệu mô phỏng (synthetic data) và thực hiện huấn luyện tự động với XGBoost.
  - Hỗ trợ các tham số dòng lệnh: `--retrain`, `--no-synthetic`, `--weight N`.
- **Huấn luyện ContextNet V2 ([models/context_net.pkl](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/models/context_net.pkl))**
  - Chạy huấn luyện thành công với 12,348 sự kiện thực tế (nhân bản x3 thành 37,044 dòng) kết hợp dữ liệu synthetic tạo thành 69,323 dòng mẫu.
  - Độ chính xác mô hình đạt **80.7%** (khắc phục trạng thái overfitting 99.9% của dữ liệu synthetic cũ).
  - Xuất ra file dữ liệu đã gộp `data/context_training_data_v2.csv` và dữ liệu thực tế `data/real_context_data.csv`.
- **Tạo công cụ quản lý Known Faces ([scripts/add_known_face.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai%20gg/camera-ai/scripts/add_known_face.py))**
  - Script tương tác dòng lệnh giúp người dùng đăng ký nhận dạng khuôn mặt.
  - Các chức năng: import ảnh có sẵn từ file/thư mục, chụp trực tiếp từ webcam, xem danh sách người đã đăng ký (`--list`), và xóa người khỏi cơ sở dữ liệu.

### **Phiên bản v2.0 (Tháng 05/2026) — Nâng cấp CPU & Persistence**
- Áp dụng **Round-Robin Scheduling** cho MediaPipe Pose trong `pipeline.py` để tối ưu CPU.
- Triển khai **Zone Persistence per-camera**: Mỗi camera (webcam, RTSP, video file) được lưu cấu hình vùng giám sát riêng biệt trong `data/zones/` (ví dụ: `rtsp_192_168_1_200_ch201.json`).
- Tích hợp **NLG Engine** với Gemini API giúp sinh cảnh báo bằng ngôn ngữ tự nhiên tiếng Việt sinh động.
- Nâng cấp **TelegramNotifier** và **AlertRecorder** đồng bộ với Dashboard UI mới.

---

## 🛠️ Chi tiết Trạng thái tính năng

### ✅ HOÀN THIỆN & ỔN ĐỊNH
- **CPU Optimization:** Đã tối ưu hóa luồng suy luận. Không còn tình trạng nghẽn CPU 100% khi có nhiều người.
- **ID Stability (ObjectTracker):** Thuật toán IoU v2 kết hợp dự đoán vận tốc giúp theo dõi mượt mà, hạn chế tối đa việc đổi ID khi người đi qua lại.
- **Vẽ vùng giám sát (Zone Editor):** Không còn bị lệch tọa độ giữa ảnh snapshot và luồng video trực tiếp.
- **Video Recorder & Telegram:** Hoạt động chuẩn xác, gửi thông báo kèm ảnh tức thì và ghi lại video clip có 5 giây đệm trước sự kiện.
- **NLG Engine:** Hoạt động rất tốt, câu văn tự nhiên và có fallback rule-based khi mất kết nối API hoặc hết quota.
- **SHAP Explanation cho ContextNet:** Đã tích hợp đầy đủ biểu đồ giải thích quyết định cảnh báo (Global Importance & Per-Object Waterfall) trực quan lên dashboard.

### 🔄 CẦN CẢI THIỆN THÊM (IN PROGRESS)
- **Độ chính xác của ContextNet V2 (Hiện tại 80.5% sau retrain lần 2):**
  - *Vấn đề tồn đọng:* Log thực tế mới vẫn còn thiếu nhiều ML features (zone_type, action chỉ 0-2% fill rate) vì camera chạy mà không vẽ zone và action recognition ít trigger.
  - *Giải pháp tiếp theo:* Cần **vẽ zone giám sát** trong Zone Editor rồi chạy camera thêm để log zone_type/zone_status đầy đủ hơn, sau đó retrain lại lần nữa để đẩy accuracy > 85-90%.
- **Known Faces Dataset:**
  - Thư mục `data/known_faces/` hiện tại chưa có dữ liệu mẫu. Cần sử dụng `add_known_face.py` để đăng ký khuôn mặt gia đình/nhân viên.
- **Action Recognition fillrate thấp:**
  - Chỉ 212/12,651 (1.6%) events có `action` được nhận dạng. Cần kiểm tra lại logic trigger của ActionRecognizer trong `pipeline.py`.
- **Lỗi không load được video trong tab Recordings (In Progress):**
  - *Vấn đề tồn đọng:* Video ghi bằng OpenCV sử dụng codec `FMP4` không tương thích với trình duyệt Chrome/Firefox (chỉ nghe tiếng hoặc không chạy được).
  - *Giải pháp tiếp theo:* Sửa đổi bộ ghi video ghi codec H.264 và xây dựng bộ transcode thời gian thực cho video cũ.

### ❌ CHƯA THỰC HIỆN / ĐỢI TRIỂN KHAI (PENDING)
- **Kiểm thử RoleNet V3 trên Camera thực tế Việt Nam:** Cần thu thập thêm video/ảnh thực tế tại Việt Nam để đánh giá domain gap của mô hình ConvNeXt-Tiny.
- **Authentication / Bảo mật API:** Hiện tại FastAPI dashboard đang mở tự do trong mạng LAN. Cần thêm Basic Auth hoặc JWT token nếu triển khai diện rộng.
- **Cải thiện Action Recognition fillrate:** Điều tra nguyên nhân ActionNet chỉ nhận dạng được ~1.6% events.

---

## 📂 Các Tệp tin Quan trọng & Vị trí

| Tên tệp | Vị trí | Vai trò |
| :--- | :--- | :--- |
| **`app.py`** | [app.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/app.py) | Điểm chạy chính (FastAPI Server + API endpoints + Dashboard) |
| **`pipeline.py`** | [pipeline.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/pipeline.py) | Điều phối luồng xử lý video (14 modules kết nối) |
| **`context_engine_ml.py`** | [context_engine_ml.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/modules/context_engine_ml.py) | ML-enhanced Context Reasoning Engine (tính toán SHAP và giải thích quyết định) |
| **`config.py`** | [config.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/config.py) | Tập trung toàn bộ cấu hình hệ thống (ngưỡng, đường dẫn, tham số) |
| **`models.py`** | [models.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/models.py) | Các lớp định nghĩa cấu trúc dữ liệu (`TrackedObject`, `AlertEvent`,...) |
| **`nlg_engine.py`** | [nlg_engine.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/modules/nlg_engine.py) | Xử lý ngôn ngữ tự nhiên sinh cảnh báo tiếng Việt |
| **`zone_detector.py`** | [zone_detector.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/modules/zone_detector.py) | Phát hiện xâm nhập vùng và tự động lưu/tải zone per-camera |
| **`add_known_face.py`** | [add_known_face.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/scripts/add_known_face.py) | Công cụ quản lý, đăng ký khuôn mặt người quen qua webcam/ảnh |
| **`export_training_data.py`** | [export_training_data.py](file:///g:/My%20Drive/DoAnTotNghiep/camera-ai gg/camera-ai/scripts/export_training_data.py) | Xuất dữ liệu log và retrain ContextNet XGBoost |

---

## 🚀 Các Bước đề xuất Tiếp theo

### ✅ ĐÃ HOÀN THÀNH (13/06/2026)
1. ~~**Tích hợp Tab Recordings & Custom Player:** Xem danh sách clip, lọc, tìm kiếm, phát, tải xuống và xóa trên Dashboard UI.~~
2. ~~**Xây dựng Video Stream với HTTP Range:** Hỗ trợ seek/tua video từ frontend.~~

### 🔜 BƯỚC TIẾP THEO ƯU TIÊN CAO

1. **Khắc phục lỗi video không chạy trên trình duyệt:**
   - Sửa file `modules/alert_recorder.py` chuyển fourcc sang ghi bằng codec `H264` hoặc `avc1`.
   - Cập nhật `/recordings/{filename}` trong `app.py` để tự động chuyển mã (transcode) file cũ định dạng `FMP4` thành `H264` sử dụng `imageio_ffmpeg` khi stream.

2. **Bước A (Vẽ Zone giám sát):**
   - Mở dashboard, vào Zone Editor, vẽ ít nhất 1 vùng `restricted` và 1 vùng `allowed`.
   - Sau đó chạy camera thêm ~30 phút để thu log có `zone_type` và `zone_status` đầy đủ.

3. **Bước B (Điều tra Action Recognition):**
   - Chỉ 1.6% events có action — kiểm tra ngưỡng trigger ActionRecognizer trong `pipeline.py` và `config.py`.
   - Cần action như `standing/walking/running` fill đủ để mô hình học tốt hơn.

4. **Bước C (Retrain lần 3 sau khi có zone data):**
   - Sau bước A+B, chạy lại retrain:
     ```bash
     python scripts/export_training_data.py --retrain
     ```
   - Mục tiêu: accuracy > 85%.

5. **Bước D (Đăng ký khuôn mặt người quen):**
   - Chạy lệnh sau để đăng ký 1-2 người vào hệ thống:
     ```bash
     python scripts/add_known_face.py
     ```
