# BỘ KHUNG MỤC LỤC CHI TIẾT
*(Dự kiến: 50 - 60 trang)*

## Lời Cảm Ơn
## Tóm Tắt Đồ Án
## Danh Mục Từ Viết Tắt
## Danh Mục Bảng Biểu
## Danh Mục Hình Ảnh

---

## Mở Đầu (Khoảng 2 trang)
1. Lý do chọn đề tài: Hạn chế của hệ thống camera truyền thống (chỉ ghi hình, báo động sai).
2. Mục tiêu giải quyết: Xây dựng hệ thống tự động "hiểu bối cảnh" (Context-aware) từ đó tự động đưa ra các mức cảnh báo chính xác, giảm thiểu sự can thiệp thủ công.
3. Đối tượng và phạm vi nghiên cứu.

---

## Chương 1: Cơ sở đề tài và Phân tích thiết kế (Khoảng 8 - 10 trang)
**1.1. Yêu cầu bài toán thực tiễn**
- 1.1.1. Yêu cầu chức năng: Phát hiện, theo dõi, nhận diện vai trò (Role), hành động (Action), đánh giá ngữ cảnh tự động, gửi cảnh báo (Telegram) và ghi hình sự kiện.
- 1.1.2. Yêu cầu phi chức năng: Xử lý video thời gian thực, độ trễ thấp, giao diện giám sát dễ dùng.

**1.2. Tại sao cần hệ thống "hiểu ngữ cảnh" (Context-aware)?**
- 1.2.1. Sự thiếu sót của hệ thống phát hiện chuyển động thông thường.
- 1.2.2. Sự khác biệt cốt lõi: Kết hợp Vai trò + Hành động + Vị trí = Mức độ rủi ro (Ngữ cảnh).

**1.3. Giới hạn và phạm vi của đồ án**
- Tập trung vào 16 vai trò xã hội và 8 hành động cơ bản.
- Chạy trên môi trường một camera (single stream) tại thời điểm hiện tại.

---

## Chương 2: Thiết kế và Xây dựng hệ thống (Khoảng 20 - 25 trang)
**2.1. Cấu trúc và luồng hoạt động tổng thể (Pipeline)**
- 2.1.1. Sơ đồ khối kiến trúc hệ thống tổng thể.
- 2.1.2. Luồng dữ liệu (Data flow) từ Frame ảnh đến Cảnh báo.
- 2.1.3. Cấu trúc cơ sở dữ liệu và lưu trữ (JSONL event log, Persistent Zones).

**2.2. Phân tích Mô hình AI và Lựa chọn công nghệ**
*(Không giải thích lý thuyết, chỉ tập trung vào lý do lựa chọn cho bài toán thực tế)*
- 2.2.1. Mô-đun Phát hiện & Theo dõi (Detection & Tracking): Tại sao kết hợp YOLOv8 và biến thể ByteTrack là tối ưu cho môi trường thực tế.
- 2.2.2. Mô-đun Nhận diện vai trò (RoleNet V3): Tại sao chọn kiến trúc ConvNeXt-Tiny (đạt 99.96% acc) thay vì MobileNet.
- 2.2.3. Mô-đun Nhận diện hành động (ActionNet): Ứng dụng MediaPipe Pose kết hợp mạng hồi quy GRU xử lý chuỗi thời gian (Time-series).
- 2.2.4. Động cơ đánh giá ngữ cảnh (ContextNet): Ứng dụng cây quyết định XGBoost phân loại 6 cấp độ cảnh báo thay vì dùng rule tĩnh (Hard-code).

**2.3. Chiến lược Dữ liệu (Data Strategy)**
- 2.3.1. Thu thập và trích xuất video từ camera an ninh thực tế.
- 2.3.2. Bộ quy tắc dán nhãn (Labeling rules) cho 16 vai trò và 8 hành động.
- 2.3.3. Xử lý dữ liệu nhiễu và chiến lược Augmentation.

---

## Chương 3: Thực nghiệm và Đánh giá kết quả (Khoảng 15 - 20 trang)
**3.1. Giao diện và Kịch bản thử nghiệm**
- 3.1.1. Giao diện Dashboard giám sát (Web UI, Zone Editor, Analytics).
- 3.1.2. Kịch bản 1: Cảnh báo người lạ xâm nhập khu vực cấm.
- 3.1.3. Kịch bản 2: Theo dõi và cảnh báo đối tượng lảng vảng đáng ngờ.

**3.2. Đánh giá hiệu năng và Yêu cầu Phần cứng**
- 3.2.1. Phân tích yêu cầu cấu hình tối thiểu và đề nghị.
- 3.2.2. Đánh giá hiệu năng thực tế: Khả năng duy trì FPS và độ trễ phản hồi cảnh báo.

**3.3. Góc nhìn Thực tiễn - Khắc phục sự cố (Troubleshooting)**
- 3.3.1. Vấn đề 1: Hiện tượng mất dấu và nhảy ID (ID Switching) trong đám đông và giải pháp bằng ByteTrack.
- 3.3.2. Vấn đề 2: Hiện tượng nghẽn luồng video (Bottleneck) khi chạy đa mô hình đồng thời và phương án tối ưu Pipeline.

---

## Kết luận và Hướng phát triển (Khoảng 2 trang)
1. Tổng kết các kết quả đạt được (Về tính năng, hiệu năng, giảm báo động giả).
2. Hạn chế còn tồn đọng của hệ thống.
3. Hướng phát triển tương lai (Multi-camera, Tối ưu ContextNet với dữ liệu lớn hơn).

---

## Tài Liệu Tham Khảo
## Phụ Lục (Nếu có)
