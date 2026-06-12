# MỞ ĐẦU

## 1. Lý do chọn đề tài

Trong bối cảnh đô thị hóa và sự phát triển kinh tế diễn ra nhanh chóng, nhu cầu bảo vệ an ninh, an toàn tài sản và tính mạng con người tại các hộ gia đình, văn phòng, cửa hàng kinh doanh ngày càng trở nên cấp thiết. Sự ra đời của các thiết bị camera giám sát (CCTV) trong những thập kỷ qua đã hỗ trợ đắc lực cho con người trong việc quan sát và ghi lại các sự kiện diễn ra xung quanh bối cảnh sống. Tuy nhiên, khi đối mặt với thực tế vận hành phức tạp ngày nay, các hệ thống camera truyền thống bộc lộ rõ hai điểm hạn chế cốt lõi sau:

*   **Tính chất giám sát thụ động:** Đa số camera hiện nay chỉ đóng vai trò như một thiết bị ghi hình và lưu trữ thuần túy. Khi có sự cố an ninh nghiêm trọng xảy ra như trộm cắp, đột nhập trái phép hay tai nạn, người dùng thường chỉ phát hiện ra sự việc sau khi mọi chuyện đã hoàn tất và buộc phải xem lại video lưu trữ (playback) để làm bằng chứng pháp lý. Hệ thống hoàn toàn không có khả năng tự động can thiệp, đưa ra cảnh báo sớm ngăn chặn hành vi phạm tội ngay từ giai đoạn manh nha.
*   **Vấn nạn báo động giả (False Alarms):** Một số dòng camera IP hiện đại đã được tích hợp tính năng phát hiện chuyển động cơ bản dựa trên sự thay đổi giá trị pixel giữa các khung hình liên tiếp. Tuy nhiên, các thuật toán này cực kỳ nhạy cảm và thiếu thông minh. Hệ thống liên tục gửi cảnh báo rít vang hoặc thông báo rác về điện thoại người dùng bất cứ khi nào có gió thổi làm lá cây rơi, thay đổi cường độ ánh sáng đột ngột, mưa bão, hoặc chó mèo chạy ngang qua. Hậu quả là người dùng rơi vào trạng thái "lờn cảnh báo", cảm thấy phiền toái và quyết định tắt bỏ hoàn toàn tính năng thông báo của thiết bị. Khi một sự cố đột nhập thực tế xảy ra, họ sẽ hoàn toàn bỏ lỡ cơ hội xử lý kịp thời.

Từ những hạn chế thực tế trên, câu hỏi đặt ra là: *Làm thế nào để xây dựng một hệ thống camera an ninh có tư duy giống con người?* Con người khi nhìn vào màn hình giám sát sẽ không báo động chỉ vì có một con mèo chạy qua, nhưng sẽ lập tức cảnh giác cao độ nếu thấy **một người lạ đeo khẩu trang (Vai trò - Role)** đang **đứng lảng vảng cạy cửa (Hành động - Action)** ở **khu vực cổng chính lúc nửa đêm (Vị trí/Thời gian - Zone & Time)**.

Đó chính là khái niệm **"Hệ thống camera an ninh hiểu ngữ cảnh (Context-aware)"**. Việc kết hợp đồng thời ba chiều thông tin: *Ai (Role)? Đang làm gì (Action)? Ở đâu (Zone)?* tạo nên một hệ thống giám sát an ninh thế hệ mới. Đề tài **"Hệ thống camera an ninh thông minh hiểu ngữ cảnh (Context-aware)"** được lựa chọn nghiên cứu nhằm hiện thực hóa giải pháp an ninh chủ động, thông minh và thân thiện cho các gia đình Việt.

---

## 2. Mục tiêu giải quyết bài toán an ninh thực tế

Mục tiêu chính của đồ án này là nghiên cứu lý thuyết, thiết kế kiến trúc và triển khai thực tế một phần mềm giám sát an ninh thông minh tích hợp trí tuệ nhân tạo, giải quyết trọn vẹn các yêu cầu sau:

1.  **Chuyển đổi từ giám sát thụ động sang chủ động:** Hệ thống có khả năng phân tích luồng video camera thời gian thực, tự động phát hiện, bám đuổi hành trình của các đối tượng và phân tích tức thì các hành vi để phát hiện nguy cơ an ninh ngay khi chúng vừa bắt đầu.
2.  **Giảm thiểu tối đa tỷ lệ báo động giả:** Thông qua việc áp dụng động cơ đánh giá ngữ cảnh thông minh, hệ thống chỉ đưa ra cảnh báo mức độ cao (Alert/Critical) khi phát hiện sự kết hợp các yếu tố gây mất an toàn (ví dụ: người lạ xâm nhập vùng cấm, hoặc đối tượng lảng vảng đáng ngờ ngoài cổng). Các chuyển động phi vật lý hoặc chuyển động của động vật, người nhà hoạt động bình thường sẽ được lọc bỏ hoặc phân loại ở mức độ an toàn (Normal).
3.  **Tự động hóa toàn bộ quy trình vận hành:**
    *   **Phát hiện và theo dõi:** Tự động vẽ bounding box, theo dấu đối tượng với độ trễ tối thiểu.
    *   **Phân loại vai trò:** Nhận diện đối tượng là người nhà, người quen, nhân viên giao hàng (shipper), thợ sửa chữa hay người lạ để có ứng xử phù hợp.
    *   **Nhận diện hành vi:** Theo dõi tư thế khớp xương để phát hiện hành vi đứng lảng vảng, chạy trốn, ngã quỵ, v.v.
    *   **Gửi cảnh báo tức thời & Ghi hình thông minh:** Khi phát hiện ngữ cảnh nguy hiểm, hệ thống tự động cắt và lưu trữ video sự kiện dài 10-15 giây để làm bằng chứng ngoại phạm, đồng thời gửi thông tin trực tiếp tới điện thoại người dùng.
4.  **Nâng cao trải nghiệm người dùng bằng AI sinh văn bản tự nhiên (Generative NLG):** Khắc phục nhược điểm các tin nhắn cảnh báo khô khan mang tính kỹ thuật của các hệ thống cũ. Hệ thống ứng dụng mô hình ngôn ngữ lớn để tự động dịch bối cảnh phát hiện thành câu văn hội thoại tự nhiên, gần gũi, giúp người dùng nắm bắt thông tin cực kỳ nhanh chóng và trực quan (ví dụ: *"Bạn ơi, có anh shipper vừa đỗ xe trước cổng chính kìa!"* thay vì *"ALERT: shipper detected at zone_1"*).

---

## 3. Đối tượng và phạm vi nghiên cứu

Để hoàn thành mục tiêu đề ra, đồ án xác định rõ đối tượng và phạm vi nghiên cứu cụ thể như sau:

### Đối tượng nghiên cứu:
*   **Thuật toán phát hiện và bám đuổi đối tượng thời gian thực:** Trọng tâm nghiên cứu là mô hình **YOLOv8** (phiên bản Nano/Small để tối ưu tốc độ suy luận) kết hợp thuật toán liên kết **ByteTrack** dựa trên Kalman Filter nhằm duy trì nhất quán ID đối tượng giữa các khung hình liên tiếp.
*   **Mô hình nhận diện vai trò dựa trên trang phục (RoleNet):** Nghiên cứu kiến trúc mạng nơ-ron tích chập hiện đại **ConvNeXt** (phiên bản ConvNeXt-Tiny) để phân loại các lớp trang phục đặc trưng cho từng vai trò xã hội trong bối cảnh thực tế tại Việt Nam.
*   **Mô hình nhận diện hành động từ chuỗi khung xương (ActionNet):** Nghiên cứu thuật toán ước lượng dáng người **MediaPipe Pose** kết hợp mạng hồi quy bộ nhớ ngắn hạn **GRU** để phân tích sự thay đổi tọa độ 33 điểm khớp xương theo thời gian thực.
*   **Mô hình đánh giá ngữ cảnh tự động (ContextNet):** Nghiên cứu thuật toán cây quyết định tăng cường **XGBoost** để phân loại các mức độ rủi ro dựa trên tổ hợp đặc trưng đầu vào đa chiều.
*   **Công nghệ sinh ngôn ngữ tự nhiên (NLG):** Ứng dụng **Gemini API** của Google và thiết kế cơ chế fallback xử lý văn bản tiếng Việt cục bộ.

### Phạm vi nghiên cứu và giới hạn thực tế:
*   **Giới hạn về dữ liệu hành vi & vai trò:** Hệ thống tập trung tối ưu hóa nhận diện trên **16 vai trò xã hội** phổ biến (shipper, công nhân, cảnh sát, người lạ, người nhà, v.v.) và **8 hành động cơ bản** (đứng, đi bộ, chạy, ngã, v.v.) có ảnh hưởng trực tiếp tới bài toán an ninh.
*   **Giới hạn môi trường triển khai:** Hệ thống hiện tại được thiết kế và tối ưu tốt nhất cho môi trường camera đơn luồng (single stream), hoạt động trong điều kiện ánh sáng đầy đủ hoặc ánh sáng nhân tạo tiêu chuẩn (không bao gồm điều kiện đêm tối hoàn toàn không có hồng ngoại).
*   **Giới hạn phần cứng:** Ứng dụng được thiết kế chạy trên các cấu hình phần cứng tầm trung (không yêu cầu siêu máy tính hay cụm GPU chuyên dụng đắt đỏ), đảm bảo khả năng tiếp cận thực tế cho các hộ gia đình hoặc văn phòng nhỏ bằng cách tối ưu hóa chuyển đổi các mô hình sang định dạng **ONNX** và lượng hóa (quantization) INT8.
