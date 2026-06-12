# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 1. Tổng kết các kết quả đạt được

Đồ án tốt nghiệp **"Hệ thống camera an ninh thông minh hiểu ngữ cảnh (Context-aware AI Security Camera)"** đã hoàn thành trọn vẹn tất cả các mục tiêu nghiên cứu và yêu cầu thực tiễn đề ra ban đầu. Những kết quả đạt được mang tính đóng góp thực tiễn lớn cho lĩnh vực giám sát an ninh thông minh:

*   **Về mặt kiến trúc hệ thống:** Nhóm đã xây dựng thành công một Pipeline thời gian thực khép kín, hoạt động cực kỳ ổn định và mượt mà. Giải pháp tách luồng đa nhiệm (Multi-threaded Pipeline) đã giải quyết triệt để bài toán trễ tích lũy, khống chế độ trễ hiển thị dưới **0.2 giây** và thời gian phản hồi cảnh báo đến điện thoại người dùng dưới **2.0 giây**.
*   **Về mặt ứng dụng Mô hình AI thực tế:**
    *   Tích hợp thành công **YOLOv8** và thuật toán **ByteTrack** cải tiến bám đuổi đối tượng ổn định, giảm thiểu tối đa hiện tượng nhảy ID (ID Switching) trong điều kiện che khuất phức tạp ngoài đời thực.
    *   Huấn luyện và triển khai mô hình **RoleNet V3 (ConvNeXt-Tiny)** đạt độ chính xác ấn tượng **99.96%** trên tập dữ liệu kiểm thử, giúp phân loại xuất sắc các vai trò shipper, công nhân, cảnh sát, người lạ qua trang phục đặc trưng tại Việt Nam.
    *   Phát triển mô hình **ActionNet (MediaPipe Pose + GRU)** siêu nhẹ chỉ **0.8 MB** chạy gần như không tốn tài nguyên (< 2ms) nhưng nhận diện chính xác 8 hành vi động, đặc biệt là hành vi té ngã nguy hiểm.
    *   Ứng dụng cây quyết định **XGBoost (ContextNet)** thay thế hoàn toàn các bộ luật tĩnh cứng nhắc, giúp hệ thống có khả năng tự động học cách suy luận ngữ cảnh an ninh thông minh và linh hoạt như tư duy của một người bảo vệ thực thụ.
*   **Về mặt trải nghiệm người dùng (NLG Engine):** Đây là một điểm sáng công nghệ đột phá của đồ án. Nhóm đã tích hợp thành công mô hình ngôn ngữ lớn **Gemini API** kết hợp với **bộ Fallback tiếng Việt nội bộ** thông minh. Sự kết hợp này mang lại khả năng sinh thông báo đẩy dạng hội thoại tự nhiên, gần gũi, giúp loại bỏ hoàn toàn cảm giác phiền toái của các thông báo kỹ thuật khô khan trước đây.
*   **Hiệu quả thực tế:** Thử nghiệm benchmark thực tế trên cấu hình máy tính cá nhân phổ thông chứng minh hệ thống đạt tốc độ xử lý mượt mà từ **12.1 FPS đến 30 FPS**, đồng thời lọc bỏ thành công hơn **92% báo động giả** so với camera truyền thống, khẳng định tính khả thi vượt trội khi đưa vào ứng dụng thực tiễn.

---

## 2. Những hạn chế còn tồn đọng của hệ thống

Mặc dù đạt được những kết quả vô cùng khả quan, nhóm nghiên cứu cũng thẳng thắn nhìn nhận hệ thống vẫn còn một số hạn chế cần được tiếp tục khắc phục và hoàn thiện:

1.  **Hạn chế về xử lý đa luồng camera:** Kiến trúc Pipeline hiện tại mới chỉ được tối ưu hóa tốt nhất cho việc xử lý **đơn luồng camera (Single Stream)**. Khi mở rộng ra chạy đồng thời 3 - 5 camera Full HD, tài nguyên CPU và GPU của máy tính cá nhân sẽ bị quá tải, gây tụt FPS.
2.  **Sự phụ thuộc vào điều kiện ánh sáng màu sắc:** Mô hình nhận diện vai trò RoleNet V3 phụ thuộc rất nhiều vào đặc điểm màu sắc và logo trên trang phục. Trong điều kiện ban đêm hoàn toàn không có ánh sáng hỗ trợ (camera chuyển sang chế độ hồng ngoại trắng đen), độ chính xác nhận diện vai trò của hệ thống sẽ bị suy giảm rõ rệt.
3.  **Hành vi phức tạp chưa được bao phủ:** Hệ thống mới chỉ nhận diện tốt các hành vi vận động cơ bản của dáng người. Các hành vi an ninh tinh vi hơn như: cạy khóa cửa (tương tác vật nhỏ), cầm nắm hung khí hoặc trộm cắp vặt tạm thời chưa nằm trong phạm vi xử lý của mô hình ActionNet hiện tại.

---

## 3. Hướng phát triển tương lai

Để nâng tầm sản phẩm và hướng tới một hệ thống giải pháp an ninh toàn diện, nhóm đề xuất các hướng nghiên cứu và phát triển tiếp theo như sau:

*   **Nâng cấp kiến trúc xử lý Đa Camera (Multi-camera Pipeline):** Ứng dụng các thư viện xử lý video song song chuyên dụng như NVIDIA DeepStream SDK để tận dụng tối đa sức mạnh của nhân Tensor trên GPU, cho phép hệ thống quản lý, suy luận đồng thời hàng chục luồng camera cùng lúc với độ trễ cực thấp.
*   **Tích hợp Nhận dạng khuôn mặt chuyên sâu (Face Recognition):** Tích hợp thêm mô hình ArcFace siêu nhẹ vào Pipeline để định danh chính xác danh tính cụ thể của từng thành viên trong gia đình hoặc nhân viên trong văn phòng, nâng mức độ "hiểu ngữ cảnh" lên tầm cao mới (ví dụ: phát hiện chính xác *"Con trai út của chủ nhà đi học về"* thay vì chỉ báo *"Người nhà đang đi bộ"*).
*   **Huấn luyện mô hình NLG Edge cục bộ (Local Small Language Model):** Nghiên cứu lượng hóa và triển khai các mô hình ngôn ngữ lớn siêu nhỏ (như Phi-3-Mini hoặc Gemma-2B) chạy hoàn toàn ngoại tuyến (Offline) ngay trên thiết bị Edge an ninh để đảm bảo tính riêng tư tuyệt đối cho dữ liệu của gia đình, không cần phụ thuộc vào mạng internet hay API đám mây bên thứ ba.
*   **Mở rộng bộ dữ liệu ContextNet:** Thu thập thêm dữ liệu ngữ cảnh thực tế từ nhiều hộ gia đình và văn phòng khác nhau để huấn luyện mô hình XGBoost đạt độ chín chắn, thông minh và bao quát hơn đối với các tình huống an ninh mập mờ ngoài đời thực.

---

# TÀI LIỆU THAM KHẢO

1.  **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** *You Only Look Once: Unified, Real-Time Object Detection.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779-788.
2.  **Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022).** *BYTETRACK: Multi-Object Tracking by Associating Every Detection Box.* European Conference on Computer Vision (ECCV).
3.  **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).** *A ConvNet for the 2020s.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11976-11986.
4.  **Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).** *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* arXiv preprint arXiv:1406.1078.
5.  **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
6.  **Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, A., Somani, M., ... & Grundmann, M. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv preprint arXiv:1906.08172.
7.  **Google AI Team (2023).** *Gemini: A Family of Highly Capable Multimodal Models.* Google DeepMind Technical Report.
