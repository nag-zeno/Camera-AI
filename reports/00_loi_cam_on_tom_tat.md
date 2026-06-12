# LỜI CẢM ƠN

Lời đầu tiên, em xin được bày tỏ lòng biết ơn chân thành và sâu sắc nhất tới thầy/cô hướng dẫn, người đã luôn dành thời gian quý báu để định hướng, tận tình chỉ bảo và truyền đạt những kiến thức vô cùng giá trị cho em trong suốt quá trình thực hiện đồ án tốt nghiệp này. Sự hỗ trợ sát sao và những ý kiến đóng góp chuyên môn của thầy/cô là kim chỉ nam giúp em hoàn thiện cấu trúc hệ thống, lựa chọn đúng hướng giải quyết cho các vấn đề kỹ thuật hóc búa, và hoàn thành đồ án một cách trọn vẹn nhất.

Em cũng xin gửi lời cảm ơn chân thành tới toàn thể các thầy cô giáo trong Khoa Công nghệ Thông tin đã tận tâm giảng dạy, truyền đạt cho em những nền tảng tri thức vững chắc trong suốt những năm học tập tại trường. Những kiến thức quý báu đó là hành trang giúp em tự tin nghiên cứu và ứng dụng vào thực tế phát triển hệ thống hôm nay.

Cuối cùng, em xin bày tỏ lòng biết ơn sâu sắc tới gia đình, bạn bè và các đồng nghiệp – những người đã luôn bên cạnh động viên, chia sẻ khó khăn và tạo mọi điều kiện tốt nhất về cả vật chất lẫn tinh thần trong suốt thời gian em thực hiện đề tài nghiên cứu này.

Dù đã dành nhiều thời gian, tâm huyết và nỗ lực để hoàn thành đồ án với tinh thần nghiêm túc nhất, hệ thống chắc chắn vẫn không tránh khỏi những thiếu sót và hạn chế nhất định. Kính mong nhận được những lời chỉ dẫn, nhận xét và đóng góp quý báu từ quý thầy cô trong Hội đồng bảo vệ để sản phẩm của em ngày càng được hoàn thiện và thực tiễn hơn.

Em xin chân thành cảm ơn!

---

# TÓM TẮT ĐỒ ÁN

Trong thời đại công nghệ số hiện nay, các hệ thống camera an ninh truyền thống đóng vai trò quan trọng trong việc giám sát và ghi lại các sự kiện. Tuy nhiên, phần lớn các hệ thống này vẫn gặp phải hai hạn chế lớn: chỉ đóng vai trò ghi hình thụ động (chỉ xem lại sau khi sự việc đã xảy ra) hoặc đưa ra các báo động sai lệch, phiền toái (kích hoạt báo động dựa trên sự thay đổi pixel đơn thuần như lá cây rơi, ánh sáng thay đổi, chó mèo chạy qua). Điều này làm giảm sút nghiêm trọng hiệu quả giám sát và gây ra hiện tượng "lờn cảnh báo" cho người sử dụng.

Đồ án tốt nghiệp này tập trung giải quyết triệt để vấn đề trên bằng cách nghiên cứu và xây dựng **"Hệ thống camera an ninh thông minh hiểu ngữ cảnh (Context-aware AI Security Camera)"**. Hệ thống này không chỉ đơn thuần là phát hiện chuyển động mà còn có khả năng tự động "đọc hiểu" bối cảnh đang diễn ra trong khung hình thời gian thực thông qua việc kết hợp đồng thời ba yếu tố cốt lõi: **Vai trò xã hội (Role)** của đối tượng, **Hành động (Action)** của đối tượng và **Vị trí/Khu vực (Zone)** mà đối tượng đang đứng. Sự kết hợp logic này tạo ra một "mức độ rủi ro ngữ cảnh" chính xác để kích hoạt các cấp độ cảnh báo phù hợp.

Hệ thống được thiết kế theo dạng đường ống xử lý (Pipeline) thời gian thực khép kín:
1. **Phát hiện & Theo dõi (Detection & Tracking):** Sử dụng mô hình **YOLOv8** siêu nhẹ kết hợp thuật toán **ByteTrack** để duy trì mã định danh đối tượng (Track ID) ổn định, giảm tối đa hiện tượng nhảy ID (ID switching).
2. **Nhận diện vai trò (Role Classification):** Huấn luyện mô hình **RoleNet V3** dựa trên kiến trúc **ConvNeXt-Tiny** tiên tiến nhằm phân loại chính xác 16 vai trò thực tế (như Shipper, Cảnh sát, Công nhân xây dựng, Người lạ, Người nhà,...) qua đặc điểm trang phục và công cụ làm việc.
3. **Nhận diện hành động (Action Recognition):** Sử dụng thư viện **MediaPipe Pose** để trích xuất 33 điểm xương trên cơ thể người, kết hợp với mạng hồi quy **GRU (ActionNet)** để phân tích chuỗi thời gian, nhận diện chính xác 8 hành vi vận động (như đứng lảng vảng, chạy gấp, đi bộ, ngã quỵ,...).
4. **Động cơ phân tích ngữ cảnh (ContextNet):** Thay vì các tập luật tĩnh `if-else` dễ lỗi và cứng nhắc, hệ thống ứng dụng mô hình cây quyết định tối ưu **XGBoost** để học cách phân loại ngữ cảnh tự động thành 6 cấp độ cảnh báo rủi ro từ bình thường đến cực kỳ nguy hiểm.
5. **Sinh ngôn ngữ tự nhiên (NLG Engine):** Tích hợp công nghệ AI sinh (sử dụng **Gemini API** kết hợp cơ chế fallback dự phòng cục bộ thông minh) để dịch các sự kiện kỹ thuật khô khan thành những câu thông báo ngắn gọn, tự nhiên và thân thiện như tin nhắn trò chuyện thông qua bot Telegram (ví dụ: *"Ting tong! Có anh Shipper giao đồ vừa đến ở cổng chính kìa bạn ơi!"*).

Hệ thống đã được kiểm thử thực tế trên hệ điều hành Windows với cấu hình phần cứng phổ thông (CPU 12 Cores, RAM 16.8 GB), đạt tốc độ xử lý mượt mà từ **12.1 FPS đến 30 FPS** tùy thuộc vào mật độ đối tượng xuất hiện trong khung hình. Kết quả thử nghiệm thực tế chứng minh hệ thống có khả năng lọc bỏ hơn 90% báo động giả của camera truyền thống, đồng thời cung cấp thông tin cảnh báo thông minh, trực quan và lập tức tới người dùng, mở ra hướng ứng dụng rộng rãi cho các hộ gia đình và văn phòng thông minh tại Việt Nam.

**Từ khóa:** *Camera AI, Hiểu ngữ cảnh, YOLOv8, ByteTrack, ConvNeXt, GRU, XGBoost, NLG, Gemini API.*

---

# ABSTRACT

In the era of digital transformation, traditional security camera systems play a vital role in monitoring and recording events. However, most of these systems still suffer from two major limitations: they are passive recording devices (only useful for reviewing after an incident occurs) or they trigger frequent false alarms (based purely on pixel changes such as falling leaves, lighting shifts, or pets passing by). This significantly reduces monitoring efficiency and causes "alarm fatigue" for users.

This graduation thesis addresses these challenges by developing a **"Context-aware Intelligent AI Security Camera System"**. Instead of merely detecting motion, the system automatically "understands" the real-time context of events in the frame by simultaneously combining three core elements: the object's **Social Role (Role)**, the object's **Action (Action)**, and the **Location/Zone (Zone)** where the object is present. This logical combination evaluates a precise "contextual risk level" to trigger appropriate alerts.

The system is designed as a closed real-time processing pipeline:
1. **Detection & Tracking:** Employs the lightweight **YOLOv8** model combined with the **ByteTrack** algorithm to maintain stable object identifiers (Track IDs), minimizing the ID switching phenomenon.
2. **Role Classification:** Trains the **RoleNet V3** model based on the advanced **ConvNeXt-Tiny** architecture to accurately classify 16 realistic roles (such as Delivery/Shipper, Police, Construction Worker, Stranger, Family Member, etc.) based on clothing features and tools.
3. **Action Recognition:** Uses the **MediaPipe Pose** library to extract 33 skeletal keypoints, combined with a **GRU (ActionNet)** recurrent network to analyze time-series data, accurately identifying 8 movements (standing, running, walking, falling, etc.).
4. **Context Engine (ContextNet):** Instead of static and fragile `if-else` rule sets, the system utilizes an optimized **XGBoost** decision tree model to automatically classify contexts into 6 warning levels, ranging from normal to highly critical.
5. **Natural Language Generation (NLG Engine):** Integrates generative AI (**Gemini API** with a smart local fallback mechanism) to translate technical events into short, natural, and friendly conversational alerts sent directly via Telegram (e.g., *"Ting tong! A delivery driver has just arrived at the main gate!"*).

The system has been evaluated in real-world environments on a Windows operating system with consumer-grade hardware (12-core CPU, 16.8 GB RAM), achieving processing speeds ranging from **12.1 FPS to 30 FPS** depending on object density. Experimental results demonstrate that the system filters out over 90% of traditional camera false alarms while delivering intelligent, intuitive, and immediate alerts to users, paving the way for widespread smart home and office applications in Vietnam.

**Keywords:** *AI Camera, Context-awareness, YOLOv8, ByteTrack, ConvNeXt, GRU, XGBoost, NLG, Gemini API.*

---

# DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Thuật ngữ đầy đủ (Tiếng Anh) | Nghĩa Tiếng Việt |
| :--- | :--- | :--- |
| **AI** | Artificial Intelligence | Trí tuệ nhân tạo |
| **NLG** | Natural Language Generation | Sinh ngôn ngữ tự nhiên |
| **YOLO** | You Only Look Once | Thuật toán phát hiện đối tượng thời gian thực |
| **GRU** | Gated Recurrent Unit | Đơn vị hồi quy có cổng |
| **XGBoost** | Extreme Gradient Boosting | Mô hình cây quyết định tăng cường độ dốc cực đại |
| **FPS** | Frames Per Second | Số khung hình trên mỗi giây |
| **API** | Application Programming Interface | Giao diện lập trình ứng dụng |
| **ONNX** | Open Neural Network Exchange | Định dạng trao đổi mạng nơ-ron mở |
| **CPU** | Central Processing Unit | Bộ vi xử lý trung tâm |
| **GPU** | Graphics Processing Unit | Bộ vi xử lý đồ họa |
| **RAM** | Random Access Memory | Bộ nhớ truy cập ngẫu nhiên |
| **IoU** | Intersection over Union | Tỷ lệ giao diện trên vùng hợp |
| **UI** | User Interface | Giao diện người dùng |
| **JSONL** | JSON Lines | Định dạng lưu trữ log dưới dạng các dòng đối tượng JSON |

---

# DANH MỤC BẢNG BIỂU

* Bảng 1.1: Yêu cầu phi chức năng của hệ thống Camera AI.
* Bảng 2.1: So sánh hiệu năng của mô hình RoleNet V3 (ConvNeXt-Tiny) so với các biến thể MobileNet.
* Bảng 2.2: Cấu trúc cơ sở dữ liệu lưu trữ sự kiện (Log event format).
* Bảng 2.3: Bảng ánh xạ kết hợp ngữ cảnh (*Role + Action + Zone*) ra 6 cấp độ rủi ro của động cơ ContextNet.
* Bảng 3.1: So sánh hiệu năng FPS thực tế theo mật độ đối tượng trên cấu hình thử nghiệm.

---

# DANH MỤC HÌNH ẢNH

* Hình 2.1: Sơ đồ khối kiến trúc hệ thống tổng thể (Real-time Pipeline).
* Hình 2.2: Luồng dữ liệu (Data flow) đi từ luồng camera trực tiếp đến thông báo Telegram.
* Hình 2.3: Sơ đồ liên kết Kalman Filter và IoU trong thuật toán ByteTrack.
* Hình 2.4: Kiến trúc ConvNeXt-Tiny được sử dụng trong RoleNet V3.
* Hình 2.5: Trích xuất 33 điểm xương thông qua MediaPipe Pose để làm đầu vào cho ActionNet.
* Hình 3.1: Giao diện Dashboard Web UI chính của hệ thống.
* Hình 3.2: Giao diện Zone Editor vẽ các vùng giám sát tùy chỉnh trực quan.
* Hình 3.3: Minh họa tin nhắn thông báo Telegram tự nhiên thông qua NLG Engine.
