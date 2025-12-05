### 📄 **Báo cáo Đánh giá Hiệu năng & Thách thức trong Phát hiện Deepfake (Image-based)**

#### **1. Đánh giá Hiện trạng (Current Status)**
Các mô hình SOTA (State-of-the-Art) gần đây như **Effort** (Efficient Orthogonal Modeling) hay **GenD** đã đạt được những kết quả ấn tượng trên các tập dữ liệu tiêu chuẩn (benchmark datasets). Tuy nhiên, khi triển khai thực tế, chúng bộc lộ những điểm yếu chí mạng về khả năng thích ứng.

#### **2. Các Thách thức Cốt lõi (Critical Failure Points)**

**A. Khoảng cách Tổng quát hóa (The Generalization Gap)**
*   Các mô hình hiện tại chỉ hoạt động tối ưu trong **miền dữ liệu đã được huấn luyện (training domain)**.
*   Nghiên cứu thực nghiệm cho thấy hiệu suất phát hiện có thể **sụt giảm từ 30-40%** chỉ sau 3-4 tháng phát hành.
*   Nguyên nhân: Các phương pháp phát hiện (Detection methods) không thể bắt kịp tốc độ cập nhật của các kiến trúc sinh ảnh thương mại đóng mã nguồn (black-box) như **Midjourney v6**, **DALL-E 3** hay mới đây nhất là **Gemini 2.5 Flash Image (Nano Banana)**.

**B. Điểm nghẽn về Dữ liệu (The Data Bottleneck)**
*   Đây là nguyên nhân gốc rễ khiến quy trình cập nhật model bị tê liệt.
*   **Sự khan hiếm:** Khác với các model nguồn mở (như Stable Diffusion) dễ dàng tạo dữ liệu training, các model thương mại mới thường bị giới hạn bởi API và chi phí, khiến việc xây dựng bộ dữ liệu quy mô lớn (Large-scale Datasets) trở nên cực kỳ khó khăn và tốn kém.
*   **Độ trễ (Latency):** Quy trình thu thập và gán nhãn dữ liệu thường chậm hơn 6-12 tháng so với tốc độ ra mắt của các model sinh ảnh, dẫn đến việc model phát hiện luôn phải "học" trên dữ liệu lỗi thời nhưng phải "thi" trên các mẫu deepfake hiện đại nhất.

#### **3. Kết luận & Kiến nghị (Conclusion & Proposal)**
Việc chỉ phụ thuộc vào các pre-trained models có sẵn mang lại rủi ro vận hành lớn do sự sai lệch phân phối dữ liệu (distribution shift). Để giải quyết vấn đề này, cần thiết phải:
*   Chuyển dịch từ mô hình tĩnh sang chiến lược **"Học liên tục" (Continuous Learning)**.
*   Xây dựng quy trình thu thập dữ liệu chủ động (Active Data Pipeline) đối với các nền tảng sinh ảnh mới ngay khi chúng vừa ra mắt.
