Dựa trên phân tích các bài báo được cung cấp, dưới đây là chi tiết về **Effort** và **GenD**, làm nổi bật những ưu điểm của chúng so với các công trình trước đó, hiệu suất benchmark so với các tiền nhiệm, và các hạn chế của chúng.

---

### **1. Mô hình: Effort (Mô hình hóa trực giao hiệu quả)**
*Từ: "Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection" (ICML 2025)*

#### **Ưu điểm so với các Mô hình trước đây**
*   **So với Tinh chỉnh Toàn bộ (Full Fine-Tuning) & LoRA:** Khác với tinh chỉnh tiêu chuẩn hoặc Thích ứng Hạng thấp (Low-Rank Adaptation - LoRA), Effort sử dụng Phân rã Giá trị Kỳ dị (SVD) để thực thi **tính trực giao nghiêm ngặt**. Nó tách không gian đặc trưng thành một không gian con "chính" bị đóng băng (bảo tồn tri thức đã huấn luyện trước) và một không gian con "dư" có thể huấn luyện (học giả mạo). Điều này ngăn chặn mô hình "quên" các khái niệm ngữ nghĩa, một vấn đề phổ biến trong các phương pháp trước đây khi không gian đặc trưng thường bị sụp đổ thành cấu trúc hạng thấp.
*   **So với UniFD (Linear Probing):** Trong khi UniFD đóng băng mạng xương sống và chỉ huấn luyện bộ phân loại, Effort cho phép thích ứng sâu hơn bằng cách tinh chỉnh các trọng số bên trong thông qua không gian con dư, dẫn đến độ chính xác cao hơn mà không tốn chi phí tính toán của việc tinh chỉnh toàn bộ.
*   **Hiệu quả Tham số:** Nó đạt kết quả SOTA chỉ với **0.19M** tham số có thể huấn luyện, nhỏ hơn khoảng **1.000 lần** so với các đối thủ như LSDA và ProDet (sử dụng ~100M tham số).

#### **Benchmark so với các Mô hình trước đây**
Effort tự so sánh với các bộ phát hiện SOTA gần đây về **Tổng quát hóa Chéo Tập dữ liệu (AUC)** (Huấn luyện trên FF++):

| Mô hình trước | Loại | AUC Trung bình (Chéo Tập dữ liệu) | Kết quả Effort | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **ProDet** (NeurIPS'24) | Tần số/Adapter | 0.867 | **0.940** | **Vượt qua** |
| **LSDA** (CVPR'24) | Không gian tiềm ẩn | 0.835 | **0.940** | **Vượt qua** |
| **UCF** (ICCV'23) | Đặc trưng chung | 0.824 | **0.940** | **Vượt qua** |
| **RECCE** (CVPR'22) | Tái tạo | 0.844 | **0.940** | **Vượt qua** |

*(Effort cũng vượt qua **UniFD** và **FatFormer** trên benchmark Phát hiện Ảnh Tổng hợp với mAP lần lượt là 99.41% so với 90.14% và 98.16%.)*

#### **Hạn chế / Điểm yếu**
*   **Giả định Lớp Nhị phân:** Phương pháp gộp tất cả các "giả mạo" vào một không gian con dư. Về cơ bản, nó coi mọi sự giả mạo là một lớp "sai lệch so với thật" duy nhất, điều này có thể gặp khó khăn nếu các loại giả mạo khác biệt đòi hỏi các tập đặc trưng loại trừ lẫn nhau.
*   **Độ nhạy Khởi tạo:** Việc chọn hạng ($r$) cho phân rã SVD là một siêu tham số cần được lựa chọn cẩn thận (mặc dù Hạng 1 hoạt động tốt trong bài báo).

---

### **2. Mô hình: GenD (Phát hiện Deepfake có khả năng Tổng quát hóa)**
*Từ: "Deepfake Detection that Generalizes Across Benchmarks" (arXiv Nov 2025)*

#### **Ưu điểm so với các Mô hình trước đây**
*   **So với các Kiến trúc Phức tạp (ForAda, Effort):** GenD thách thức sự cần thiết của các bổ sung kiến trúc (như mạng song song của ForAda) hoặc phân rã tham số phức tạp (như SVD của Effort). Nó chứng minh rằng chỉ cần tinh chỉnh các tham số **Chuẩn hóa Lớp (Layer Normalization - LN)** (0.03% của mô hình) kết hợp với **Học Metric** (Chiếu lên siêu cầu - Hypersphere projection) là vượt trội hơn.
*   **So với Huấn luyện Không cặp:** GenD chứng minh bằng thực nghiệm rằng các mô hình trước đây thường chịu ảnh hưởng của "học đường tắt" (học nền/danh tính). GenD sử dụng **Huấn luyện Có cặp** (Thật vs. Giả được tạo từ cùng một nguồn), buộc mô hình phải học các artifact thao tác cấp thấp thay vì nội dung ngữ nghĩa.
*   **Quy mô Đánh giá:** Trong khi các mô hình trước (bao gồm cả Effort) thường đánh giá trên 3–5 tập dữ liệu, GenD đánh giá trên **14 tập dữ liệu**, cung cấp một bằng chứng mạnh mẽ hơn nhiều về khả năng tổng quát hóa.

#### **Benchmark so với các Mô hình trước đây**
GenD so sánh trực tiếp với **Effort** và **ForAda** (bài báo khác trong phân tích này) trên 14 tập dữ liệu.

| Mô hình trước | Loại | AUC Trung bình (14 Tập dữ liệu) | Kết quả GenD | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **Effort** (ICML'25) | SVD Adapter | 88.5% | **91.6%** (với DINOv3) | **Vượt qua** |
| **ForAda** (CVPR'25) | Parallel Adapter | 88.4% | **91.2%** (với CLIP) | **Vượt qua** |

*(Lưu ý: GenD vượt qua Effort cụ thể trên tập dữ liệu DFDC [87.1% so với 84.3%] và FFIW [92.8% so với 92.1%], mặc dù kết quả cạnh tranh trên Celeb-DF).*

#### **Hạn chế / Điểm yếu**
*   **Mù Thời gian:** Giống như Effort, GenD hoạt động theo từng khung hình và sử dụng tính trung bình đơn giản. Nó bỏ qua các dấu hiệu thời gian (ví dụ: sự không nhất quán đồng bộ môi hoặc nhấp nháy) vốn rất quan trọng đối với một số deepfake video tiên tiến.
*   **Phụ thuộc vào Phát hiện Khuôn mặt:** Quy trình dựa hoàn toàn vào việc cắt khuôn mặt chính xác (sử dụng RetinaFace). Nếu bộ phát hiện khuôn mặt thất bại do khẩu trang, góc cực đoan hoặc bị che khuất, quy trình phân loại sẽ bị phá vỡ.
*   **Thiên kiến Nhân khẩu học:** Bài báo ghi nhận tỷ lệ lỗi cao hơn trên các nhóm nhân khẩu học cụ thể (ví dụ: tông màu da tối hơn) và phụ kiện (kính mắt).

---

### **Tóm tắt So sánh**

| Tính năng | **Effort** | **GenD** |
| :--- | :--- | :--- |
| **Kỹ thuật** | Phân rã SVD (Không gian con Trực giao) | Tinh chỉnh Layer Norm + Học Metric |
| **Triết lý** | "Giữ ngữ nghĩa đóng băng, học giả mạo trong không gian con riêng biệt." | "Tinh chỉnh các lớp chuẩn hóa để chiếu đặc trưng lên siêu cầu." |
| **Chiến lược Dữ liệu** | Huấn luyện FF++ tiêu chuẩn. | Huấn luyện Thật/Giả **Có cặp** (Đổi mới then chốt). |
| **Hiệu suất** | Xuất sắc (0.940 trên TB 7 tập dữ liệu). | **Vượt trội** (91.6% trên TB 14 tập dữ liệu). |
| **Người chiến thắng?** | | **GenD** dường như là mô hình vượt trội dựa trên các benchmark được cung cấp trong bài báo GenD, bao gồm Effort như một đường cơ sở. |
