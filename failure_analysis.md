    Phân tích Nguyên nhân Thất bại của các Mô hình Pre-trained (Off-the-shelf Models)

    Mặc dù áp dụng các kiến trúc SOTA, kết quả thực nghiệm trên các bộ trọng số có sẵn (available pre-trained weights) cho thấy hiệu năng thấp đáng kể. Nguyên nhân kỹ thuật được xác định do 3 yếu tố chính:

    1. Vấn đề về Độ phân giải (Resolution Mismatch):

        Hầu hết các model có sẵn (như Effort, ResNet-based) yêu cầu nén ảnh đầu vào xuống kích thước chuẩn 224x224 hoặc 256x256 để xử lý.

        Hệ quả: Quá trình Down-sampling này vô tình triệt tiêu các "dấu vết pháp y" (forensic artifacts) tần số cao vốn chỉ tồn tại ở độ phân giải gốc (1024x1024 trở lên) của các công cụ như Midjourney v6 hay Flux. Model bị buộc phải đưa ra quyết định dựa trên dữ liệu đã bị làm mờ, dẫn đến tỷ lệ False Negative cao.

    2. Sự "Hóa thạch" của Dữ liệu Huấn luyện (Dataset Fossilization):

        Các bộ trọng số công khai thường được huấn luyện trên các tập dữ liệu cũ (như GenImage với Stable Diffusion v1.4/1.5 hoặc FaceForensics++ từ 2019).

        Các model này học cách tìm kiếm các lỗi tạo sinh cũ (ví dụ: biến dạng mống mắt, lỗi nền grid noise). Các model sinh ảnh hiện đại (Modern Generators) đã khắc phục hoàn toàn các lỗi này, khiến bộ phát hiện trở nên "mù" trước các giả mạo tinh vi mới.

    3. Hiện tượng Overfitting vào Benchmark:

        Các model nghiên cứu thường được tối ưu hóa cực đoan (hyper-parameter tuning) cho một tập dữ liệu cụ thể để đạt điểm số cao trong bài báo khoa học. Khi áp dụng vào môi trường thực tế (In-the-wild testing) với phân phối dữ liệu khác biệt, khả năng suy luận của model giảm sút mạnh do thiếu tính tổng quát (Robustness).

    Kết luận:
    Không thể sử dụng nguyên trạng các model pre-trained cho môi trường sản phẩm (Production). Cần thiết phải thực hiện chiến lược Fine-tuning lại các kiến trúc này trên tập dữ liệu mới tự thu thập để khôi phục độ chính xác.
