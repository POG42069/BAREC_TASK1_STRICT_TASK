# CHỈ THỊ HỆ THỐNG: Kế hoạch thực thi BAREC Shared Task 2026

## 1. Tổng quan dự án
*   **Cuộc thi:** BAREC Shared Task 2026 (Đánh giá mức độ dễ đọc của tiếng Ả Rập).
*   **Nhiệm vụ mục tiêu:** Task 1 - Đánh giá cấp độ câu (Sentence-Level Classification).
*   **Nhánh thi đấu:** Strict Track (Nhánh Nghiêm ngặt).
*   **Mục tiêu:** Dự đoán mức độ khó đọc của từng câu văn tiếng Ả Rập theo thang điểm chi tiết gồm 19 cấp độ.

## 2. Giới hạn của Strict Track (CỰC KỲ QUAN TRỌNG)
*   **Luật lệ:** Các mô hình bắt buộc phải được huấn luyện ĐỘC QUYỀN trên bộ dữ liệu BAREC Corpus do ban tổ chức cung cấp. 
*   **Nghiêm cấm:** Không được phép sử dụng bất kỳ dữ liệu bên ngoài nào (như SAMER Corpus, từ điển bên ngoài, hoặc dữ liệu sinh ra từ LLM).

## 3. Chiến lược Kỹ thuật (Dựa trên Baseline II)
Hãy triển khai hệ thống theo đúng cấu hình của Baseline II như sau:
*   **Mô hình cốt lõi (Pretrained Model):** Sử dụng kiến trúc mô hình **AraBERTv2**.
*   **Tiền xử lý văn bản (Input Variant):** Bắt buộc sử dụng công cụ **D3Tok**. Công cụ này giúp token hóa (băm nhỏ) các từ tiếng Ả Rập thành từ gốc (base forms) và các phụ tố đi kèm (clitic forms) dựa trên đặc điểm ngôn ngữ học.
*   **Hàm tính lỗi (Loss Function):** Sử dụng hàm **Cross-entropy loss (CE)** tiêu chuẩn cho bài toán phân loại đa lớp.

## 4. Định dạng nộp bài (Submission Format)
Hệ thống phải xuất kết quả dự đoán tuân thủ tuyệt đối định dạng sau để nộp lên hệ thống Codabench:
1.  **Loại file:** Tạo một file định dạng `.csv`.
2.  **Tên file:** Bắt buộc đặt tên là `prediction.csv`.
3.  **Cấu trúc cột:** File CSV phải chứa đúng 2 cột với tiêu đề (header) như sau:
    *   `Sentence ID`: Mã số ID của câu văn trong tập dữ liệu kiểm thử (Test set).
    *   `Prediction`: Nhãn dự đoán của mô hình (là một số nguyên từ `1` đến `19`).
4.  **Đóng gói:** Nén file `prediction.csv` này thành một file `.zip` duy nhất trước khi tải lên nền tảng Codabench.

## 5. Hành động yêu cầu (Action Items)
1.  Tải bộ dữ liệu BAREC Corpus.
2.  Tạo môi trường ảo (`venv`) bằng lệnh `python -m venv venv` và kích hoạt nó để quản lý các thư viện cài đặt một cách độc lập, tránh xung đột với hệ thống.
3.  Khởi tạo môi trường Python với các thư viện cần thiết (Hugging Face `transformers`, `CAMEL Tools` cho D3Tok, `torch`, `scikit-learn`, `pandas`).
4.  Tạo một file `requirements.txt` ghi lại đầy đủ danh sách các thư viện và phiên bản đã cài đặt để đảm bảo khả năng tái lập (reproducibility) của dự án.
5.  Xây dựng mã nguồn huấn luyện AraBERTv2 tuân thủ đúng 3 yếu tố của Baseline II (AraBERTv2 + D3Tok + CE Loss).
6.  Đẩy toàn bộ mã nguồn (bao gồm file code và `requirements.txt`) lên repository GitHub sau: https://github.com/POG42069/BAREC_TASK1_STRICT_TASK
7.  Dừng lại và chờ xác nhận trước khi bắt đầu quá trình huấn luyện (training).