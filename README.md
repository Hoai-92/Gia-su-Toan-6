# 🎓 Gia sư Toán 6 AI Agent

Dự án phát triển một AI Agent đóng vai trò là gia sư cá nhân, hỗ trợ học sinh học môn Toán lớp 6 (bám sát sách giáo khoa). Ứng dụng được xây dựng dựa trên kiến trúc **ReAct Agent** của LangChain và giao diện trực quan bằng Streamlit.

## 🌟 Tính năng nổi bật

Thay vì chỉ trả lời như một Chatbot thông thường (dễ bị ảo giác kiến thức), Agent này hoạt động theo chu trình tự chủ 3 bước sử dụng dữ liệu thực tế:

1. **Trích xuất lý thuyết thật (ReadData):** Tự động lật và quét file sách giáo khoa `toan_lop_6_tap_1.pdf` để lấy định nghĩa chính xác.
2. **Sinh ví dụ minh họa (GenExamples):** Dựa trên lý thuyết gốc, gọi AI sinh ra bài tập ví dụ từ cơ bản để học sinh làm quen.
3. **Tạo đề kiểm tra (CreateTest):** Tự động thiết kế một bài test 15 phút ngắn để củng cố lại kiến thức vừa học.

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python
* **Giao diện Web:** Streamlit
* **Khung logic AI:** LangChain (ReAct Agent Architecture)
* **Mô hình AI:** Google Gemini (`gemini-3.5-flash-lite`)
* **Xử lý tài liệu:** `pypdf` (hỗ trợ đọc hiểu file PDF nội bộ)

## 🚀 Hướng dẫn cài đặt và chạy trên máy cá nhân (Local)

**Bước 1: Clone dự án về máy**
Tải toàn bộ mã nguồn về và đảm bảo có đủ các file: `app.py`, `ai_service.py`, `requirements.txt`, và `toan_lop_6_tap_1.pdf`.

**Bước 2: Cài đặt thư viện**
Mở Terminal và chạy lệnh:

```bash
pip install -r requirements.txt
```

# 🎓 Gia sư Toán 6 AI Agent

Ứng dụng Streamlit tích hợp AI Agent (sử dụng kiến trúc ReAct của LangChain và Google Gemini) đóng vai trò làm gia sư Toán lớp 6. Agent có khả năng tự động suy luận để gọi 3 công cụ:

1. Đọc lý thuyết từ dữ liệu gốc
2. Sinh bài tập ví dụ tương tự
3. Tạo đề kiểm tra ôn tập nhanh

## 1. Cài đặt môi trường (Windows)

Mở terminal tại thư mục chứa dự án và chạy các lệnh sau:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Cấu hình bảo mật API Key

Khóa API được bảo mật hoàn toàn thông qua file `.env`.

- Đảm bảo file `.env` nằm cùng thư mục với `app.py`.
- Lệnh `load_dotenv()` trong mã nguồn sẽ tự động nạp khóa từ file này.
- **Lưu ý:** Tuyệt đối không chia sẻ file `.env` hoặc tải nó lên Github.

## 3. Khởi chạy ứng dụng

Đảm bảo môi trường ảo (venv) đang được kích hoạt, gõ lệnh sau để chạy:

```bash
python -m streamlit run app.py
```

Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.
Khóa API của bạn sẽ được tự động điền sẵn ở thanh cấu hình bên trái nhờ file `.env`. Bạn chỉ cần bấm "Chạy thử Demo" để xem Agent tự động tư duy và gọi các công cụ (Tools).

Link deploy dự án: [gia-su-toan-6.streamlit.app](https://gia-su-toan-6.streamlit.app/)
