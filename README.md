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
