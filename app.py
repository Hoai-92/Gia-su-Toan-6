import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
QUESTION = "Thầy ơi, hôm nay dạy em Bài 1 về Tập hợp nhé. Cho em xin lý thuyết, ví dụ và 1 bài test nhỏ để em tự làm."

st.set_page_config(page_title="Gia sư Toán 6 AI", page_icon="🎓", layout="centered")
st.title("🎓 Gia sư Toán 6 AI Agent")
st.caption("Agent tự động: Trích xuất lý thuyết → Sinh ví dụ → Tạo đề kiểm tra")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Cấu hình AI")
    api_key = st.text_input("Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    model = st.text_input("Model", value="gemini-3.5-flash-lite")
    
    show_steps = st.checkbox("Hiển thị các bước AI tư duy (Tools)", value=True)
    if st.button("Xóa lịch sử bài học", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

def render_message(message):
    st.markdown(message["content"])
    if show_steps and message.get("steps"):
        with st.expander("Quy trình Agent gọi Tools"):
            for step in message["steps"]:
                st.markdown(f"**🛠️ Công cụ đã dùng: {step['tool']}**")
                st.code(f"Input: {step['input']}", language="text")
                st.text(f"Kết quả (Observation):\n{step['observation']}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

quick = st.button("🚀 Chạy thử Demo: Dạy bài Tập Hợp Toán 6", use_container_width=True)
typed = st.chat_input("Hỏi gia sư bài toán bạn cần học...")
question = QUESTION if quick else typed

if question:
    if not api_key.strip():
        st.warning("Bạn chưa nhập mã Gemini API key ở thanh bên trái!")
    else:
        previous = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Gia sư đang tra cứu sách và soạn bài..."):
                    from ai_service import answer_question
                    # Dùng mã khóa hoạt động bình thường của bạn
                    result = answer_question(question, previous, api_key.strip(), model.strip())
                
                message = {
                    "role": "assistant", 
                    "content": result["output"], 
                    "steps": result["steps"]
                }
                render_message(message)
                st.session_state.messages.append(message)
            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối: {e}")