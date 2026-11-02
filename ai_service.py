import json
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- TEST TOOL 1: Đọc dữ liệu gốc ---
def read_original_data(query: str) -> str:
    """Đọc lý thuyết từ sách giáo khoa Toán 6."""
    query = query.lower()
    if "tập hợp" in query:
        return "Bài 1: Tập hợp. Khái niệm cốt lõi: Tập hợp là sự gom nhóm các đối tượng (phần tử) có chung tính chất. Ký hiệu thường dùng: A = {1, 2, 3}. Mỗi phần tử liệt kê 1 lần, cách nhau bởi dấu phẩy hoặc chấm phẩy."
    return f"Đã tìm thấy lý thuyết cơ bản cho phần: {query} trong SGK Toán 6 Tập 1."

# --- TEST TOOL 2: Sinh ví dụ tương tự ---
def generate_similar_examples(query: str) -> str:
    """Tạo các bài tập ví dụ cơ bản cho học sinh làm quen."""
    return f"Ví dụ minh họa cho phần '{query}': Hãy viết tập hợp B gồm các số tự nhiên chẵn nhỏ hơn 10. Đáp án mẫu: B = {{0; 2; 4; 6; 8}}."

# --- TEST TOOL 3: Tạo đề ôn tập ---
def create_review_test(query: str) -> str:
    """Tạo đề kiểm tra ôn tập ngắn để củng cố kiến thức."""
    return f"Đề luyện tập nhanh chủ đề '{query}':\nCâu 1: Viết tập hợp C các chữ cái trong từ 'HANOI'.\nCâu 2: Điền ký hiệu thuộc (∈) hoặc không thuộc (∉) thích hợp: 5 ... {{1, 3, 5, 7}}."

def build_executor(api_key: str, model: str):
    # Sử dụng Google Gemini thay vì OpenAI
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.2, # Giữ độ sáng tạo thấp để toán học được chính xác
        max_retries=2,
    )

    tools = [
        Tool(name="ReadData", func=read_original_data, description="Luôn dùng công cụ này đầu tiên để lấy lý thuyết Toán 6 gốc."),
        Tool(name="GenExamples", func=generate_similar_examples, description="Dùng công cụ này sau khi đã có lý thuyết để tạo bài tập ví dụ."),
        Tool(name="CreateTest", func=create_review_test, description="Dùng công cụ này cuối cùng để tạo bài test 15 phút cho học sinh.")
    ]

    # Cấu hình Prompt (Đã đưa phong cách dạy từ cơ bản của bạn vào làm kim chỉ nam)
    prompt = PromptTemplate.from_template("""
Bạn là một AI Agent đóng vai trò Gia sư Toán lớp 6.
Quy tắc dạy học: Luôn dạy bài cơ bản trước, tuyệt đối không đưa bài tập nâng cao trừ khi học sinh yêu cầu.
Lịch sử chat: {chat_history}

Bạn có các công cụ sau:
{tools}

BẠN BẮT BUỘC PHẢI DÙNG ĐỊNH DẠNG NÀY ĐỂ GỌI CÔNG CỤ (Suy nghĩ -> Hành động -> Kết quả):
Thought: Bạn đang nghĩ xem cần làm bước gì tiếp theo
Action: Tên công cụ (chỉ được chọn 1 trong: [{tool_names}])
Action Input: Từ khóa cần tìm (ví dụ: tập hợp)
Observation: Kết quả trả về từ công cụ
(Lặp lại chu trình Thought/Action/Action Input/Observation cho đến khi đủ thông tin)

Khi đã có đủ thông tin, hãy trả lời học sinh theo định dạng:
Final Answer: Trình bày bài giảng hoàn chỉnh, thân thiện.

Câu hỏi của học sinh: {input}
Thought:{agent_scratchpad}
""")
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )
    return agent_executor

def answer_question(question, history, api_key, model):
    executor = build_executor(api_key, model)
    context = "\n".join(f"{m['role']}: {m['content'][:1000]}" for m in history[-4:])
    result = executor.invoke({"input": question, "chat_history": context or "(empty)"})
    steps = [
        {"tool": action.tool, "input": str(action.tool_input), "observation": str(observation)}
        for action, observation in result.get("intermediate_steps", [])
    ]
    return {"output": result["output"], "steps": steps}