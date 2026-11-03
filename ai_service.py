import os
import re
import unicodedata
from pypdf import PdfReader
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

def normalize_text(text: str) -> str:
    """Hàm chuẩn hóa: Khử dấu tiếng Việt và xóa khoảng trắng thừa để so khớp tuyệt đối."""
    if not text:
        return ""
    # Ép về chữ không dấu (ví dụ: "Bài 5" -> "bai 5")
    text_no_accents = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    # Xóa dấu xuống dòng và các khoảng trắng kép
    return re.sub(r'\s+', ' ', text_no_accents.lower()).strip()

def build_executor(api_key: str, model: str):
    # Khởi tạo LLM chính điều phối Agent
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.2, # Giữ độ sáng tạo thấp để logic toán học chính xác
        max_retries=2,
    )

    # --- TOOL 1 (HÀNG THẬT V2): Quét file PDF Toán 6 với tính năng Chuẩn hóa ---
    def read_original_data(query: str) -> str:
        """Luôn dùng công cụ này đầu tiên để lấy lý thuyết Toán 6 gốc từ sách."""
        file_path = "toan_lop_6_tap_1.pdf"
        if not os.path.exists(file_path):
            return "Lỗi: Không tìm thấy file toan_lop_6_tap_1.pdf trong thư mục."
        
        try:
            reader = PdfReader(file_path)
            # 1. Chuẩn hóa từ khóa của học sinh
            query_clean = normalize_text(query)
            
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if raw_text:
                    # 2. Chuẩn hóa toàn bộ text của trang sách PDF để tìm kiếm
                    page_clean = normalize_text(raw_text)
                    
                    # 3. So khớp không dấu
                    if query_clean in page_clean:
                        # Nếu trúng mục tiêu, giao toàn bộ nội dung thô của trang đó (cắt bớt để tránh vượt giới hạn token)
                        snippet = raw_text.replace('\n', ' ')[:2000]
                        return f"Đã tìm thấy dữ liệu ở trang {i+1} SGK. Dưới đây là nội dung trang sách:\n{snippet}"
            
            # Bơm định hướng nếu không tìm thấy
            return f"Đã quét 100% sách nhưng không thấy cụm từ '{query}'. Yêu cầu học sinh cung cấp tên chủ đề cụ thể (ví dụ: tập hợp, lũy thừa, số nguyên âm)."
        except Exception as e:
            return f"Lỗi đọc PDF: {e}"

    # --- TOOL 2 (HÀNG THẬT): Gọi Gemini phụ sinh bài tập ---
    def generate_similar_examples(query: str) -> str:
        """Dùng công cụ này sau khi đã có lý thuyết để tạo bài tập ví dụ."""
        llm_phu = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.5)
        prompt = f"Dựa trên kiến thức Toán lớp 6, hãy sáng tác 1 ví dụ cơ bản, dễ hiểu về chủ đề '{query}' và kèm theo lời giải chi tiết. Tuyệt đối không cho bài tập nâng cao."
        response = llm_phu.invoke(prompt)
        return response.content

    # --- TOOL 3 (HÀNG THẬT): Gọi Gemini phụ sinh đề kiểm tra ---
    def create_review_test(query: str) -> str:
        """Dùng công cụ này cuối cùng để tạo bài test 15 phút cho học sinh."""
        llm_phu = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.5)
        prompt = f"Dựa trên kiến thức Toán lớp 6, hãy tạo 1 đề kiểm tra 15 phút (gồm 3 câu hỏi trắc nghiệm hoặc tự luận ngắn) về chủ đề '{query}'. Chỉ đưa ra đề bài, không đưa đáp án."
        response = llm_phu.invoke(prompt)
        return response.content

    tools = [
        Tool(name="ReadData", func=read_original_data, description="Luôn dùng công cụ này đầu tiên để lấy lý thuyết Toán 6 gốc."),
        Tool(name="GenExamples", func=generate_similar_examples, description="Dùng công cụ này sau khi đã có lý thuyết để tạo bài tập ví dụ."),
        Tool(name="CreateTest", func=create_review_test, description="Dùng công cụ này cuối cùng để tạo bài test 15 phút cho học sinh.")
    ]

    # --- HỆ THỐNG PROMPT MỚI: Kỷ luật sắt với AI ---
    prompt = PromptTemplate.from_template("""
Bạn là một AI Agent đóng vai trò Gia sư Toán lớp 6. 
Quy tắc dạy học BẮT BUỘC:
1. Bạn đang dạy theo đúng MỘT cuốn sách duy nhất (đã được nạp trong công cụ ReadData). TUYỆT ĐỐI KHÔNG phân bua hay hỏi học sinh về các bộ sách khác (Kết nối tri thức, Cánh diều...).
2. Khi học sinh hỏi bất cứ chủ đề hay số bài nào (ví dụ: "bài 5", "số nguyên"), BẠN PHẢI GỌI CÔNG CỤ ReadData để quét tài liệu gốc trước khi tự suy diễn.
3. Luôn dạy bài cơ bản trước, tuyệt đối không đưa bài tập nâng cao trừ khi học sinh yêu cầu.

Lịch sử chat: {chat_history}

Bạn có các công cụ sau:
{tools}

BẠN BẮT BUỘC PHẢI DÙNG ĐỊNH DẠNG NÀY ĐỂ GỌI CÔNG CỤ (Suy nghĩ -> Hành động -> Kết quả):
Thought: Bạn đang nghĩ xem cần làm bước gì tiếp theo
Action: Tên công cụ (chỉ được chọn 1 trong: [{tool_names}])
Action Input: Từ khóa cần tìm (ví dụ: bài 5)
Observation: Kết quả trả về từ công cụ
(Lặp lại chu trình này cho đến khi đủ thông tin)

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