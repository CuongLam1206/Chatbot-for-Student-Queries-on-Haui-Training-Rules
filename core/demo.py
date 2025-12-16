"""
Demo Agentic RAG System
Tạo giao diện Gradio để test và demo hệ thống
"""

import gradio as gr
from core.agentic_rag import AgenticRAG, load_agentic_rag
from core.config import system_config
import os


def create_demo_interface():
    """Tạo Gradio interface cho Agentic RAG"""
    
    # Initialize Agentic RAG
    print("🚀 Initializing Agentic RAG System...")
    agentic_rag = load_agentic_rag()
    
    # Print config
    agentic_rag.print_config()
    
    # Tạo interface
    with gr.Blocks(title="Agentic RAG - HaUI Regulations") as demo:
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🤖 Agentic RAG System</h1>
                <h3>Hệ thống Tư vấn Quy chế Đào tạo - Đại học Công nghiệp Hà Nội</h3>
                <p><i>Powered by LangGraph & GPT-4o-mini</i></p>
            </div>
        """)
        
        # Description
        with gr.Row():
            gr.Markdown("""
            ## 🎯 Giới thiệu
            
            Đây là **Agentic RAG** - hệ thống RAG nâng cao với các agents thông minh có khả năng:
            - 🔍 **Phân tích câu hỏi** để hiểu intent và độ phức tạp
            - 📋 **Lập kế hoạch retrieval** tự động
            - 🔎 **Tìm kiếm thông minh** với multi-query và query expansion
            - 🧠 **Suy luận đa bước** (Chain-of-Thought) cho câu hỏi phức tạp
            - ✓ **Tự kiểm tra** và cải thiện câu trả lời
            
            ### 💡 Hướng dẫn sử dụng:
            1. Nhập câu hỏi về quy chế đào tạo
            2. Hệ thống sẽ tự động phân tích và tìm kiếm thông tin
            3. Nhận câu trả lời với độ tin cậy và nguồn trích dẫn
            
            ### 📚 Ví dụ câu hỏi:
            - Sinh viên bị điểm F phải làm gì?
            - Điều kiện để được xét tốt nghiệp là gì?
            - Thời gian tối đa để hoàn thành chương trình là bao lâu?
            - Sinh viên có thể học bao nhiêu tín chỉ mỗi học kỳ?
            """)
        
        # Chatbot interface
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Chat với Agentic RAG",
                    height=500,
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Câu hỏi của bạn",
                        placeholder="Nhập câu hỏi về quy chế đào tạo...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Gửi", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Xóa lịch sử", scale=1)
                    examples_btn = gr.Button("💡 Xem ví dụ", scale=1)
            
            # Sidebar với thông tin
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Thông tin hệ thống")
                
                info_text = gr.Textbox(
                    label="Trạng thái",
                    value=f"""
✅ Model: {agentic_rag.llm.model_name}
✅ Vectorstore: Loaded
✅ Agents: Active
✅ Mode: Agentic RAG
                    """.strip(),
                    lines=6,
                    interactive=False
                )
                
                gr.Markdown("### 🎛️ Cấu hình")
                verbose_checkbox = gr.Checkbox(
                    label="Hiển thị logs chi tiết",
                    value=system_config.verbose
                )
                
                gr.Markdown("### 📊 Thống kê")
                stats_text = gr.Textbox(
                    label="Số liệu",
                    value="Chưa có câu hỏi nào",
                    lines=4,
                    interactive=False
                )
        
        # Examples
        with gr.Row():
            gr.Examples(
                examples=[
                    "Sinh viên bị điểm F phải làm gì và có được học lại không?",
                    "Điều kiện để được xét tốt nghiệp là gì?",
                    "Thời gian tối đa để hoàn thành chương trình đại học là bao lâu?",
                    "Sinh viên có thể đăng ký bao nhiêu tín chỉ mỗi học kỳ?",
                    "Quy định về nghỉ học tạm thời là gì?",
                    "Khi nào sinh viên bị buộc thôi học?",
                ],
                inputs=msg,
                label="📝 Câu hỏi mẫu (click để dùng)"
            )
        
        # Functions
        def respond(message, chat_history):
            """Xử lý câu hỏi và trả lời"""
            # Update verbose setting
            system_config.verbose = verbose_checkbox.value
            
            # Get response from agentic RAG
            result = agentic_rag.query(message)
            
            bot_message = result["answer"]
            
            # Thêm thông tin metadata nếu verbose
            if verbose_checkbox.value:
                metadata = result.get("metadata", {})
                bot_message += f"\n\n---\n*Debug Info:*"
                bot_message += f"\n- Documents retrieved: {metadata.get('num_documents', 0)}"
                bot_message += f"\n- Strategy: {metadata.get('retrieval_strategy', 'N/A')}"
                bot_message += f"\n- Retries: {metadata.get('retry_count', 0)}"
            
            # Gradio internally expects messages format
            # Always append as message dictionaries
            chat_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": bot_message}
            ]
            
            # Update stats
            confidence = result.get("confidence", 0.0)
            num_docs = result.get("metadata", {}).get("num_documents", 0)
            stats = f"""
Câu hỏi cuối: ✓
Độ tin cậy: {confidence:.0%}
Documents: {num_docs}
            """.strip()
            
            return "", chat_history, stats
        
        def clear_history():
            """Xóa lịch sử chat"""
            return [], "Chưa có câu hỏi nào"
        
        # Event handlers
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, stats_text]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, stats_text]
        )
        
        clear_btn.click(
            clear_history,
            outputs=[chatbot, stats_text]
        )
        
        verbose_checkbox.change(
            lambda x: system_config.__setattr__("verbose", x),
            inputs=[verbose_checkbox]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch demo
    print("\n" + "="*60)
    print("🚀 LAUNCHING AGENTIC RAG DEMO")
    print("="*60 + "\n")
    
    try:
        demo = create_demo_interface()
        
        # Launch - let Gradio find available port automatically
        demo.launch(
            share=False,  # Set to True to create public link
            server_name="127.0.0.1",
            server_port=None,  # Auto-select available port
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("\nMake sure:")
        print("1. You have created the vectorstore (run notebook first)")
        print("2. All dependencies are installed (pip install -r requirements.txt)")
        print("3. OpenAI API key is set in .env file")
