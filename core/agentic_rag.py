"""
Agentic RAG System - Main Implementation
Hệ thống RAG với agents thông minh sử dụng LangGraph
"""

from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
import os

from .config import model_config, vectorstore_config, agent_config, system_config
from .agents import (
    AgentState,
    QueryAnalyzerAgent,
    RetrievalPlannerAgent,
    RetrievalAgent,
    ReasoningAgent,
    ValidationAgent,
    ResponseFormatterAgent
)


class AgenticRAG:
    """
    Agentic RAG System
    
    Workflow:
    1. Query Analysis - Phân tích câu hỏi
    2. Retrieval Planning - Lập kế hoạch tìm kiếm
    3. Retrieval - Tìm kiếm thông tin
    4. Reasoning - Suy luận trả lời
    5. Validation - Kiểm tra chất lượng
    6. Response Formatting - Format câu trả lời
    """
    
    def __init__(self, vectorstore: Chroma = None):
        """
        Khởi tạo Agentic RAG System
        
        Args:
            vectorstore: Chroma vectorstore (nếu None, sẽ load từ config)
        """
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_config.chat_model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            api_key=model_config.openai_api_key
        )
        
        # Initialize or load vectorstore
        if vectorstore is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_config.embedding_model
            )
            
            if os.path.exists(vectorstore_config.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=vectorstore_config.persist_directory,
                    embedding_function=embeddings
                )
                print(f"✅ Loaded vectorstore from {vectorstore_config.persist_directory}")
            else:
                raise ValueError(f"Vectorstore not found at {vectorstore_config.persist_directory}. Please create it first.")
        else:
            self.vectorstore = vectorstore
        
        # Initialize agents
        self.query_analyzer = QueryAnalyzerAgent(self.llm)
        self.retrieval_planner = RetrievalPlannerAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self.vectorstore)
        self.reasoning_agent = ReasoningAgent(self.llm)
        self.validation_agent = ValidationAgent(self.llm)
        self.formatter = ResponseFormatterAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("✅ AgenticRAG initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Xây dựng LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (các bước trong workflow)
        workflow.add_node("analyze_query", self.query_analyzer.analyze)
        workflow.add_node("plan_retrieval", self.retrieval_planner.plan)
        workflow.add_node("retrieve", self.retrieval_agent.retrieve)
        workflow.add_node("reason", self.reasoning_agent.reason)
        workflow.add_node("validate", self.validation_agent.validate)
        workflow.add_node("format_response", self.formatter.format)
        workflow.add_node("direct_response", self._handle_direct_response)
        
        # Define edges (luồng chạy)
        workflow.set_entry_point("analyze_query")
        
        # Conditional edge: skip retrieval nếu là greeting/chitchat
        workflow.add_conditional_edges(
            "analyze_query",
            self._needs_retrieval,
            {
                "retrieval": "plan_retrieval",
                "direct": "direct_response"
            }
        )
        
        workflow.add_edge("plan_retrieval", "retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "validate")
        
        # Conditional edge: retry nếu validation fail
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "retry": "plan_retrieval",  # Retry từ planning
                "continue": "format_response"
            }
        )
        
        workflow.add_edge("format_response", END)
        workflow.add_edge("direct_response", END)
        
        # Compile
        return workflow.compile()
    
    def _needs_retrieval(self, state: AgentState) -> Literal["retrieval", "direct"]:
        """Quyết định có cần retrieval hay trả lời trực tiếp"""
        analysis = state.get("query_analysis", {})
        needs_retrieval = analysis.get("needs_retrieval", True)
        
        if needs_retrieval:
            return "retrieval"
        else:
            return "direct"
    
    def _handle_direct_response(self, state: AgentState) -> AgentState:
        """Xử lý direct response (không cần retrieval)"""
        analysis = state.get("query_analysis", {})
        direct_response = analysis.get("direct_response", "")
        
        state["final_answer"] = direct_response
        state["confidence_score"] = 1.0
        state["citations"] = []
        state["current_step"] = "direct_response_completed"
        
        return state

    
    def _should_retry(self, state: AgentState) -> Literal["retry", "continue"]:
        """Quyết định có retry hay không"""
        if state.get("needs_retry", False):
            # Tăng retry count
            state["retry_count"] = state.get("retry_count", 0) + 1
            return "retry"
        return "continue"
    
    def query(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Xử lý câu hỏi từ user
        
        Args:
            question: Câu hỏi
            conversation_history: Lịch sử hội thoại (optional)
        
        Returns:
            Dict chứa:
            - answer: Câu trả lời
            - confidence: Độ tin cậy
            - citations: Nguồn tham khảo
            - metadata: Thông tin debug
        """
        if system_config.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 AGENTIC RAG PROCESSING")
            print(f"{'='*60}")
            print(f"Question: {question}")
        
        # Initialize state
        initial_state: AgentState = {
            "original_query": question,
            "conversation_history": conversation_history or [],
            "query_analysis": None,
            "reformulated_queries": [],
            "retrieved_documents": [],
            "retrieval_strategy": "",
            "reasoning_steps": [],
            "intermediate_answers": [],
            "final_answer": "",
            "confidence_score": 0.0,
            "citations": [],
            "validation_result": None,
            "needs_retry": False,
            "retry_count": 0,
            "current_step": "initialized",
            "error_message": None
        }
        
        try:
            # Run workflow với recursion limit cao hơn
            final_state = self.workflow.invoke(
                initial_state,
                {"recursion_limit": 50}  # Tăng từ 25 default
            )
            
            if system_config.verbose:
                print(f"\n{'='*60}")
                print(f"✅ PROCESSING COMPLETE")
                print(f"{'='*60}\n")
            
            # Extract results
            return {
                "answer": final_state["final_answer"],
                "confidence": final_state["confidence_score"],
                "citations": final_state["citations"],
                "metadata": {
                    "query_analysis": final_state.get("query_analysis"),
                    "num_documents": len(final_state.get("retrieved_documents", [])),
                    "retrieval_strategy": final_state.get("retrieval_strategy"),
                    "retry_count": final_state.get("retry_count", 0),
                    "validation": final_state.get("validation_result")
                }
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "answer": f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "metadata": {
                    "error": error_msg
                }
            }
    
    def chat(self, message: str, history: List[List[str]]) -> str:
        """
        Interface cho Gradio chatbot
        
        Args:
            message: Tin nhắn từ user
            history: Lịch sử chat [[user, bot], [user, bot], ...]
        
        Returns:
            Câu trả lời
        """
        # Convert history to conversation format
        conversation_history = []
        for user_msg, bot_msg in history:
            conversation_history.append({"role": "user", "content": user_msg})
            conversation_history.append({"role": "assistant", "content": bot_msg})
        
        # Query
        result = self.query(message, conversation_history)
        
        return result["answer"]
    
    def print_config(self):
        """In ra cấu hình hiện tại"""
        from .config import get_config_summary
        import json
        
        print("\n" + "="*60)
        print("AGENTIC RAG CONFIGURATION")
        print("="*60)
        print(json.dumps(get_config_summary(), indent=2, ensure_ascii=False))
        print("="*60 + "\n")


def load_agentic_rag(vectorstore_path: str = None) -> AgenticRAG:
    """
    Tiện ích để load AgenticRAG system
    
    Args:
        vectorstore_path: Đường dẫn đến vectorstore (optional)
    
    Returns:
        AgenticRAG instance
    """
    if vectorstore_path:
        # Load custom vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name=model_config.embedding_model
        )
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings
        )
        return AgenticRAG(vectorstore)
    else:
        # Load từ config
        return AgenticRAG()


if __name__ == "__main__":
    # Test the system
    print("🚀 Testing Agentic RAG System\n")
    
    try:
        # Initialize
        agentic_rag = AgenticRAG()
        agentic_rag.print_config()
        
        # Test query
        test_question = "Sinh viên bị điểm F phải làm gì?"
        
        print(f"\n📝 Test Question: {test_question}\n")
        
        result = agentic_rag.query(test_question)
        
        print(f"\n📄 Answer:\n{result['answer']}")
        print(f"\n🎯 Confidence: {result['confidence']:.2%}")
        print(f"\n📚 Citations: {', '.join(result['citations'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: Make sure you have created the vectorstore first by running the notebook!")
