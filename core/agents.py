"""
Agents cho Agentic RAG System
Định nghĩa các agents thực hiện các nhiệm vụ cụ thể
"""

from typing import List, Dict, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from .tools import (
    VectorSearchTool, 
    QueryReformulationTool, 
    QueryAnalysisTool,
    InformationExtractionTool,
    ValidationTool
)
from .config import model_config, agent_config, system_config


class AgentState(TypedDict):
    """State được share giữa các agents trong workflow"""
    # Input
    original_query: str
    conversation_history: List[Dict[str, str]]
    
    # Query Analysis
    query_analysis: Optional[Dict[str, Any]]
    reformulated_queries: List[str]
    
    # Retrieval
    retrieved_documents: List[Dict[str, Any]]
    retrieval_strategy: str
    
    # Reasoning
    reasoning_steps: List[str]
    intermediate_answers: List[str]
    
    # Response
    final_answer: str
    confidence_score: float
    citations: List[str]
    
    # Validation
    validation_result: Optional[Dict[str, Any]]
    needs_retry: bool
    retry_count: int
    
    # Metadata
    current_step: str
    error_message: Optional[str]


class QueryAnalyzerAgent:
    """
    Agent phân tích query của user
    Nhiệm vụ: Hiểu intent, trích xuất entities, xác định complexity
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.analysis_tool = QueryAnalysisTool(llm)
        
        # Import normalizer
        try:
            from .query_normalizer import normalizer
            self.normalizer = normalizer
        except ImportError:
            print("⚠️ Query normalizer not found, slang/abbreviation support disabled")
            self.normalizer = None
    
    def analyze(self, state: AgentState) -> AgentState:
        """Phân tích query và cập nhật state"""
        query = state["original_query"].strip()
        conversation_history = state.get("conversation_history", [])
        
        # NORMALIZE QUERY - Chuẩn hóa từ lóng và viết tắt
        original_query = query
        if self.normalizer:
            query = self.normalizer.normalize(query)
            
            # Log nếu có thay đổi
            if query != original_query and system_config.verbose:
                print(f"\n📝 [QueryNormalizer] Original: {original_query}")
                print(f"✅ [QueryNormalizer] Normalized: {query}")
                explanations = self.normalizer.get_explanation(original_query)
                if explanations:
                    print(f"   Terms normalized: {dict(explanations)}")
        
        if system_config.verbose:
            print(f"\n🔍 [QueryAnalyzer] Analyzing query: {query}")
        
        # QUERY CLASSIFICATION - Phân loại query trước khi retrieval
        query_type = self._classify_query(query, conversation_history)
        
        if system_config.verbose:
            print(f"   Query type: {query_type}")
        
        # Xử lý theo loại query
        if query_type == "greeting":
            # Chào hỏi - không cần retrieval
            state["query_analysis"] = {
                "intent": "greeting",
                "complexity": "simple",
                "needs_retrieval": False,
                "direct_response": self._handle_greeting(query)
            }
        elif query_type == "meta_conversation":
            # Câu hỏi về chính cuộc hội thoại
            state["query_analysis"] = {
                "intent": "meta_conversation",
                "complexity": "simple",
                "needs_retrieval": False,
                "direct_response": self._handle_meta_question(query, conversation_history)
            }
        elif query_type == "chitchat":
            # Chitchat không liên quan tài liệu
            state["query_analysis"] = {
                "intent": "chitchat",
                "complexity": "simple",
                "needs_retrieval": False,
                "direct_response": self._handle_chitchat(query)
            }
        elif query_type == "out_of_domain":
            # Câu hỏi ngoài domain - từ chối lịch sự
            state["query_analysis"] = {
                "intent": "out_of_domain",
                "complexity": "simple",
                "needs_retrieval": False,
                "direct_response": self._handle_out_of_domain(query)
            }
        else:
            # Document-related query - tiến hành phân tích bình thường
            analysis = self.analysis_tool.analyze(query)
            analysis["needs_retrieval"] = True
            state["query_analysis"] = analysis
            
            if system_config.verbose:
                print(f"   Intent: {analysis.get('intent', 'unknown')}")
                print(f"   Complexity: {analysis.get('complexity', 'unknown')}")
                print(f"   Key terms: {analysis.get('key_terms', [])}")
        
        state["current_step"] = "query_analyzed"
        return state
    
    def _classify_query(self, query:str, history: List[Dict[str, str]]) -> str:
        """Phân loại query: greeting, meta_conversation, chitchat, out_of_domain, document_related"""
        query_lower = query.lower()
        
        # Greetings
        greeting_patterns = [
            "xin chào", "chào", "hello", "hi", "hey",
            "chào bạn", "chào bot", "buổi sáng", "buổi chiều", "buổi tối"
        ]
        if any(pattern in query_lower for pattern in greeting_patterns) and len(query.split()) <= 5:
            return "greeting"
        
        # Meta-conversation questions (về chính cuộc hội thoại)
        meta_patterns = [
            "tôi vừa hỏi", "câu hỏi trước", "bạn vừa nói",
            "tôi hỏi gì", "tôi đã hỏi", "câu trước",
            "what did i ask", "previous question"
        ]
        if any(pattern in query_lower for pattern in meta_patterns):
            return "meta_conversation"
        
        # Chitchat không liên quan tài liệu
        chitchat_patterns = [
            "bạn là ai", "tên bạn là gì", "bạn làm được gì",
            "who are you", "what's your name", "how are you",
            "cảm ơn", "thank you", "thanks", "ok", "tạm biệt", "bye"
        ]
        if any(pattern in query_lower for pattern in chitchat_patterns):
            return "chitchat"
        
        # OUT OF DOMAIN - Câu hỏi hoàn toàn không liên quan quy chế đào tạo
        out_of_domain_patterns = [
            # Toán học
            "phương trình", "đạo hàm", "tích phân", "hình học", "đại số",
            "logarit", "lượng giác", "ma trận", "vector", "tổ hợp",
            # Vật lý, hóa học
            "lực", "gia tốc", "năng lượng", "nguyên tử", "phản ứng hóa học",
            # Lịch sử, địa lý
            "chiến tranh", "vua", "triều đại", "lãnh thổ", "đất nước",
            # Thời tiết, ẩm thực
            "thời tiết", "nấu ăn", "món ăn", "công thức nấu",
            # Thể thao, giải trí
            "bóng đá", "ca sĩ", "phim", "âm nhạc",
            # Lập trình (nếu không liên quan đào tạo)
            "code python", "lập trình java", "debug", "algorithm",
            # Y tế
            "bệnh", "thuốc", "triệu chứng", "điều trị"
        ]
        if any(pattern in query_lower for pattern in out_of_domain_patterns):
            return "out_of_domain"
        
        # Kiểm tra các từ khóa TRONG domain (quy chế đào tạo HaUI)
        domain_keywords = [
            "sinh viên", "học phần", "tín chỉ", "điểm", "thi", "tốt nghiệp",
            "đào tạo", "học kỳ", "chương trình", "quy chế", "điều", "chương",
            "đăng ký", "rút bớt", "nghỉ học", "bảo lưu", "kỷ luật",
            "gpa", "cpa", "haui", "đại học công nghiệp"
        ]
        
        # Nếu có từ khóa domain -> chắc chắn là document_related
        if any(keyword in query_lower for keyword in domain_keywords):
            return "document_related"
        
        # Nếu không match gì cả, dùng LLM để kiểm tra (fallback)
        # Tạm thời return document_related, nhưng có thể cải thiện sau
        return "document_related"
    
    def _handle_greeting(self, query: str) -> str:
        """Xử lý câu chào hỏi"""
        greetings = [
            "Xin chào! Tôi là trợ lý AI của Trường Đại học Công nghiệp Hà Nội. Tôi có thể giúp bạn tìm hiểu về quy chế đào tạo. Bạn có câu hỏi gì không?",
            "Chào bạn! Tôi sẵn sàng hỗ trợ bạn về các vấn đề liên quan đến quy định đào tạo tại HaUI. Hãy đặt câu hỏi nhé!",
            "Xin chào! Rất vui được hỗ trợ bạn. Tôi có thể trả lời các câu hỏi về quy chế đào tạo, điều kiện tốt nghiệp, và các quy định khác của trường. Bạn cần hỏi gì?"
        ]
        import random
        return random.choice(greetings)
    
    def _handle_meta_question(self, query: str, history: List[Dict[str, str]]) -> str:
        """Xử lý câu hỏi về chính cuộc hội thoại"""
        if not history or len(history) < 2:
            return "Bạn chưa hỏi câu nào trước đó trong cuộc hội thoại này."
        
        # Lấy TẤT CẢ câu hỏi của user
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        
        if not user_messages:
            return "Tôi không tìm thấy câu hỏi nào của bạn trong cuộc hội thoại này."
        
        query_lower = query.lower()
        
        # Phân biệt: hỏi TẤT CẢ vs chỉ câu TRƯỚC
        all_questions_patterns = [
            "tất cả", "all", "toàn bộ", "những câu", "các câu",
            "danh sách", "list", "lịch sử"
        ]
        
        ask_for_all = any(pattern in query_lower for pattern in all_questions_patterns)
        
        if ask_for_all and len(user_messages) > 1:
            # Trả về TẤT CẢ câu hỏi
            response = f"📝 Bạn đã hỏi tổng cộng {len(user_messages)} câu hỏi trong cuộc hội thoại này:\n\n"
            
            for idx, msg in enumerate(user_messages, 1):
                question = msg.get("content", "")
                # Giới hạn độ dài hiển thị
                if len(question) > 80:
                    question = question[:77] + "..."
                response += f"{idx}. {question}\n"
            
            response += "\nBạn muốn hỏi thêm về vấn đề nào không?"
            return response
        else:
            # Chỉ trả về câu CUỐI CÙNG
            last_question = user_messages[-1].get("content", "")
            return f'Câu hỏi trước đó của bạn là: "{last_question}"\n\nBạn có muốn hỏi thêm về vấn đề này không?'

    
    def _handle_chitchat(self, query: str) -> str:
        """Xử lý chitchat"""
        query_lower = query.lower()
        
        if "bạn là ai" in query_lower or "tên bạn" in query_lower:
            return "Tôi là trợ lý AI của Trường Đại học Công nghiệp Hà Nội, được thiết kế để hỗ trợ sinh viên và giảng viên về các quy định đào tạo. Tôi có thể giúp bạn tìm hiểu về quy chế đào tạo, điều kiện tốt nghiệp, và các quy định khác của trường."
        elif "cảm ơn" in query_lower or "thank" in query_lower:
            return "Rất vui được giúp đỡ bạn! Nếu có câu hỏi gì khác về quy chế đào tạo, đừng ngần ngại hỏi nhé. 😊"
        elif "tạm biệt" in query_lower or "bye" in query_lower:
            return "Tạm biệt! Chúc bạn học tập tốt. Hẹn gặp lại! 👋"
        else:
            return "Tôi được thiết kế để trả lời các câu hỏi về quy chế đào tạo tại ĐH Công nghiệp Hà Nội. Bạn có câu hỏi gì về quy định đào tạo, điều kiện tốt nghiệp, hoặc các vấn đề học tập không?"
    
    def _handle_out_of_domain(self, query: str) -> str:
        """Xử lý câu hỏi ngoài domain"""
        return """Xin lỗi, câu hỏi của bạn không thuộc phạm vi chuyên môn của tôi. 

Tôi là trợ lý AI chuyên về **Quy chế Đào tạo của Đại học Công nghiệp Hà Nội**. Tôi có thể giúp bạn với các vấn đề như:
• Quy định về học tập, thi cử, và tốt nghiệp
• Điều kiện, thủ tục liên quan đến đào tạo
• Các quy chế, quy định của trường
• Câu hỏi về học phần, tín chỉ, GPA/CPA

Bạn có câu hỏi nào liên quan đến đào tạo tại HaUI mà tôi có thể giúp không?"""


class RetrievalPlannerAgent:
    """
    Agent lập kế hoạch retrieval
    Nhiệm vụ: Quyết định chiến lược retrieval dựa trên query analysis
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.reformulation_tool = QueryReformulationTool(llm)
    
    def plan(self, state: AgentState) -> AgentState:
        """Lập kế hoạch retrieval"""
        query = state["original_query"]
        analysis = state.get("query_analysis", {})
        complexity = analysis.get("complexity", "medium")
        
        if system_config.verbose:
            print(f"\n📋 [RetrievalPlanner] Planning retrieval strategy...")
        
        # Xác định strategy dựa trên complexity
        if complexity == "simple":
            strategy = "single_query"
            queries = [query]
        elif complexity == "medium":
            strategy = "multi_query"
            if agent_config.enable_multi_query:
                queries = self.reformulation_tool.reformulate(query)
            else:
                queries = [query]
        else:  # complex
            strategy = "multi_query_with_expansion"
            if agent_config.enable_multi_query:
                queries = self.reformulation_tool.reformulate(query)
                if agent_config.enable_query_expansion:
                    expanded = self.reformulation_tool.expand_query(query)
                    queries.append(expanded)
            else:
                queries = [query]
        
        if system_config.verbose:
            print(f"   Strategy: {strategy}")
            print(f"   Generated {len(queries)} queries")
        
        # Cập nhật state
        state["retrieval_strategy"] = strategy
        state["reformulated_queries"] = queries
        state["current_step"] = "retrieval_planned"
        
        return state


class RetrievalAgent:
    """
    Agent thực hiện retrieval
    Nhiệm vụ: Tìm kiếm documents từ vector store
    """
    
    def __init__(self, vectorstore: Chroma):
        self.search_tool = VectorSearchTool(vectorstore)
    
    def retrieve(self, state: AgentState) -> AgentState:
        """Thực hiện retrieval"""
        queries = state.get("reformulated_queries", [state["original_query"]])
        
        if system_config.verbose:
            print(f"\n🔎 [Retrieval] Searching with {len(queries)} queries...")
        
        all_documents = []
        seen_contents = set()  # Để tránh duplicate
        
        for query in queries:
            results = self.search_tool.search(query)
            
            for doc in results:
                # Chỉ thêm nếu chưa có (dựa vào content)
                content_hash = hash(doc["content"])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_documents.append(doc)
        
        # Sắp xếp theo similarity score
        all_documents.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Giới hạn số lượng
        top_documents = all_documents[:agent_config.top_k if hasattr(agent_config, 'top_k') else 10]
        
        if system_config.verbose:
            print(f"   Retrieved {len(top_documents)} unique documents")
            if top_documents:
                print(f"   Top similarity: {top_documents[0].get('similarity_score', 0):.3f}")
        
        # Cập nhật state
        state["retrieved_documents"] = top_documents
        state["current_step"] = "documents_retrieved"
        
        return state


class ReasoningAgent:
    """
    Agent thực hiện reasoning
    Nhiệm vụ: Suy luận từ documents để trả lời câu hỏi
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.extraction_tool = InformationExtractionTool(llm)
    
    def reason(self, state: AgentState) -> AgentState:
        """Thực hiện reasoning"""
        query = state["original_query"]
        documents = state.get("retrieved_documents", [])
        analysis = state.get("query_analysis", {})
        
        if system_config.verbose:
            print(f"\n🧠 [Reasoning] Processing {len(documents)} documents...")
        
        if not documents:
            state["final_answer"] = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            state["confidence_score"] = 0.0
            state["current_step"] = "reasoning_completed"
            return state
        
        # Chain of Thought reasoning nếu câu hỏi phức tạp
        if agent_config.enable_chain_of_thought and analysis.get("complexity") == "complex":
            answer = self._chain_of_thought_reasoning(query, documents, analysis)
        else:
            answer = self._direct_reasoning(query, documents)
        
        # Trích xuất citations
        citations = self._extract_citations(documents)
        
        # Tính confidence score dựa trên similarity scores
        if documents:
            avg_similarity = sum(doc.get("similarity_score", 0) for doc in documents[:3]) / min(3, len(documents))
            confidence = min(0.95, avg_similarity)
        else:
            confidence = 0.0
        
        if system_config.verbose:
            print(f"   Confidence: {confidence:.2f}")
        
        # Cập nhật state
        state["final_answer"] = answer
        state["confidence_score"] = confidence
        state["citations"] = citations
        state["current_step"] = "reasoning_completed"
        
        return state
    
    def _direct_reasoning(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Reasoning trực tiếp từ documents"""
        # Tạo context
        context = "\n\n---\n\n".join([
            f"[Nguồn: {doc.get('doc_type', 'Unknown')}]\n{doc['content']}" 
            for doc in documents[:5]
        ])
        
        prompt = f"""{system_config.system_role}

Dựa vào các tài liệu sau, hãy trả lời câu hỏi một cách chính xác và đầy đủ.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {query}

YÊU CẦU:
1. Trả lời chính xác dựa trên tài liệu
2. Trích dẫn cụ thể (Điều số, Chương số)
3. Nếu có nhiều điều kiện, liệt kê rõ ràng
4. Nếu không chắc chắn, nói rõ

TRẢ LỜI:"""
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def _chain_of_thought_reasoning(self, query: str, documents: List[Dict[str, Any]], analysis: Dict) -> str:
        """Chain of Thought reasoning cho câu hỏi phức tạp"""
        sub_questions = analysis.get("sub_questions", [])
        
        if not sub_questions:
            return self._direct_reasoning(query, documents)
        
        # Trả lời từng câu hỏi con
        intermediate_answers = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            if system_config.verbose:
                print(f"   Sub-question {i}: {sub_q}")
            
            answer = self._direct_reasoning(sub_q, documents)
            intermediate_answers.append(f"**Câu hỏi {i}:** {sub_q}\n**Trả lời:** {answer}")
        
        # Tổng hợp câu trả lời
        context = "\n\n".join(intermediate_answers)
        
        synthesis_prompt = f"""Dựa vào các câu trả lời cho các câu hỏi con, hãy tổng hợp thành một câu trả lời hoàn chỉnh cho câu hỏi gốc.

CÂU HỎI GỐC: {query}

CÁC CÂU TRẢ LỜI CON:
{context}

Hãy tổng hợp thành câu trả lời mạch lạc, đầy đủ và dễ hiểu."""
        
        response = self.llm.invoke(synthesis_prompt)
        return response.content.strip()
    
    def _extract_citations(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Trích xuất citations từ documents"""
        citations = []
        for doc in documents[:3]:  # Top 3 documents
            source = doc.get("doc_type", "Unknown")
            citations.append(source)
        return list(set(citations))  # Remove duplicates


class ValidationAgent:
    """
    Agent validate câu trả lời
    Nhiệm vụ: Kiểm tra chất lượng câu trả lời
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.validation_tool = ValidationTool(llm)
    
    def validate(self, state: AgentState) -> AgentState:
        """Validate câu trả lời"""
        if not agent_config.enable_answer_validation:
            state["validation_result"] = {"is_valid": True, "confidence": 1.0}
            state["needs_retry"] = False
            state["current_step"] = "validation_completed"
            return state
        
        query = state["original_query"]
        answer = state.get("final_answer", "")
        documents = state.get("retrieved_documents", [])
        confidence = state.get("confidence_score", 0.0)
        
        if system_config.verbose:
            print(f"\n✓ [Validation] Validating answer...")
        
        # Validate
        validation_result = self.validation_tool.validate(query, answer, documents)
        
        is_valid = validation_result.get("is_valid", False)
        val_confidence = validation_result.get("confidence", 0.0)
        
        # Quyết định có cần retry không
        needs_retry = (
            not is_valid or 
            val_confidence < agent_config.min_confidence_score or
            confidence < agent_config.min_confidence_score
        ) and state.get("retry_count", 0) < agent_config.max_retries
        
        if system_config.verbose:
            print(f"   Valid: {is_valid}, Confidence: {val_confidence:.2f}")
            if needs_retry:
                print(f"   ⚠️ Needs retry (attempt {state.get('retry_count', 0) + 1}/{agent_config.max_retries})")
        
        # Cập nhật state
        state["validation_result"] = validation_result
        state["needs_retry"] = needs_retry
        state["current_step"] = "validation_completed"
        
        return state


class ResponseFormatterAgent:
    """
    Agent format câu trả lời cuối cùng
    Nhiệm vụ: Format câu trả lời với citations, confidence, etc.
    """
    
    def format(self, state: AgentState) -> AgentState:
        """Format câu trả lời"""
        answer = state.get("final_answer", "")
        citations = state.get("citations", [])
        confidence = state.get("confidence_score", 0.0)
        
        if system_config.verbose:
            print(f"\n📝 [Formatter] Formatting final response...")
        
        # Format với citations nếu được yêu cầu
        if agent_config.require_citations and citations:
            formatted_answer = f"{answer}\n\n---\n**Nguồn tham khảo:** {', '.join(citations)}"
        else:
            formatted_answer = answer
        
        # Thêm confidence warning nếu thấp
        if confidence < 0.7:
            formatted_answer += f"\n\n*Lưu ý: Độ tin cậy của câu trả lời này là {confidence:.0%}. Vui lòng kiểm tra lại hoặc hỏi cụ thể hơn.*"
        
        state["final_answer"] = formatted_answer
        state["current_step"] = "response_formatted"
        
        return state


if __name__ == "__main__":
    print("✅ Agents module loaded successfully")
    print(f"📦 Available agents:")
    print(f"   - QueryAnalyzerAgent")
    print(f"   - RetrievalPlannerAgent")
    print(f"   - RetrievalAgent")
    print(f"   - ReasoningAgent")
    print(f"   - ValidationAgent")
    print(f"   - ResponseFormatterAgent")
