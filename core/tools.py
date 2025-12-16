"""
Tools cho Agentic RAG System
Định nghĩa các công cụ mà agents có thể sử dụng
"""

from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import re
from .config import model_config, vectorstore_config, agent_config


class VectorSearchTool:
    """Tool để search trong vector database"""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.top_k = vectorstore_config.top_k
    
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Tìm kiếm trong vector database
        
        Args:
            query: Câu query
            k: Số lượng kết quả (mặc định dùng config)
        
        Returns:
            List các documents với metadata và similarity scores
        """
        k = k or self.top_k
        
        # Similarity search với scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Unknown")
            })
        
        return formatted_results
    
    def search_with_filter(self, query: str, filter_dict: Dict[str, Any], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search với metadata filtering
        
        Args:
            query: Câu query
            filter_dict: Điều kiện lọc, ví dụ: {"doc_type": "Chapter I"}
            k: Số lượng kết quả
        """
        k = k or self.top_k
        
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return formatted_results


class QueryReformulationTool:
    """Tool để cải thiện và reformulate queries"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def reformulate(self, original_query: str, context: str = "") -> List[str]:
        """
        Tạo các phiên bản query khác nhau để tăng khả năng tìm được thông tin
        
        Args:
            original_query: Query gốc
            context: Context bổ sung (nếu có)
        
        Returns:
            List các query đã được reformulate
        """
        prompt = f"""Bạn là chuyên gia về quy chế đào tạo. Hãy tạo {agent_config.max_query_reformulations} cách diễn đạt khác nhau cho câu hỏi sau để tìm kiếm thông tin hiệu quả hơn.

Câu hỏi gốc: {original_query}

Yêu cầu:
1. Giữ nguyên ý nghĩa câu hỏi
2. Sử dụng từ khóa và thuật ngữ chính thức trong quy chế
3. Mỗi cách diễn đạt nên tập trung vào khía cạnh khác nhau của câu hỏi

Trả về {agent_config.max_query_reformulations} câu hỏi, mỗi câu trên một dòng, không đánh số."""

        response = self.llm.invoke(prompt)
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        
        # Luôn bao gồm query gốc
        if original_query not in queries:
            queries.insert(0, original_query)
        
        return queries[:agent_config.max_query_reformulations + 1]
    
    def expand_query(self, query: str) -> str:
        """
        Mở rộng query với các từ đồng nghĩa và thuật ngữ liên quan
        """
        prompt = f"""Hãy mở rộng câu hỏi sau bằng cách thêm các từ đồng nghĩa, thuật ngữ liên quan trong quy chế đào tạo:

Câu hỏi: {query}

Trả về câu hỏi đã được mở rộng (chỉ 1 câu duy nhất)."""

        response = self.llm.invoke(prompt)
        return response.content.strip()


class QueryAnalysisTool:
    """Tool để phân tích query của user"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Phân tích câu hỏi để hiểu intent và trích xuất thông tin quan trọng
        
        Returns:
            Dict chứa:
            - intent: Mục đích câu hỏi (query, definition, procedure, comparison, etc.)
            - key_terms: Các từ khóa quan trọng
            - entities: Các thực thể (Điều X, Chương Y, etc.)
            - complexity: Độ phức tạp (simple, medium, complex)
            - sub_questions: Các câu hỏi con (nếu có)
        """
        prompt = f"""Phân tích câu hỏi sau về quy chế đào tạo:

Câu hỏi: {query}

Hãy trả về phân tích theo format JSON với các trường:
- intent: Loại câu hỏi (query: hỏi thông tin, definition: hỏi định nghĩa, procedure: hỏi quy trình, comparison: so sánh, calculation: tính toán)
- key_terms: List các từ khóa quan trọng
- entities: List các thực thể cụ thể (Điều số, Chương số, học phần, điểm số, etc.)
- complexity: simple/medium/complex
- sub_questions: Nếu câu hỏi phức tạp, chia thành các câu hỏi con (list)

Chỉ trả về JSON, không giải thích thêm."""

        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        import json
        try:
            result = json.loads(response.content.strip())
        except:
            # Fallback nếu không parse được
            result = {
                "intent": "query",
                "key_terms": self._extract_keywords(query),
                "entities": self._extract_entities(query),
                "complexity": "medium",
                "sub_questions": []
            }
        
        return result
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords đơn giản bằng regex"""
        # Loại bỏ stop words tiếng Việt cơ bản
        stop_words = {'là', 'của', 'và', 'có', 'được', 'trong', 'cho', 'với', 'để', 'khi', 'nào', 'như', 'về'}
        words = re.findall(r'\w+', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities như Điều X, Chương Y"""
        entities = []
        
        # Extract Điều X
        dieu_matches = re.findall(r'Điều\s+\d+', query)
        entities.extend(dieu_matches)
        
        # Extract Chương X
        chuong_matches = re.findall(r'Chương\s+[IVX]+', query, re.IGNORECASE)
        entities.extend(chuong_matches)
        
        return entities


class InformationExtractionTool:
    """Tool để trích xuất thông tin cụ thể từ documents"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def extract(self, documents: List[Dict[str, Any]], query: str) -> str:
        """
        Trích xuất thông tin liên quan đến query từ documents
        
        Args:
            documents: List các documents từ retrieval
            query: Câu hỏi gốc
        
        Returns:
            Thông tin đã được trích xuất và tổng hợp
        """
        # Tạo context từ documents
        context = "\n\n---\n\n".join([
            f"[{doc['doc_type']}]\n{doc['content']}" 
            for doc in documents
        ])
        
        prompt = f"""Dựa vào các đoạn văn bản sau từ quy chế đào tạo, hãy trích xuất thông tin trả lời câu hỏi.

CÂU HỎI: {query}

TÀI LIỆU:
{context}

Hãy trích xuất và tổng hợp thông tin có liên quan. Chỉ sử dụng thông tin từ tài liệu, không bịa thêm."""

        response = self.llm.invoke(prompt)
        return response.content.strip()


class ValidationTool:
    """Tool để validate câu trả lời"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def validate(self, query: str, answer: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate xem câu trả lời có đầy đủ và chính xác không
        
        Returns:
            Dict chứa:
            - is_valid: bool
            - confidence: float (0-1)
            - issues: List các vấn đề (nếu có)
            - suggestions: Gợi ý cải thiện
        """
        context = "\n".join([doc['content'] for doc in source_documents])
        
        prompt = f"""Đánh giá chất lượng câu trả lời sau:

CÂU HỎI: {query}

CÂU TRẢ LỜI: {answer}

TÀI LIỆU THAM KHẢO:
{context}

Hãy đánh giá theo các tiêu chí:
1. Câu trả lời có trả lời đầy đủ câu hỏi không?
2. Thông tin có chính xác dựa trên tài liệu không?
3. Có thiếu thông tin quan trọng nào không?
4. Có thông tin sai lệch hoặc bịa đặt không?

Trả về JSON với format:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "issues": ["vấn đề 1", "vấn đề 2"],
  "suggestions": ["gợi ý 1", "gợi ý 2"]
}}

Chỉ trả về JSON, không giải thích."""

        response = self.llm.invoke(prompt)
        
        import json
        try:
            result = json.loads(response.content.strip())
        except:
            # Fallback
            result = {
                "is_valid": True,
                "confidence": 0.7,
                "issues": [],
                "suggestions": []
            }
        
        return result


if __name__ == "__main__":
    # Test tools
    print("✅ Tools module loaded successfully")
    print(f"📦 Available tool classes: VectorSearchTool, QueryReformulationTool, QueryAnalysisTool, InformationExtractionTool, ValidationTool")
