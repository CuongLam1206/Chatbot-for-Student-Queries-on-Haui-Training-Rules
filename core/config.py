"""
Configuration cho Agentic RAG System
Quản lý tất cả các cấu hình cho agents, models, và retrieval
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Cấu hình cho các mô hình AI"""
    # OpenAI models
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class VectorStoreConfig:
    """Cấu hình cho vector database"""
    persist_directory: str = "vector_db"
    collection_name: str = "haui_regulations"
    
    # Retrieval settings
    top_k: int = 5  # Số lượng chunks lấy về
    similarity_threshold: float = 0.5  # Ngưỡng similarity tối thiểu
    
    # Chunk settings (đã có data, nhưng lưu lại cho tham khảo)
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class AgentConfig:
    """Cấu hình cho Agents"""
    # Query Analysis
    min_query_length: int = 5
    max_query_reformulations: int = 3  # Số lần tối đa reformulate query
    
    # Retrieval Planning
    enable_multi_query: bool = True  # Tạo nhiều queries từ 1 query
    enable_query_expansion: bool = True  # Mở rộng query với từ đồng nghĩa
    
    # Reasoning
    enable_chain_of_thought: bool = True  # Suy luận từng bước
    enable_self_reflection: bool = True  # Tự đánh giá kết quả
    max_reasoning_steps: int = 5  # Số bước suy luận tối đa
    
    # Response Generation
    require_citations: bool = True  # Yêu cầu trích dẫn nguồn
    min_confidence_score: float = 0.5  # Lowered from 0.6 để giảm retry
    
    # Validation
    enable_answer_validation: bool = True
    max_retries: int = 1  # Giảm từ 2 xuống 1 để tránh loop


@dataclass
class SystemConfig:
    """Cấu hình tổng thể của hệ thống"""
    # System prompt
    system_role: str = """Bạn là một chuyên gia tư vấn đào tạo tại Trường Đại học Công nghiệp Hà Nội.
Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến quy chế đào tạo đại học và cao đẳng hệ chính quy theo học chế tín chỉ.
Bạn cần:
1. Phân tích câu hỏi kỹ lưỡng
2. Tìm kiếm thông tin chính xác từ tài liệu
3. Suy luận logic để đưa ra câu trả lời đầy đủ
4. Trích dẫn nguồn cụ thể (Điều, Chương)
5. Thừa nhận nếu không tìm thấy thông tin"""
    
    # Logging
    verbose: bool = True  # In ra logs để debug
    log_file: Optional[str] = "agentic_rag.log"
    
    # Performance
    enable_caching: bool = True  # Cache kết quả retrieval
    parallel_tool_execution: bool = False  # Chạy tools song song (experimental)


@dataclass
class MongoDBConfig:
    """Cấu hình cho MongoDB"""
    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database: str = os.getenv("MONGODB_DATABASE", "agentic_rag_db")
    sessions_collection: str = "sessions"
    messages_collection: str = "messages"



# Global config instances
model_config = ModelConfig()
vectorstore_config = VectorStoreConfig()
agent_config = AgentConfig()
system_config = SystemConfig()


def get_config_summary() -> dict:
    """Lấy summary của tất cả configs"""
    return {
        "model": {
            "chat_model": model_config.chat_model,
            "temperature": model_config.temperature,
            "embedding_model": model_config.embedding_model,
        },
        "vectorstore": {
            "persist_directory": vectorstore_config.persist_directory,
            "top_k": vectorstore_config.top_k,
        },
        "agent": {
            "multi_query": agent_config.enable_multi_query,
            "chain_of_thought": agent_config.enable_chain_of_thought,
            "self_reflection": agent_config.enable_self_reflection,
        },
        "system": {
            "verbose": system_config.verbose,
        }
    }


if __name__ == "__main__":
    # Test config
    import json
    print("🔧 Configuration Summary:")
    print(json.dumps(get_config_summary(), indent=2, ensure_ascii=False))
