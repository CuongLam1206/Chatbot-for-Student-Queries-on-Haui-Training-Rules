"""
MongoDB database integration for chat history and conversation management
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "agentic_rag_db")

# OpenAI for title generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Database:
    """MongoDB database manager"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DATABASE]
        self.sessions = self.db["sessions"]
        self.messages = self.db["messages"]
        
        # Create indexes for performance
        self.sessions.create_index("session_id", unique=True)
        self.sessions.create_index([("updated_at", DESCENDING)])
        self.messages.create_index("session_id")
        self.messages.create_index([("timestamp", DESCENDING)])
        
        # Initialize LLM for title generation
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=50,
            api_key=OPENAI_API_KEY
        )
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create new conversation session
        
        Args:
            session_id: Unique session ID
            
        Returns:
            Created session document
        """
        now = datetime.now()
        session = {
            "session_id": session_id,
            "title": None,  # Will be set after first message
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }
        
        self.sessions.insert_one(session)
        return session
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Get all conversations sorted by last update
        
        Returns:
            List of conversation summaries
        """
        conversations = list(
            self.sessions.find(
                {},
                {"_id": 0, "session_id": 1, "title": 1, "updated_at": 1, "message_count": 1}
            ).sort("updated_at", DESCENDING)
        )
        
        # Set default title for conversations without one
        for conv in conversations:
            if not conv.get("title"):
                conv["title"] = "New Conversation"
        
        return conversations
    
    def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full conversation with messages
        
        Args:
            session_id: Session ID
            
        Returns:
            Conversation with messages or None if not found
        """
        session = self.sessions.find_one({"session_id": session_id}, {"_id": 0})
        if not session:
            return None
        
        messages = list(
            self.messages.find(
                {"session_id": session_id},
                {"_id": 0}
            ).sort("timestamp", 1)
        )
        
        return {
            "session": session,
            "messages": messages
        }
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete conversation and all its messages
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted successfully
        """
        # Delete session
        result = self.sessions.delete_one({"session_id": session_id})
        
        # Delete all messages
        self.messages.delete_many({"session_id": session_id})
        
        return result.deleted_count > 0
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save message to database and auto-generate title if first message
        
        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Saved message document
        """
        now = datetime.now()
        
        # Create session if it doesn't exist
        session = self.sessions.find_one({"session_id": session_id})
        if not session:
            self.create_session(session_id)
            session = self.sessions.find_one({"session_id": session_id})
        
        # Save message
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "metadata": metadata or {}
        }
        self.messages.insert_one(message)
        
        # Update session
        message_count = self.messages.count_documents({"session_id": session_id})
        
        update_data = {
            "updated_at": now,
            "message_count": message_count
        }
        
        # Auto-generate title if this is the first user message
        if role == "user" and not session.get("title"):
            title = self.generate_conversation_title(content)
            update_data["title"] = title
        
        self.sessions.update_one(
            {"session_id": session_id},
            {"$set": update_data}
        )
        
        return message
    
    def generate_conversation_title(self, first_message: str) -> str:
        """
        Generate concise conversation title from first message using LLM
        
        Args:
            first_message: First user message
            
        Returns:
            Generated title (max 50 chars)
        """
        try:
            prompt = f"""Tạo tiêu đề ngắn gọn (tối đa 50 ký tự) cho cuộc hội thoại dựa trên câu hỏi đầu tiên:

Câu hỏi: "{first_message}"

Tiêu đề nên:
- Ngắn gọn, súc tích
- Phản ánh nội dung chính
- Không có dấu ngoặc kép
- Viết hoa chữ cái đầu

Chỉ trả về tiêu đề, không giải thích."""
            
            response = self.llm.invoke(prompt)
            title = response.content.strip()
            
            # Ensure max 50 characters
            if len(title) > 50:
                title = title[:47] + "..."
            
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            # Fallback: Use first 50 chars of message
            return first_message[:47] + "..." if len(first_message) > 50 else first_message
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for Agentic RAG
        
        Args:
            session_id: Session ID
            
        Returns:
            List of messages in format [{"role": "user/assistant", "content": "..."}]
        """
        messages = list(
            self.messages.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1}
            ).sort("timestamp", 1)
        )
        
        return messages
    
    def close(self):
        """Close database connection"""
        self.client.close()


# Global database instance
db = Database()
