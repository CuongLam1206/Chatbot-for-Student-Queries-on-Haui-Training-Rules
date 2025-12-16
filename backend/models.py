"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class Message(BaseModel):
    """Individual message model"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ChatRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session/Conversation ID")


class ChatResponse(BaseModel):
    """Chat response with metadata"""
    answer: str = Field(..., description="Bot response")
    session_id: str = Field(..., description="Session/Conversation ID")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    citations: Optional[List[str]] = Field(default=None, description="Source citations")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class Session(BaseModel):
    """Chat session model"""
    session_id: str = Field(..., description="Unique session ID")
    title: Optional[str] = Field(default=None, description="Conversation title")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages in conversation")


class ConversationListItem(BaseModel):
    """Conversation summary for sidebar"""
    session_id: str = Field(..., description="Unique session ID")
    title: str = Field(default="New Conversation", description="Conversation title")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages")


class NewConversationResponse(BaseModel):
    """Response when creating new conversation"""
    session_id: str = Field(..., description="New session ID")
    message: str = Field(default="New conversation created", description="Success message")
