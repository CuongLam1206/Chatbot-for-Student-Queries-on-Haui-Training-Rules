"""
FastAPI backend for Agentic RAG Web Application
Provides REST API for chat and conversation management
"""

import os
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys

# Add parent directory to path to import agentic_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agentic_rag import AgenticRAG, load_agentic_rag
from backend.models import (
    ChatRequest,
    ChatResponse,
    ConversationListItem,
    NewConversationResponse,
    Message
)
from backend.database import db

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="REST API for Agentic RAG chatbot with conversation management",
    version="1.0.0"
)

# CORS middleware - allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agentic RAG system (lazy loading)
agentic_rag: AgenticRAG = None


def get_agentic_rag() -> AgenticRAG:
    """Get or initialize Agentic RAG system"""
    global agentic_rag
    if agentic_rag is None:
        print("🚀 Initializing Agentic RAG System...")
        agentic_rag = load_agentic_rag()
        print("✅ Agentic RAG System ready!")
    return agentic_rag


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("🚀 Starting Agentic RAG Backend API")
    print("=" * 60)
    get_agentic_rag()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Agentic RAG Backend API")
    db.close()


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Agentic RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "conversations": "/api/conversations",
            "new_conversation": "/api/conversations/new",
            "delete_conversation": "/api/conversations/{session_id}"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_agentic_rag()
        return {
            "status": "healthy",
            "agentic_rag": "initialized",
            "model": rag.llm.model_name,
            "database": "connected"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    rag = get_agentic_rag()
    return {
        "model": rag.llm.model_name,
        "temperature": rag.llm.temperature,
        "vectorstore": "ChromaDB",
        "database": "MongoDB"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send message and get response
    
    Args:
        request: ChatRequest with message and session_id
        
    Returns:
        ChatResponse with answer and metadata
    """
    try:
        rag = get_agentic_rag()
        
        # Get conversation history
        conversation_history = db.get_conversation_history(request.session_id)
        
        # Save user message
        db.save_message(
            session_id=request.session_id,
            role="user",
            content=request.message,
            metadata={}
        )
        
        # Get response from Agentic RAG
        result = rag.query(
            question=request.message,
            conversation_history=conversation_history
        )
        
        # Save assistant response
        db.save_message(
            session_id=request.session_id,
            role="assistant",
            content=result["answer"],
            metadata={
                "confidence": result.get("confidence"),
                "citations": result.get("citations"),
                "metadata": result.get("metadata")
            }
        )
        
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            confidence=result.get("confidence"),
            citations=result.get("citations"),
            metadata=result.get("metadata")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/api/conversations", response_model=List[ConversationListItem])
async def get_conversations():
    """
    Get all conversations list
    
    Returns:
        List of conversation summaries
    """
    try:
        conversations = db.get_all_conversations()
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """
    Get specific conversation with full history
    
    Args:
        session_id: Conversation ID
        
    Returns:
        Conversation with messages
    """
    try:
        conversation = db.get_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")


@app.post("/api/conversations/new", response_model=NewConversationResponse)
async def create_new_conversation():
    """
    Create new conversation
    
    Returns:
        New session ID
    """
    try:
        session_id = str(uuid.uuid4())
        db.create_session(session_id)
        return NewConversationResponse(
            session_id=session_id,
            message="New conversation created"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """
    Delete conversation permanently
    
    Args:
        session_id: Conversation ID to delete
        
    Returns:
        Success message
    """
    try:
        success = db.delete_conversation(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting Agentic RAG Backend Server")
    print("=" * 60)
    print("\n📍 Server will be available at: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
