from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from app.schemas.booking import BookingRead


class ChatIntentEnum(str, Enum):
    RAG_QUERY = "rag_query"
    INTERVIEW_BOOKING = "interview_booking"
    INTERVIEW_CANCELLATION = "interview_cancellation"
    GENERAL_CONVERSATION = "general_conversation"


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="The message content")
    timestamp: Optional[str] = None


class RetrievedSource(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    section_title: Optional[str] = None
    score: float
    snippet: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique session ID for maintaining Redis memory")
    message: str = Field(..., min_length=1, description="User's question or message")
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    intent: ChatIntentEnum
    reply: str
    sources: List[RetrievedSource] = []
    booking: Optional[BookingRead] = None
    booking_status: Optional[str] = None
