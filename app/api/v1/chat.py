from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.services.rag_pipeline import rag_pipeline
from app.services.redis_memory import chat_memory

router = APIRouter(prefix="/chat", tags=["Conversational RAG & Booking"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Conversational RAG query & Interview Booking endpoint"
)
def chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        response = rag_pipeline.process_chat(request=request, db=db)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing error: {str(e)}"
        )


@router.get(
    "/history/{session_id}",
    response_model=List[ChatMessage],
    summary="Retrieve multi-turn chat history from Redis"
)
def get_session_history(session_id: str):
    raw_history = chat_memory.get_history(session_id)
    return [ChatMessage(role=m["role"], content=m["content"]) for m in raw_history]


@router.delete(
    "/history/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation history from Redis"
)
def clear_session_history(session_id: str):
    chat_memory.clear_history(session_id)
    return {"success": True, "message": f"History for session '{session_id}' cleared."}
