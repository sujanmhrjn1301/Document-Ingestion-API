import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatIntentEnum,
    RetrievedSource,
)
from app.schemas.booking import BookingRead
from app.models.booking import InterviewBooking
from app.services.redis_memory import chat_memory
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service
from app.config import settings

logger = logging.getLogger(__name__)


class CustomRAGPipeline:
    """
    Custom RAG Pipeline orchestrator built from scratch (no RetrievalQAChain).
    Features:
    - Redis multi-turn conversation memory
    - Semantic similarity retrieval from Pinecone
    - Integrated interview booking tool calling
    - SQL storage for bookings
    """

    def process_chat(self, request: ChatRequest, db: Session) -> ChatResponse:
        session_id = request.session_id
        user_message = request.message.strip()

        history = chat_memory.get_history(session_id, limit=settings.MAX_HISTORY_TURNS)

        query_embedding = embedding_service.get_embedding(user_message)
        matches = vector_store.query_similar(query_vector=query_embedding, top_k=4)

        retrieved_sources: List[RetrievedSource] = []
        context_snippets: List[str] = []

        for match in matches:
            metadata = match.get("metadata", {})
            score = match.get("score", 0.0)
            chunk_content = metadata.get("content", "")
            doc_name = metadata.get("document_name", "Document")
            section = metadata.get("section_title", "General")
            doc_id = metadata.get("document_id", "")

            if score >= 0.35 and chunk_content:
                context_snippets.append(
                    f"[Source: {doc_name} | Section: {section}]\n{chunk_content}"
                )
                retrieved_sources.append(
                    RetrievedSource(
                        chunk_id=match.get("id", ""),
                        document_id=doc_id,
                        document_name=doc_name,
                        section_title=section,
                        score=score,
                        snippet=chunk_content[:200] + ("..." if len(chunk_content) > 200 else "")
                    )
                )

        retrieved_context_str = "\n\n---\n\n".join(context_snippets) if context_snippets else None

        booking_context = None
        if any(kw in user_message.lower() for kw in ["interview", "book", "schedule", "appointment", "when", "cancel", "remove", "delete"]):
            existing_bookings = db.query(InterviewBooking).filter(
                InterviewBooking.session_id == session_id
            ).order_by(InterviewBooking.created_at.desc()).all()
            if existing_bookings:
                booking_lines = []
                for b in existing_bookings:
                    booking_lines.append(
                        f"- ID: {b.id}, Name: {b.candidate_name}, Email: {b.candidate_email}, "
                        f"Date: {b.interview_date}, Time: {b.interview_time}, "
                        f"Status: {b.status}, Notes: {b.notes or 'N/A'}"
                    )
                booking_context = (
                    "### EXISTING CONFIRMED BOOKINGS FOR THIS SESSION:\n"
                    + "\n".join(booking_lines)
                    + "\n\nWhen cancelling, always use the exact Date values shown above (YYYY-MM-DD format) in tool calls."
                )
        reply_text, booking_data, cancel_data = llm_service.generate_chat_response(
            user_message=user_message,
            chat_history=history,
            retrieved_context=retrieved_context_str,
            booking_context=booking_context
        )

        booking_record_read: Optional[BookingRead] = None
        intent = ChatIntentEnum.RAG_QUERY
        booking_status = None

        if cancel_data:
            intent = ChatIntentEnum.INTERVIEW_CANCELLATION
            cancel_mode = cancel_data.get("cancel_mode", "")
            cancelled_count = 0
            try:
                session_bookings = db.query(InterviewBooking).filter(
                    InterviewBooking.session_id == session_id
                ).all()

                def normalize_date(date_str: str) -> str:
                    """Try multiple date formats to normalize to YYYY-MM-DD."""
                    from datetime import datetime as dt
                    formats = [
                        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                        "%B %d, %Y", "%b %d, %Y",
                        "%d %B %Y", "%d %b %Y",
                        "%Y-%m-%dT%H:%M:%S",
                    ]
                    for fmt in formats:
                        try:
                            return dt.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                    return date_str.strip()

                if cancel_mode == "by_date":
                    target_date = normalize_date(cancel_data.get("target_date", ""))
                    bookings_to_cancel = [
                        b for b in session_bookings
                        if normalize_date(b.interview_date) == target_date
                    ]

                elif cancel_mode == "all":
                    bookings_to_cancel = session_bookings

                elif cancel_mode == "all_except_date":
                    keep_date = normalize_date(cancel_data.get("keep_date", ""))
                    bookings_to_cancel = [
                        b for b in session_bookings
                        if normalize_date(b.interview_date) != keep_date
                    ]
                else:
                    bookings_to_cancel = []

                cancelled_details = []
                for b in bookings_to_cancel:
                    cancelled_details.append(f"{b.candidate_name} on {b.interview_date}")
                    db.delete(b)
                    cancelled_count += 1
                db.commit()

                booking_status = f"CANCELLED ({cancelled_count} booking(s) removed)"
                if cancelled_count > 0 and not reply_text.strip():
                    reply_text = f"Done! Cancelled {cancelled_count} booking(s): {', '.join(cancelled_details)}."
                elif cancelled_count == 0:
                    booking_status = "NO_MATCH"
                    if not reply_text.strip():
                        reply_text = "No matching bookings were found to cancel."

            except Exception as e:
                logger.error(f"Failed to cancel bookings: {e}")
                db.rollback()
                booking_status = "CANCELLATION_FAILED"

        elif booking_data:
            intent = ChatIntentEnum.INTERVIEW_BOOKING
            try:
                booking_obj = InterviewBooking(
                    session_id=session_id,
                    candidate_name=booking_data.get("candidate_name", "Unknown"),
                    candidate_email=booking_data.get("candidate_email", ""),
                    interview_date=booking_data.get("interview_date", ""),
                    interview_time=booking_data.get("interview_time", ""),
                    notes=booking_data.get("notes", None),
                    status="CONFIRMED"
                )
                db.add(booking_obj)
                db.commit()
                db.refresh(booking_obj)
                booking_record_read = BookingRead.model_validate(booking_obj)
                booking_status = "CONFIRMED"
            except Exception as e:
                logger.error(f"Failed to persist booking: {e}")
                db.rollback()
                booking_status = "FAILED"
        elif any(kw in user_message.lower() for kw in ["cancel", "remove", "delete"]):
            intent = ChatIntentEnum.INTERVIEW_CANCELLATION
            booking_status = "IN_PROGRESS"
        elif "interview" in user_message.lower() or "book" in user_message.lower() or "schedule" in user_message.lower():
            intent = ChatIntentEnum.INTERVIEW_BOOKING
            booking_status = "IN_PROGRESS"
        elif not retrieved_sources:
            intent = ChatIntentEnum.GENERAL_CONVERSATION

        chat_memory.add_message(session_id=session_id, role="user", content=user_message)
        chat_memory.add_message(session_id=session_id, role="assistant", content=reply_text)

        return ChatResponse(
            session_id=session_id,
            intent=intent,
            reply=reply_text,
            sources=retrieved_sources if intent == ChatIntentEnum.RAG_QUERY else [],
            booking=booking_record_read,
            booking_status=booking_status
        )

rag_pipeline = CustomRAGPipeline()
