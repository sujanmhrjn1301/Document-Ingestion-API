import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text
from app.database import Base


class InterviewBooking(Base):
    """Represents an interview booking scheduled by the user via the Conversational RAG agent."""
    __tablename__ = "interview_bookings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), nullable=True, index=True)
    candidate_name = Column(String(150), nullable=False)
    candidate_email = Column(String(255), nullable=False, index=True)
    interview_date = Column(String(50), nullable=False)
    interview_time = Column(String(50), nullable=False) 
    notes = Column(Text, nullable=True)
    status = Column(String(50), default="CONFIRMED", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
