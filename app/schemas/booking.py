from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class BookingCreate(BaseModel):
    candidate_name: str = Field(..., description="Full name of the candidate")
    candidate_email: EmailStr = Field(..., description="Valid email address of the candidate")
    interview_date: str = Field(..., description="Target interview date, e.g. '2026-10-15'")
    interview_time: str = Field(..., description="Target interview time, e.g. '14:00' or '2:00 PM'")
    notes: Optional[str] = Field(None, description="Optional extra notes or discussion topic")
    session_id: Optional[str] = Field(None, description="Chat session ID associated with this booking")


class BookingRead(BaseModel):
    id: str
    session_id: Optional[str]
    candidate_name: str
    candidate_email: str
    interview_date: str
    interview_time: str
    notes: Optional[str]
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
