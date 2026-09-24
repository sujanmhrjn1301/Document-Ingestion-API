from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.booking import InterviewBooking
from app.schemas.booking import BookingRead

router = APIRouter(prefix="/bookings", tags=["Interview Bookings"])


@router.get(
    "",
    response_model=List[BookingRead],
    summary="List all scheduled interview bookings"
)
def list_bookings(db: Session = Depends(get_db)):
    bookings = db.query(InterviewBooking).order_by(InterviewBooking.created_at.desc()).all()
    return [BookingRead.model_validate(b) for b in bookings]


@router.get(
    "/{booking_id}",
    response_model=BookingRead,
    summary="Get details of a specific booking"
)
def get_booking(booking_id: str, db: Session = Depends(get_db)):
    booking = db.query(InterviewBooking).filter(InterviewBooking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Booking not found")
    return BookingRead.model_validate(booking)


@router.delete(
    "/{booking_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel/delete a specific interview booking"
)
def cancel_booking(booking_id: str, db: Session = Depends(get_db)):
    booking = db.query(InterviewBooking).filter(InterviewBooking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Booking not found")
    db.delete(booking)
    db.commit()
    return {"success": True, "message": f"Booking for {booking.candidate_name} on {booking.interview_date} cancelled."}
