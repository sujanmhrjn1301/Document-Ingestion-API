import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from app.config import settings


BOOKING_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "book_interview",
        "description": "Schedule and confirm an interview booking when candidate details (name, email, date, time) are provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_name": {
                    "type": "string",
                    "description": "Full name of the candidate."
                },
                "candidate_email": {
                    "type": "string",
                    "description": "Valid email address of the candidate."
                },
                "interview_date": {
                    "type": "string",
                    "description": "Date for the interview (e.g., '2026-10-15' or 'October 15, 2026')."
                },
                "interview_time": {
                    "type": "string",
                    "description": "Time for the interview (e.g., '14:00' or '2:30 PM')."
                },
                "notes": {
                    "type": "string",
                    "description": "Any additional notes, interview topic, or role description."
                }
            },
            "required": ["candidate_name", "candidate_email", "interview_date", "interview_time"]
        }
    }
}


CANCEL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "cancel_interview",
        "description": "Cancel one or more interview bookings. Use cancel_mode to specify what to cancel.",
        "parameters": {
            "type": "object",
            "properties": {
                "cancel_mode": {
                    "type": "string",
                    "enum": ["by_date", "all", "all_except_date"],
                    "description": "'by_date': cancel bookings on a specific date. 'all': cancel all bookings. 'all_except_date': cancel all bookings EXCEPT the one on keep_date."
                },
                "target_date": {
                    "type": "string",
                    "description": "The date of bookings to cancel (for 'by_date' mode). Format: YYYY-MM-DD."
                },
                "keep_date": {
                    "type": "string",
                    "description": "The date of the booking to KEEP (for 'all_except_date' mode). Format: YYYY-MM-DD."
                }
            },
            "required": ["cancel_mode"]
        }
    }
}


SYSTEM_PROMPT = """You are an intelligent AI Assistant specialized in the Constitution of Nepal and Interview Scheduling.

Your capabilities:
1. **Constitution RAG Specialist**: Answer user questions accurately using the provided Constitution of Nepal document context. Cite article numbers and parts whenever available. If the provided context does not contain enough information to answer, state so politely rather than hallucinating.
2. **Interview Booking System**: You ARE directly connected to the organization's backend booking system via the `book_interview` tool. NEVER say "I'm unable to proceed with the booking directly" or "Please contact the organization". You HAVE the tool to complete the booking!
   
   Required fields to book an interview:
   - `candidate_name` (Full Name)
   - `candidate_email` (Email Address)
   - `interview_date` (Date, e.g., '2026-09-30' or 'September 30, 2026')
   - `interview_time` (Time, e.g., '16:00' or '4:00 PM')
   - `notes` (optional notes, e.g., role)

   RULES FOR INTERVIEW BOOKING:
   - If any required field is missing, ask the candidate politely for the missing detail.
   - When all 4 required fields are provided (either in the current message OR accumulated across previous messages in the conversation history, or when the user confirms with 'yes', 'proceed', etc.), you MUST call the `book_interview` tool with the details.
   - NEVER refuse to book when details are available. Always call `book_interview`.

3. **Interview Cancellation System**: You can cancel bookings using the `cancel_interview` tool.
   - Use cancel_mode='by_date' with target_date to cancel a specific booking by date.
   - Use cancel_mode='all' to cancel all bookings for the session.
   - Use cancel_mode='all_except_date' with keep_date to cancel all bookings EXCEPT the one on a given date (e.g., 'cancel all except tomorrow's interview').
   - When EXISTING CONFIRMED BOOKINGS context is provided, use it to determine which bookings to cancel.
   - NEVER refuse to cancel. Always call `cancel_interview` when the user asks to cancel.
"""


class LLMService:
    """Service to handle LLM reasoning, conversational RAG, and function-calling for interview booking."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY
        )
        self.model = settings.LLM_MODEL

    def _build_system_prompt(self) -> str:
        """Build system prompt with current date/time so the LLM can resolve relative dates."""
        # Nepal timezone UTC+5:45
        nepal_tz = timezone(timedelta(hours=5, minutes=45))
        now = datetime.now(nepal_tz)
        today_str = now.strftime("%Y-%m-%d (%A)")
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d (%A)")
        current_time = now.strftime("%I:%M %p")

        date_context = (
            f"\n\nIMPORTANT DATE/TIME CONTEXT:\n"
            f"- Today's date: {today_str}\n"
            f"- Tomorrow's date: {tomorrow_str}\n"
            f"- Current time: {current_time}\n"
            f"When the user says 'tomorrow', use {tomorrow_str}. "
            f"Always resolve relative dates (today, tomorrow, next Monday, etc.) to exact YYYY-MM-DD format when calling tools.\n"
        )
        return SYSTEM_PROMPT + date_context

    def generate_chat_response(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        retrieved_context: Optional[str] = None,
        booking_context: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Processes multi-turn dialogue with retrieved context and tools.
        Returns (assistant_reply, booking_tool_arguments_or_None, cancel_tool_arguments_or_None).
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        context_parts = []
        if retrieved_context and retrieved_context.strip():
            context_parts.append(f"### RETRIEVED DOCUMENT CONTEXT:\n{retrieved_context}")
        if booking_context and booking_context.strip():
            context_parts.append(booking_context)

        if context_parts:
            user_content_with_context = (
                "\n\n".join(context_parts) + f"\n\n### USER QUERY:\n{user_message}"
            )
        else:
            user_content_with_context = user_message

        messages.append({"role": "user", "content": user_content_with_context})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[BOOKING_TOOL_DEFINITION, CANCEL_TOOL_DEFINITION],
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        booking_data = None
        cancel_data = None

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "book_interview":
                    try:
                        booking_data = json.loads(tool_call.function.arguments)
                    except Exception:
                        booking_data = None
                elif tool_call.function.name == "cancel_interview":
                    try:
                        cancel_data = json.loads(tool_call.function.arguments)
                    except Exception:
                        cancel_data = None

        reply_content = response_message.content or ""
        
        if not booking_data and not cancel_data:
            full_convo = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]) + f"\nuser: {user_message}"
            if any(k in full_convo.lower() for k in ["interview", "book", "schedule"]) and any(k in user_message.lower() for k in ["yes", "proceed", "confirm", "ok", "book", "sure"]):
                try:
                    extract_resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "Extract booking details from the conversation and call book_interview tool if name, email, date, time are present. If not enough details, return nothing."},
                            {"role": "user", "content": full_convo}
                        ],
                        tools=[BOOKING_TOOL_DEFINITION],
                        tool_choice={"type": "function", "function": {"name": "book_interview"}}
                    )
                    if extract_resp.choices[0].message.tool_calls:
                        tool_call = extract_resp.choices[0].message.tool_calls[0]
                        extracted = json.loads(tool_call.function.arguments)
                        if all(k in extracted and extracted[k] for k in ["candidate_name", "candidate_email", "interview_date", "interview_time"]):
                            booking_data = extracted
                except Exception:
                    pass

        if booking_data:
            if not reply_content.strip() or "unable" in reply_content.lower() or "contact" in reply_content.lower():
                reply_content = (
                    f"Thank you, {booking_data.get('candidate_name')}! Your interview has been successfully scheduled "
                    f"for {booking_data.get('interview_date')} at {booking_data.get('interview_time')}. "
                    f"A confirmation has been saved and sent to {booking_data.get('candidate_email')}."
                )

        return reply_content, booking_data, cancel_data

llm_service = LLMService()

