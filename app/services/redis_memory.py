import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import redis
from app.config import settings

logger = logging.getLogger(__name__)


class RedisChatMemory:
    """
    Manages conversational memory in Redis for multi-turn conversations.
    Supports TTL expiration and retrieval of recent conversation turns.
    """

    def __init__(self):
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self._connected = True
        except Exception as e:
            logger.warning(f"Redis connection warning: {e}. In-memory fallback will be active.")
            self.redis_client = None
            self._connected = False
            self._fallback_store: Dict[str, List[Dict[str, Any]]] = {}

    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}:history"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message turn to Redis for the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        key = self._get_key(session_id)

        if self._connected and self.redis_client:
            try:
                self.redis_client.rpush(key, json.dumps(message))
                self.redis_client.expire(key, settings.REDIS_CHAT_TTL_SECONDS)
                return
            except Exception as e:
                logger.error(f"Redis add_message failed: {e}")

        if session_id not in self._fallback_store:
            self._fallback_store[session_id] = []
        self._fallback_store[session_id].append(message)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve messages for a session formatted for LLM context:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        key = self._get_key(session_id)
        raw_messages: List[Dict[str, Any]] = []

        if self._connected and self.redis_client:
            try:
                items = self.redis_client.lrange(key, 0, -1)
                for item in items:
                    raw_messages.append(json.loads(item))
            except Exception as e:
                logger.error(f"Redis get_history failed: {e}")
                raw_messages = self._fallback_store.get(session_id, [])
        else:
            raw_messages = self._fallback_store.get(session_id, [])

        if limit and len(raw_messages) > limit:
            raw_messages = raw_messages[-limit:]

        return [{"role": m["role"], "content": m["content"]} for m in raw_messages]

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        key = self._get_key(session_id)
        if self._connected and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")
        self._fallback_store.pop(session_id, None)


chat_memory = RedisChatMemory()
