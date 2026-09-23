from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError

class Settings(BaseSettings):
    APP_NAME: str = "FastAPI Document Ingestion and RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # pinecone vectorDB
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_ENVIRONMENT: Optional[str] = None

    # openrouter
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "openai/gpt-4o-mini"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # redis
    REDIS_URL: str
    REDIS_CHAT_TTL_SECONDS: int = 86400
    MAX_HISTORY_TURNS: int = 8

    # sql
    DATABASE_URL: str = "sqlite:///./app.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

try:
    settings = Settings()
except ValidationError as error:
    print("Configuration validation failed:")
    print(error)
    raise