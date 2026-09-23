from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ChunkingStrategyEnum(str, Enum):
    FIXED_SIZE = "fixed_size"
    ARTICLE_SEMANTIC = "article_semantic"


class DocumentChunkRead(BaseModel):
    id: str
    vector_id: str
    chunk_index: int
    content: str
    char_count: int
    section_title: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunking_strategy: str
    total_chunks: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailRead(DocumentRead):
    chunks: List[DocumentChunkRead] = []


class IngestResponse(BaseModel):
    success: bool
    message: str
    document: DocumentRead
    chunks_created: int
    vector_ids: List[str]
