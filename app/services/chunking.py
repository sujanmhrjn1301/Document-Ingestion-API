import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from app.schemas.document import ChunkingStrategyEnum


class Chunk:
    def __init__(self, content: str, chunk_index: int, section_title: Optional[str] = None):
        self.content = content.strip()
        self.chunk_index = chunk_index
        self.section_title = section_title
        self.char_count = len(self.content)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into a list of Chunk objects."""
        pass


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size sliding window chunker with configurable chunk size and overlap.
    Splits text while respecting sentence/paragraph boundaries when possible.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []
        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        paragraphs = re.split(r"\n\s*\n", cleaned_text)
        chunks: List[Chunk] = []
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(Chunk(content=current_chunk, chunk_index=chunk_idx))
                    chunk_idx += 1

                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        overlap_start = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_start + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    words = para.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                            temp_chunk = f"{temp_chunk} {word}".strip()
                        else:
                            if temp_chunk:
                                chunks.append(Chunk(content=temp_chunk, chunk_index=chunk_idx))
                                chunk_idx += 1
                                overlap = temp_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(temp_chunk) else ""
                                temp_chunk = f"{overlap} {word}".strip()
                            else:
                                temp_chunk = word
                    current_chunk = temp_chunk

        if current_chunk.strip():
            chunks.append(Chunk(content=current_chunk, chunk_index=chunk_idx))

        return chunks


class ArticleSemanticChunker(BaseChunker):
    """
    Semantic chunker specifically designed for constitutional/legal documents.
    Splits text hierarchically by Parts, Articles, Sections, and Schedules,
    preserving the full legal context, article numbers, and section headers.
    """
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        
        pattern = r"(?i)(?=(?:^|\n)(?:Part\s+\d+[\s\S]*?\n|Article\s+\d+[\s\.:]|Schedule\s*[-–\d]+|Preamble\b))"
        
        sections = re.split(pattern, cleaned_text, flags=re.MULTILINE)
        
        chunks: List[Chunk] = []
        chunk_idx = 0
        current_part = "General Section"

        for section in sections:
            section_text = section.strip()
            if not section_text or len(section_text) < 15:
                continue

            first_line = section_text.split("\n")[0].strip()
            
            if re.match(r"(?i)^Part\s+\d+", first_line):
                current_part = first_line
            
            section_title = f"{current_part} - {first_line[:100]}" if current_part != first_line else first_line[:100]

            if len(section_text) > self.max_chunk_size:
                sub_chunker = FixedSizeChunker(chunk_size=self.max_chunk_size, chunk_overlap=150)
                sub_chunks = sub_chunker.chunk(section_text)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        content=sc.content,
                        chunk_index=chunk_idx,
                        section_title=section_title
                    ))
                    chunk_idx += 1
            else:
                chunks.append(Chunk(
                    content=section_text,
                    chunk_index=chunk_idx,
                    section_title=section_title
                ))
                chunk_idx += 1

        if not chunks:
            fallback = FixedSizeChunker(chunk_size=800, chunk_overlap=150)
            return fallback.chunk(text)

        return chunks


def get_chunker(strategy: Any) -> BaseChunker:
    """Factory function to get selected chunking strategy."""
    strat_str = str(strategy.value if hasattr(strategy, "value") else strategy).lower()
    
    if strat_str in ["fixed_size", "recursive", "sliding_window"]:
        return FixedSizeChunker(chunk_size=800, chunk_overlap=150)
    elif strat_str in ["article_semantic", "semantic", "hierarchical"]:
        return ArticleSemanticChunker(max_chunk_size=1200)
    else:
        return ArticleSemanticChunker(max_chunk_size=1200)

