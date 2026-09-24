from typing import List
from openai import OpenAI
from app.config import settings

class EmbeddingService:
    """Service to generate embeddings using OpenRouter / OpenAI API."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY
        )
        self.model = settings.EMBEDDING_MODEL

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text query."""
        text = text.replace("\n", " ").strip()
        if not text:
            text = "empty"
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches."""
        cleaned_texts = [t.replace("\n", " ").strip() or "empty" for t in texts]
        embeddings: List[List[float]] = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings
embedding_service = EmbeddingService()
