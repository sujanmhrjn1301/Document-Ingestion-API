from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from app.config import settings


class PineconeVectorStore:
    """Manages vector storage and similarity retrieval via Pinecone."""

    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self) -> None:
        """Create index if it does not already exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 50) -> List[str]:
        """
        Upsert a batch of vectors with metadata.
        vectors format: [{"id": str, "values": List[float], "metadata": Dict[str, Any]}]
        """
        upserted_ids: List[str] = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)
            upserted_ids.extend([item["id"] for item in batch])
        return upserted_ids

    def query_similar(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for top-k similar vectors.
        Returns list of matches with metadata and similarity score.
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": float(match.score),
                "metadata": match.metadata or {}
            })
        return matches

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors associated with a document_id."""
        try:
            self.index.delete(filter={"document_id": {"$eq": document_id}})
        except Exception:
            pass


vector_store = PineconeVectorStore()
