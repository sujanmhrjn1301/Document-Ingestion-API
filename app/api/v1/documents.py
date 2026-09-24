import io
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, Query, HTTPException, UploadFile, status
from sqlalchemy.orm import Session


from app.database import get_db
from app.models.document import Document, DocumentChunk
from app.schemas.document import (
    ChunkingStrategyEnum,
    DocumentDetailRead,
    DocumentRead,
    IngestResponse,
)
from app.services.chunking import get_chunker
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store

router = APIRouter(tags=["Document Ingestion"])


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF or TXT bytes with PyMuPDF and pypdf support."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_pages = []
            for page in doc:
                text = page.get_text() or ""
                if text.strip():
                    text_pages.append(text.strip())
            extracted_text = "\n\n".join(text_pages)
            if extracted_text.strip():
                return extracted_text
        except Exception:
            pass

        # Fallback to pypdf
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_pages.append(page_text.strip())
            extracted_text = "\n\n".join(text_pages)
            if extracted_text.strip():
                return extracted_text
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse PDF file: {str(e)}"
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract readable text from the PDF file. Please ensure the PDF is text-based."
        )
    elif lower_name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to decode text file: {str(e)}"
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only .pdf and .txt files are accepted."
        )



@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest PDF/TXT document with selectable chunking strategy"
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or TXT document to ingest"),
    chunking_strategy: ChunkingStrategyEnum = Query(
        default=ChunkingStrategyEnum.ARTICLE_SEMANTIC,
        description="Choose chunking strategy from dropdown"
    ),
    db: Session = Depends(get_db)
):
    filename = file.filename or "uploaded_doc"
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    raw_text = extract_text_from_file(file_bytes, filename)
    if not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract any readable text from the uploaded file."
        )

    chunker = get_chunker(chunking_strategy)
    chunks = chunker.chunk(raw_text)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks were generated from the document content."
        )

    document_id = str(uuid.uuid4())
    doc_obj = Document(
        id=document_id,
        filename=filename,
        file_type="PDF" if filename.lower().endswith(".pdf") else "TXT",
        file_size_bytes=file_size,
        chunking_strategy=chunking_strategy.value,
        total_chunks=len(chunks)
    )
    db.add(doc_obj)
    db.flush()

    chunk_texts = [c.content for c in chunks]
    embeddings = embedding_service.get_embeddings_batch(chunk_texts)

    vectors_to_upsert = []
    vector_ids = []
    chunk_db_objects = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{document_id}_chunk_{i}"
        vector_ids.append(vector_id)

        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "document_name": filename,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title or "General",
                "content": chunk.content,
                "char_count": chunk.char_count,
                "chunking_strategy": chunking_strategy.value
            }
        })

        chunk_db_obj = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            vector_id=vector_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            char_count=chunk.char_count,
            section_title=chunk.section_title
        )
        chunk_db_objects.append(chunk_db_obj)

    try:
        vector_store.upsert_vectors(vectors_to_upsert)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upsert embeddings into Pinecone: {str(e)}"
        )

    db.bulk_save_objects(chunk_db_objects)
    db.commit()
    db.refresh(doc_obj)

    return IngestResponse(
        success=True,
        message=f"Successfully ingested '{filename}' with {len(chunks)} chunks using strategy '{chunking_strategy.value}'.",
        document=DocumentRead.model_validate(doc_obj),
        chunks_created=len(chunks),
        vector_ids=vector_ids
    )


@router.get(
    "/documents",
    response_model=List[DocumentRead],
    summary="List all ingested documents and metadata"
)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentRead.model_validate(d) for d in docs]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailRead,
    summary="Get document details including chunk metadata"
)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentDetailRead.model_validate(doc)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete document metadata and vectors"
)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    vector_store.delete_by_document_id(document_id)
    
    db.delete(doc)
    db.commit()
    return {"success": True, "message": f"Document '{doc.filename}' deleted successfully."}

