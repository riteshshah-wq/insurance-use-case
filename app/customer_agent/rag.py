"""Simple RAG (Retrieval Augmented Generation) for insurance policy docs."""

import os
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory document store (for demo purposes)
_document_store: List[dict] = []


def load_policy_documents(docs_dir: str = "data/sample_policies") -> int:
    """Load policy documents from disk into the in-memory store.

    In production, this would use GoldenRay's RAG endpoint or a vector DB.

    Args:
        docs_dir: Directory containing policy text files.

    Returns:
        Number of documents loaded.
    """
    global _document_store
    _document_store = []

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning(f"Policy docs directory not found: {docs_dir}")
        return 0

    for file_path in docs_path.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        _document_store.append({
            "id": file_path.stem,
            "filename": file_path.name,
            "content": content,
            "chunks": _chunk_text(content, chunk_size=500),
        })
        logger.info(f"Loaded policy document: {file_path.name}")

    return len(_document_store)


def retrieve(query: str, top_k: int = 3) -> List[dict]:
    """Retrieve relevant document chunks for a query.

    Uses simple keyword matching for demo. In production, use:
    - GoldenRay RAG endpoint for vector similarity search
    - Direct vector DB (Milvus, ChromaDB) for embeddings-based retrieval

    Args:
        query: The search query.
        top_k: Number of chunks to return.

    Returns:
        List of relevant chunks with metadata.
    """
    if not _document_store:
        load_policy_documents()

    query_terms = set(query.lower().split())
    scored_chunks = []

    for doc in _document_store:
        for i, chunk in enumerate(doc["chunks"]):
            chunk_terms = set(chunk.lower().split())
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                scored_chunks.append({
                    "document": doc["filename"],
                    "chunk_index": i,
                    "content": chunk,
                    "score": overlap / len(query_terms) if query_terms else 0,
                })

    # Sort by relevance and return top_k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50-word overlap
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
