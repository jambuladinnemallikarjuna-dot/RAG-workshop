from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .chunker import chunk_text
from .embeddings import LocalEmbeddingModel
from .retriever import top_k_matches
from .text_cleaner import clean_text


class RAGPipeline:
    """Simple in-memory RAG pipeline for workshop use."""

    def __init__(self, chunk_size: int = 250, overlap: int = 40, top_k: int = 4, threshold: float = 0.15) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.threshold = threshold
        self.embedding_model = LocalEmbeddingModel()
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_name: Optional[str] = None

    def index_document(self, filename: str, text: str) -> Dict[str, Any]:
        cleaned = clean_text(text)
        self.document_name = filename
        self.chunks = chunk_text(cleaned, chunk_size=self.chunk_size, overlap=self.overlap, source_name=filename)
        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return {"document": filename, "chunks": 0, "indexed": False}

        chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(chunk_texts)
        return {"document": filename, "chunks": len(self.chunks), "indexed": True}

    def retrieve(self, question: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.chunks or self.embeddings is None or self.embeddings.size == 0:
            return []

        query_vector = self.embedding_model.encode([question])[0]
        matches = top_k_matches(query_vector, self.embeddings.tolist(), k=top_k or self.top_k)

        relevant_matches: List[Dict[str, Any]] = []
        min_score = threshold if threshold is not None else self.threshold
        for match in matches:
            if match["score"] < min_score:
                continue
            chunk = self.chunks[match["index"]]
            relevant_matches.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(match["score"]),
                    "source": chunk["source"],
                    "page_number": chunk.get("page_number"),
                    "text": chunk["text"],
                }
            )
        return relevant_matches

    def build_context(self, matches: List[Dict[str, Any]]) -> str:
        if not matches:
            return ""
        context_blocks = []
        for match in matches:
            snippet = match["text"].strip()
            source = match.get("source", "document")
            page = match.get("page_number")
            page_label = f" (page {page})" if page is not None else ""
            context_blocks.append(f"[Source: {source}{page_label}]\n{snippet}")
        return "\n\n---\n\n".join(context_blocks)
