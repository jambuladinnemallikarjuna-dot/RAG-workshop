import re
from typing import Any, Dict, List


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30, source_name: str = "document") -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with lightweight metadata."""
    clean = (text or "").strip()
    if not clean:
        return []

    words = re.findall(r"\b\S+\b", clean)
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[Dict[str, Any]] = []
    for index in range(0, len(words), step):
        segment = words[index : index + chunk_size]
        if not segment:
            continue
        chunk_text = " ".join(segment)
        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks)}",
                "text": chunk_text,
                "source": source_name,
                "page_number": None,
                "start_word": index,
                "end_word": index + len(segment) - 1,
                "word_count": len(segment),
            }
        )

    if not chunks:
        return []

    # Guarantee the final chunk is not empty and keep metadata consistent.
    if chunks[-1]["text"].strip() == "":
        chunks.pop()

    return chunks
