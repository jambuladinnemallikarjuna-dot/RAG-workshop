import io
import os
import re
from typing import Any, Dict, Optional, Union

from docx import Document as DocxDocument
import fitz

from .text_cleaner import clean_text


def extract_text_from_file(file_data: Union[str, bytes, None], filename: str = "document.txt") -> Dict[str, Any]:
    """Extract text from a text, PDF, or DOCX payload."""
    if file_data is None:
        raise ValueError("No document content was provided.")

    extension = os.path.splitext(filename or "")[1].lower()

    if isinstance(file_data, str):
        text = clean_text(file_data)
        return {"text": text, "filename": filename, "pages": 1, "source_type": extension or ".txt"}

    if extension == ".pdf":
        return _extract_pdf_text(file_data, filename)

    if extension == ".docx":
        return _extract_docx_text(file_data, filename)

    if extension in {".txt", "", ".md"}:
        text = clean_text(file_data.decode("utf-8", errors="ignore"))
        return {"text": text, "filename": filename, "pages": 1, "source_type": ".txt"}

    raise ValueError(f"Unsupported file type: {filename}")


def _extract_pdf_text(file_data: bytes, filename: str) -> Dict[str, Any]:
    try:
        with fitz.open(stream=file_data, filetype="pdf") as doc:
            pages = []
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                text = page.get_text("text")
                if text:
                    pages.append((page_number + 1, text))

            text = "\n\n".join(page_text for _, page_text in pages)
            cleaned = clean_text(text)
            return {
                "text": cleaned,
                "filename": filename,
                "pages": len(doc),
                "source_type": ".pdf",
            }
    except Exception as exc:  # pragma: no cover - exercised in higher-level app
        raise ValueError(f"Could not read PDF content from {filename}: {exc}") from exc


def _extract_docx_text(file_data: bytes, filename: str) -> Dict[str, Any]:
    try:
        document = DocxDocument(io.BytesIO(file_data))
        paragraphs = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)

        text = "\n\n".join(paragraphs)
        cleaned = clean_text(text)
        return {
            "text": cleaned,
            "filename": filename,
            "pages": max(1, len(paragraphs) // 20 + 1),
            "source_type": ".docx",
        }
    except Exception as exc:  # pragma: no cover - exercised in higher-level app
        raise ValueError(f"Could not read DOCX content from {filename}: {exc}") from exc


def load_document_from_path(path: str) -> Dict[str, Any]:
    """Load a document from a file path and return extracted text metadata."""
    with open(path, "rb") as handle:
        payload = handle.read()

    filename = os.path.basename(path)
    return extract_text_from_file(payload, filename=filename)
