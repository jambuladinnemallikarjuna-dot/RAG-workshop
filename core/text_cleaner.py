import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """Normalize extracted text while preserving meaning and punctuation."""
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned
