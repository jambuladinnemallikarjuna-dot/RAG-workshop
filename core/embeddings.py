import hashlib
import re
from typing import Sequence

import numpy as np


class LocalEmbeddingModel:
    """Lightweight local embedding model for offline semantic retrieval.

    This intentionally avoids the blocked scikit-learn/transformers import chain on
    Windows while still producing deterministic vectors for chunk matching.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        self.embedding_dim = max(8, int(embedding_dim))

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", str(text).lower())
        return [token for token in tokens if token]

    def _text_to_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        if not tokens:
            return vector

        frequencies: dict[int, float] = {}
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.embedding_dim
            frequencies[index] = frequencies.get(index, 0.0) + 1.0

        for index, weight in frequencies.items():
            vector[index] = float(weight)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts = [str(item).strip() for item in texts if str(item).strip()]
        if not clean_texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = [self._text_to_vector(text) for text in clean_texts]
        return np.vstack(vectors).astype(np.float32)
