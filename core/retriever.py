import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_matches(query_vector: np.ndarray, candidate_vectors: list[np.ndarray]) -> list[dict]:
    """Return similarity matches sorted from highest to lowest."""
    matches = []
    for idx, vector in enumerate(candidate_vectors):
        score = cosine_similarity(query_vector, vector)
        matches.append({"index": idx, "score": score})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches


def top_k_matches(query_vector: np.ndarray, candidate_vectors: list[np.ndarray], k: int = 3) -> list[dict]:
    """Return the top k matches only."""
    matches = rank_matches(query_vector, candidate_vectors)
    return matches[:max(1, int(k))]
