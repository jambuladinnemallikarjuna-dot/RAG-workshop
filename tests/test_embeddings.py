import numpy as np

from core.embeddings import LocalEmbeddingModel


def test_embedding_model_returns_vector_rows_and_finite_values():
    model = LocalEmbeddingModel()
    vectors = model.encode(["AI workshop", "RAG pipeline", "local retrieval"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 256)
    assert np.all(np.isfinite(vectors))
