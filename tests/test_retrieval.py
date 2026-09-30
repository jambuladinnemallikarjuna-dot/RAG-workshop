import numpy as np

from core.retriever import cosine_similarity, rank_matches, top_k_matches


def test_cosine_similarity_basic_values():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)

    c = np.array([1.0, 0.0])
    assert np.isclose(cosine_similarity(a, c), 1.0)


def test_rank_matches_returns_sorted_scores():
    query = np.array([1.0, 0.0])
    vectors = [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 1.0])]
    ranked = rank_matches(query, vectors)
    assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"]


def test_top_k_matches_limits_results():
    query = np.array([1.0, 0.0])
    vectors = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([0.0, 1.0])]
    results = top_k_matches(query, vectors, k=2)
    assert len(results) == 2
