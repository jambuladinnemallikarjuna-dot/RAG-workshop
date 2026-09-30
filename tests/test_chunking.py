from core.chunker import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 20 + 5 for chunk in chunks)


def test_chunk_text_handles_empty_input():
    chunks = chunk_text("   \n\t", chunk_size=20, overlap=5)
    assert chunks == []
