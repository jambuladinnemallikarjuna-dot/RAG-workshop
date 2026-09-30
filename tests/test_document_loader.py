from core.document_loader import extract_text_from_file


def test_extract_text_from_file_handles_text_content():
    text = "AI workshop\n\nThis is a test document."
    result = extract_text_from_file(text, filename="sample.txt")
    assert "AI workshop" in result["text"]
    assert result["filename"] == "sample.txt"
    assert result["pages"] == 1
