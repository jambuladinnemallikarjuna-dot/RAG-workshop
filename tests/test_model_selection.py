from core.model_selector import detect_ollama_models, resolve_model_name


def test_detect_ollama_models_handles_empty_list():
    assert detect_ollama_models([]) == {"qwen": None, "llama": None}


def test_resolve_model_name_prefers_qwen():
    available = ["qwen2.5:7b", "llama3.2:3b"]
    assert resolve_model_name(available) == "qwen2.5:7b"


def test_resolve_model_name_falls_back_to_llama():
    available = ["llama3.2:3b"]
    assert resolve_model_name(available) == "llama3.2:3b"


def test_resolve_model_name_when_no_support_model_is_available():
    available = ["mistral:7b"]
    assert resolve_model_name(available) is None
