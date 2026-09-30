import re
from typing import Iterable, List, Optional


def detect_ollama_models(model_names: Optional[Iterable[str]] = None) -> dict:
    """Return the best-known Qwen and Llama model names if present."""
    qwen_name = None
    llama_name = None

    for item in model_names or []:
        name = str(item).strip()
        if not name:
            continue
        lower_name = name.lower()

        if "qwen" in lower_name and qwen_name is None:
            qwen_name = name
        elif "llama" in lower_name and llama_name is None:
            llama_name = name

    return {"qwen": qwen_name, "llama": llama_name}


def resolve_model_name(available_models: Optional[Iterable[str]]) -> Optional[str]:
    """Prefer a Qwen model, then a Llama model, otherwise return None."""
    detected = detect_ollama_models(available_models)
    if detected["qwen"]:
        return detected["qwen"]
    if detected["llama"]:
        return detected["llama"]
    return None
