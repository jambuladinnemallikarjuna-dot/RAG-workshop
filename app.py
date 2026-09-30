from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st

try:
    from core.document_loader import extract_text_from_file
    from core.model_selector import resolve_model_name
    from core.ollama_client import generate_answer, is_ollama_running, list_local_models
    from core.rag_pipeline import RAGPipeline
    from ui.components import metric_box, status_badge
    from ui.styles import CSS
except Exception as exc:  # pragma: no cover - runtime guard for local environment issues
    st.set_page_config(page_title="Local RAG Workshop", page_icon="📚", layout="wide")
    st.error(
        "The app could not initialize its local AI dependencies. "
        "This is usually caused by a blocked local DLL or a missing runtime dependency. "
        "Please reinstall the project environment and retry."
    )
    st.code(str(exc))
    raise SystemExit(0)

st.set_page_config(page_title="Local RAG Workshop", page_icon="📚", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_meta" not in st.session_state:
    st.session_state.index_meta = {}


# --- Header ---
ollama_connected = is_ollama_running()
available_models = list_local_models() if ollama_connected else []
selected_model = resolve_model_name(available_models)

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<div class="hero-title">Local RAG Workshop</div>', unsafe_allow_html=True)
    st.caption("Upload a document, retrieve the relevant context, and ask grounded questions using Ollama locally.")
with header_col2:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    status_badge("Connected" if ollama_connected else "Not Connected", ollama_connected)
    status_badge(f"{selected_model or 'No Local Model'}" if selected_model else "No Local Model", bool(selected_model))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

if not ollama_connected:
    st.warning(
        "Ollama is not running locally. Start the Ollama service before asking questions. "
        "Example: `ollama serve` in a terminal."
    )

if ollama_connected and not selected_model:
    st.warning(
        "No Qwen or Llama model is installed locally. Install one, for example: `ollama pull qwen2.5` or `ollama pull llama3.2`."
    )

# --- Sidebar settings ---
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk size", 80, 500, 250, 10)
overlap = st.sidebar.slider("Chunk overlap", 10, 120, 40, 5)
top_k = st.sidebar.slider("Top K", 1, 8, 4, 1)
threshold = st.sidebar.slider("Similarity threshold", 0.0, 0.9, 0.15, 0.01)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

if st.session_state.pipeline is None:
    pipeline = RAGPipeline(chunk_size=chunk_size, overlap=overlap, top_k=top_k, threshold=threshold)
    st.session_state.pipeline = pipeline
else:
    st.session_state.pipeline.chunk_size = chunk_size
    st.session_state.pipeline.overlap = overlap
    st.session_state.pipeline.top_k = top_k
    st.session_state.pipeline.threshold = threshold
    pipeline = st.session_state.pipeline

# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    try:
        extracted = extract_text_from_file(uploaded_file.getvalue(), filename=file_name)
        document_text = extracted["text"]
        st.session_state.index_meta = {
            "filename": file_name,
            "size": file_size,
            "pages": extracted.get("pages", 1),
            "characters": len(document_text),
        }
        pipeline = st.session_state.pipeline
        pipeline.index_document(file_name, document_text)
        st.success(f"Document indexed: {file_name}")
        st.session_state.index_meta["chunks"] = len(pipeline.chunks)
        st.session_state.index_meta["embedding_status"] = "Ready" if pipeline.embeddings is not None else "Not Ready"
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.index_meta = {}

# --- Knowledge base status ---
status_cols = st.columns(4)
with status_cols[0]:
    metric_box("Document count", str(1 if st.session_state.index_meta else 0))
with status_cols[1]:
    metric_box("Chunk count", str(st.session_state.index_meta.get("chunks", 0)))
with status_cols[2]:
    metric_box("Embedding status", "Ready" if st.session_state.index_meta else "Not Ready")
with status_cols[3]:
    metric_box("Retrieval", "Ready" if st.session_state.index_meta else "Not Ready")

document_details = st.session_state.index_meta
if document_details:
    info_cols = st.columns(4)
    with info_cols[0]:
        metric_box("Filename", document_details.get("filename", "-"))
    with info_cols[1]:
        metric_box("File size", f"{document_details.get('size', 0)} bytes")
    with info_cols[2]:
        metric_box("Pages", str(document_details.get("pages", 0)))
    with info_cols[3]:
        metric_box("Characters", f"{document_details.get('characters', 0):,}")

st.markdown("---")

# --- Chat UI ---
question = st.text_input("Ask a question about the uploaded document", placeholder="Example: What are the eligibility requirements?")
chat_cols = st.columns([1, 1])
with chat_cols[0]:
    ask_button = st.button("Ask the document")
with chat_cols[1]:
    if st.button("Clear chat"):
        st.session_state.chat_history = []

if ask_button:
    if not st.session_state.index_meta:
        st.warning("Upload a document before asking a question.")
    elif not question.strip():
        st.warning("Please enter a question first.")
    else:
        matches = pipeline.retrieve(question, top_k=top_k, threshold=threshold)
        if not matches:
            st.warning("No relevant information was found in the uploaded document for that question. Try asking a question related to the uploaded text.")
        else:
            context = pipeline.build_context(matches)
            answer = generate_answer(selected_model or "", question, context, temperature=temperature)
            st.session_state.chat_history.append({"question": question, "answer": answer, "matches": matches})
            st.success("Grounded answer generated from the retrieved context.")

for turn in reversed(st.session_state.chat_history):
    with st.container():
        st.markdown("### User question")
        st.write(turn["question"])
        st.markdown("### Assistant answer")
        st.write(turn["answer"])
        with st.expander("View retrieved context and scores"):
            for match in turn["matches"]:
                st.markdown(f"**Chunk:** {match['chunk_id']} | **Score:** {match['score']:.3f} | **Source:** {match['source']}")
                st.caption(f"Page {match['page_number']}" if match.get("page_number") else "Page unavailable")
                st.write(match["text"])
        st.markdown("---")

# --- Model and environment info ---
with st.sidebar:
    st.header("Ollama status")
    if ollama_connected:
        st.success("Ollama is running")
    else:
        st.error("Ollama is not running")

    if selected_model:
        st.info(f"Using model: {selected_model}")
    else:
        st.warning("No Qwen or Llama model detected. Install a model with Ollama.")

    st.markdown("### Setup instructions")
    st.code("ollama serve\nollama pull qwen2.5\nollama pull llama3.2")

st.markdown('</div>', unsafe_allow_html=True)
