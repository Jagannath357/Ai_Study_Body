from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_buddy.config import ConfigError, Settings, load_settings, require_api_key
from study_buddy.ingest import FREE_TIER_BATCH_SIZE, FREE_TIER_SLEEP_SECONDS, run_ingestion
from study_buddy.qa import answer_question
from study_buddy.vectorstore import get_vectorstore, manifest_path


def _save_uploaded_pdfs(files: list[Any], data_dir: Path) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uploaded in files:
        target = data_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved += 1
    return saved


def _load_manifest_summary(db_dir: Path) -> tuple[int, int]:
    path = manifest_path(db_dir)
    if not path.exists():
        return 0, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, 0
    files = data.get("files", {})
    if not isinstance(files, dict):
        return 0, 0
    file_count = len(files)
    chunk_count = 0
    for item in files.values():
        if isinstance(item, dict):
            chunk_count += int(item.get("num_chunks", 0))
    return file_count, chunk_count


def _bootstrap_settings() -> Settings:
    settings = load_settings(data_dir=Path("data"), db_dir=Path("chroma_db"))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _ensure_session_defaults(settings: Settings) -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = settings
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        try:
            require_api_key(settings)
            st.session_state.vectorstore = get_vectorstore(settings)
        except Exception:
            st.session_state.vectorstore = None
    if "last_ingest_stats" not in st.session_state:
        st.session_state.last_ingest_stats = None


def _render_sidebar(settings: Settings) -> tuple[int, list[Any], bool, bool, bool]:
    st.sidebar.header("Document Management")
    st.sidebar.caption("Drag-and-drop PDFs, then process to update ChromaDB.")
    st.sidebar.info(
        "Free Tier Optimized mode: ingestion sends "
        f"{FREE_TIER_BATCH_SIZE} chunk(s) per batch and sleeps "
        f"{FREE_TIER_SLEEP_SECONDS} seconds between batches."
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are saved under data/ and ingested into chroma_db/.",
    )
    reset_index = st.sidebar.toggle("Reset existing index", value=False)
    top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=12, value=settings.top_k)

    col_a, col_b = st.sidebar.columns(2)
    process_clicked = col_a.button("Process", type="primary", use_container_width=True)
    clear_chat_clicked = col_b.button("Clear Chat", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.markdown("**Paths**")
    st.sidebar.code(f"data_dir: {settings.data_dir}", language="text")
    st.sidebar.code(f"db_dir: {settings.db_dir}", language="text")

    return top_k, uploaded_files or [], reset_index, process_clicked, clear_chat_clicked


def _handle_sidebar_actions(
    *,
    settings: Settings,
    uploaded_files: list[Any],
    reset_index: bool,
    process_clicked: bool,
    clear_chat_clicked: bool,
) -> None:
    if clear_chat_clicked:
        st.session_state.chat_history = []
        st.success("Conversation cleared.")

    if not process_clicked:
        return

    existing_pdfs = list(settings.data_dir.rglob("*.pdf"))
    if not uploaded_files and not existing_pdfs:
        st.warning("No PDFs found. Upload at least one PDF in the sidebar.")
        return

    try:
        require_api_key(settings)
        with st.status("Processing documents...", expanded=True) as status:
            if uploaded_files:
                status.write("Saving uploaded PDF files...")
                saved_count = _save_uploaded_pdfs(uploaded_files, settings.data_dir)
                status.write(f"Saved {saved_count} PDF file(s) to data/.")
            else:
                status.write("No new uploads. Using PDFs already in data/.")

            status.write("Waiting 5 seconds before ingestion...")
            time.sleep(5)

            status.write(
                "Free Tier throttle active: "
                f"{FREE_TIER_SLEEP_SECONDS}s pause after every {FREE_TIER_BATCH_SIZE} chunks."
            )
            status.write("Running ingestion pipeline...")
            stats = run_ingestion(settings=settings, reset=reset_index)
            st.session_state.last_ingest_stats = stats
            st.session_state.vectorstore = get_vectorstore(settings)

            status.update(label="Ingestion complete", state="complete")
        st.success("Vector database updated successfully.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Processing failed: {exc}")


def _render_header_metrics(settings: Settings, top_k: int) -> None:
    file_count, chunk_count = _load_manifest_summary(settings.db_dir)
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed PDFs", file_count)
    col2.metric("Indexed Chunks", chunk_count)
    col3.metric("Top-K", top_k)

    stats = st.session_state.get("last_ingest_stats")
    if stats:
        st.info(
            "Last ingestion: "
            f"files_seen={stats.files_seen}, files_indexed={stats.files_indexed}, "
            f"chunks_added={stats.chunks_added}, chunks_updated={stats.chunks_updated}, "
            f"duration_s={stats.duration_s}"
        )


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            citations = message.get("citations", [])
            if citations:
                st.caption("Citations: " + ", ".join(citations))


def _answer_user_question(question: str, settings: Settings, top_k: int) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if st.session_state.vectorstore is None:
                require_api_key(settings)
                st.session_state.vectorstore = get_vectorstore(settings)

            with st.spinner("Generating answer from indexed context..."):
                result = answer_question(
                    question=question,
                    settings=settings,
                    k=top_k,
                    vectorstore=st.session_state.vectorstore,
                )

            st.markdown(result.answer)
            if result.citations:
                st.caption("Citations: " + ", ".join(result.citations))

            with st.expander("Retrieval diagnostics"):
                for chunk in result.retrieved_chunks:
                    st.write(
                        f"{chunk.source_file} | page={chunk.page} | "
                        f"score={chunk.score:.4f} | chunk={chunk.chunk_index}"
                    )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "citations": result.citations,
                }
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"I hit an error: {exc}"
            st.error(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})


def main() -> None:
    st.set_page_config(page_title="AI Study Buddy", layout="wide")
    st.title("AI Study Buddy")
    st.caption("RAG-powered study assistant with Gemini + ChromaDB")

    try:
        settings = _bootstrap_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    _ensure_session_defaults(settings)

    top_k, uploaded_files, reset_index, process_clicked, clear_chat_clicked = _render_sidebar(settings)

    _handle_sidebar_actions(
        settings=settings,
        uploaded_files=uploaded_files,
        reset_index=reset_index,
        process_clicked=process_clicked,
        clear_chat_clicked=clear_chat_clicked,
    )

    _render_header_metrics(settings, top_k=top_k)
    st.divider()
    _render_chat_history()

    prompt = st.chat_input("Ask a question about your indexed PDFs...")
    if prompt:
        _answer_user_question(prompt, settings=settings, top_k=top_k)


if __name__ == "__main__":
    main()
