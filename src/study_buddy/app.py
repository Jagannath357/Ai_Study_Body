from __future__ import annotations

import copy
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_buddy.config import ConfigError, Settings, load_settings, require_api_key
from study_buddy.ingest import FREE_TIER_BATCH_SIZE, FREE_TIER_SLEEP_SECONDS, UploadedPDF, run_ingestion
from study_buddy.qa import answer_question
from study_buddy.vectorstore import get_vectorstore, manifest_path

VISUALIZATION_KEYWORDS = ("chart", "plot", "graph", "visualize", "visualization", "trend", "line", "bar")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _make_empty_session(session_number: int) -> dict[str, Any]:
    now = _now_iso()
    return {
        "id": f"session-{session_number}-{int(datetime.now(UTC).timestamp() * 1000)}",
        "title": f"Session {session_number}",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def _find_session_index(session_id: str) -> int | None:
    for idx, session in enumerate(st.session_state.all_sessions):
        if session.get("id") == session_id:
            return idx
    return None


def _derive_session_title(messages: list[dict[str, Any]], fallback: str) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        normalized = " ".join(content.split())
        return normalized[:52] + ("..." if len(normalized) > 52 else "")
    return fallback


def _sync_active_session() -> None:
    session_id = st.session_state.get("active_session_id")
    if not session_id:
        return
    idx = _find_session_index(session_id)
    if idx is None:
        return

    current = st.session_state.all_sessions[idx]
    messages = copy.deepcopy(st.session_state.get("chat_history", []))
    fallback_title = str(current.get("title", "Session"))
    current["messages"] = messages
    current["title"] = _derive_session_title(messages, fallback_title)
    current["updated_at"] = _now_iso()
    st.session_state.all_sessions[idx] = current


def _activate_session(session_id: str) -> None:
    _sync_active_session()
    idx = _find_session_index(session_id)
    if idx is None:
        return
    st.session_state.active_session_id = session_id
    st.session_state.chat_history = copy.deepcopy(st.session_state.all_sessions[idx].get("messages", []))


def _start_new_session() -> None:
    _sync_active_session()
    session_number = st.session_state.next_session_number
    new_session = _make_empty_session(session_number)
    st.session_state.next_session_number = session_number + 1
    st.session_state.all_sessions.append(new_session)
    st.session_state.active_session_id = new_session["id"]
    st.session_state.chat_history = []


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
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _ensure_session_defaults(settings: Settings) -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = settings
    if "all_sessions" not in st.session_state:
        st.session_state.all_sessions = []
    if "next_session_number" not in st.session_state:
        st.session_state.next_session_number = 1

    if not st.session_state.all_sessions:
        first_session = _make_empty_session(st.session_state.next_session_number)
        st.session_state.next_session_number += 1
        st.session_state.all_sessions.append(first_session)
        st.session_state.active_session_id = first_session["id"]

    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = st.session_state.all_sessions[0]["id"]

    active_idx = _find_session_index(st.session_state.active_session_id)
    if active_idx is None:
        st.session_state.active_session_id = st.session_state.all_sessions[0]["id"]
        active_idx = 0

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = copy.deepcopy(st.session_state.all_sessions[active_idx].get("messages", []))

    if "vectorstore" not in st.session_state:
        try:
            require_api_key(settings)
            st.session_state.vectorstore = get_vectorstore(settings)
        except Exception:
            st.session_state.vectorstore = None

    if "last_ingest_stats" not in st.session_state:
        st.session_state.last_ingest_stats = None


def _render_sidebar(settings: Settings) -> tuple[int, list[Any], bool, bool, bool, str | None]:
    st.sidebar.header("Study Sessions")
    new_chat_clicked = st.sidebar.button("New Chat", use_container_width=True)
    st.sidebar.caption("Load any previous conversation from this browser session.")

    selected_session_id: str | None = None
    active_session_id = st.session_state.active_session_id
    previous_sessions = [s for s in reversed(st.session_state.all_sessions) if s.get("id") != active_session_id]

    if previous_sessions:
        for session in previous_sessions:
            title = str(session.get("title", "Session"))
            message_count = len(session.get("messages", []))
            label = f"{title} ({message_count})"
            if st.sidebar.button(label, key=f"session_{session.get('id')}", use_container_width=True):
                selected_session_id = str(session.get("id"))
    else:
        st.sidebar.caption("No previous sessions yet.")

    st.sidebar.divider()
    st.sidebar.header("Document Management")
    st.sidebar.caption("Upload PDFs, then process the currently listed files into ChromaDB.")
    st.sidebar.info(
        "Free Tier Optimized mode: ingestion sends "
        f"{FREE_TIER_BATCH_SIZE} chunk(s) per batch and sleeps "
        f"{FREE_TIER_SLEEP_SECONDS} seconds between batches."
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only files currently shown here are processed when you click Process.",
    )
    reset_index = st.sidebar.toggle("Reset existing index", value=False)
    top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=12, value=settings.top_k)
    process_clicked = st.sidebar.button("Process", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.markdown("**Paths**")
    st.sidebar.code(f"db_dir: {settings.db_dir}", language="text")

    return top_k, uploaded_files or [], reset_index, process_clicked, new_chat_clicked, selected_session_id


def _handle_sidebar_actions(
    *,
    settings: Settings,
    uploaded_files: list[Any],
    reset_index: bool,
    process_clicked: bool,
    new_chat_clicked: bool,
    selected_session_id: str | None,
) -> None:
    if selected_session_id:
        _activate_session(selected_session_id)
        st.rerun()

    if new_chat_clicked:
        _start_new_session()
        st.rerun()

    if not process_clicked:
        return

    if not uploaded_files:
        st.warning("No uploaded PDFs to process. Add files in the sidebar first.")
        return

    try:
        require_api_key(settings)
        with st.status("Processing documents...", expanded=True) as status:
            status.write("Reading uploaded files from memory...")
            uploaded_pdfs = [
                UploadedPDF(name=uploaded.name, data=uploaded.getvalue()) for uploaded in uploaded_files
            ]
            status.write(f"Prepared {len(uploaded_pdfs)} PDF file(s) from the current uploader list.")

            status.write(
                "Free Tier throttle active: "
                f"{FREE_TIER_SLEEP_SECONDS}s pause after every {FREE_TIER_BATCH_SIZE} chunks."
            )
            status.write("Running ingestion pipeline...")
            stats = run_ingestion(settings=settings, reset=reset_index, uploaded_pdfs=uploaded_pdfs)
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


def _render_chart(chart: dict[str, Any]) -> None:
    chart_title = str(chart.get("title", "")).strip()
    if chart_title:
        st.caption(chart_title)

    chart_type = str(chart.get("chart_type", "bar")).lower()
    chart_data = chart.get("data")
    if chart_type == "line":
        st.line_chart(chart_data)
    else:
        st.bar_chart(chart_data)


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            citations = message.get("citations", [])
            if citations:
                st.caption("Citations: " + ", ".join(citations))
            chart = message.get("chart")
            if isinstance(chart, dict):
                _render_chart(chart)


def _is_visualization_request(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in VISUALIZATION_KEYWORDS)


def _extract_response_text(response: object) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict) and "text" in item:
                lines.append(str(item["text"]))
        return "\n".join(lines).strip()
    return str(response).strip()


def _coerce_chart_data(data: Any) -> dict[str, float] | list[float] | None:
    if isinstance(data, dict):
        normalized: dict[str, float] = {}
        for key, value in data.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized or None

    if isinstance(data, list):
        if not data:
            return None
        if all(isinstance(item, (int, float)) for item in data):
            return [float(item) for item in data]

        mapped: dict[str, float] = {}
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            x = item.get("label", item.get("x", f"p{idx}"))
            y = item.get("value", item.get("y"))
            try:
                mapped[str(x)] = float(y)
            except (TypeError, ValueError):
                continue
        return mapped or None

    return None


def _parse_chart_payload(raw_text: str) -> dict[str, Any] | None:
    payload: dict[str, Any] | None = None
    try:
        candidate = json.loads(raw_text)
        if isinstance(candidate, dict):
            payload = candidate
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, dict):
                    payload = candidate
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        return None

    chart_type = str(payload.get("chart_type", "bar")).lower()
    if chart_type not in {"bar", "line"}:
        chart_type = "bar"

    chart_data = _coerce_chart_data(payload.get("data"))
    if chart_data is None:
        return None

    return {
        "chart_type": chart_type,
        "title": str(payload.get("title", "")).strip(),
        "data": chart_data,
    }


def _build_chart_context(chunks: list[Any]) -> str:
    snippets: list[str] = []
    for chunk in chunks[:8]:
        chunk_text = " ".join(str(getattr(chunk, "text", "")).split())
        if len(chunk_text) > 450:
            chunk_text = chunk_text[:450] + "..."
        snippets.append(
            f"source={getattr(chunk, 'source_file', 'unknown')} "
            f"page={getattr(chunk, 'page', 0)} text={chunk_text}"
        )
    return "\n".join(snippets)


def _generate_chart_payload(question: str, settings: Settings, retrieved_chunks: list[Any]) -> dict[str, Any] | None:
    if not retrieved_chunks:
        return None

    require_api_key(settings)
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )
    prompt = (
        "You are generating chart-ready structured data for Streamlit. "
        "Return ONLY valid JSON with keys: chart_type, title, data. "
        "chart_type must be 'bar' or 'line'. "
        "data must be either a dictionary of label->number, a list of numbers, "
        "or a list of objects with label/value (or x/y). "
        "Use only facts supported by context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{_build_chart_context(retrieved_chunks)}"
    )
    response = llm.invoke(prompt)
    return _parse_chart_payload(_extract_response_text(response))


def _answer_user_question(question: str, settings: Settings, top_k: int) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question, "created_at": _now_iso()})
    _sync_active_session()

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

            chart_payload: dict[str, Any] | None = None
            if _is_visualization_request(question):
                try:
                    with st.spinner("Preparing visualization data..."):
                        chart_payload = _generate_chart_payload(
                            question=question,
                            settings=settings,
                            retrieved_chunks=result.retrieved_chunks,
                        )
                    if chart_payload:
                        _render_chart(chart_payload)
                except Exception:
                    chart_payload = None

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
                    "chart": chart_payload,
                    "created_at": _now_iso(),
                }
            )
            _sync_active_session()
        except Exception as exc:  # noqa: BLE001
            msg = f"I hit an error: {exc}"
            st.error(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg, "created_at": _now_iso()})
            _sync_active_session()


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
    _sync_active_session()

    top_k, uploaded_files, reset_index, process_clicked, new_chat_clicked, selected_session_id = _render_sidebar(
        settings
    )

    _handle_sidebar_actions(
        settings=settings,
        uploaded_files=uploaded_files,
        reset_index=reset_index,
        process_clicked=process_clicked,
        new_chat_clicked=new_chat_clicked,
        selected_session_id=selected_session_id,
    )

    _render_header_metrics(settings, top_k=top_k)
    st.divider()
    _render_chat_history()

    prompt = st.chat_input("Ask a question about your indexed PDFs...")
    if prompt:
        _answer_user_question(prompt, settings=settings, top_k=top_k)

    _sync_active_session()


if __name__ == "__main__":
    main()
