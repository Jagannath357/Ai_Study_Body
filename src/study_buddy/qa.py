from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from study_buddy.config import Settings, require_api_key
from study_buddy.models import AnswerResult
from study_buddy.retriever import retrieve_chunks


PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_prompt.txt"


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_context(chunks: list) -> str:
    if not chunks:
        return "No context available."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            (
                f"[{i}] source={chunk.source_file} page={chunk.page} "
                f"score={chunk.score:.4f}\n{chunk.text}"
            )
        )
    return "\n\n".join(parts)


def _extract_text(response: object) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        joined: list[str] = []
        for item in content:
            if isinstance(item, str):
                joined.append(item)
            elif isinstance(item, dict) and "text" in item:
                joined.append(str(item["text"]))
        return "\n".join(joined).strip()
    return str(response)


def answer_question(
    question: str,
    settings: Settings,
    k: int | None = None,
    vectorstore: Any | None = None,
) -> AnswerResult:
    started = time.perf_counter()
    chunks = retrieve_chunks(question=question, settings=settings, k=k, vectorstore=vectorstore)
    if not chunks:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AnswerResult(
            answer="I could not find relevant context in the indexed PDFs.",
            citations=[],
            retrieved_chunks=[],
            latency_ms=latency_ms,
        )

    require_api_key(settings)
    template = _load_prompt_template()
    context = _build_context(chunks)
    prompt = template.format(context=context, question=question)
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )
    response = llm.invoke(prompt)
    answer_text = _extract_text(response).strip()

    seen: set[str] = set()
    citations: list[str] = []
    for chunk in chunks:
        citation = f"{chunk.source_file}#page={chunk.page}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    latency_ms = int((time.perf_counter() - started) * 1000)
    return AnswerResult(
        answer=answer_text,
        citations=citations,
        retrieved_chunks=chunks,
        latency_ms=latency_ms,
    )
