from __future__ import annotations

from typing import Any

from study_buddy.config import Settings
from study_buddy.models import RetrievedChunk
from study_buddy.vectorstore import get_vectorstore


def _to_retrieved_chunk(doc: Any, score: float) -> RetrievedChunk:
    metadata = doc.metadata or {}
    return RetrievedChunk(
        doc_id=str(metadata.get("doc_id", "")),
        source_file=str(metadata.get("source_file", "unknown")),
        page=int(metadata.get("page", 0)),
        chunk_index=int(metadata.get("chunk_index", 0)),
        score=float(score),
        text=doc.page_content,
    )


def retrieve_chunks(
    question: str,
    settings: Settings,
    k: int | None = None,
    vectorstore: Any | None = None,
) -> list[RetrievedChunk]:
    top_k = k or settings.top_k
    store = vectorstore or get_vectorstore(settings)

    if hasattr(store, "similarity_search_with_relevance_scores"):
        results = store.similarity_search_with_relevance_scores(question, k=top_k)
        return [_to_retrieved_chunk(doc, score) for doc, score in results]

    results = store.similarity_search_with_score(question, k=top_k)
    normalized: list[RetrievedChunk] = []
    for doc, distance in results:
        score = 1.0 / (1.0 + float(distance))
        normalized.append(_to_retrieved_chunk(doc, score))
    return normalized
