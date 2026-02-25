from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IngestStats:
    files_seen: int = 0
    files_indexed: int = 0
    chunks_added: int = 0
    chunks_updated: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    doc_id: str
    source_file: str
    page: int
    chunk_index: int
    score: float
    text: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: int

