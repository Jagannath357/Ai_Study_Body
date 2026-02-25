from __future__ import annotations

from pathlib import Path

import pytest

from study_buddy.config import load_settings
from study_buddy.retriever import retrieve_chunks


class FakeDoc:
    def __init__(self, text: str, metadata: dict) -> None:
        self.page_content = text
        self.metadata = metadata


class FakeVectorStore:
    def similarity_search_with_score(self, question: str, k: int = 6):  # noqa: ARG002
        return [
            (
                FakeDoc(
                    "chunk text",
                    {
                        "doc_id": "abc:1:0",
                        "source_file": "notes.pdf",
                        "page": 1,
                        "chunk_index": 0,
                    },
                ),
                0.2,
            )
        ]


def test_retrieve_chunks_maps_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("study_buddy.retriever.get_vectorstore", lambda settings: FakeVectorStore())
    settings = load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")
    chunks = retrieve_chunks(question="test question", settings=settings, k=3)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_id == "abc:1:0"
    assert chunk.source_file == "notes.pdf"
    assert chunk.page == 1
    assert chunk.chunk_index == 0
    assert chunk.score > 0
    assert "chunk text" in chunk.text

