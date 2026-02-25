from __future__ import annotations

from study_buddy.ingest import split_text


def test_split_text_respects_chunk_size() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text(text=text, chunk_size=10, chunk_overlap=2)
    assert chunks
    assert all(len(c) <= 10 for c in chunks)
    assert chunks == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]


def test_split_text_empty_input() -> None:
    chunks = split_text(text="   ", chunk_size=10, chunk_overlap=2)
    assert chunks == []

