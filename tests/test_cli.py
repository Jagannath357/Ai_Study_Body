from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from study_buddy.cli import app
from study_buddy.config import load_settings
from study_buddy.models import AnswerResult, IngestStats, RetrievedChunk

runner = CliRunner()


def test_doctor_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"
    data_dir.mkdir()
    db_dir.mkdir()

    result = runner.invoke(
        app,
        ["doctor", "--data-dir", str(data_dir), "--db-dir", str(db_dir)],
    )

    assert result.exit_code == 0
    assert "Doctor checks passed" in result.stdout


def test_ask_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    settings = load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")

    monkeypatch.setattr("study_buddy.cli.load_settings", lambda data_dir, db_dir: settings)
    monkeypatch.setattr(
        "study_buddy.cli.answer_question",
        lambda question, settings, k: AnswerResult(
            answer="test answer",
            citations=["notes.pdf#page=1"],
            retrieved_chunks=[
                RetrievedChunk(
                    doc_id="abc:1:0",
                    source_file="notes.pdf",
                    page=1,
                    chunk_index=0,
                    score=0.9,
                    text="chunk text",
                )
            ],
            latency_ms=42,
        ),
    )

    result = runner.invoke(app, ["ask", "--question", "What is this?", "--k", "6"])
    assert result.exit_code == 0
    assert "test answer" in result.stdout
    assert "notes.pdf#page=1" in result.stdout


def test_ingest_reset_string(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    settings = load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")
    state = {"reset": None}

    monkeypatch.setattr("study_buddy.cli.load_settings", lambda data_dir, db_dir: settings)
    monkeypatch.setattr("study_buddy.cli.require_api_key", lambda settings: None)

    def _fake_ingest(settings, reset):
        state["reset"] = reset
        return IngestStats(files_seen=0, files_indexed=0, chunks_added=0, chunks_updated=0, duration_s=0.1)

    monkeypatch.setattr("study_buddy.cli.run_ingestion", _fake_ingest)
    result = runner.invoke(app, ["ingest", "--reset", "true"])
    assert result.exit_code == 0
    assert state["reset"] is True
