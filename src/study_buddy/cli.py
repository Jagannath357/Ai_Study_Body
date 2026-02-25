from __future__ import annotations

from pathlib import Path

import typer

from study_buddy.config import ConfigError, load_settings, require_api_key
from study_buddy.ingest import run_ingestion
from study_buddy.qa import answer_question
from study_buddy.vectorstore import ensure_db_readable

app = typer.Typer(add_completion=False, help="AI Study Buddy CLI")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise typer.BadParameter("Use true/false for --reset.")


@app.command()
def ingest(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory containing PDF files."),
    db_dir: Path = typer.Option(Path("chroma_db"), "--db-dir", help="Directory for Chroma persistence."),
    reset: str = typer.Option("false", "--reset", help="Reset the vector collection before indexing (true/false)."),
) -> None:
    """Index PDFs from the data folder into ChromaDB."""
    try:
        reset_value = _parse_bool(reset)
        settings = load_settings(data_dir=data_dir, db_dir=db_dir)
        require_api_key(settings)
        stats = run_ingestion(settings=settings, reset=reset_value)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Ingest failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("Ingestion complete")
    typer.echo(f"files_seen={stats.files_seen}")
    typer.echo(f"files_indexed={stats.files_indexed}")
    typer.echo(f"chunks_added={stats.chunks_added}")
    typer.echo(f"chunks_updated={stats.chunks_updated}")
    typer.echo(f"duration_s={stats.duration_s}")
    if stats.errors:
        typer.echo("errors:")
        for err in stats.errors:
            typer.echo(f"- {err}")


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q", help="Question to ask over indexed PDFs."),
    k: int = typer.Option(6, "--k", help="Number of chunks to retrieve."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory containing PDF files."),
    db_dir: Path = typer.Option(Path("chroma_db"), "--db-dir", help="Directory for Chroma persistence."),
) -> None:
    """Answer a question using retrieved chunks from ChromaDB."""
    try:
        settings = load_settings(data_dir=data_dir, db_dir=db_dir)
        result = answer_question(question=question, settings=settings, k=k)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Ask failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(result.answer)
    typer.echo("")
    typer.echo("Citations:")
    if result.citations:
        for citation in result.citations:
            typer.echo(f"- {citation}")
    else:
        typer.echo("- None")
    typer.echo("")
    typer.echo("Retrieved chunks:")
    if result.retrieved_chunks:
        for chunk in result.retrieved_chunks:
            typer.echo(
                f"- {chunk.source_file} page={chunk.page} score={chunk.score:.4f} chunk={chunk.chunk_index}"
            )
    else:
        typer.echo("- None")
    typer.echo(f"latency_ms={result.latency_ms}")


@app.command()
def repl(
    k: int = typer.Option(6, "--k", help="Number of chunks to retrieve."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory containing PDF files."),
    db_dir: Path = typer.Option(Path("chroma_db"), "--db-dir", help="Directory for Chroma persistence."),
) -> None:
    """Interactive Q&A loop."""
    try:
        settings = load_settings(data_dir=data_dir, db_dir=db_dir)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}")
        raise typer.Exit(code=1)

    typer.echo("AI Study Buddy REPL. Type 'exit' or 'quit' to stop.")
    while True:
        question = typer.prompt("Question").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = answer_question(question=question, settings=settings, k=k)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Error: {exc}")
            continue
        typer.echo(result.answer)
        if result.citations:
            typer.echo("Citations: " + ", ".join(result.citations))
        typer.echo("")


@app.command()
def doctor(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory containing PDF files."),
    db_dir: Path = typer.Option(Path("chroma_db"), "--db-dir", help="Directory for Chroma persistence."),
) -> None:
    """Run environment checks for API key, paths, and DB readability."""
    try:
        settings = load_settings(data_dir=data_dir, db_dir=db_dir)
        data_dir_exists = settings.data_dir.exists()
        db_dir_exists = settings.db_dir.exists()
        issues: list[str] = []
        if not data_dir_exists:
            issues.append(f"data_dir does not exist: {settings.data_dir}")
        if not db_dir_exists:
            issues.append(f"db_dir does not exist: {settings.db_dir}")
        if issues:
            raise ConfigError("; ".join(issues))
        ensure_db_readable(settings)
        require_api_key(settings)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Doctor failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("Doctor checks passed")
    typer.echo(f"GOOGLE_API_KEY=set")
    typer.echo(f"data_dir={settings.data_dir} exists={data_dir_exists}")
    typer.echo(f"db_dir={settings.db_dir} exists={db_dir_exists}")
    typer.echo("chroma_db=readable")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
