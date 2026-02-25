from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

from study_buddy.config import Settings
from study_buddy.models import IngestStats
from study_buddy.vectorstore import (
    build_embeddings,
    get_vectorstore,
    manifest_path,
    reset_collection,
)

FREE_TIER_BATCH_SIZE = 2
FREE_TIER_SLEEP_SECONDS = 10


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    files = data.get("files")
    if not isinstance(files, dict):
        return {"version": 1, "files": {}}
    data.setdefault("version", 1)
    return data


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _read_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((idx, text))
    return pages


def _discover_pdfs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.pdf"))


def _add_documents_throttled(store: Any, docs: list[Document], ids: list[str]) -> None:
    if not docs:
        return
    for start in range(0, len(docs), FREE_TIER_BATCH_SIZE):
        end = start + FREE_TIER_BATCH_SIZE
        store.add_documents(documents=docs[start:end], ids=ids[start:end])
        if end < len(docs):
            time.sleep(FREE_TIER_SLEEP_SECONDS)


def run_ingestion(settings: Settings, reset: bool = False) -> IngestStats:
    started = time.perf_counter()
    stats = IngestStats()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.db_dir.mkdir(parents=True, exist_ok=True)

    m_path = manifest_path(settings.db_dir)
    manifest: dict[str, Any]
    if reset:
        reset_collection(settings)
        manifest = {"version": 1, "files": {}}
    else:
        manifest = _load_manifest(m_path)

    embeddings = build_embeddings(settings)
    store = get_vectorstore(settings, embeddings=embeddings)
    known_files: dict[str, Any] = manifest.setdefault("files", {})

    pdf_paths = _discover_pdfs(settings.data_dir)
    stats.files_seen = len(pdf_paths)
    current_rel_paths = {p.relative_to(settings.data_dir).as_posix() for p in pdf_paths}

    for stale_rel_path in list(known_files.keys()):
        if stale_rel_path in current_rel_paths:
            continue
        stale = known_files.get(stale_rel_path, {})
        stale_ids = stale.get("chunk_ids", [])
        if stale_ids:
            store.delete(ids=stale_ids)
        del known_files[stale_rel_path]

    for pdf_path in pdf_paths:
        rel_path = pdf_path.relative_to(settings.data_dir).as_posix()
        try:
            file_hash = _sha256(pdf_path)
            previous = known_files.get(rel_path)
            if previous and previous.get("file_hash") == file_hash:
                continue

            if previous and previous.get("chunk_ids"):
                store.delete(ids=previous["chunk_ids"])

            ingested_at = datetime.now(UTC).isoformat()
            docs: list[Document] = []
            ids: list[str] = []
            for page_number, page_text in _read_pdf_pages(pdf_path):
                chunks = split_text(
                    text=page_text,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                for chunk_index, chunk in enumerate(chunks):
                    doc_id = f"{file_hash}:{page_number}:{chunk_index}"
                    ids.append(doc_id)
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "doc_id": doc_id,
                                "source_file": rel_path,
                                "page": page_number,
                                "chunk_index": chunk_index,
                                "file_hash": file_hash,
                                "ingested_at": ingested_at,
                            },
                        )
                    )

            if docs:
                _add_documents_throttled(store=store, docs=docs, ids=ids)

            known_files[rel_path] = {
                "file_hash": file_hash,
                "chunk_ids": ids,
                "updated_at": ingested_at,
                "num_chunks": len(ids),
            }
            stats.files_indexed += 1
            if previous is None:
                stats.chunks_added += len(ids)
            else:
                stats.chunks_updated += len(ids)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"{rel_path}: {exc}")

    _save_manifest(m_path, manifest)
    stats.duration_s = round(time.perf_counter() - started, 3)
    return stats
