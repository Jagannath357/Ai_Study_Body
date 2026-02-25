from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from study_buddy.config import Settings, require_api_key


def build_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    require_api_key(settings)
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embed_model,
        google_api_key=settings.google_api_key,
    )


def get_vectorstore(settings: Settings, embeddings: GoogleGenerativeAIEmbeddings | None = None) -> Chroma:
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    if embeddings is None:
        embeddings = build_embeddings(settings)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.db_dir),
    )


def reset_collection(settings: Settings) -> None:
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.db_dir))
    try:
        client.delete_collection(name=settings.collection_name)
    except Exception:
        pass


def ensure_db_readable(settings: Settings) -> None:
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    _ = chromadb.PersistentClient(path=str(settings.db_dir))


def manifest_path(db_dir: Path) -> Path:
    return db_dir / "ingestion_manifest.json"

