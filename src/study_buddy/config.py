from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when required config is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_chat_model: str
    gemini_embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    data_dir: Path
    db_dir: Path
    collection_name: str


def _get_int_env(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got: {raw}") from exc


def load_settings(data_dir: Path | None = None, db_dir: Path | None = None) -> Settings:
    load_dotenv()
    settings = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        chunk_size=_get_int_env("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int_env("CHUNK_OVERLAP", 150),
        top_k=_get_int_env("TOP_K", 6),
        data_dir=(data_dir or Path("data")).resolve(),
        db_dir=(db_dir or Path("chroma_db")).resolve(),
        collection_name=os.getenv("CHROMA_COLLECTION", "study_buddy"),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if settings.chunk_size <= 0:
        raise ConfigError("CHUNK_SIZE must be > 0")
    if settings.chunk_overlap < 0:
        raise ConfigError("CHUNK_OVERLAP must be >= 0")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if settings.top_k <= 0:
        raise ConfigError("TOP_K must be > 0")


def require_api_key(settings: Settings) -> None:
    if not settings.google_api_key:
        raise ConfigError("GOOGLE_API_KEY is required but not set.")

