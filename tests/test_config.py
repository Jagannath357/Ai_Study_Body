from __future__ import annotations

from pathlib import Path

import pytest

from study_buddy.config import ConfigError, load_settings, require_api_key


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBED_MODEL", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)

    settings = load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")

    assert settings.gemini_chat_model == "gemini-2.0-flash"
    assert settings.gemini_embed_model == "text-embedding-004"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.top_k == 6


def test_invalid_chunk_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "not-an-int")
    with pytest.raises(ConfigError):
        load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")


def test_require_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = load_settings(data_dir=tmp_path / "data", db_dir=tmp_path / "db")
    with pytest.raises(ConfigError):
        require_api_key(settings)

