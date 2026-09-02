import os

import pytest

from rag_chatbot.core.config import LLMSettings, QdrantSettings, Settings

PREFIXES = ("QDRANT_", "EMBEDDING_", "LLM_", "RETRIEVAL_", "CHUNKING_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Settings read the real environment; a developer's shell must not leak into the assertions."""
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            monkeypatch.delenv(key, raising=False)


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "7000")
    monkeypatch.setenv("QDRANT_COLLECTION", "docs-test")
    s = QdrantSettings()
    assert (s.port, s.collection) == (7000, "docs-test")


def test_provider_keys_are_read_from_their_conventional_env_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert LLMSettings().openai_api_key == "sk-test"


def test_settings_load_from_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("qdrant:\n  port: 6444\nchunking:\n  chunk_size: 128\nlog_level: DEBUG\n")
    s = Settings.from_yaml(cfg)
    assert s.qdrant.port == 6444
    assert s.chunking.chunk_size == 128
    assert s.log_level == "DEBUG"
    assert s.retrieval.top_k == 5, "untouched sections keep their defaults"
