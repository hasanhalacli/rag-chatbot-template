"""Configuration management."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """Qdrant vector store settings."""

    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False
    collection: str = "documents"

    class Config:
        env_prefix = "QDRANT_"


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32

    class Config:
        env_prefix = "EMBEDDING_"


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    provider: str = "openai"  # openai, azure, anthropic, xai
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1000

    # Provider-specific
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    azure_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = "2024-02-01"
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    xai_api_key: Optional[str] = Field(default=None, alias="XAI_API_KEY")

    class Config:
        env_prefix = "LLM_"
        populate_by_name = True


class RetrievalSettings(BaseSettings):
    """Retrieval settings."""

    top_k: int = 5
    score_threshold: float = 0.7
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    class Config:
        env_prefix = "RETRIEVAL_"


class ChunkingSettings(BaseSettings):
    """Document chunking settings."""

    strategy: str = "recursive"  # recursive, sentence, semantic
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    class Config:
        env_prefix = "CHUNKING_"


class Settings(BaseSettings):
    """Main application settings."""

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # General
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Settings instance.
        """
        with open(path) as f:
            config = yaml.safe_load(f) or {}

        return cls(
            qdrant=QdrantSettings(**config.get("qdrant", {})),
            embedding=EmbeddingSettings(**config.get("embedding", {})),
            llm=LLMSettings(**config.get("llm", {})),
            retrieval=RetrievalSettings(**config.get("retrieval", {})),
            chunking=ChunkingSettings(**config.get("chunking", {})),
            log_level=config.get("log_level", "INFO"),
            data_dir=Path(config.get("data_dir", "data")),
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance.
    """
    return Settings()
