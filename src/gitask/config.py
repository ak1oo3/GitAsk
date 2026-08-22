"""Central configuration, loaded from environment / .env.

Every layer reads settings from here (`from gitask.config import settings`)
rather than calling os.environ directly, so the whole app is configured in one
place. See .env.example for the full list.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Database ----
    database_url: str = "postgresql://gitask:gitask@localhost:5432/gitask"

    # ---- Embeddings ----
    embedding_provider: str = "local"           # "local" | "openai"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- LLM ----
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # ---- Ingestion ----
    clone_dir: str = "./.repos"
    max_file_bytes: int = 1_000_000

    # ---- Retrieval ----
    top_k_vector: int = 20      # candidates pulled from vector search before re-rank
    top_k_final: int = 8        # chunks kept for the grounded prompt

    @property
    def embedding_dim(self) -> int:
        """Vector dimension for the active embedding model.

        Keep this authoritative — the pgvector column width must match.
        """
        if self.embedding_provider == "openai":
            # text-embedding-3-small
            return 1536
        # bge-small-en-v1.5 -> 384, bge-base-en-v1.5 -> 768
        return 768 if "base" in self.local_embedding_model else 384


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
