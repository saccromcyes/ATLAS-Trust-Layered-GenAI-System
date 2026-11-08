"""Runtime configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (any OpenAI-compatible endpoint: Groq, OpenAI, Together, vLLM, Ollama) ---
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"  # writes the answer
    llm_fast_model: str = "llama-3.1-8b-instant"  # extracts and checks claims
    llm_timeout_s: float = 30.0

    # --- Retrieval models ---
    # "fastembed" uses local ONNX models; "hash" is a dependency-free fallback for tests/offline.
    embedder: str = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "cross-encoder"  # or "lexical"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = str(ROOT / ".cache" / "models")

    # --- Storage ---
    qdrant_url: str = ""  # empty = embedded Qdrant on local disk
    qdrant_path: str = str(ROOT / ".data" / "qdrant")
    database_url: str = f"sqlite:///{ROOT / '.data' / 'atlas.db'}"
    corpus_dir: str = str(ROOT / "corpus")

    # --- Retrieval behaviour ---
    chunk_size: int = 700  # characters
    chunk_overlap: int = 120
    candidates_per_retriever: int = 20
    context_chunks: int = 5
    rrf_k: int = 60

    # --- Trust layer ---
    min_relevance: float = 0.20  # reranker probability below which we refuse before generating
    min_groundedness: float = 0.60  # share of verified claims below which the answer is withheld

    # --- Public demo guard rails ---
    max_upload_mb: int = 10
    max_uploads_per_session: int = 5
    session_ttl_minutes: int = 120
    rate_limit_per_10min: int = 30
    max_question_chars: int = 500

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
