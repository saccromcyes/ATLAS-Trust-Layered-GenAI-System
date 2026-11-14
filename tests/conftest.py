from __future__ import annotations

import json

import pytest

from atlas.config import ROOT, Settings
from atlas.ingestion import load_directory
from atlas.llm import LLMError
from atlas.models import HashEmbedder, LexicalReranker
from atlas.pipeline import AnswerPipeline
from atlas.retrieval import KnowledgeBase
from atlas.store import MetadataStore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="",
        embedder="hash",
        reranker="lexical",
        qdrant_path=":memory:",
        database_url=f"sqlite:///{tmp_path / 'atlas.db'}",
        corpus_dir=str(ROOT / "corpus"),
        min_relevance=0.2,
        min_similarity=1.0,  # hash embeddings: let the lexical reranker alone decide in tests
        rate_limit_per_10min=1000,
    )


@pytest.fixture
def kb(settings) -> KnowledgeBase:
    store = MetadataStore(settings.database_url)
    kb = KnowledgeBase(settings, HashEmbedder(), LexicalReranker(), store)
    for doc in load_directory(settings.corpus_dir):
        kb.add_document(doc)
    return kb


class FakeLLM:
    """Scripted stand-in for an OpenAI-compatible model."""

    mode = "llm"

    def __init__(self, answer: str = "", claims: list[dict] | None = None, fail_stream: bool = False):
        self.answer, self.claims, self.fail_stream = answer, claims or [], fail_stream
        self.calls: list[str] = []

    def stream(self, messages, model):
        self.calls.append("stream")
        if self.fail_stream:
            yield "Partial "
            raise LLMError("429 rate limited")
        for i in range(0, len(self.answer), 7):
            yield self.answer[i : i + 7]

    def complete_json(self, messages, model):
        self.calls.append("json")
        return json.loads(json.dumps({"claims": self.claims}))


@pytest.fixture
def make_pipeline(settings, kb):
    def _make(llm) -> AnswerPipeline:
        return AnswerPipeline(settings, kb, llm, kb.store)

    return _make
