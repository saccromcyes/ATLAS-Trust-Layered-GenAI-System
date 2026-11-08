"""Embedding and reranking models, each with an offline fallback.

The production path uses small ONNX models through FastEmbed, so the whole app runs on a
2-vCPU machine with no GPU and no embedding API bill. The fallbacks keep tests and offline
development fast and dependency-free.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

import numpy as np

from atlas.config import Settings

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,'%-][a-z0-9%]+)*")
STOPWORDS = frozenset(
    (
        "a an and are as at be by can do does for from has have how i if in is it its many "
        "much my of on or our should the their there this to was we what when where which who why will with you your"
    ).split()
)


def _fold(token: str) -> str:
    """Minimal plural folding so 'passwords' matches 'password' (no stemmer dependency)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [_fold(t) for t in tokens if not (drop_stopwords and t in STOPWORDS)]


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Feature-hashing embedder over unigrams and bigrams. Deterministic, no downloads."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class FastEmbedEmbedder:
    """BGE-small (384-d) via FastEmbed/ONNX Runtime."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack(list(self._model.embed(texts, batch_size=32))) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self._model.query_embed(text)))


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance probability in [0, 1] for each passage."""


class LexicalReranker:
    """Query-term coverage, weighted by term rarity. Used offline and in tests."""

    name = "lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return [0.0] * len(passages)
        doc_terms = [set(tokenize(p)) for p in passages]
        n = len(passages) or 1
        idf = {t: math.log(1 + n / (1 + sum(t in d for d in doc_terms))) for t in q_terms}
        total = sum(idf.values())
        return [sum(idf[t] for t in q_terms & d) / total for d in doc_terms]


class CrossEncoderReranker:
    """MS-MARCO MiniLM cross-encoder: reads query and passage together, far sharper than bi-encoders."""

    name = "cross-encoder"

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        logits = list(self._model.rerank(query, passages, batch_size=16))
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fastembed":
        try:
            return FastEmbedEmbedder(settings.embed_model, settings.model_cache_dir)
        except Exception as exc:  # model download blocked, no disk, etc.
            log.warning("FastEmbed unavailable (%s); falling back to hash embeddings", exc)
    return HashEmbedder()


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        try:
            return CrossEncoderReranker(settings.rerank_model, settings.model_cache_dir)
        except Exception as exc:
            log.warning("Cross-encoder unavailable (%s); falling back to lexical reranking", exc)
    return LexicalReranker()
