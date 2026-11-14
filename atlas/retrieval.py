"""Hybrid retrieval: dense vectors (Qdrant) + BM25, fused with Reciprocal Rank Fusion, then
re-scored by a cross-encoder. Every stage's ranks and scores are kept so the UI can show
exactly why a passage was chosen."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from atlas.config import Settings
from atlas.ingestion import Chunk, Document, chunk_document
from atlas.models import Embedder, Reranker, tokenize
from atlas.store import MetadataStore

log = logging.getLogger(__name__)
COLLECTION = "atlas_chunks"


@dataclass
class Candidate:
    chunk: Chunk
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["chunk"] = asdict(self.chunk)
        return data


@dataclass
class RetrievalResult:
    query: str
    candidates: list[Candidate]  # all fused candidates, sorted by rerank score
    context: list[Candidate]  # the top passages sent to the LLM
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_relevance(self) -> float:
        return self.context[0].rerank_score if self.context else 0.0

    @property
    def top_similarity(self) -> float:
        """Best embedding (cosine) similarity among the candidates: the paraphrase-tolerant signal."""
        return max((c.vector_score or 0.0 for c in self.candidates), default=0.0)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF (Cormack et al., 2009): sum of 1 / (k + rank) across rankings. Scale-free, so BM25
    scores and cosine similarities never need to be calibrated against each other."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def _embedding_text(chunk: Chunk) -> str:
    # Contextual header: the title and section travel with the passage, so "the limit is $300"
    # still matches a question about London hotels.
    header = chunk.doc_title + (f" > {chunk.heading}" if chunk.heading and chunk.heading != chunk.doc_title else "")
    return f"{header}\n{chunk.text}"


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder, reranker: Reranker, store: MetadataStore):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        if settings.qdrant_url:
            self.qdrant = QdrantClient(url=settings.qdrant_url)
        elif settings.qdrant_path == ":memory:":
            self.qdrant = QdrantClient(":memory:")
        else:
            Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=settings.qdrant_path)
        self._ensure_collection()
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in store.chunks()}
        self._bm25_cache: dict[tuple[str, ...], tuple[BM25Okapi, list[str]]] = {}

    # --- indexing ------------------------------------------------------------
    def _ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(COLLECTION):
            self.qdrant.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add_document(self, doc: Document) -> tuple[list[Chunk], bool]:
        """Index a document. Returns (chunks, changed); unchanged re-ingests are skipped."""
        existing = self.store.find_document(doc.session_id, doc.doc_id)
        if existing and existing.content_hash == doc.content_hash:
            return [c for c in self._chunks.values() if c.doc_id == doc.doc_id and c.session_id == doc.session_id], False
        if existing:
            self._delete_doc_vectors(doc.session_id, doc.doc_id)

        chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("The document produced no indexable text.")
        vectors = self.embedder.embed_documents([_embedding_text(c) for c in chunks])
        self.qdrant.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=self._point_id(c.chunk_id),
                    vector=np.asarray(v, dtype=np.float32).tolist(),
                    payload={"chunk_id": c.chunk_id, "session_id": c.session_id, "doc_id": c.doc_id},
                )
                for c, v in zip(chunks, vectors)
            ],
        )
        self.store.save_document(doc, chunks)
        for chunk_id in [k for k, c in self._chunks.items() if c.doc_id == doc.doc_id and c.session_id == doc.session_id]:
            del self._chunks[chunk_id]
        self._chunks.update({c.chunk_id: c for c in chunks})
        self._bm25_cache.clear()
        return chunks, True

    def _delete_doc_vectors(self, session_id: str, doc_id: str) -> None:
        self.qdrant.delete(
            COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]
            ),
        )

    def delete_session(self, session_id: str) -> None:
        self.qdrant.delete(
            COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]),
        )
        self.store.delete_session(session_id)
        self._chunks = {k: c for k, c in self._chunks.items() if c.session_id != session_id}
        self._bm25_cache.clear()

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    # --- retrieval -----------------------------------------------------------
    def _visible(self, session_ids: list[str]) -> list[Chunk]:
        return [c for c in self._chunks.values() if c.session_id in session_ids]

    def _bm25(self, visible: list[Chunk]) -> tuple[BM25Okapi, list[str]]:
        key = tuple(sorted(c.chunk_id for c in visible))
        if key not in self._bm25_cache:
            ids = [c.chunk_id for c in visible]
            corpus = [tokenize(_embedding_text(c)) or ["_"] for c in visible]
            self._bm25_cache = {key: (BM25Okapi(corpus), ids)}  # keep one index; sessions change rarely
        return self._bm25_cache[key]

    def retrieve(self, query: str, session_id: str = "demo") -> RetrievalResult:
        s = self.settings
        sessions = ["demo"] if session_id == "demo" else ["demo", session_id]
        visible = self._visible(sessions)
        timings: dict[str, float] = {}
        if not visible:
            return RetrievalResult(query, [], [], timings)

        t0 = time.perf_counter()
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32).tolist()
        hits = self.qdrant.query_points(
            COLLECTION,
            query=query_vector,
            limit=s.candidates_per_retriever,
            query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchAny(any=sessions))]),
        ).points
        vector_ranking = [(h.payload["chunk_id"], float(h.score)) for h in hits if h.payload["chunk_id"] in self._chunks]
        timings["vector_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25, ids = self._bm25(visible)
        scores = bm25.get_scores(tokenize(query) or ["_"])
        order = np.argsort(scores)[::-1][: s.candidates_per_retriever]
        bm25_ranking = [(ids[i], float(scores[i])) for i in order if scores[i] > 0]
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        fused = reciprocal_rank_fusion([[c for c, _ in vector_ranking], [c for c, _ in bm25_ranking]], k=s.rrf_k)
        candidates: dict[str, Candidate] = {cid: Candidate(self._chunks[cid], rrf_score=sc) for cid, sc in fused.items()}
        for rank, (cid, score) in enumerate(vector_ranking, start=1):
            candidates[cid].vector_rank, candidates[cid].vector_score = rank, round(score, 4)
        for rank, (cid, score) in enumerate(bm25_ranking, start=1):
            candidates[cid].bm25_rank, candidates[cid].bm25_score = rank, round(score, 3)
        pool = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)[: s.candidates_per_retriever]

        t0 = time.perf_counter()
        rerank_scores = self.reranker.score(query, [_embedding_text(c.chunk) for c in pool])
        for cand, score in zip(pool, rerank_scores):
            cand.rerank_score = round(float(score), 4)
        pool.sort(key=lambda c: (c.rerank_score, c.rrf_score), reverse=True)
        for rank, cand in enumerate(pool, start=1):
            cand.final_rank = rank
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(query, pool, pool[: s.context_chunks], {k: round(v, 1) for k, v in timings.items()})

    def retrieve_ablation(self, query: str, mode: str, k: int) -> list[Chunk]:
        """Passage rankings for one retrieval strategy, used by the evaluation harness."""
        result = self.retrieve(query)
        if mode == "hybrid+rerank":
            ordered = result.candidates
        elif mode == "hybrid":
            ordered = sorted(result.candidates, key=lambda c: c.rrf_score, reverse=True)
        elif mode == "vector":
            ordered = sorted([c for c in result.candidates if c.vector_rank], key=lambda c: c.vector_rank)
        elif mode == "bm25":
            ordered = sorted([c for c in result.candidates if c.bm25_rank], key=lambda c: c.bm25_rank)
        else:
            raise ValueError(mode)
        return [c.chunk for c in ordered[:k]]
