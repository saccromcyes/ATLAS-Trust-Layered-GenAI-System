from typing import List, Dict
from qdrant_client import QdrantClient
from ..settings import settings
from ..llm.client import OpenAIClient
from .bm25 import BM25Index
from .rerank import rerank

class HybridRetriever:
    def __init__(self, bm25_docs: List[Dict]):
        self.llm = OpenAIClient()
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.bm25 = BM25Index(bm25_docs)

    def vector_search(self, query: str, top_k: int) -> List[Dict]:
        qv = self.llm.embed([query])[0]
        hits = self.qdrant.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=qv,
            limit=top_k,
            with_payload=True,
        )
        out = []
        for h in hits:
            pl = h.payload or {}
            out.append({
                "chunk_id": pl.get("chunk_id", str(h.id)),
                "source_path": pl.get("source_path", ""),
                "text": pl.get("text", ""),
                "score": float(h.score),
            })
        return out

    def bm25_search(self, query: str, top_k: int) -> List[Dict]:
        hits = self.bm25.search(query, top_k=top_k)
        return [{
            "chunk_id": d["chunk_id"],
            "source_path": d["source_path"],
            "text": d["text"],
            "score": float(s),
        } for s, d in hits]

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        v = self.vector_search(query, top_k=top_k)
        b = self.bm25_search(query, top_k=top_k)

        merged: Dict[str, Dict] = {}
        for item in v + b:
            cid = item["chunk_id"]
            if cid not in merged or item["score"] > merged[cid]["score"]:
                merged[cid] = item

        return rerank(query, list(merged.values()))[:top_k]
