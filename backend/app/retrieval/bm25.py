from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 8) -> List[Tuple[float, Dict]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
        return [(float(s), self.docs[i]) for i, s in ranked]
