from typing import List, Dict

def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    # Lightweight rerank: lexical overlap booster.
    q = set(query.lower().split())

    def s(c):
        t = set((c.get("text") or "").lower().split())
        return len(q & t)

    return sorted(candidates, key=s, reverse=True)
