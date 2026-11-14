_Last run 2026-11-14 · commit `5f6329c` · retrieval only · BAAI/bge-small-en-v1.5 · Xenova/ms-marco-MiniLM-L-6-v2_

**Retrieval ablation**: is the passage that states the answer in the top *k*? (43 questions, 59 passages)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 84% | 88% | 91% | 0.866 |
| Vector only | 86% | 93% | 98% | 0.909 |
| Hybrid (RRF) | 91% | 93% | 93% | 0.925 |
| **Hybrid + cross-encoder** | 93% | 100% | 100% | 0.961 |
