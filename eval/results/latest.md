_Last run 2026-11-16 · commit `59f1d98` · openai/gpt-oss-120b · BAAI/bge-small-en-v1.5 · Xenova/ms-marco-MiniLM-L-6-v2_

| Metric | Result |
|---|---|
| Answer accuracy (47 answerable questions) | **100%** |
| Multi-hop accuracy | 100% |
| Correct refusals (8 unanswerable questions) | **100%** |
| Hallucinated answers to unanswerable questions | 0 |
| Citation accuracy | 100% |
| Mean groundedness of shown answers | 99% |
| Latency p50 / p95 | 1249 ms / 1750 ms |

**Refusal gate**: which questions each rule would refuse before calling the model. The rest are answered, or refused by the model itself.

| Refuse when | Answerable questions let through | Unanswerable questions stopped early |
|---|---|---|
| reranker < 0.2 | 87% | 50% |
| reranker < 0.2 and similarity < 0.55 | 100% | 12% |
| **reranker < 0.2 and similarity < 0.6 (current)** | 100% | 12% |
| reranker < 0.2 and similarity < 0.65 | 92% | 25% |
| reranker < 0.2 and similarity < 0.7 | 89% | 50% |

**Retrieval ablation**: is the passage that states the answer in the top *k*? (43 questions, 59 passages)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 84% | 88% | 91% | 0.866 |
| Vector only | 86% | 93% | 98% | 0.909 |
| Hybrid (RRF) | 91% | 93% | 93% | 0.925 |
| **Hybrid + cross-encoder** | 93% | 100% | 100% | 0.961 |
