# ATLAS — Trust-Layered GenAI System  
**Hybrid RAG + Verification + Evaluation**

ATLAS is a production-style GenAI application that answers questions over document corpora using **hybrid retrieval**, **citations**, and a **claim-verification layer** to reduce hallucinations and improve trust.

---

## Key Features

- **Hybrid RAG**
  - Vector search (Qdrant)
  - Keyword search (BM25)
  - Merge + rerank

- **Evidence-based answers**
  - Stable chunk IDs
  - Source citations for every answer

- **Claim Verification**
  - Extracts atomic claims
  - Verifies each claim against retrieved evidence
  - Computes a **groundedness score (0–1)**

- **Safe Refusal**
  - Automatically refuses to answer when evidence is insufficient

- **Evaluation Harness**
  - Golden-set regression tests
  - Measures answer correctness and citation coverage

- **Full Stack (Dockerized)**
  - FastAPI (backend)
  - Postgres (metadata + run logs)
  - Qdrant (vector database)
  - Streamlit (UI)

---

## Tech Stack

| Layer | Technology |
|-----|-----------|
| Backend | FastAPI, Python |
| Vector DB | Qdrant |
| Metadata DB | PostgreSQL |
| Retrieval | BM25 + Embeddings |
| LLM | OpenAI-compatible API |
| UI | Streamlit |
| Infra | Docker, Docker Compose |

---
