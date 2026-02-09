# ATLAS — Trust-Layered GenAI System  
**Hybrid RAG + Verification + Evaluation**

ATLAS is a production-style GenAI application that answers questions over document corpora using **hybrid retrieval**, **citations**, and a **claim-verification layer** to reduce hallucinations and improve trust.

---

## 🔥 Key Features

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

## 🧱 Tech Stack

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

## Project Structure

ATLAS_FULL_ATLAS/
├── backend/
│ ├── app/
│ │ ├── ingestion/ # loaders, chunking, embeddings
│ │ ├── retrieval/ # BM25 + hybrid retriever
│ │ ├── llm/ # LLM client + verifier
│ │ ├── eval/ # golden set evaluation
│ │ ├── main.py # FastAPI entrypoint
│ │ └── models.py
│ └── Dockerfile
├── ui/
│ ├── app.py # Streamlit UI
│ └── Dockerfile
├── sample_docs/ # Example documents
├── docker-compose.yml
├── .env.example
└── README.md


---

## ⚙️ Prerequisites

Make sure you have:

- **Docker** (v20+)
- **Docker Compose**
- An **OpenAI-compatible API key**


## How to Run the Project

###  Clone / Unzip the Repository

git clone <your-repo-url>
cd ATLAS_FULL_ATLAS
or unzip and cd into the folder

Create Environment File
cp .env.example .env
Edit .env and add your API key:

OPENAI_API_KEY=your_api_key_here
Do NOT commit .env to GitHub.

Start the Full Stack
docker compose up --build
This starts:

Backend API → http://localhost:8000

Streamlit UI → http://localhost:8501

Postgres

Qdrant

Open the UI
Go to:

http://localhost:8501

### Local Dev (without Docker)
If you don’t have Docker available, you can run the backend and UI locally.

1) Create and activate a virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install backend dependencies
```bash
pip install -r backend/requirements.txt
```

3) Install UI dependencies
```bash
pip install -r ui/requirements.txt
```

4) Start Postgres and Qdrant separately (Docker or local installs)
- Postgres must be reachable at the host/port in `.env`
- Qdrant must be reachable at the host/port in `.env`

5) Run the API
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

6) Run the UI (point it at the API)
```bash
export ATLAS_API_URL=http://localhost:8000
streamlit run ui/app.py --server.port 8501
```
Ingest Documents
From the UI:

Click “Ingest sample_docs”

OR via API:

curl -X POST "http://localhost:8000/ingest/folder?path=/app/sample_docs"
Ask Questions
Try:

What are the working hours?

How many leave days can be carry-forwarded?

What security controls are required?

You’ll see:

Answer

Citations

Groundedness score

Claim verification details

Run Evaluation (Regression Tests)
docker compose exec backend python -m app.eval.run_eval
This runs the golden set and prints:

PASS / FAIL per question

Groundedness scores

How Verification Works (High Level)
Retrieve evidence (hybrid search)

Generate answer only from evidence

Split answer into atomic claims

Verify each claim against evidence

Compute groundedness score

Refuse if groundedness < threshold

Why This Project Is Production-Grade
Prevents hallucinations (verification + refusal)

Measures quality (evaluation harness)

Scales cleanly (Dockerized services)

Clean separation of concerns

Extendable (rerankers, auth, feedback loop)

 Supported File Types
.txt

.md

.pdf

