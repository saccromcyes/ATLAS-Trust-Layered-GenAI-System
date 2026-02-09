from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pathlib import Path
from typing import List, Dict
import hashlib

from .db import engine, get_db, Base
from .models import Document, Chunk, RunLog
from .schemas import IngestResponse, AskRequest, AskResponse, Citation
from .settings import settings
from .utils import timer

from .ingestion.loader import load_text, list_files
from .ingestion.chunker import chunk_text
from .ingestion.embedder import Embedder
from .ingestion.indexer import upsert_chunks

from .retrieval.hybrid import HybridRetriever
from .llm.client import OpenAIClient
from .llm.prompts import SYSTEM_ANSWER
from .llm.verifier import Verifier

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ATLAS API", version="1.0.0")

def build_bm25_docs(db: Session) -> List[Dict]:
    rows = db.query(Chunk).all()
    return [{
        "chunk_id": r.chunk_id,
        "source_path": (r.meta or {}).get("source_path", ""),
        "text": r.text,
    } for r in rows]

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.APP_ENV}

@app.post("/ingest/folder", response_model=IngestResponse)
def ingest_folder(path: str, db: Session = Depends(get_db)):
    folder = Path(path)
    if not folder.exists():
        raise HTTPException(400, f"Folder not found: {path}")

    embedder = Embedder()
    doc_count = 0
    chunk_count = 0
    payloads: List[Dict] = []
    texts: List[str] = []

    for fp in list_files(folder):
        sp = str(fp)
        if db.query(Document).filter(Document.source_path == sp).first():
            continue

        try:
            text = load_text(fp)
        except Exception as e:
            raise HTTPException(400, f"Failed reading {fp.name}: {e}")

        doc = Document(source_path=sp, title=fp.name)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_count += 1

        chunks = chunk_text(text)
        for c in chunks:
            db.add(Chunk(
                doc_id=doc.id,
                chunk_id=c["chunk_id"],
                text=c["text"],
                meta={"source_path": sp, "title": fp.name},
            ))
            payloads.append({"chunk_id": c["chunk_id"], "source_path": sp, "title": fp.name, "text": c["text"]})
            texts.append(c["text"])
        db.commit()
        chunk_count += len(chunks)

    if chunk_count == 0:
        return {"documents_ingested": 0, "chunks_indexed": 0}

    vectors = embedder.embed(texts)
    upsert_chunks(vectors, payloads)
    return {"documents_ingested": doc_count, "chunks_indexed": chunk_count}

@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max bytes: {settings.MAX_UPLOAD_BYTES}")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".txt", ".md", ".pdf"]:
        raise HTTPException(400, "Unsupported file type. Use .txt/.md/.pdf")

    tmp = Path("/tmp") / f"atlas_{hashlib.md5(data).hexdigest()}{suffix}"
    tmp.write_bytes(data)
    return ingest_folder(str(tmp.parent), db)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    with timer() as t:
        bm25_docs = build_bm25_docs(db)
        if not bm25_docs:
            raise HTTPException(400, "No documents indexed. Ingest first.")

        retriever = HybridRetriever(bm25_docs=bm25_docs)
        hits = retriever.search(req.question, top_k=req.top_k)

        evidence = []
        citations: List[Citation] = []
        for h in hits:
            snippet = (h["text"][:260] + "...") if len(h["text"]) > 260 else h["text"]
            citations.append(Citation(
                chunk_id=h["chunk_id"],
                source_path=h["source_path"],
                score=float(h["score"]),
                snippet=snippet,
            ))
            evidence.append({"chunk_id": h["chunk_id"], "text": h["text"]})

        context = "\n\n".join([f"[{e['chunk_id']}] {e['text']}" for e in evidence])

        llm = OpenAIClient()
        answer = llm.chat(
            [{"role":"system","content":SYSTEM_ANSWER},
             {"role":"user","content": f"Question: {req.question}\n\nEvidence:\n{context}\n\nReturn an answer strictly grounded in evidence."}],
            temperature=0.2,
        )

        verifier = Verifier()
        claims = verifier.extract_claims(answer)
        verification = verifier.verify(claims, evidence)

        groundedness = float(verification.get("groundedness") or 0.0)
        if groundedness < 0.55:
            answer = "I can’t answer confidently from the available evidence. Please add documents or rephrase."

        latency = t()

    db.add(RunLog(
        question=req.question,
        answer=answer,
        citations=[c.model_dump() for c in citations],
        verification=verification,
        latency_sec=latency,
        cost_est=None,
    ))
    db.commit()

    return AskResponse(answer=answer, citations=citations, verification=verification)
