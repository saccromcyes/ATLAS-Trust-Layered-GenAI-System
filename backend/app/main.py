from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path
from typing import List, Dict
import hashlib

from .db import engine, get_db, Base
from .models import Document, Chunk, RunLog
from .schemas import (
    IngestResponse,
    AskRequest,
    AskResponse,
    Citation,
    VerificationSummary,
    RetrievalSummary,
    QualitySummary,
    RunSummary,
    StatsResponse,
    DocumentSummary,
)
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

    verification_summary = _build_verification_summary(verification)
    retrieval_summary = _build_retrieval_summary(citations, req.top_k)
    quality_summary = _build_quality_summary(verification, verification_summary)
    return AskResponse(
        answer=answer,
        citations=citations,
        verification=verification,
        verification_summary=verification_summary,
        retrieval_summary=retrieval_summary,
        quality_summary=quality_summary,
        latency_sec=latency,
    )

@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    doc_count = db.query(func.count(Document.id)).scalar() or 0
    chunk_count = db.query(func.count(Chunk.id)).scalar() or 0
    run_count = db.query(func.count(RunLog.id)).scalar() or 0
    avg_latency = db.query(func.avg(RunLog.latency_sec)).scalar()
    rows = db.query(RunLog.verification).all()
    groundedness_values = []
    for (verification,) in rows:
        if isinstance(verification, dict):
            value = verification.get("groundedness")
            if value is not None:
                groundedness_values.append(float(value))
    avg_groundedness = None
    if groundedness_values:
        avg_groundedness = sum(groundedness_values) / len(groundedness_values)
    last_run_at = db.query(func.max(RunLog.created_at)).scalar()
    return StatsResponse(
        documents=int(doc_count),
        chunks=int(chunk_count),
        runs=int(run_count),
        avg_latency_sec=float(avg_latency) if avg_latency is not None else None,
        avg_groundedness=float(avg_groundedness) if avg_groundedness is not None else None,
        last_run_at=str(last_run_at) if last_run_at is not None else None,
    )

@app.get("/runs/recent", response_model=List[RunSummary])
def recent_runs(limit: int = 8, db: Session = Depends(get_db)):
    limit = min(max(limit, 1), 20)
    rows = (
        db.query(RunLog)
        .order_by(RunLog.created_at.desc())
        .limit(limit)
        .all()
    )
    summaries = []
    for row in rows:
        groundedness = 0.0
        if isinstance(row.verification, dict):
            groundedness = float(row.verification.get("groundedness") or 0.0)
        summaries.append(RunSummary(
            id=row.id,
            question=row.question,
            answer=row.answer,
            groundedness=groundedness,
            latency_sec=row.latency_sec,
            created_at=str(row.created_at),
        ))
    return summaries

@app.get("/documents", response_model=List[DocumentSummary])
def list_documents(db: Session = Depends(get_db)):
    rows = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        DocumentSummary(
            id=row.id,
            source_path=row.source_path,
            title=row.title,
            created_at=str(row.created_at),
        )
        for row in rows
    ]

def _build_verification_summary(verification: Dict) -> VerificationSummary:
    verdicts = verification.get("verdicts") or []
    total_claims = len(verdicts)
    supported_claims = sum(1 for v in verdicts if v.get("supported"))
    support_ratio = (supported_claims / total_claims) if total_claims else 0.0
    return VerificationSummary(
        total_claims=total_claims,
        supported_claims=supported_claims,
        support_ratio=float(support_ratio),
    )

def _build_retrieval_summary(citations: List[Citation], top_k: int) -> RetrievalSummary:
    top_score = max((c.score for c in citations), default=0.0)
    return RetrievalSummary(
        top_k=top_k,
        citation_count=len(citations),
        top_score=float(top_score),
    )

def _build_quality_summary(
    verification: Dict,
    verification_summary: VerificationSummary,
) -> QualitySummary:
    groundedness = float(verification.get("groundedness") or 0.0)
    support_ratio = float(verification_summary.support_ratio)
    trust_score = (0.65 * groundedness) + (0.35 * support_ratio)
    return QualitySummary(
        trust_score=float(trust_score),
        groundedness=groundedness,
        support_ratio=support_ratio,
    )
