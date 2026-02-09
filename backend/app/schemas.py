from pydantic import BaseModel
from typing import Any, List, Dict, Optional

class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_indexed: int

class AskRequest(BaseModel):
    question: str
    top_k: int = 8

class Citation(BaseModel):
    chunk_id: str
    source_path: str
    score: float
    snippet: str

class VerificationSummary(BaseModel):
    total_claims: int
    supported_claims: int
    support_ratio: float

class RetrievalSummary(BaseModel):
    top_k: int
    citation_count: int
    top_score: float

class QualitySummary(BaseModel):
    trust_score: float
    groundedness: float
    support_ratio: float

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    verification: Dict[str, Any]
    verification_summary: Optional[VerificationSummary] = None
    retrieval_summary: Optional[RetrievalSummary] = None
    quality_summary: Optional[QualitySummary] = None
    latency_sec: Optional[float] = None

class RunSummary(BaseModel):
    id: int
    question: str
    answer: str
    groundedness: float
    latency_sec: Optional[float]
    created_at: str

class DocumentSummary(BaseModel):
    id: int
    source_path: str
    title: Optional[str]
    created_at: str

class StatsResponse(BaseModel):
    documents: int
    chunks: int
    runs: int
    avg_latency_sec: Optional[float]
    avg_groundedness: Optional[float]
    last_run_at: Optional[str]
