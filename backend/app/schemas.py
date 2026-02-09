from pydantic import BaseModel
from typing import Any, List, Dict

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

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    verification: Dict[str, Any]
