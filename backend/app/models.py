from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Index
from sqlalchemy.sql import func
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source_path = Column(String(500), unique=True, nullable=False)
    title = Column(String(300), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, nullable=False)
    chunk_id = Column(String(100), unique=True, nullable=False)  # stable id for citations
    text = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (Index("idx_chunks_doc_id", "doc_id"),)

class RunLog(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    verification = Column(JSON, nullable=True)
    latency_sec = Column(Float, nullable=True)
    cost_est = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
