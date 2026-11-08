"""Relational metadata: documents, chunks and the log of every question answered."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from atlas.ingestion import Chunk, Document


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(300))
    content_hash: Mapped[str] = mapped_column(String(32))
    n_chunks: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String(240), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    doc_id: Mapped[str] = mapped_column(String(200), index=True)
    doc_title: Mapped[str] = mapped_column(String(300))
    heading: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RunRow(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    session_id: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32))  # answered | withheld | refused | error
    mode: Mapped[str] = mapped_column(String(32))  # llm | fallback
    groundedness: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_claims: Mapped[int] = mapped_column(Integer, default=0)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    timings: Mapped[dict] = mapped_column(JSON, default=dict)


class MetadataStore:
    def __init__(self, database_url: str):
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session()

    # --- documents ---------------------------------------------------------
    def find_document(self, session_id: str, doc_id: str) -> DocumentRow | None:
        with self.session() as s:
            return s.get(DocumentRow, (session_id, doc_id))

    def save_document(self, doc: Document, chunks: list[Chunk]) -> None:
        with self.session() as s, s.begin():
            s.execute(delete(ChunkRow).where(ChunkRow.session_id == doc.session_id, ChunkRow.doc_id == doc.doc_id))
            s.merge(
                DocumentRow(
                    session_id=doc.session_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    content_hash=doc.content_hash,
                    n_chunks=len(chunks),
                )
            )
            s.add_all(
                ChunkRow(
                    chunk_id=c.chunk_id,
                    session_id=c.session_id,
                    doc_id=c.doc_id,
                    doc_title=c.doc_title,
                    heading=c.heading,
                    text=c.text,
                    position=c.position,
                    page=c.page,
                )
                for c in chunks
            )

    def list_documents(self, session_ids: list[str]) -> list[DocumentRow]:
        with self.session() as s:
            rows = s.scalars(select(DocumentRow).where(DocumentRow.session_id.in_(session_ids)))
            return sorted(rows, key=lambda r: (r.session_id != "demo", r.title))

    def count_documents(self, session_id: str) -> int:
        with self.session() as s:
            return s.scalar(select(func.count()).select_from(DocumentRow).where(DocumentRow.session_id == session_id))

    def chunks(self, session_ids: list[str] | None = None) -> list[Chunk]:
        with self.session() as s:
            query = select(ChunkRow)
            if session_ids is not None:
                query = query.where(ChunkRow.session_id.in_(session_ids))
            return [
                Chunk(r.chunk_id, r.doc_id, r.doc_title, r.text, r.heading, r.position, r.session_id, r.page)
                for r in s.scalars(query.order_by(ChunkRow.doc_id, ChunkRow.position))
            ]

    def expired_sessions(self, ttl_minutes: int) -> list[str]:
        cutoff = utcnow() - timedelta(minutes=ttl_minutes)
        with self.session() as s:
            rows = s.execute(
                select(DocumentRow.session_id)
                .where(DocumentRow.session_id != "demo")
                .group_by(DocumentRow.session_id)
                .having(func.max(DocumentRow.created_at) < cutoff)
            )
            return [r[0] for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self.session() as s, s.begin():
            s.execute(delete(ChunkRow).where(ChunkRow.session_id == session_id))
            s.execute(delete(DocumentRow).where(DocumentRow.session_id == session_id))

    # --- runs --------------------------------------------------------------
    def log_run(self, **fields) -> int:
        with self.session() as s, s.begin():
            row = RunRow(**fields)
            s.add(row)
            s.flush()
            return row.id

    def stats(self) -> dict:
        with self.session() as s:
            total = s.scalar(select(func.count()).select_from(RunRow)) or 0
            by_status = dict(s.execute(select(RunRow.status, func.count()).group_by(RunRow.status)).all())
            avg_ground = s.scalar(select(func.avg(RunRow.groundedness)).where(RunRow.groundedness.is_not(None)))
            return {
                "questions_answered": total,
                "by_status": by_status,
                "avg_groundedness": round(avg_ground, 3) if avg_ground is not None else None,
            }
