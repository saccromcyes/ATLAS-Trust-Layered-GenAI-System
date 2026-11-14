"""FastAPI application: JSON + server-sent-events API and the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from atlas import __version__
from atlas.config import ROOT, Settings, get_settings
from atlas.ingestion import load_bytes, load_directory
from atlas.llm import build_llm
from atlas.models import build_embedder, build_reranker
from atlas.pipeline import AnswerPipeline
from atlas.prompts import PROMPT_VERSION
from atlas.retrieval import KnowledgeBase
from atlas.store import MetadataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("atlas")

WEB_DIR = ROOT / "web"
EVAL_RESULTS = ROOT / "eval" / "results" / "latest.json"
_SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MetadataStore(settings.database_url)
        self.embedder = build_embedder(settings)
        self.reranker = build_reranker(settings)
        self.kb = KnowledgeBase(settings, self.embedder, self.reranker, self.store)
        self.llm = build_llm(settings)
        self.pipeline = AnswerPipeline(settings, self.kb, self.llm, self.store)

    def bootstrap_corpus(self) -> int:
        added = 0
        corpus = Path(self.settings.corpus_dir)
        if corpus.exists():
            for doc in load_directory(corpus):
                _, changed = self.kb.add_document(doc)
                added += changed
        log.info("Demo corpus ready (%d documents re-indexed)", added)
        return added


class RateLimiter:
    """Sliding-window limiter per client IP. In-memory is enough for a single-instance demo."""

    def __init__(self, limit: int, window_s: int = 600):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self.hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry = int(self.window - (now - hits[0])) + 1
            raise HTTPException(429, f"Rate limit reached. Try again in {retry} seconds.", headers={"Retry-After": str(retry)})
        hits.append(now)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


def session_of(x_session_id: str | None) -> str:
    return x_session_id if x_session_id and _SESSION_RE.match(x_session_id) and x_session_id != "demo" else "demo"


async def _cleanup_loop(services: Services) -> None:
    while True:
        await asyncio.sleep(300)
        try:
            for session_id in services.store.expired_sessions(services.settings.session_ttl_minutes):
                services.kb.delete_session(session_id)
                log.info("Deleted expired upload session %s", session_id)
        except Exception:  # never let cleanup crash the app
            log.exception("Session cleanup failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = Services(settings)
        services.bootstrap_corpus()
        app.state.services = services
        app.state.limiter = RateLimiter(settings.rate_limit_per_10min)
        task = asyncio.create_task(_cleanup_loop(services))
        yield
        task.cancel()

    app = FastAPI(title="ATLAS", version=__version__, lifespan=lifespan, docs_url="/api/docs", redoc_url=None)

    def services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/api/health")
    def health(request: Request):
        svc = services(request)
        return {"status": "ok", "version": __version__, "mode": svc.pipeline.mode}

    @app.get("/api/config")
    def config(request: Request):
        svc = services(request)
        return {
            "version": __version__,
            "mode": svc.pipeline.mode,
            "llm_model": settings.llm_model if svc.pipeline.mode == "llm" else None,
            "verifier_model": settings.llm_fast_model if svc.pipeline.mode == "llm" else None,
            "embedder": type(svc.embedder).__name__,
            "embed_model": settings.embed_model if type(svc.embedder).__name__ == "FastEmbedEmbedder" else "hashing",
            "reranker": svc.reranker.name,
            "rerank_model": settings.rerank_model if svc.reranker.name == "cross-encoder" else "lexical",
            "min_relevance": settings.min_relevance,
            "min_similarity": settings.min_similarity,
            "min_groundedness": settings.min_groundedness,
            "prompt_version": PROMPT_VERSION,
            "max_upload_mb": settings.max_upload_mb,
        }

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request, x_session_id: str | None = Header(default=None)):
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(422, f"Questions must be under {settings.max_question_chars} characters.")
        request.app.state.limiter.check(client_ip(request))
        pipeline = services(request).pipeline
        session_id = session_of(x_session_id)

        def events():
            try:
                for event in pipeline.run(body.question.strip(), session_id):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except Exception:
                log.exception("Pipeline failed")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong while answering.'})}\n\n"

        return StreamingResponse(
            iterate_in_threadpool(events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/documents")
    def documents(request: Request, x_session_id: str | None = Header(default=None)):
        session_id = session_of(x_session_id)
        rows = services(request).store.list_documents(["demo", session_id])
        return [
            {"doc_id": r.doc_id, "title": r.title, "source": r.source, "chunks": r.n_chunks, "uploaded": r.session_id != "demo"}
            for r in rows
        ]

    @app.post("/api/documents")
    async def upload(request: Request, file: UploadFile = File(...), x_session_id: str | None = Header(default=None)):
        session_id = session_of(x_session_id)
        if session_id == "demo":
            raise HTTPException(400, "Missing session id; reload the page and try again.")
        request.app.state.limiter.check(client_ip(request))
        svc = services(request)
        if svc.store.count_documents(session_id) >= settings.max_uploads_per_session:
            raise HTTPException(400, f"You can upload up to {settings.max_uploads_per_session} documents per session.")
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Files must be under {settings.max_upload_mb} MB.")
        try:
            doc = load_bytes(file.filename or "upload.txt", data, session_id)
            chunks, _ = await run_in_threadpool(svc.kb.add_document, doc)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Upload failed")
            raise HTTPException(400, "Could not read this file.") from exc
        return {"doc_id": doc.doc_id, "title": doc.title, "chunks": len(chunks)}

    @app.delete("/api/session")
    def clear_session(request: Request, x_session_id: str | None = Header(default=None)):
        session_id = session_of(x_session_id)
        if session_id != "demo":
            services(request).kb.delete_session(session_id)
        return {"cleared": session_id != "demo"}

    @app.get("/api/eval")
    def eval_results():
        if not EVAL_RESULTS.exists():
            return JSONResponse({"available": False})
        return {"available": True, **json.loads(EVAL_RESULTS.read_text())}

    @app.get("/api/stats")
    def stats(request: Request):
        return services(request).store.stats()

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
