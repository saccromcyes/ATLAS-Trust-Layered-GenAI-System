"""The trust-layered answer pipeline.

retrieve -> relevance gate -> stream a cited answer -> verify each claim -> groundedness gate

Each step emits an event so the UI can show the work as it happens, and the same generator
powers the evaluation harness.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from atlas.config import Settings
from atlas.llm import REFUSAL_TOKEN, ExtractiveLLM, LLMError, split_sentences
from atlas.models import tokenize
from atlas.prompts import answer_messages, verify_messages
from atlas.retrieval import Candidate, KnowledgeBase
from atlas.store import MetadataStore

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d{1,2})\]")
VERDICT_WEIGHT = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


@dataclass
class Verification:
    claims: list[dict]
    groundedness: float | None
    method: str


def _is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL_TOKEN)


def _maybe_refusal(text: str) -> bool:
    """True while the streamed prefix could still turn out to be the refusal token."""
    stripped = text.strip()
    return REFUSAL_TOKEN.startswith(stripped) or stripped.startswith(REFUSAL_TOKEN)


def extract_citations(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def strip_invalid_citations(answer: str, n_sources: int) -> str:
    return _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)


def groundedness(claims: list[dict]) -> float | None:
    if not claims:
        return None
    return round(sum(VERDICT_WEIGHT.get(c.get("verdict", "unsupported"), 0.0) for c in claims) / len(claims), 3)


def lexical_verify(answer: str, context: list[Candidate]) -> Verification:
    """Fallback verifier: a sentence counts as supported when most of its content words appear in
    the sources it cites. Strict, cheap, and good enough for extractive answers."""
    claims = []
    for sentence in split_sentences(answer):
        cited = extract_citations(sentence, len(context)) or list(range(1, len(context) + 1))
        claim = _CITATION.sub("", sentence).strip()
        terms = set(tokenize(claim))
        if not terms:
            continue
        source_terms = set().union(*(tokenize(context[n - 1].chunk.text) for n in cited))
        coverage = len(terms & source_terms) / len(terms)
        verdict = "supported" if coverage >= 0.8 else "partial" if coverage >= 0.5 else "unsupported"
        claims.append({"claim": claim, "verdict": verdict, "sources": cited, "reason": f"{coverage:.0%} term overlap"})
    return Verification(claims, groundedness(claims), "lexical")


class AnswerPipeline:
    def __init__(self, settings: Settings, kb: KnowledgeBase, llm, store: MetadataStore):
        self.settings = settings
        self.kb = kb
        self.llm = llm
        self.store = store
        self.extractive = ExtractiveLLM()

    @property
    def mode(self) -> str:
        return self.llm.mode

    def _verify(self, answer: str, context: list[Candidate]) -> Verification:
        if isinstance(self.llm, ExtractiveLLM):
            return lexical_verify(answer, context)
        try:
            data = self.llm.complete_json(verify_messages(answer, context), self.settings.llm_fast_model)
            claims = [
                {
                    "claim": str(c.get("claim", "")).strip(),
                    "verdict": c.get("verdict") if c.get("verdict") in VERDICT_WEIGHT else "unsupported",
                    "sources": [int(n) for n in c.get("sources", []) if str(n).isdigit() and 1 <= int(n) <= len(context)],
                    "reason": str(c.get("reason", ""))[:200],
                }
                for c in data.get("claims", [])
                if str(c.get("claim", "")).strip()
            ]
            if claims:
                return Verification(claims, groundedness(claims), "llm-judge")
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("LLM verification failed (%s); using lexical verifier", exc)
        return lexical_verify(answer, context)

    def run(self, question: str, session_id: str = "demo") -> Iterator[dict]:
        s = self.settings
        started = time.perf_counter()
        timings: dict[str, float] = {}
        mode = self.mode

        retrieval = self.kb.retrieve(question, session_id)
        timings.update(retrieval.timings_ms)
        yield {
            "type": "retrieval",
            "candidates": [c.to_dict() for c in retrieval.candidates[:10]],
            "context_ids": [c.chunk.chunk_id for c in retrieval.context],
            "top_relevance": retrieval.top_relevance,
            "reranker": self.kb.reranker.name,
            "timings": retrieval.timings_ms,
        }

        def finish(status: str, answer: str = "", verification: Verification | None = None, citations=None):
            timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            run_id = self.store.log_run(
                session_id=session_id,
                question=question,
                answer=answer,
                status=status,
                mode=mode,
                groundedness=verification.groundedness if verification else None,
                n_claims=len(verification.claims) if verification else 0,
                citations=citations or [],
                timings=timings,
            )
            return {"type": "done", "status": status, "run_id": run_id, "mode": mode, "timings": timings}

        off_topic = retrieval.top_relevance < s.min_relevance and retrieval.top_similarity < s.min_similarity
        if not retrieval.context or off_topic:
            yield {
                "type": "refusal",
                "reason": "low_relevance",
                "message": "I couldn't find anything in the documents that answers this, so I won't guess.",
                "top_relevance": retrieval.top_relevance,
                "threshold": s.min_relevance,
                "top_similarity": retrieval.top_similarity,
                "similarity_threshold": s.min_similarity,
            }
            yield finish("refused")
            return

        context = retrieval.context
        t0 = time.perf_counter()
        answer = ""
        if isinstance(self.llm, ExtractiveLLM):
            answer = self.llm.answer(question, [c.chunk.text for c in context])
            if not _is_refusal(answer):
                yield {"type": "token", "text": answer}
        else:
            buffer, emitting = "", False
            try:
                for delta in self.llm.stream(answer_messages(question, context), s.llm_model):
                    answer += delta
                    if emitting:
                        yield {"type": "token", "text": delta}
                        continue
                    buffer += delta
                    # Hold back the opening characters until we know the model is not refusing.
                    if not _maybe_refusal(buffer):
                        emitting = True
                        yield {"type": "token", "text": buffer}
                if not emitting and buffer.strip() and not _is_refusal(buffer):
                    yield {"type": "token", "text": buffer}  # very short answer
            except LLMError as exc:
                log.warning("Generation failed (%s); falling back to extractive answer", exc)
                mode = "fallback"
                yield {
                    "type": "notice",
                    "message": "The language model is unavailable, so this answer is quoted directly from the sources.",
                }
                yield {"type": "reset"}
                answer = self.extractive.answer(question, [c.chunk.text for c in context])
                if not _is_refusal(answer):
                    yield {"type": "token", "text": answer}
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if _is_refusal(answer) or not answer.strip():
            yield {
                "type": "refusal",
                "reason": "model_refused",
                "message": "The retrieved passages don't contain this information, so I won't guess.",
                "top_relevance": retrieval.top_relevance,
            }
            yield finish("refused")
            return

        answer = strip_invalid_citations(answer, len(context))
        cited = extract_citations(answer, len(context))
        citations = [
            {
                "n": n,
                "chunk_id": context[n - 1].chunk.chunk_id,
                "doc_id": context[n - 1].chunk.doc_id,
                "doc_title": context[n - 1].chunk.doc_title,
                "heading": context[n - 1].chunk.heading,
                "page": context[n - 1].chunk.page,
                "text": context[n - 1].chunk.text,
                "session_id": context[n - 1].chunk.session_id,
            }
            for n in cited
        ]
        yield {"type": "citations", "items": citations}

        t0 = time.perf_counter()
        verification = self._verify(answer, context) if mode == "llm" else lexical_verify(answer, context)
        timings["verification_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        yield {
            "type": "verification",
            "claims": verification.claims,
            "groundedness": verification.groundedness,
            "method": verification.method,
            "threshold": s.min_groundedness,
        }

        if not citations:
            status = "withheld"
            yield {"type": "withheld", "message": "The answer cited no sources, so it can't be trusted."}
        elif verification.groundedness is not None and verification.groundedness < s.min_groundedness:
            status = "withheld"
            yield {
                "type": "withheld",
                "message": f"Only {verification.groundedness:.0%} of the answer could be verified against the sources, so it is hidden by default.",
            }
        else:
            status = "answered"
        yield finish(status, answer, verification, [c["chunk_id"] for c in citations])

    def answer(self, question: str, session_id: str = "demo") -> dict:
        """Run the pipeline to completion and collect the events (used by evaluation)."""
        result: dict = {"answer": "", "claims": [], "groundedness": None, "citations": [], "retrieved": []}
        for event in self.run(question, session_id):
            kind = event["type"]
            if kind == "retrieval":
                result["retrieved"] = [c["chunk"]["doc_id"] for c in event["candidates"]]
                result["context_ids"] = event["context_ids"]
                result["top_relevance"] = event["top_relevance"]
            elif kind == "reset":
                result["answer"] = ""
            elif kind == "token":
                result["answer"] += event["text"]
            elif kind == "citations":
                result["citations"] = event["items"]
            elif kind == "verification":
                result["claims"], result["groundedness"] = event["claims"], event["groundedness"]
            elif kind == "done":
                result.update(status=event["status"], timings=event["timings"], mode=event["mode"])
        return result
