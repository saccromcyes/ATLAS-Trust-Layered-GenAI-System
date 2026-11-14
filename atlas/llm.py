"""LLM access through any OpenAI-compatible API, plus an extractive fallback that needs no LLM.

Swapping providers is a config change (base URL + model), not a code change. If the provider
is down, rate-limited or not configured, ATLAS degrades to extractive answers instead of failing.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Iterator
from typing import Protocol

from atlas.config import Settings
from atlas.models import tokenize

log = logging.getLogger(__name__)

REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    mode: str

    def stream(self, messages: list[dict], model: str) -> Iterator[str]: ...

    def complete_json(self, messages: list[dict], model: str) -> dict: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def provider_options(base_url: str, reasoning_effort: str) -> dict:
    """Extra request options. gpt-oss models reason before answering: keep that short, and on Groq
    leave the reasoning text out of the response (it still counts toward the output tokens)."""
    options: dict = {}
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    if "groq.com" in base_url:
        options["extra_body"] = {"include_reasoning": False}
    return options


_FULLWIDTH_CITATION = re.compile(r"【(\d{1,2})(?:†[^】]*)?】")
_PLAIN_CHARS = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-"})


_SPACE_BEFORE_PERCENT = re.compile(r"[\u202f\u00a0](?=%)")


def _tidy(text: str) -> str:
    text = _SPACE_BEFORE_PERCENT.sub("", text)
    return _FULLWIDTH_CITATION.sub(r"[\1]", text.translate(_PLAIN_CHARS)).replace("**", "")


def normalise_stream(deltas: Iterable[str]) -> Iterator[str]:
    """Rewrite model output into the plain form the pipeline parses, as it streams.

    gpt-oss models cite as 【1】 or 【1†L3-L5】 (their training format) rather than [1], use
    narrow no-break spaces and hyphens inside numbers ("4\u202f%"), and bold with **. A
    citation, a "**" or a space before "%" can arrive split across deltas, so the unfinished
    part is held back until the next delta shows what it is.
    """
    pending = ""
    for delta in deltas:
        pending += delta
        cut = len(pending)
        opened = pending.rfind("【")
        if opened != -1 and "】" not in pending[opened:] and cut - opened < 24:
            cut = opened
        elif pending.endswith(("*", "\u202f", "\u00a0")):
            cut -= 1
        if cut:
            yield _tidy(pending[:cut])
            pending = pending[cut:]
    if pending:
        yield _tidy(pending)


class OpenAICompatibleLLM:
    mode = "llm"

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        self._options = provider_options(settings.llm_base_url, settings.llm_reasoning_effort)

    def stream(self, messages: list[dict], model: str) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=model, messages=messages, temperature=0.1, max_tokens=2000, stream=True, **self._options
            )
            yield from normalise_stream(
                event.choices[0].delta.content for event in response if event.choices and event.choices[0].delta.content
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def complete_json(self, messages: list[dict], model: str) -> dict:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2500,
                response_format={"type": "json_object"},
                **self._options,
            )
            return parse_json(response.choices[0].message.content or "{}")
        except Exception as exc:
            raise LLMError(str(exc)) from exc


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    # Move citations that follow a full stop ("... taxes. [1]") inside the sentence they support.
    text = re.sub(r"([.!?])\s*((?:\[\d{1,2}\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)
    cleaned = re.sub(r"[*_`#|]+", " ", text)
    return [s.strip(" -•\t") for s in _SENTENCE.split(cleaned) if len(s.strip()) > 15]


class ExtractiveLLM:
    """Answers by quoting the source sentences that best cover the question.

    Used when no API key is configured or the provider fails. It cannot synthesise across
    sources, but every sentence it returns is copied from evidence, so it never hallucinates.
    """

    mode = "fallback"

    min_coverage = 0.3  # tuned on eval/golden.jsonl

    def answer(self, question: str, sources: list[str]) -> str:
        q_terms = set(tokenize(question))
        sentences = [(n, s) for n, source in enumerate(sources, start=1) for s in split_sentences(source)]
        if not q_terms or not sentences:
            return REFUSAL_TOKEN
        # Weight query terms by rarity across the evidence: a question about the "CEO" is not
        # answered by sentences that merely mention the company name.
        term_sets = [set(tokenize(s)) for _, s in sentences]
        n_sent = len(sentences)
        idf = {t: math.log(1 + n_sent / (1 + sum(t in ts for ts in term_sets))) for t in q_terms}
        total = sum(idf.values())
        scored = []
        for (n, sentence), terms in zip(sentences, term_sets):
            coverage = sum(idf[t] for t in q_terms & terms) / total
            if coverage > 0:
                scored.append((coverage, -n, sentence, n))
        if not scored:
            return REFUSAL_TOKEN
        scored.sort(reverse=True)
        best = scored[0][0]
        if best < self.min_coverage:
            return REFUSAL_TOKEN
        picked = [item for item in scored if item[0] >= best * 0.6][:3]
        return " ".join(f"{sentence.rstrip('.')}. [{n}]" for _, _, sentence, n in picked)

    def stream(self, messages: list[dict], model: str) -> Iterator[str]:
        raise NotImplementedError("Use answer() for extractive mode")

    def complete_json(self, messages: list[dict], model: str) -> dict:
        raise NotImplementedError


def build_llm(settings: Settings) -> LLM | ExtractiveLLM:
    if settings.llm_enabled:
        return OpenAICompatibleLLM(settings)
    log.warning("No LLM_API_KEY set: running in extractive fallback mode")
    return ExtractiveLLM()
