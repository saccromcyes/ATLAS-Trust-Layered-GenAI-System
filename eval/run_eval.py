"""Evaluate ATLAS on the golden set.

    python -m eval.run_eval --retrieval-only        # no API key needed; compares retrieval strategies
    python -m eval.run_eval                         # full run: answers, citations, refusals, groundedness
    python -m eval.run_eval --fail-under-hit5 0.9   # exit non-zero on regression (used in CI)

Results go to eval/results/latest.json (served by /api/eval) and eval/results/latest.md, and the
summary table in README.md is refreshed between the EVAL markers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

from atlas.config import ROOT, Settings
from atlas.ingestion import load_directory
from atlas.llm import build_llm
from atlas.models import build_embedder, build_reranker
from atlas.pipeline import AnswerPipeline
from atlas.prompts import PROMPT_VERSION
from atlas.retrieval import KnowledgeBase
from atlas.store import MetadataStore

GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS = ROOT / "eval" / "results"
README = ROOT / "README.md"
MODES = ["bm25", "vector", "hybrid", "hybrid+rerank"]


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def includes_all(answer: str, groups: list[str]) -> bool:
    text = normalise(answer)
    return all(any(alt.strip() in text for alt in group.split("|")) for group in groups)


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def build_kb(settings: Settings) -> KnowledgeBase:
    store = MetadataStore(settings.database_url)
    kb = KnowledgeBase(settings, build_embedder(settings), build_reranker(settings), store)
    for doc in load_directory(settings.corpus_dir):
        kb.add_document(doc)
    return kb


def is_evidence(chunk, g: dict) -> bool:
    """A passage is evidence when it comes from an expected document and states every key fact."""
    return chunk.doc_id in g["expected_docs"] and includes_all(chunk.text, g["must_include"])


def evaluate_retrieval(kb: KnowledgeBase, golden: list[dict]) -> dict:
    all_chunks = kb.store.chunks(["demo"])
    # Only questions whose answer is stated in a single passage (computed answers are excluded).
    answerable = [g for g in golden if g["expected_docs"] and any(is_evidence(c, g) for c in all_chunks)]
    table = {}
    for mode in MODES:
        hits = {1: 0, 3: 0, 5: 0}
        reciprocal_ranks = []
        for g in answerable:
            ranking = kb.retrieve_ablation(g["question"], mode, k=10)
            first = next((i for i, c in enumerate(ranking, start=1) if is_evidence(c, g)), None)
            reciprocal_ranks.append(1 / first if first else 0.0)
            for k in hits:
                hits[k] += bool(first and first <= k)
        n = len(answerable)
        table[mode] = {
            "hit@1": round(hits[1] / n, 3),
            "hit@3": round(hits[3] / n, 3),
            "hit@5": round(hits[5] / n, 3),
            "mrr@10": round(sum(reciprocal_ranks) / n, 3),
        }
    return {"n_questions": len(answerable), "n_passages": len(all_chunks), "by_mode": table}


SIMILARITY_THRESHOLDS = (0.55, 0.6, 0.65, 0.7)


def evaluate_relevance_gate(kb: KnowledgeBase, golden: list[dict], min_relevance: float) -> dict:
    """Score every question's best passage, to show what each refusal-gate setting would do.

    The gate refuses without calling the model. Set too strictly it refuses reworded questions
    it could answer; set too loosely the model does all the refusing (which it can, at a cost).
    """
    scores = []
    for g in golden:
        r = kb.retrieve(g["question"])
        scores.append(
            {
                "id": g["id"],
                "type": g["type"],
                "top_relevance": round(r.top_relevance, 4),
                "top_similarity": round(r.top_similarity, 4),
            }
        )
    answerable = [x for x in scores if x["type"] != "unanswerable"]
    unanswerable = [x for x in scores if x["type"] == "unanswerable"]

    def row(rule: str, stops, similarity: float | None) -> dict:
        return {
            "rule": rule,
            "min_similarity": similarity,
            "answerable_passed": round(sum(not stops(x) for x in answerable) / len(answerable), 3),
            "unanswerable_stopped": round(sum(stops(x) for x in unanswerable) / len(unanswerable), 3),
        }

    table = [row(f"reranker < {min_relevance:g}", lambda x: x["top_relevance"] < min_relevance, None)]
    for sim in SIMILARITY_THRESHOLDS:
        table.append(
            row(
                f"reranker < {min_relevance:g} and similarity < {sim:g}",
                lambda x, sim=sim: x["top_relevance"] < min_relevance and x["top_similarity"] < sim,
                sim,
            )
        )
    return {"by_rule": table, "questions": scores}


def evaluate_answers(pipeline: AnswerPipeline, golden: list[dict], pause_s: float) -> dict:
    rows = []
    for g in golden:
        result = pipeline.answer(g["question"])
        refused = result["status"] in ("refused", "withheld")
        cited_docs = [c["doc_id"] for c in result["citations"]]
        row = {
            "id": g["id"],
            "type": g["type"],
            "question": g["question"],
            "status": result["status"],
            "answer": result["answer"],
            "groundedness": result["groundedness"],
            "cited_docs": cited_docs,
            "top_relevance": result.get("top_relevance"),
            "latency_ms": result["timings"].get("total_ms"),
            "mode": result["mode"],
        }
        if g["type"] == "unanswerable":
            row["correct"] = refused
        else:
            row["correct"] = (not refused) and includes_all(result["answer"], g["must_include"])
            row["citation_accuracy"] = sum(d in g["expected_docs"] for d in cited_docs) / len(cited_docs) if cited_docs else None
        rows.append(row)
        print(f"  {g['id']:>4}  {'✓' if row['correct'] else '✗'}  {result['status']:<9} {g['question'][:70]}", flush=True)
        time.sleep(pause_s)

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in rows if r["status"] == "answered"]
    citation_scores = [r["citation_accuracy"] for r in answerable if r.get("citation_accuracy") is not None]
    ground = [r["groundedness"] for r in answered if r["groundedness"] is not None]
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    false_answers = sum(r["status"] == "answered" for r in unanswerable)
    return {
        "metrics": {
            "answer_accuracy": round(sum(r["correct"] for r in answerable) / len(answerable), 3),
            "multi_hop_accuracy": round(
                sum(r["correct"] for r in answerable if r["type"] == "multi_hop")
                / max(1, sum(r["type"] == "multi_hop" for r in answerable)),
                3,
            ),
            "refusal_accuracy": round(sum(r["correct"] for r in unanswerable) / len(unanswerable), 3),
            "false_refusal_rate": round(sum(r["status"] != "answered" for r in answerable) / len(answerable), 3),
            "hallucinated_answers": false_answers,
            "citation_accuracy": round(statistics.mean(citation_scores), 3) if citation_scores else None,
            "mean_groundedness": round(statistics.mean(ground), 3) if ground else None,
            "latency_p50_ms": round(statistics.median(latencies)) if latencies else None,
            "latency_p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))]) if latencies else None,
        },
        "questions": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [f"_Last run {report['run_at'][:10]} · commit `{report['commit']}` · {report['config']['summary']}_", ""]
    if "answers" in report:
        m = report["answers"]["metrics"]
        lines += [
            "| Metric | Result |",
            "|---|---|",
            f"| Answer accuracy ({report['answers']['n_answerable']} answerable questions) | **{pct(m['answer_accuracy'])}** |",
            f"| Multi-hop accuracy | {pct(m['multi_hop_accuracy'])} |",
            f"| Correct refusals ({report['answers']['n_unanswerable']} unanswerable questions) | **{pct(m['refusal_accuracy'])}** |",
            f"| Hallucinated answers to unanswerable questions | {m['hallucinated_answers']} |",
            f"| Citation accuracy | {pct(m['citation_accuracy'])} |",
            f"| Mean groundedness of shown answers | {pct(m['mean_groundedness'])} |",
            f"| Latency p50 / p95 | {m['latency_p50_ms']} ms / {m['latency_p95_ms']} ms |",
            "",
        ]
    gate = report.get("relevance_gate")
    if gate and "by_rule" in gate:
        current = report["config"].get("min_similarity")
        lines += [
            "**Refusal gate**: which questions each rule would refuse before calling the model. The rest are answered, or refused by the model itself.",
            "",
            "| Refuse when | Answerable questions let through | Unanswerable questions stopped early |",
            "|---|---|---|",
        ]
        for row in gate["by_rule"]:
            on = row["min_similarity"] is not None and current is not None and abs(row["min_similarity"] - current) < 1e-9
            mark = "**" if on else ""
            lines.append(
                f"| {mark}{row['rule']}{' (current)' if on else ''}{mark} | {pct(row['answerable_passed'])} | {pct(row['unanswerable_stopped'])} |"
            )
        lines.append("")
    r = report["retrieval"]["by_mode"]
    lines += [
        f"**Retrieval ablation**: is the passage that states the answer in the top *k*? ({report['retrieval']['n_questions']} questions, {report['retrieval'].get('n_passages', '?')} passages)",
        "",
        "| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    names = {"bm25": "BM25 only", "vector": "Vector only", "hybrid": "Hybrid (RRF)", "hybrid+rerank": "Hybrid + cross-encoder"}
    for mode in MODES:
        row = r[mode]
        bold = "**" if mode == "hybrid+rerank" else ""
        lines.append(
            f"| {bold}{names[mode]}{bold} | {pct(row['hit@1'])} | {pct(row['hit@3'])} | {pct(row['hit@5'])} | {row['mrr@10']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def update_readme(markdown: str) -> None:
    if not README.exists():
        return
    text = README.read_text()
    pattern = re.compile(r"(<!-- EVAL:START -->\n).*?(<!-- EVAL:END -->)", flags=re.DOTALL)
    if pattern.search(text):
        README.write_text(pattern.sub(lambda m: m.group(1) + markdown + m.group(2), text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--fail-under-hit5", type=float, default=None)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between questions (rate limits)")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp()
    settings = Settings(qdrant_path=":memory:", database_url=f"sqlite:///{tmp}/eval.db")
    kb = build_kb(settings)
    golden = load_golden()
    llm = build_llm(settings)
    config = {
        "embedder": settings.embed_model if type(kb.embedder).__name__ == "FastEmbedEmbedder" else "hashing",
        "reranker": settings.rerank_model if kb.reranker.name == "cross-encoder" else "lexical",
        "llm": settings.llm_model if llm.mode == "llm" else "extractive fallback",
        "verifier": settings.llm_fast_model if llm.mode == "llm" else "lexical",
        "prompt_version": PROMPT_VERSION,
        "min_relevance": settings.min_relevance,
        "min_similarity": settings.min_similarity,
        "min_groundedness": settings.min_groundedness,
    }
    config["summary"] = f"{config['llm']} · {config['embedder']} · {config['reranker']}"

    print("Retrieval ablation…")
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "config": config,
        "retrieval": evaluate_retrieval(kb, golden),
    }
    for mode, row in report["retrieval"]["by_mode"].items():
        print(f"  {mode:<14} {row}")
    print("Refusal gate…")
    report["relevance_gate"] = evaluate_relevance_gate(kb, golden, settings.min_relevance)
    for row in report["relevance_gate"]["by_rule"]:
        print(f"  {row}")
    for q in sorted(report["relevance_gate"]["questions"], key=lambda q: q["top_relevance"]):
        print(f"  {q['top_relevance']:.4f}  sim {q['top_similarity']:.4f}  {q['type']:<12} {q['id']}")

    if not args.retrieval_only:
        print("Answer evaluation…")
        pipeline = AnswerPipeline(settings, kb, llm, kb.store)
        answers = evaluate_answers(pipeline, golden, args.pause)
        answers["n_answerable"] = sum(g["type"] != "unanswerable" for g in golden)
        answers["n_unanswerable"] = sum(g["type"] == "unanswerable" for g in golden)
        print(json.dumps(answers["metrics"], indent=2))
        # Never publish LLM metrics that were really produced by the extractive fallback.
        fell_back = [q["id"] for q in answers["questions"] if q["mode"] == "fallback"]
        if llm.mode == "llm" and fell_back:
            print(
                f"{len(fell_back)} of {len(golden)} answers fell back to extractive mode: {', '.join(fell_back)}", file=sys.stderr
            )
            if len(fell_back) > 0.1 * len(golden):
                print("FAIL: the language model was unavailable for too many questions; results not published.", file=sys.stderr)
                return 1
        # An answer withheld for citing nothing means the model's citation format changed, not that it was wrong.
        uncited = [q["id"] for q in answers["questions"] if q["status"] == "withheld" and not q["cited_docs"]]
        if llm.mode == "llm" and len(uncited) > 0.1 * len(golden):
            print(
                f"FAIL: {len(uncited)} answers cited no sources ({', '.join(uncited)}); check the citation format.",
                file=sys.stderr,
            )
            return 1
        report["answers"] = answers

    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        previous = json.loads((RESULTS / "latest.json").read_text()) if (RESULTS / "latest.json").exists() else {}
        if args.retrieval_only and "answers" in previous:
            report["answers"] = previous["answers"]  # keep the last full answer evaluation
            report["answers_run_at"] = previous.get("answers_run_at", previous.get("run_at"))
        elif "answers" in report:
            report["answers_run_at"] = report["run_at"]
        (RESULTS / "latest.json").write_text(json.dumps(report, indent=2))
        markdown = render_markdown(report)
        (RESULTS / "latest.md").write_text(markdown)
        update_readme(markdown)
        print(f"Wrote {RESULTS / 'latest.json'}")

    hit5 = report["retrieval"]["by_mode"]["hybrid+rerank"]["hit@5"]
    if args.fail_under_hit5 is not None and hit5 < args.fail_under_hit5:
        print(f"FAIL: hybrid+rerank hit@5 {hit5:.3f} < {args.fail_under_hit5}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
