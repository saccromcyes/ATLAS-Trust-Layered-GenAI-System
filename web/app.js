"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);

// ---------- Session (uploads are scoped to it) ----------
function sessionId() {
  let id = null;
  try { id = localStorage.getItem("atlas-session"); } catch (_) { /* storage blocked */ }
  if (!id || !/^[A-Za-z0-9-]{8,64}$/.test(id)) {
    id = (crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try { localStorage.setItem("atlas-session", id); } catch (_) { /* fine: session lasts this page view */ }
  }
  return id;
}
const SESSION = sessionId();
const headers = (extra = {}) => ({ "X-Session-Id": SESSION, ...extra });

// ---------- Tabs ----------
const tabs = ["ask", "eval", "docs"];
function showTab(name, push = true) {
  for (const t of tabs) {
    const selected = t === name;
    const tab = $(`#tab-${t}`);
    tab.setAttribute("aria-selected", selected);
    tab.tabIndex = selected ? 0 : -1;
    $(`#panel-${t}`).hidden = !selected;
  }
  if (name === "eval") loadEval();
  if (name === "docs") loadDocs();
  if (push) history.replaceState(null, "", name === "ask" ? "#" : `#${name}`);
}
for (const t of tabs) $(`#tab-${t}`).addEventListener("click", () => showTab(t));
$(".tabs").addEventListener("keydown", (e) => {
  if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
  const current = tabs.findIndex((t) => $(`#tab-${t}`).getAttribute("aria-selected") === "true");
  const next = tabs[(current + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
  showTab(next);
  $(`#tab-${next}`).focus();
});
document.querySelectorAll("[data-goto]").forEach((el) => el.addEventListener("click", () => showTab(el.dataset.goto)));

// ---------- Config / mode badge ----------
let CONFIG = null;
async function loadConfig() {
  try {
    CONFIG = await (await fetch("/api/config")).json();
    const mode = $("#mode");
    if (CONFIG.mode === "llm") {
      mode.innerHTML = `<span class="dot"></span><span>Live: ${esc(CONFIG.llm_model)}</span>`;
    } else {
      mode.classList.add("fallback");
      mode.innerHTML = `<span class="dot"></span><span>Quote mode (no language model)</span>`;
      mode.title = "No LLM key is configured, so answers are quoted directly from the sources.";
    }
    $("#drop-limit").textContent = `PDF, Markdown or text, up to ${CONFIG.max_upload_mb} MB`;
  } catch (_) { /* the page still works without it */ }
}

// ---------- Answer rendering ----------
function renderAnswerText(text) {
  const lines = esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").split(/\n+/);
  let html = "", inList = false;
  for (const line of lines) {
    const item = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (item) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${item[1]}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim()) html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html.replace(/\[(\d{1,2})\]/g, (_, n) => `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`);
}

function focusSource(n, scroll = true) {
  document.querySelectorAll(".cite, .evidence li").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(`.cite[data-n="${n}"]`).forEach((el) => el.classList.add("active"));
  const card = document.querySelector(`.evidence li[data-n="${n}"]`);
  if (card) {
    card.classList.add("active");
    if (scroll) card.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  }
}
$("#result").addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (cite) focusSource(cite.dataset.n);
});
$("#result").addEventListener("mouseover", (e) => {
  const cite = e.target.closest(".cite");
  if (cite) focusSource(cite.dataset.n, false);
});

// Passages keep their markdown heading for retrieval; the heading is already shown above the text.
const displayText = (t) => t.split("\n").filter((l) => !/^\s*#{1,6}\s/.test(l)).join("\n").replace(/\*\*(.+?)\*\*/g, "$1").trim();

function renderEvidence(candidates, contextIds) {
  const byId = Object.fromEntries(candidates.map((c) => [c.chunk.chunk_id, c]));
  const list = $("#evidence");
  list.innerHTML = contextIds.map((id, i) => {
    const c = byId[id];
    if (!c) return "";
    const ch = c.chunk;
    const where = [ch.heading && ch.heading !== ch.doc_title ? ch.heading : null, ch.page ? `page ${ch.page}` : null].filter(Boolean).join(", ");
    return `<li data-n="${i + 1}">
      <div class="ev-head"><span class="ev-num">${i + 1}</span><div><div class="ev-title">${esc(ch.doc_title)}${ch.session_id !== "demo" ? ' <span class="tag">your upload</span>' : ""}</div>${where ? `<div class="ev-sub">${esc(where)}</div>` : ""}</div></div>
      <p class="ev-text clamped">${esc(displayText(ch.text))}</p>
      <div class="ev-foot"><span class="relevance" title="Cross-encoder relevance">Relevance <i><b style="width:${Math.round(c.rerank_score * 100)}%"></b></i> ${pct(c.rerank_score)}</span><button type="button" class="more">Show full passage</button></div>
    </li>`;
  }).join("");
  list.querySelectorAll(".more").forEach((btn) => btn.addEventListener("click", () => {
    const text = btn.closest("li").querySelector(".ev-text");
    const clamped = text.classList.toggle("clamped");
    btn.textContent = clamped ? "Show full passage" : "Show less";
  }));

  const contextSet = new Set(contextIds);
  const rank = (r) => (r == null ? "—" : r);
  $("#trace-table").innerHTML = `<thead><tr><th>Passage</th><th class="num">BM25 rank</th><th class="num">Vector rank</th><th class="num">Fused</th><th class="num">Final relevance</th></tr></thead><tbody>${
    candidates.map((c) => `<tr class="${contextSet.has(c.chunk.chunk_id) ? "in-context" : ""}">
      <td>${esc(c.chunk.doc_title)}${c.chunk.heading && c.chunk.heading !== c.chunk.doc_title ? `<div class="muted">${esc(c.chunk.heading)}</div>` : ""}</td>
      <td class="num">${rank(c.bm25_rank)}</td><td class="num">${rank(c.vector_rank)}</td>
      <td class="num">${(c.rrf_score * 1000).toFixed(1)}</td><td class="num">${pct(c.rerank_score)}</td></tr>`).join("")
  }</tbody>`;
}

function markCited(citations) {
  const cited = new Set(citations.map((c) => String(c.n)));
  document.querySelectorAll(".evidence li").forEach((li) => li.classList.toggle("cited", cited.has(li.dataset.n)));
}

function renderVerification(v) {
  const claims = v.claims || [];
  if (!claims.length) return;
  $("#proof").hidden = false;
  const counts = { supported: 0, partial: 0, unsupported: 0 };
  claims.forEach((c) => counts[c.verdict]++);
  $("#proof-score").textContent = `${pct(v.groundedness)} grounded`;
  $("#proof-bar").innerHTML = claims.map((c) => `<span class="v-${c.verdict}" title="${esc(c.verdict)}: ${esc(c.claim)}"></span>`).join("");
  $("#proof-bar").setAttribute("aria-label", `${counts.supported} supported, ${counts.partial} partly supported, ${counts.unsupported} unsupported claims`);
  const label = { supported: "Supported", partial: "Partly supported", unsupported: "Not supported" };
  $("#claims").innerHTML = claims.map((c) => `<li>
    <span class="mark v-${c.verdict}" aria-label="${label[c.verdict]}"></span>
    <div><span class="claim-text">${esc(c.claim)}</span>
    <span class="claim-meta">${label[c.verdict]}${c.sources?.length ? ` by source ${c.sources.map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`).join("")}` : ""}</span>
    ${c.reason ? `<span class="claim-reason">${esc(c.reason)}</span>` : ""}</div>
  </li>`).join("");
  $("#proof-note").textContent = v.method === "llm-judge"
    ? `Each claim was checked against the evidence by ${CONFIG?.verifier_model || "a second model"}. Answers below ${pct(v.threshold)} grounded are hidden.`
    : `Claims were checked by word overlap with the cited passages. Answers below ${pct(v.threshold)} grounded are hidden.`;
}

function renderTimings(t, mode) {
  const s = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
  const parts = [];
  const retrieval = (t.vector_ms || 0) + (t.bm25_ms || 0);
  if (retrieval) parts.push(`searched in ${s(retrieval)}`);
  if (t.rerank_ms) parts.push(`reranked in ${s(t.rerank_ms)}`);
  if (t.generation_ms) parts.push(`${mode === "llm" ? "answered" : "quoted"} in ${s(t.generation_ms)}`);
  if (t.verification_ms) parts.push(`verified in ${s(t.verification_ms)}`);
  $("#timings").textContent = parts.length ? `${parts.join(", ").replace(/^./, (c) => c.toUpperCase())}. Total ${s(t.total_ms)}.` : "";
}

function setStatus(text, busy = false) {
  $("#status").innerHTML = text ? `${busy ? '<span class="spinner" aria-hidden="true"></span>' : ""}<span>${esc(text)}</span>` : "";
}

function resetResult() {
  $("#result").hidden = false;
  for (const id of ["#refusal", "#withheld", "#proof"]) $(id).hidden = true;
  $("#answer").innerHTML = "";
  $("#answer").className = "answer";
  $("#evidence").innerHTML = "";
  $("#trace-table").innerHTML = "";
  $("#claims").innerHTML = "";
  $("#timings").textContent = "";
}

// ---------- Ask ----------
let busy = false;
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  $("#ask-btn").disabled = true;
  resetResult();
  setStatus("Searching the documents…", true);
  let answer = "";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = Array.isArray(body.detail) ? "Questions need between 3 and 500 characters." : body.detail;
      throw new Error(detail || `Request failed (${res.status}).`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const ev = JSON.parse(part.slice(6));
        switch (ev.type) {
          case "retrieval":
            renderEvidence(ev.candidates, ev.context_ids);
            setStatus("Writing an answer from the evidence…", true);
            break;
          case "notice":
            setStatus(ev.message, true);
            break;
          case "reset":
            answer = "";
            $("#answer").innerHTML = "";
            break;
          case "token":
            answer += ev.text;
            $("#answer").classList.add("streaming");
            $("#answer").innerHTML = renderAnswerText(answer);
            break;
          case "citations":
            $("#answer").classList.remove("streaming");
            markCited(ev.items);
            setStatus("Checking each claim against the evidence…", true);
            break;
          case "verification":
            renderVerification(ev);
            break;
          case "refusal": {
            $("#answer").classList.remove("streaming");
            const r = $("#refusal");
            r.hidden = false;
            const gauge = ev.reason === "low_relevance"
              ? `<div class="gauge">Best match relevance ${pct(ev.top_relevance)}, threshold ${pct(ev.threshold)}<div class="gauge-track"><div class="gauge-fill" style="width:${Math.min(100, Math.round(ev.top_relevance * 100))}%"></div><div class="gauge-mark" style="left:${Math.round(ev.threshold * 100)}%"></div></div></div>`
              : "";
            r.innerHTML = `<h3>Not in the documents</h3><p>${esc(ev.message)} The closest passages are listed under Evidence.</p>${gauge}`;
            break;
          }
          case "withheld": {
            const w = $("#withheld");
            w.hidden = false;
            $("#answer").hidden = true;
            w.innerHTML = `<h3>Answer hidden: not enough of it checks out</h3><p>${esc(ev.message)}</p><button type="button" class="ghost" id="reveal">Show the unverified answer</button>`;
            $("#reveal").addEventListener("click", () => {
              $("#answer").hidden = false;
              $("#answer").classList.add("dimmed");
              $("#reveal").remove();
            });
            break;
          }
          case "done":
            setStatus("");
            renderTimings(ev.timings, ev.mode);
            break;
          case "error":
            throw new Error(ev.message);
        }
      }
    }
  } catch (err) {
    setStatus("");
    const r = $("#refusal");
    r.hidden = false;
    r.innerHTML = `<h3>Couldn't get an answer</h3><p>${esc(err.message || "The connection dropped.")} Try again in a moment.</p>`;
  } finally {
    $("#answer").classList.remove("streaming");
    busy = false;
    $("#ask-btn").disabled = false;
  }
}

const q = $("#question");
function autosize() { q.style.height = "auto"; q.style.height = `${Math.min(q.scrollHeight, 160)}px`; }
q.addEventListener("input", autosize);
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});
$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  $("#answer").hidden = false;
  ask(q.value.trim());
});
$("#suggestions").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  q.value = btn.textContent;
  autosize();
  $("#answer").hidden = false;
  ask(q.value);
});

// ---------- Evaluation ----------
let evalLoaded = false;
async function loadEval() {
  if (evalLoaded) return;
  const body = $("#eval-body");
  try {
    const data = await (await fetch("/api/eval")).json();
    if (!data.available) {
      body.innerHTML = `<p class="muted">No evaluation results have been published yet. They appear here after the first automated test run.</p>`;
      return;
    }
    evalLoaded = true;
    body.innerHTML = renderEval(data);
    bindEvalFilter(data);
  } catch (_) {
    body.innerHTML = `<p class="muted">Couldn't load the results. Reload the page to try again.</p>`;
  }
}

function renderEval(d) {
  const date = (iso) => new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  let html = "";
  if (d.answers) {
    const m = d.answers.metrics;
    html += `<dl class="metrics">
      <div class="metric"><dt>Correct answers</dt><dd>${pct(m.answer_accuracy)}<small>${d.answers.n_answerable} questions the documents can answer</small></dd></div>
      <div class="metric"><dt>Correct refusals</dt><dd>${pct(m.refusal_accuracy)}<small>${d.answers.n_unanswerable} questions they can't answer</small></dd></div>
      <div class="metric"><dt>Citation accuracy</dt><dd>${pct(m.citation_accuracy)}<small>citations pointing at the right document</small></dd></div>
      <div class="metric"><dt>Median response time</dt><dd>${m.latency_p50_ms != null ? (m.latency_p50_ms / 1000).toFixed(1) + " s" : "—"}<small>95th percentile ${m.latency_p95_ms != null ? (m.latency_p95_ms / 1000).toFixed(1) + " s" : "—"}</small></dd></div>
    </dl>`;
  }
  const names = { bm25: "Keyword search (BM25)", vector: "Meaning search (embeddings)", hybrid: "Both, merged (RRF)", "hybrid+rerank": "Both, merged, then reranked" };
  const modes = Object.keys(d.retrieval.by_mode);
  const best = modes.reduce((a, b) => (d.retrieval.by_mode[b]["mrr@10"] > d.retrieval.by_mode[a]["mrr@10"] ? b : a));
  const bar = (x) => `<span class="bar-cell"><i><b style="width:${Math.round(x * 100)}%"></b></i>${pct(x)}</span>`;
  html += `<section class="eval-section"><h2>Which retrieval strategy finds the evidence</h2>
    <p>For each question, is the passage that states the answer among the top results? Measured on ${d.retrieval.n_questions} questions across ${d.retrieval.n_passages} passages.</p>
    <div class="table-wrap"><table><thead><tr><th>Strategy</th><th>Top 1</th><th>Top 3</th><th>Top 5</th><th class="num">MRR@10</th></tr></thead><tbody>${
      modes.map((mode) => { const r = d.retrieval.by_mode[mode]; return `<tr class="${mode === best ? "best" : ""}"><td>${names[mode] || mode}</td><td>${bar(r["hit@1"])}</td><td>${bar(r["hit@3"])}</td><td>${bar(r["hit@5"])}</td><td class="num">${r["mrr@10"].toFixed(3)}</td></tr>`; }).join("")
    }</tbody></table></div></section>`;

  if (d.answers) {
    html += `<section class="eval-section"><h2>Every test question</h2>
      <div class="filter" id="eval-filter">${["all", "answerable", "multi_hop", "paraphrase", "unanswerable", "failed"].map((f, i) => `<button type="button" data-f="${f}" aria-pressed="${i === 0}">${{ all: "All", answerable: "Direct", multi_hop: "Two facts combined", paraphrase: "Paraphrased", unanswerable: "Not answerable", failed: "Failed only" }[f]}</button>`).join("")}</div>
      <div class="table-wrap"><table><thead><tr><th>Question</th><th>Result</th><th>Outcome</th></tr></thead><tbody id="eval-rows"></tbody></table></div></section>`;
  }
  const c = d.config;
  html += `<section class="eval-section"><h2>Setup tested</h2><dl class="config-list">
    <dt>Answer model</dt><dd>${esc(c.llm)}</dd><dt>Claim checker</dt><dd>${esc(c.verifier)}</dd>
    <dt>Embeddings</dt><dd>${esc(c.embedder)}</dd><dt>Reranker</dt><dd>${esc(c.reranker)}</dd>
    <dt>Prompt version</dt><dd>${esc(c.prompt_version)}</dd>
    <dt>Retrieval tested</dt><dd>${date(d.run_at)}, commit ${esc(d.commit)}</dd>
    ${d.answers_run_at ? `<dt>Answers tested</dt><dd>${date(d.answers_run_at)}</dd>` : ""}
  </dl></section>`;
  return html;
}

function bindEvalFilter(d) {
  if (!d.answers) return;
  const rows = d.answers.questions;
  const outcome = { answered: "Answered", refused: "Declined", withheld: "Hidden (low groundedness)", error: "Error" };
  const draw = (f) => {
    const shown = rows.filter((r) => f === "all" || (f === "failed" ? !r.correct : r.type === f));
    $("#eval-rows").innerHTML = shown.map((r) => `<tr><td>${esc(r.question)}</td><td><span class="pill ${r.correct ? "ok" : "bad"}">${r.correct ? "Pass" : "Fail"}</span></td><td>${outcome[r.status] || esc(r.status)}${r.groundedness != null && r.status === "answered" ? `, ${pct(r.groundedness)} grounded` : ""}</td></tr>`).join("")
      || `<tr><td colspan="3" class="muted">No questions in this group.</td></tr>`;
  };
  draw("all");
  $("#eval-filter").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    document.querySelectorAll("#eval-filter button").forEach((b) => b.setAttribute("aria-pressed", b === btn));
    draw(btn.dataset.f);
  });
}

// ---------- Documents ----------
async function loadDocs() {
  try {
    const docs = await (await fetch("/api/documents", { headers: headers() })).json();
    $("#docs").innerHTML = docs.map((d) => `<li><span><span class="doc-title">${esc(d.title)}</span>${d.uploaded ? '<span class="tag">your upload</span>' : ""}</span><span class="muted">${d.chunks} passages</span></li>`).join("");
    $("#clear-uploads").hidden = !docs.some((d) => d.uploaded);
  } catch (_) {
    $("#docs").innerHTML = `<li class="muted">Couldn't load the document list.</li>`;
  }
}

async function upload(file) {
  const status = $("#upload-status");
  status.className = "upload-status";
  status.textContent = `Reading and indexing ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/documents", { method: "POST", headers: headers(), body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Upload failed (${res.status}).`);
    status.textContent = `Added “${body.title}” (${body.chunks} passages). Ask a question about it on the Ask tab.`;
    loadDocs();
  } catch (err) {
    status.className = "upload-status error";
    status.textContent = err.message;
  }
}
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; });
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
$("#clear-uploads").addEventListener("click", async () => {
  await fetch("/api/session", { method: "DELETE", headers: headers() });
  $("#upload-status").textContent = "Your uploads were deleted.";
  loadDocs();
});

// ---------- Boot ----------
loadConfig();
const initial = location.hash.slice(1);
showTab(tabs.includes(initial) ? initial : "ask", false);
