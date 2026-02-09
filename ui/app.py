import os
from datetime import datetime

import requests
import streamlit as st

API = os.getenv("ATLAS_API_URL", "http://backend:8000")

st.set_page_config(page_title="ATLAS", layout="wide")
st.markdown(
    """
    <style>
        .atlas-hero {
            background: linear-gradient(120deg, #1b2338, #101827);
            padding: 24px 28px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            color: #f8fafc;
        }
        .atlas-chip {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(56,189,248,0.18);
            color: #7dd3fc;
            font-weight: 600;
            margin-right: 6px;
            font-size: 0.8rem;
        }
        .atlas-card {
            background: rgba(15,23,42,0.8);
            padding: 18px;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="atlas-hero">
        <div class="atlas-chip">Hybrid RAG</div>
        <div class="atlas-chip">Verification</div>
        <div class="atlas-chip">Evaluation</div>
        <h1>ATLAS Mission Control</h1>
        <p>Evidence-first answers with measurable groundedness, latency, and claim-level validation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def _safe_get(url: str, **kwargs):
    try:
        r = requests.get(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.warning(f"Unable to reach backend: {exc}")
        return None

def _safe_post(url: str, **kwargs):
    try:
        r = requests.post(url, timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None

stats = _safe_get(f"{API}/stats") or {}
documents = _safe_get(f"{API}/documents") or []
recent_runs = _safe_get(f"{API}/runs/recent") or []

with st.sidebar:
    st.header("Indexing")
    st.caption("Ingest sample docs or upload your own PDF/TXT/MD.")

    if st.button("Ingest sample_docs"):
        out = _safe_post(f"{API}/ingest/folder", params={"path": "/app/sample_docs"})
        if out:
            st.success(out)

    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Ingest uploaded file"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        out = _safe_post(f"{API}/ingest/upload", files=files)
        if out:
            st.success(out)

    top_k = st.slider("Top-K evidence", 3, 12, 8)
    st.divider()
    st.subheader("System Snapshot")
    st.metric("Documents", stats.get("documents", 0))
    st.metric("Chunks", stats.get("chunks", 0))
    st.metric("Runs", stats.get("runs", 0))
    st.metric("Avg latency (s)", stats.get("avg_latency_sec") or 0)
    st.metric("Avg groundedness", stats.get("avg_groundedness") or 0)
    health = _safe_get(f"{API}/health") or {}
    st.caption(f"Backend status: {health.get('status', 'unknown')}")
    st.caption(f"Environment: {health.get('env', 'unknown')}")

col_a, col_b, col_c = st.columns([1.3, 1, 1], gap="large")

with col_a:
    st.subheader("Ask the system")
    st.write("Probe the knowledge base and see proof-backed answers.")
    q = st.text_input(
        "Question",
        placeholder="e.g., What are the working hours and PTO carry-forward rules?",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("Ask ATLAS", type="primary")

with col_b:
    st.subheader("Evidence dial")
    st.write("Adjust how much evidence the system pulls for each answer.")
    st.metric("Top-K evidence", top_k)
    st.progress(min(top_k / 12, 1.0))
    st.caption("Higher = more evidence, but slower response.")
    if stats:
        st.caption(f"Avg groundedness: {stats.get('avg_groundedness') or 0:.2f}")

with col_c:
    st.subheader("System heartbeat")
    st.write("Latest run timing and reliability.")
    last_run_at = stats.get("last_run_at")
    if last_run_at:
        st.caption(f"Last run at {last_run_at}")
    st.metric("Avg latency (s)", stats.get("avg_latency_sec") or 0)
    st.metric("Avg groundedness", stats.get("avg_groundedness") or 0)

st.divider()

intro_left, intro_right = st.columns([1.4, 1], gap="large")
with intro_left:
    st.markdown(
        """
        <div class="atlas-card">
            <h3>Showcase Checklist</h3>
            <ul>
                <li>Ingest sample docs to light up hybrid retrieval.</li>
                <li>Ask a policy or process question to trigger verification.</li>
                <li>Inspect citations and claim verdicts to prove grounding.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with intro_right:
    st.markdown(
        """
        <div class="atlas-card">
            <h3>Demo Prompts</h3>
            <ul>
                <li>“What are the working hours?”</li>
                <li>“How many leave days can be carried forward?”</li>
                <li>“Which security controls are required?”</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "history" not in st.session_state:
    st.session_state.history = []

if ask_clicked and q.strip():
    with st.spinner("Thinking with evidence..."):
        out = _safe_post(f"{API}/ask", json={"question": q, "top_k": top_k})
    if out:
        st.session_state.history.append((q, out))

tab_overview, tab_docs, tab_runs = st.tabs(["Answer Workspace", "Evidence Library", "Run Ledger"])

with tab_overview:
    if not st.session_state.history:
        st.info("Ask a question to see answers with citations and verification details.")
    for q, out in reversed(st.session_state.history[-8:]):
        st.subheader("Question")
        st.write(q)

        ver = out.get("verification", {}) or {}
        col1, col2 = st.columns([2, 1], gap="large")

        summary = out.get("verification_summary") or {}
        retrieval = out.get("retrieval_summary") or {}
        quality = out.get("quality_summary") or {}
        support_ratio = summary.get("support_ratio", 0.0)
        supported_claims = summary.get("supported_claims", 0)
        total_claims = summary.get("total_claims", 0)
        trust_score = quality.get("trust_score", 0.0)

        with col1:
            st.subheader("Answer")
            st.write(out.get("answer", ""))
            st.caption(f"Latency: {out.get('latency_sec') or 0:.2f} sec")

            st.subheader("Verification")
            st.metric("Trust score", float(trust_score or 0.0))
            st.metric("Groundedness", float(ver.get("groundedness", 0.0) or 0.0))
            st.metric("Claims supported", f"{supported_claims}/{total_claims}")
            st.progress(min(float(support_ratio), 1.0))
            st.json(ver)

        with col2:
            st.subheader("Citations")
            st.caption(
                f"Top score: {retrieval.get('top_score', 0):.3f} "
                f"| Evidence used: {retrieval.get('citation_count', 0)}"
            )
            for c in out.get("citations", []) or []:
                with st.expander(f"{c['chunk_id']}  |  score={c['score']:.3f}"):
                    st.write(c["source_path"])
                    st.write(c["snippet"])

        st.divider()

with tab_docs:
    st.subheader("Indexed documents")
    if not documents:
        st.info("No documents indexed yet.")
    for doc in documents:
        with st.expander(doc.get("title") or doc.get("source_path")):
            st.write(doc.get("source_path"))
            st.caption(f"Ingested at {doc.get('created_at')}")

with tab_runs:
    st.subheader("Recent runs")
    if not recent_runs:
        st.info("No runs yet.")
    for run in recent_runs:
        created = run.get("created_at") or ""
        try:
            created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            created = created
        with st.expander(f"{created} | groundedness {run.get('groundedness', 0):.2f}"):
            st.write(run.get("question"))
            st.write(run.get("answer"))
            st.caption(f"Latency: {run.get('latency_sec') or 0:.2f} sec")
