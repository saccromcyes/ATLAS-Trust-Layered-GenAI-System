import streamlit as st
import requests

API = "http://backend:8000"

st.set_page_config(page_title="ATLAS", layout="wide")
st.title("ATLAS — Hybrid RAG + Verification + Eval")

with st.sidebar:
    st.header("Indexing")
    st.caption("Ingest sample docs or upload your own PDF/TXT/MD.")

    if st.button("Ingest sample_docs"):
        r = requests.post(f"{API}/ingest/folder", params={"path": "/app/sample_docs"}, timeout=180)
        r.raise_for_status()
        st.success(r.json())

    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Ingest uploaded file"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{API}/ingest/upload", files=files, timeout=300)
        r.raise_for_status()
        st.success(r.json())

    top_k = st.slider("Top-K evidence", 3, 12, 8)

st.divider()

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("Ask a question", placeholder="e.g., What does the handbook say about carry-forward leave?")
if st.button("Ask") and q.strip():
    with st.spinner("Thinking..."):
        r = requests.post(f"{API}/ask", json={"question": q, "top_k": top_k}, timeout=300)
        r.raise_for_status()
        out = r.json()
    st.session_state.history.append((q, out))

for q, out in reversed(st.session_state.history[-8:]):
    st.subheader("Question")
    st.write(q)

    ver = out.get("verification", {}) or {}
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Answer")
        st.write(out.get("answer", ""))

        st.subheader("Verification")
        st.metric("Groundedness", float(ver.get("groundedness", 0.0) or 0.0))
        st.json(ver)

    with col2:
        st.subheader("Citations")
        for c in out.get("citations", []) or []:
            with st.expander(f"{c['chunk_id']}  |  score={c['score']:.3f}"):
                st.write(c["source_path"])
                st.write(c["snippet"])

    st.divider()
