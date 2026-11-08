from atlas.ingestion import Document, chunk_document, load_bytes
from atlas.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_agreement_between_retrievers():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert max(scores, key=scores.get) == "b"
    assert scores["b"] == 1 / 61 + 1 / 62


def test_chunks_have_stable_ids_and_headings():
    doc = load_bytes("policy.md", b"# Policy\n\n## Hotels\n\nThe limit is $220 per night in most cities.\n")
    first, second = chunk_document(doc), chunk_document(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].heading == "Hotels"
    assert first[0].doc_title == "Policy"


def test_long_sections_split_with_overlap():
    text = "# T\n\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_document(Document("t", "T", text, "t.md"), size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)


def test_hybrid_retrieval_finds_the_right_document(kb):
    result = kb.retrieve("What is the hotel limit per night in London?")
    assert result.context[0].chunk.doc_id == "expense-and-travel-policy"
    assert "300" in result.context[0].chunk.text
    assert set(result.timings_ms) == {"vector_ms", "bm25_ms", "rerank_ms"}


def test_every_stage_is_traced(kb):
    top = kb.retrieve("minimum password length").candidates[0]
    assert top.final_rank == 1
    assert top.rrf_score > 0
    assert top.vector_rank is not None or top.bm25_rank is not None


def test_uploaded_documents_are_private_to_their_session(kb):
    secret = b"# Project Nightjar\n\nProject Nightjar launches on 14 February with a budget of 4.2 million."
    kb.add_document(load_bytes("nightjar.md", secret, session_id="session-aaaa1111"))

    mine = kb.retrieve("When does Project Nightjar launch?", session_id="session-aaaa1111")
    assert mine.context[0].chunk.doc_id == "nightjar"

    others = kb.retrieve("When does Project Nightjar launch?", session_id="session-bbbb2222")
    assert all(c.chunk.doc_id != "nightjar" for c in others.candidates)

    kb.delete_session("session-aaaa1111")
    assert all(c.chunk.doc_id != "nightjar" for c in kb.retrieve("Project Nightjar", "session-aaaa1111").candidates)


def test_reingesting_unchanged_document_is_a_no_op(kb, settings):
    from atlas.ingestion import load_directory

    doc = load_directory(settings.corpus_dir)[0]
    _, changed = kb.add_document(doc)
    assert changed is False
