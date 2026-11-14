from atlas.llm import REFUSAL_TOKEN, ExtractiveLLM, parse_json
from atlas.pipeline import AnswerPipeline, extract_citations, groundedness, strip_invalid_citations
from tests.conftest import FakeLLM


def events_of(pipeline, question, session_id="demo"):
    return list(pipeline.run(question, session_id))


def kinds(events):
    return [e["type"] for e in events]


def test_fallback_mode_answers_with_citations(make_pipeline):
    result = make_pipeline(ExtractiveLLM()).answer("How many characters must a password have?")
    assert result["status"] == "answered"
    assert "14" in result["answer"]
    assert result["citations"] and result["citations"][0]["doc_id"] == "information-security-policy"
    assert result["mode"] == "fallback"


def test_off_topic_question_is_refused_before_generation(make_pipeline):
    llm = FakeLLM(answer="Canberra [1].")
    events = events_of(make_pipeline(llm), "What is the capital of Australia?")
    assert "refusal" in kinds(events)
    assert events[-1]["status"] == "refused"
    assert llm.calls == []  # the LLM was never asked


def test_gate_lets_a_question_through_when_embeddings_find_it_relevant(settings, kb):
    # Reworded questions can score near 0 with the cross-encoder; the embedding similarity
    # catches them, so the gate refuses only when both signals are low.
    settings.min_similarity = 0.0
    llm = FakeLLM(answer=REFUSAL_TOKEN)
    result = AnswerPipeline(settings, kb, llm, kb.store).answer("What is the capital of Australia?")
    assert llm.calls == ["stream"]  # the model was asked, and it declined
    assert result["status"] == "refused"


def test_llm_answer_is_streamed_cited_and_verified(make_pipeline):
    llm = FakeLLM(
        answer="Passwords must be at least 14 characters long [1]. MFA is mandatory [1].",
        claims=[
            {"claim": "Passwords must be at least 14 characters", "verdict": "supported", "sources": [1]},
            {"claim": "MFA is mandatory", "verdict": "supported", "sources": [1]},
        ],
    )
    events = events_of(make_pipeline(llm), "What is the minimum password length?")
    assert kinds(events)[0] == "retrieval"
    assert kinds(events).count("token") > 1  # streamed in pieces
    verification = next(e for e in events if e["type"] == "verification")
    assert verification["groundedness"] == 1.0 and verification["method"] == "llm-judge"
    assert events[-1]["status"] == "answered"


def test_low_groundedness_answer_is_withheld(make_pipeline):
    llm = FakeLLM(
        answer="Passwords need 14 characters [1]. They must be changed every 30 days [1].",
        claims=[
            {"claim": "Passwords need 14 characters", "verdict": "supported", "sources": [1]},
            {"claim": "Passwords must be changed every 30 days", "verdict": "unsupported", "sources": []},
        ],
    )
    events = events_of(make_pipeline(llm), "What is the minimum password length?")
    assert "withheld" in kinds(events)
    assert events[-1]["status"] == "withheld"


def test_model_refusal_token_is_never_shown(make_pipeline):
    llm = FakeLLM(answer=REFUSAL_TOKEN)
    events = events_of(make_pipeline(llm), "What is the hotel limit for business trips?")
    assert "token" not in kinds(events)
    assert events[-1]["status"] == "refused"


def test_provider_failure_degrades_to_extractive_answer(make_pipeline):
    llm = FakeLLM(fail_stream=True)
    events = events_of(make_pipeline(llm), "How many characters must a password have?")
    assert "notice" in kinds(events) and "reset" in kinds(events)
    assert events[-1]["mode"] == "fallback"
    assert events[-1]["status"] == "answered"


def test_citation_helpers():
    assert extract_citations("A [2]. B [1][2]. C [9].", 3) == [2, 1]
    assert strip_invalid_citations("A [2]. C [9].", 3) == "A [2]. C ."
    assert groundedness([{"verdict": "supported"}, {"verdict": "partial"}]) == 0.75
    assert groundedness([]) is None


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"claims": []}\n```') == {"claims": []}
    assert parse_json('Here you go: {"a": 1} hope that helps') == {"a": 1}


def test_runs_are_logged(make_pipeline, kb):
    make_pipeline(ExtractiveLLM()).answer("What is the minimum password length?")
    assert kb.store.stats()["questions_answered"] == 1


def test_provider_options_for_reasoning_models():
    from atlas.llm import provider_options

    groq = provider_options("https://api.groq.com/openai/v1", "low")
    assert groq == {"reasoning_effort": "low", "extra_body": {"include_reasoning": False}}
    assert provider_options("https://api.openai.com/v1", "") == {}
