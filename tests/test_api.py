import json

import pytest
from fastapi.testclient import TestClient

from atlas.main import create_app

SESSION = {"X-Session-Id": "test-session-1234"}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def sse_events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    config = client.get("/api/config").json()
    assert config["mode"] == "fallback"
    assert config["reranker"] == "lexical"


def test_ask_streams_server_sent_events(client):
    response = client.post("/api/ask", json={"question": "How many sick days do employees get?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[0]["type"] == "retrieval"
    assert events[-1]["type"] == "done"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "10" in answer


def test_question_validation(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 600}).status_code == 422


def test_demo_corpus_is_listed(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 7
    assert not any(d["uploaded"] for d in docs)


def test_upload_then_ask_about_it(client):
    content = b"# Orchard Trial\n\nThe orchard trial in Nashik used 14 drones and finished on 3 June."
    up = client.post("/api/documents", files={"file": ("trial.md", content, "text/markdown")}, headers=SESSION)
    assert up.status_code == 200, up.text
    assert any(d["uploaded"] for d in client.get("/api/documents", headers=SESSION).json())

    events = sse_events(client.post("/api/ask", json={"question": "How many drones did the orchard trial use?"}, headers=SESSION))
    citations = next(e for e in events if e["type"] == "citations")["items"]
    assert citations[0]["doc_id"] == "trial"

    assert client.delete("/api/session", headers=SESSION).json() == {"cleared": True}


def test_upload_rejects_bad_files(client):
    bad = client.post("/api/documents", files={"file": ("x.exe", b"MZ...", "application/octet-stream")}, headers=SESSION)
    assert bad.status_code == 400
    no_session = client.post("/api/documents", files={"file": ("a.md", b"# A\n\nSome text here.", "text/markdown")})
    assert no_session.status_code == 400


def test_rate_limit(settings):
    settings.rate_limit_per_10min = 2
    with TestClient(create_app(settings)) as c:
        for _ in range(2):
            assert c.post("/api/ask", json={"question": "sick days?"}).status_code == 200
        limited = c.post("/api/ask", json={"question": "sick days?"})
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers


def test_eval_endpoint(client):
    assert "available" in client.get("/api/eval").json()
