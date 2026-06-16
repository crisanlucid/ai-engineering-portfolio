"""Unit tests for api.py — health endpoint and /ask endpoint."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from judge import JudgeResult, Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, source: str = "docs/test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _scored_docs(content: str = "relevant content", score: float = 0.85):
    return [(_doc(content), score)]


def _make_app(vectorstore=None, adapter=None):
    """Return a TestClient with pre-loaded state (skips lifespan startup)."""
    from api import app, _state
    _state.clear()
    if vectorstore is not None:
        _state["vectorstore"] = vectorstore
    if adapter is not None:
        _state["adapter"] = adapter
    return TestClient(app, raise_server_exceptions=True)


def _mock_vectorstore(scored_pairs=None):
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = (
        scored_pairs if scored_pairs is not None else _scored_docs()
    )
    return vs


def _mock_adapter(response: str = "the answer"):
    a = MagicMock()
    a.complete.return_value = response
    return a


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok_when_ready(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_returns_503_when_not_ready(self):
        client = _make_app()  # no state loaded
        resp = client.get("/health")
        assert resp.status_code == 503

    def test_response_contains_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "agy")
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.get("/health")
        assert resp.json()["provider"] == "agy"

    def test_response_contains_language(self, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "de")
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.get("/health")
        # LANGUAGE is read at import time; just assert key exists
        assert "language" in resp.json()


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_returns_200_with_answer(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter("the answer"),
        )
        resp = client.post("/ask", json={"question": "What is this?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "the answer"

    def test_returns_503_when_not_ready(self):
        client = _make_app()
        resp = client.post("/ask", json={"question": "hello"})
        assert resp.status_code == 503

    def test_sources_are_unique_and_present(self):
        pairs = [
            (_doc("a", "docs/foo.md"), 0.9),
            (_doc("b", "docs/foo.md"), 0.8),  # duplicate source
            (_doc("c", "docs/bar.md"), 0.7),
        ]
        client = _make_app(
            vectorstore=_mock_vectorstore(pairs),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": "q"})
        sources = resp.json()["sources"]
        assert sources == ["docs/foo.md", "docs/bar.md"]  # unique, ordered

    def test_chunks_contain_score_and_preview(self):
        pairs = [(_doc("hello world content", "docs/x.md"), 0.75)]
        client = _make_app(
            vectorstore=_mock_vectorstore(pairs),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": "q"})
        chunk = resp.json()["chunks"][0]
        assert chunk["source"] == "docs/x.md"
        assert chunk["score"] == pytest.approx(0.625, abs=0.01)  # 1 - 0.75/2
        assert "hello world" in chunk["preview"]

    def test_preview_truncated_to_200_chars(self):
        long_content = "x" * 500
        pairs = [(_doc(long_content, "docs/long.md"), 0.5)]
        client = _make_app(
            vectorstore=_mock_vectorstore(pairs),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": "q"})
        assert len(resp.json()["chunks"][0]["preview"]) <= 200

    def test_k_parameter_forwarded_to_vectorstore(self):
        vs = _mock_vectorstore()
        client = _make_app(vectorstore=vs, adapter=_mock_adapter())
        client.post("/ask", json={"question": "q", "k": 5})
        vs.similarity_search_with_score.assert_called_once_with("q", k=5)

    def test_verdict_none_when_judge_disabled(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": "q", "judge": False})
        assert resp.json()["verdict"] is None

    def test_verdict_present_when_judge_enabled(self):
        fake_result = JudgeResult(verdict=Verdict.SUPPORTED, reason="all good")
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        with patch("api.llm_judge", return_value=fake_result):
            resp = client.post("/ask", json={"question": "q", "judge": True})
        verdict = resp.json()["verdict"]
        assert verdict["verdict"] == "SUPPORTED"
        assert verdict["reason"] == "all good"

    def test_empty_question_returns_422(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    def test_k_out_of_range_returns_422(self):
        client = _make_app(
            vectorstore=_mock_vectorstore(),
            adapter=_mock_adapter(),
        )
        resp = client.post("/ask", json={"question": "q", "k": 0})
        assert resp.status_code == 422
