"""Unit tests for judge.py — parsing, badge, and judge() integration."""

from unittest.mock import MagicMock

from judge import JudgeResult, Verdict, _build_judge_prompt, _parse_verdict, judge
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, source: str = "docs/test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _pair(content: str, score: float = 0.8) -> tuple[Document, float]:
    return (_doc(content), score)


def _adapter(response: str) -> MagicMock:
    a = MagicMock()
    a.complete.return_value = response
    return a


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_supported(self):
        raw = "VERDICT: SUPPORTED\nREASON: The answer matches the context."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.SUPPORTED
        assert "matches" in result.reason

    def test_partial(self):
        raw = "VERDICT: PARTIAL\nREASON: Only half the claims are backed."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.PARTIAL

    def test_not_supported(self):
        raw = "VERDICT: NOT_SUPPORTED\nREASON: The answer fabricates information."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.NOT_SUPPORTED

    def test_not_supported_with_space(self):
        """Model may write 'NOT SUPPORTED' instead of 'NOT_SUPPORTED'."""
        raw = "VERDICT: NOT SUPPORTED\nREASON: No evidence found."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.NOT_SUPPORTED

    def test_case_insensitive(self):
        raw = "verdict: supported\nreason: looks good."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.SUPPORTED

    def test_fallback_to_partial_on_unknown_verdict(self):
        raw = "VERDICT: UNKNOWN\nREASON: Something weird."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.PARTIAL  # safe default

    def test_fallback_reason_is_raw_text_when_no_reason_line(self):
        raw = "VERDICT: SUPPORTED"
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.SUPPORTED
        # reason should be the full raw text as fallback
        assert len(result.reason) > 0

    def test_extra_whitespace_around_colon(self):
        raw = "VERDICT :  PARTIAL\nREASON  :  Incomplete."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.PARTIAL
        assert result.reason == "Incomplete."

    def test_garbled_response_defaults_to_partial(self):
        raw = "I cannot determine the verdict at this time."
        result = _parse_verdict(raw)
        assert result.verdict == Verdict.PARTIAL


# ---------------------------------------------------------------------------
# JudgeResult.badge
# ---------------------------------------------------------------------------


class TestJudgeResultBadge:
    def test_supported_badge_contains_text(self):
        r = JudgeResult(verdict=Verdict.SUPPORTED, reason="good")
        assert "SUPPORTED" in r.badge()
        assert "good" in r.badge()

    def test_partial_badge_contains_text(self):
        r = JudgeResult(verdict=Verdict.PARTIAL, reason="half")
        assert "PARTIAL" in r.badge()

    def test_not_supported_badge_contains_text(self):
        r = JudgeResult(verdict=Verdict.NOT_SUPPORTED, reason="bad")
        assert "NOT SUPPORTED" in r.badge()


# ---------------------------------------------------------------------------
# _build_judge_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_includes_chunk_content(self):
        pairs = [_pair("chunk content here")]
        prompt = _build_judge_prompt(pairs, "some answer")
        assert "chunk content here" in prompt

    def test_includes_answer(self):
        pairs = [_pair("ctx")]
        prompt = _build_judge_prompt(pairs, "the assistant answer")
        assert "the assistant answer" in prompt

    def test_includes_source(self):
        doc = Document(page_content="x", metadata={"source": "docs/foo.md"})
        pairs = [(doc, 0.9)]
        prompt = _build_judge_prompt(pairs, "answer")
        assert "docs/foo.md" in prompt

    def test_includes_cosine_score(self):
        pairs = [_pair("ctx", score=0.8765)]
        prompt = _build_judge_prompt(pairs, "answer")
        assert "0.8765" in prompt

    def test_multiple_chunks_all_present(self):
        pairs = [_pair(f"chunk {i}") for i in range(3)]
        prompt = _build_judge_prompt(pairs, "answer")
        for i in range(3):
            assert f"chunk {i}" in prompt


# ---------------------------------------------------------------------------
# judge()
# ---------------------------------------------------------------------------


class TestJudge:
    def test_returns_judge_result(self):
        pairs = [_pair("relevant context")]
        adapter = _adapter("VERDICT: SUPPORTED\nREASON: Matches context.")
        result = judge(pairs, "the answer", adapter)
        assert isinstance(result, JudgeResult)

    def test_adapter_called_once(self):
        pairs = [_pair("ctx")]
        adapter = _adapter("VERDICT: SUPPORTED\nREASON: ok.")
        judge(pairs, "answer", adapter)
        adapter.complete.assert_called_once()

    def test_judge_system_prompt_is_passed(self):

        pairs = [_pair("ctx")]
        adapter = _adapter("VERDICT: PARTIAL\nREASON: partial.")
        judge(pairs, "answer", adapter)
        system_arg = adapter.complete.call_args[0][0]
        assert "SUPPORTED" in system_arg
        assert "PARTIAL" in system_arg
        assert "NOT_SUPPORTED" in system_arg or "NOT SUPPORTED" in system_arg

    def test_verdict_propagated_correctly(self):
        pairs = [_pair("ctx")]
        adapter = _adapter("VERDICT: NOT_SUPPORTED\nREASON: Fabricated.")
        result = judge(pairs, "answer", adapter)
        assert result.verdict == Verdict.NOT_SUPPORTED
        assert "Fabricated" in result.reason

    def test_partial_on_garbled_llm_response(self):
        pairs = [_pair("ctx")]
        adapter = _adapter("I'm unable to determine this.")
        result = judge(pairs, "answer", adapter)
        assert result.verdict == Verdict.PARTIAL
