"""Unit tests for chat.py — retrieval, scoring, formatting and ask()."""
import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, source: str = "docs/test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _scored(content: str, l2_dist: float, source: str = "docs/test.md"):
    """Return a (Document, l2_distance) pair as Chroma would."""
    return (_doc(content, source), l2_dist)


# ---------------------------------------------------------------------------
# retrieve() — L2 → cosine conversion
# ---------------------------------------------------------------------------

class TestRetrieve:
    """retrieve() wraps similarity_search_with_score and converts distances."""

    def _make_vectorstore(self, pairs):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = pairs
        return vs

    def test_perfect_match_gives_score_of_one(self):
        import chat
        vs = self._make_vectorstore([_scored("exact", l2_dist=0.0)])
        results = chat.retrieve(vs, "exact", k=1)
        doc, score = results[0]
        assert score == 1.0

    def test_score_is_clamped_to_zero(self):
        import chat
        # l2_dist > 2.0 would produce negative cosine — must clamp to 0
        vs = self._make_vectorstore([_scored("far", l2_dist=3.0)])
        _, score = chat.retrieve(vs, "far", k=1)[0]
        assert score == 0.0

    def test_known_l2_distance_converts_correctly(self):
        import chat
        # cosine_sim = 1 - (l2 / 2)  →  l2=0.4 → 1 - 0.2 = 0.8
        vs = self._make_vectorstore([_scored("mid", l2_dist=0.4)])
        _, score = chat.retrieve(vs, "mid", k=1)[0]
        assert score == pytest.approx(0.8, abs=1e-4)

    def test_returns_requested_k_results(self):
        import chat
        pairs = [_scored(f"doc{i}", l2_dist=0.1 * i) for i in range(5)]
        vs = self._make_vectorstore(pairs)
        results = chat.retrieve(vs, "q", k=3)
        vs.similarity_search_with_score.assert_called_once_with("q", k=3)
        assert len(results) == 5  # returns whatever the mock gives back

    def test_scores_are_rounded_to_4dp(self):
        import chat
        vs = self._make_vectorstore([_scored("x", l2_dist=0.333333)])
        _, score = chat.retrieve(vs, "x", k=1)[0]
        assert score == round(1.0 - 0.333333 / 2.0, 4)

    def test_document_is_preserved(self):
        import chat
        doc = _doc("hello", source="src/foo.md")
        vs = self._make_vectorstore([(doc, 0.5)])
        results = chat.retrieve(vs, "hello", k=1)
        assert results[0][0].page_content == "hello"
        assert results[0][0].metadata["source"] == "src/foo.md"


# ---------------------------------------------------------------------------
# format_context()
# ---------------------------------------------------------------------------

class TestFormatContext:
    def _pairs(self):
        return [
            (_doc("content A", "a.md"), 0.9),
            (_doc("content B", "b.md"), 0.7),
        ]

    def test_contains_all_chunk_content(self):
        import chat
        with patch.object(chat, "AGENT_DEBUG", False):
            ctx = chat.format_context(self._pairs())
        assert "content A" in ctx
        assert "content B" in ctx

    def test_chunk_headers_include_source(self):
        import chat
        with patch.object(chat, "AGENT_DEBUG", False):
            ctx = chat.format_context(self._pairs())
        assert "source: a.md" in ctx
        assert "source: b.md" in ctx

    def test_no_score_tag_when_debug_off(self):
        import chat
        with patch.object(chat, "AGENT_DEBUG", False):
            ctx = chat.format_context(self._pairs())
        assert "cosine" not in ctx

    def test_score_tag_present_when_debug_on(self):
        import chat
        with patch.object(chat, "AGENT_DEBUG", True):
            ctx = chat.format_context(self._pairs())
        assert "cosine: 0.9000" in ctx
        assert "cosine: 0.7000" in ctx

    def test_chunk_numbering_starts_at_one(self):
        import chat
        with patch.object(chat, "AGENT_DEBUG", False):
            ctx = chat.format_context(self._pairs())
        assert "Chunk 1" in ctx
        assert "Chunk 2" in ctx


# ---------------------------------------------------------------------------
# _print_debug_scores()
# ---------------------------------------------------------------------------

class TestPrintDebugScores:
    def _pairs(self):
        return [
            (_doc("a", "file_a.md"), 0.95),
            (_doc("b", "file_b.md"), 0.60),
        ]

    def test_outputs_to_stderr(self):
        import chat
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            chat._print_debug_scores(self._pairs())
        output = buf.getvalue()
        assert len(output) > 0

    def test_each_source_is_printed(self):
        import chat
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            chat._print_debug_scores(self._pairs())
        output = buf.getvalue()
        assert "file_a.md" in output
        assert "file_b.md" in output

    def test_scores_are_printed(self):
        import chat
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            chat._print_debug_scores(self._pairs())
        output = buf.getvalue()
        assert "0.9500" in output
        assert "0.6000" in output

    def test_debug_header_is_present(self):
        import chat
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            chat._print_debug_scores(self._pairs())
        assert "[DEBUG]" in buf.getvalue()


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------

class TestAsk:
    def _setup(self, l2_dist: float = 0.4):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            (_doc("relevant content", "docs/a.md"), l2_dist)
        ]
        adapter = MagicMock()
        adapter.complete.return_value = "the answer"
        return vs, adapter

    def test_returns_adapter_response(self):
        import chat
        vs, adapter = self._setup()
        with patch.object(chat, "AGENT_DEBUG", False), \
             patch("builtins.print"):
            result = chat.ask(vs, "what?", adapter)
        assert result == "the answer"

    def test_adapter_complete_called_once(self):
        import chat
        vs, adapter = self._setup()
        with patch.object(chat, "AGENT_DEBUG", False), \
             patch("builtins.print"):
            chat.ask(vs, "what?", adapter)
        adapter.complete.assert_called_once()

    def test_context_included_in_user_prompt(self):
        import chat
        vs, adapter = self._setup()
        with patch.object(chat, "AGENT_DEBUG", False), \
             patch("builtins.print"):
            chat.ask(vs, "what?", adapter)
        _, user_prompt = adapter.complete.call_args[0]
        assert "relevant content" in user_prompt
        assert "what?" in user_prompt

    def test_debug_scores_printed_when_debug_on(self):
        import chat
        vs, adapter = self._setup()
        with patch.object(chat, "AGENT_DEBUG", True), \
             patch("builtins.print"), \
             patch("chat._print_debug_scores") as mock_debug:
            chat.ask(vs, "what?", adapter)
        mock_debug.assert_called_once()

    def test_debug_scores_not_printed_when_debug_off(self):
        import chat
        vs, adapter = self._setup()
        with patch.object(chat, "AGENT_DEBUG", False), \
             patch("builtins.print"), \
             patch("chat._print_debug_scores") as mock_debug:
            chat.ask(vs, "what?", adapter)
        mock_debug.assert_not_called()
