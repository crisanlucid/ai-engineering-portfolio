"""LLM-as-judge: validates whether an answer is supported by the retrieved context."""

import contextlib
import re
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.documents import Document
from llm import LLMAdapter


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"


@dataclass
class JudgeResult:
    verdict: Verdict
    reason: str

    def badge(self) -> str:
        """Colour-coded one-liner for terminal output."""
        icons = {
            Verdict.SUPPORTED: "\033[32m✔ SUPPORTED\033[0m",
            Verdict.PARTIAL: "\033[33m~ PARTIAL\033[0m",
            Verdict.NOT_SUPPORTED: "\033[31m✘ NOT SUPPORTED\033[0m",
        }
        return f"{icons[self.verdict]}  —  {self.reason}"


JUDGE_SYSTEM = """\
You are a strict factual judge. Your job is to decide whether an AI assistant's
answer is supported by the provided source chunks.

Verdict definitions
───────────────────
SUPPORTED     The answer is fully backed by the chunks.
              This includes a correct "I don't know" when the topic is absent.
PARTIAL       Some claims are supported; others are not, or the answer is
              incomplete despite relevant information being present.
NOT_SUPPORTED The answer makes claims absent from the chunks, or misses
              critical information that IS in the chunks.

Response format — return exactly two lines, nothing else:
VERDICT: <SUPPORTED|PARTIAL|NOT_SUPPORTED>
REASON: <one concise sentence>\
"""


def _build_judge_prompt(
    scored_docs: list[tuple[Document, float]],
    answer: str,
) -> str:
    chunks = ""
    for i, (doc, score) in enumerate(scored_docs):
        src = doc.metadata.get("source", "unknown")
        chunks += f"\n--- Chunk {i + 1} | source: {src} | cosine: {score:.4f} ---\n"
        chunks += doc.page_content.strip()
        chunks += "\n"

    return (
        f"SOURCE CHUNKS:\n{chunks}\n"
        f"ASSISTANT ANSWER:\n{answer.strip()}\n\n"
        "Evaluate the answer against the source chunks and respond with the "
        "two-line format described in your instructions."
    )


def _parse_verdict(raw: str) -> JudgeResult:
    """Parse the judge's raw response into a JudgeResult.

    Falls back gracefully if the model doesn't follow the exact format.
    """
    verdict = Verdict.PARTIAL  # safe default
    reason = raw.strip()  # keep full text as fallback reason

    verdict_match = re.search(
        r"VERDICT\s*:\s*(SUPPORTED|PARTIAL|NOT[_\s]SUPPORTED)",
        raw,
        re.IGNORECASE,
    )
    if verdict_match:
        raw_verdict = verdict_match.group(1).upper().replace(" ", "_")
        with contextlib.suppress(ValueError):
            verdict = Verdict(raw_verdict)

    reason_match = re.search(r"REASON\s*:\s*(.+)", raw, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return JudgeResult(verdict=verdict, reason=reason)


def judge(
    scored_docs: list[tuple[Document, float]],
    answer: str,
    adapter: LLMAdapter,
) -> JudgeResult:
    """Call the LLM judge and return a structured verdict."""
    prompt = _build_judge_prompt(scored_docs, answer)
    raw = adapter.complete(JUDGE_SYSTEM, prompt)
    return _parse_verdict(raw)
