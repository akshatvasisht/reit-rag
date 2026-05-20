"""Tests for conflict-intent routing: deduplication bypass and prompt injection."""

from __future__ import annotations

from uuid import uuid4

import httpx
from anthropic import APIConnectionError

from src.generation import generator
from src.generation.prompts import _CONFLICT_INSTRUCTION, build_user_message
from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import RetrievalResult
from src.versioning.chains import dedupe_by_version_group


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Acme REIT",
    doc_type: str = "investor_presentation",
    doc_version: str = "2026-03",
    report_date: str = "2026-03",
    page_number: int = 10,
    chunk_text: str = "Revenue was 100M",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="ACR",
        doc_type=doc_type,
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=doc_version,
        section_title="Financials",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


# ---------------------------------------------------------------------------
# dedupe_by_version_group — conflict intent
# ---------------------------------------------------------------------------


def test_dedupe_conflict_returns_all_chunks_including_older_versions() -> None:
    """conflict intent must not suppress any chunk, even when versions differ."""
    older = _make_rc(_make_chunk(doc_version="2025-12", report_date="2025-12"))
    newer = _make_rc(_make_chunk(doc_version="2026-03", report_date="2026-03"))
    duplicate_newer = _make_rc(_make_chunk(doc_version="2026-03", report_date="2026-03", page_number=20))

    result = dedupe_by_version_group([older, newer, duplicate_newer], intent="conflict")

    assert result == [older, newer, duplicate_newer]


def test_dedupe_conflict_preserves_input_order() -> None:
    """Output order must match input order when intent is conflict."""
    chunks = [
        _make_rc(_make_chunk(doc_version=f"202{i}-0{i+1}", report_date=f"202{i}-0{i+1}"))
        for i in range(3)
    ]
    result = dedupe_by_version_group(chunks, intent="conflict")
    assert result == chunks


# ---------------------------------------------------------------------------
# dedupe_by_version_group — latest intent regression
# ---------------------------------------------------------------------------


def test_dedupe_latest_suppresses_older_versions() -> None:
    """latest intent must keep only the most recent doc_version per group."""
    older = _make_rc(_make_chunk(doc_version="2025-12", report_date="2025-12"))
    newer = _make_rc(_make_chunk(doc_version="2026-03", report_date="2026-03"))

    result = dedupe_by_version_group([older, newer], intent="latest")

    assert newer in result
    assert older not in result


def test_dedupe_latest_single_version_is_passthrough() -> None:
    """When only one doc_version exists, latest is a no-op."""
    rc = _make_rc(_make_chunk(doc_version="2026-03", report_date="2026-03"))
    result = dedupe_by_version_group([rc], intent="latest")
    assert result == [rc]


# ---------------------------------------------------------------------------
# build_user_message — conflict instruction block
# ---------------------------------------------------------------------------


def test_build_user_message_conflict_includes_instruction_block() -> None:
    """conflict intent must append the attribution instruction block."""
    chunk = _make_chunk()
    contexts = [_make_rc(chunk)]
    msg = build_user_message("test query", contexts, intent="conflict")
    assert _CONFLICT_INSTRUCTION in msg


def test_build_user_message_latest_excludes_conflict_instruction() -> None:
    """Non-conflict intents must not include the conflict instruction block."""
    chunk = _make_chunk()
    contexts = [_make_rc(chunk)]
    for intent in ("latest", "historical", "comparison"):
        msg = build_user_message("test query", contexts, intent=intent)  # type: ignore[arg-type]
        assert _CONFLICT_INSTRUCTION not in msg, f"Conflict block leaked into intent={intent}"


# ---------------------------------------------------------------------------
# answer_structured — intent threaded through to build_user_message
# ---------------------------------------------------------------------------


def _make_conflict_retrieval() -> RetrievalResult:
    older = _make_rc(_make_chunk(doc_version="2025-12", report_date="2025-12", chunk_text="Revenue 95M"))
    newer = _make_rc(_make_chunk(doc_version="2026-03", report_date="2026-03", chunk_text="Revenue 100M"))
    return RetrievalResult(
        query="Are there conflicting revenue figures?",
        intent="conflict",
        contexts=[older, newer],
        companies=["Acme REIT"],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
    )


class _CapturingMessages:
    """Records the user message passed to create() so tests can assert on it."""

    def __init__(self) -> None:
        self.last_user_content: str | None = None

    def create(self, **kwargs):  # noqa: ANN003
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "user":
                self.last_user_content = msg["content"]
        raise APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )


class _CapturingClient:
    def __init__(self) -> None:
        self.messages = _CapturingMessages()


def test_answer_structured_passes_conflict_intent_to_prompt(monkeypatch) -> None:
    """When retrieval returns conflict intent, the user message must contain the
    conflict instruction block."""
    capturing_client = _CapturingClient()
    monkeypatch.setattr(generator, "retrieve", lambda query: _make_conflict_retrieval())
    monkeypatch.setattr(generator, "_get_client", lambda: capturing_client)

    result = generator.answer_structured("Are there conflicting revenue figures?")

    # The call raises APIConnectionError so we get an abstained answer.
    assert result.abstained is True
    # But the captured user message must contain the conflict instructions.
    assert capturing_client.messages.last_user_content is not None
    assert _CONFLICT_INSTRUCTION in capturing_client.messages.last_user_content
