"""Unit tests for src/retrieval/reranker.py.

Regression guard for contextualized_text-preferred scoring: the reranker must
score a chunk whose contextualized_text bridges the vocab gap higher than a
chunk that only contains the raw (non-matching) text.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.models import Chunk, RetrievedChunk


def _make_chunk(
    chunk_text: str,
    contextualized_text: str | None = None,
    content_type: str = "text",
    company: str = "EastGroup Properties",
    doc_type: str = "investor_presentation",
    page_number: int | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="EGP",
        doc_type=doc_type,
        report_date="2025-12",
        doc_version="2025-12",
        chunk_text=chunk_text,
        content_type=content_type,  # type: ignore[arg-type]
        contextualized_text=contextualized_text,
        page_number=page_number,
    )


def _make_rc(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk)


# ---------------------------------------------------------------------------
# Contextualized-text preference — EGP vocab-gap regression
# ---------------------------------------------------------------------------


def test_contextualized_text_preferred_over_raw_chunk_text(monkeypatch) -> None:
    """Chunk with 'pipeline' in contextualized_text must outscore bare-Program chunk.

    This is a regression guard for the EGP p.26 case where the cross-encoder
    scored (query='pipeline', passage='Current Development and Value-Add Program...')
    at -4.9 and pushed the correct chunk past top-5.  The fix: feed
    contextualized_text (which contains 'pipeline') to the cross-encoder instead
    of raw chunk_text.
    """
    from src.retrieval import reranker as _reranker_module

    # Chunk A: contextualized_text bridges the vocab gap.
    chunk_a = _make_chunk(
        chunk_text="Current Development and Value-Add Program consists of ...",
        contextualized_text="active development and value-add acquisition pipeline",
    )
    # Chunk B: no contextualized_text, only raw "Program" text.
    chunk_b = _make_chunk(
        chunk_text="Current Development and Value-Add Program consists of ...",
        contextualized_text=None,
    )

    rc_a = _make_rc(chunk_a)
    rc_b = _make_rc(chunk_b)

    # Track which passage strings reach the model.
    captured_pairs: list[list[tuple[str, str]]] = []

    class _FakeEncoder:
        def predict(self, pairs: Any) -> list[float]:
            captured_pairs.append(list(pairs))
            # Score by whether the passage contains "pipeline".
            return [5.0 if "pipeline" in p.lower() else -4.9 for _, p in pairs]

    monkeypatch.setattr(_reranker_module, "_cross_encoder", _FakeEncoder())

    query = "What is EastGroup's current development pipeline?"
    results = _reranker_module.rerank(query, [rc_a, rc_b], top_n=2)

    assert len(captured_pairs) == 1, "predict() should be called exactly once"
    passages = [p for _, p in captured_pairs[0]]

    # Chunk A must use contextualized_text (vocab-bridged).
    assert passages[0] == chunk_a.contextualized_text, (
        "reranker should feed contextualized_text for chunk A, not raw chunk_text"
    )
    # Chunk B must fall back to chunk_text (no contextualized_text).
    assert passages[1] == chunk_b.chunk_text, (
        "reranker should fall back to chunk_text when contextualized_text is None"
    )

    # Chunk A (with "pipeline" in contextualized_text) must rank first.
    assert results[0].chunk.id == chunk_a.id, (
        "chunk with pipeline in contextualized_text should outscore bare-Program chunk"
    )
    assert results[1].chunk.id == chunk_b.id


def test_rerank_fallback_when_no_contextualized_text(monkeypatch) -> None:
    """When all chunks lack contextualized_text, raw chunk_text is used unchanged."""
    from src.retrieval import reranker as _reranker_module

    chunk = _make_chunk(chunk_text="some raw text", contextualized_text=None)
    rc = _make_rc(chunk)

    captured: list[list[tuple[str, str]]] = []

    class _FakeEncoder:
        def predict(self, pairs: Any) -> list[float]:
            captured.append(list(pairs))
            return [1.0] * len(pairs)

    monkeypatch.setattr(_reranker_module, "_cross_encoder", _FakeEncoder())

    _reranker_module.rerank("some query", [rc], top_n=1)

    assert captured[0][0][1] == "some raw text", (
        "should fall back to chunk_text when contextualized_text is None"
    )


# ---------------------------------------------------------------------------
# _passage_for_rerank — chart-aware passage prefixing
# ---------------------------------------------------------------------------


def test_passage_for_rerank_prefixes_chart_description() -> None:
    """chart_description chunks must get a structured chart marker prepended.

    Vision-generated prose often lacks the company name and the word
    'chart' itself, so the cross-encoder loses lexical handles a natural-
    language query relies on.
    """
    from src.retrieval.reranker import _passage_for_rerank

    chunk = _make_chunk(
        chunk_text="Domestic miles trending up against international from 2023 onward.",
        content_type="chart_description",
        company="Simon Property Group",
        doc_type="thematic_report",
        page_number=17,
    )
    rc = _make_rc(chunk)

    passage = _passage_for_rerank(rc)

    assert passage.startswith("Chart from Simon Property Group (thematic_report, p.17): "), (
        f"chart_description should be prefixed; got: {passage!r}"
    )
    assert chunk.chunk_text in passage, "original chunk_text must still be present after prefix"


def test_passage_for_rerank_prefixes_chart_context() -> None:
    """chart_context (low-confidence vision fallback) gets the same prefix."""
    from src.retrieval.reranker import _passage_for_rerank

    chunk = _make_chunk(
        chunk_text="A bar chart showing quarterly metrics; values are not legible.",
        content_type="chart_context",
        company="Digital Realty Trust",
        doc_type="investor_presentation",
        page_number=4,
    )

    passage = _passage_for_rerank(_make_rc(chunk))

    assert passage.startswith("Chart from Digital Realty Trust (investor_presentation, p.4): "), (
        f"chart_context should be prefixed; got: {passage!r}"
    )


def test_passage_for_rerank_leaves_text_chunks_unchanged() -> None:
    """Non-vision chunks must not receive the chart prefix."""
    from src.retrieval.reranker import _passage_for_rerank

    chunk = _make_chunk(
        chunk_text="EGP's same-store NOI guidance for 2026 is 8-12%.",
        content_type="text",
    )

    passage = _passage_for_rerank(_make_rc(chunk))

    assert passage == "EGP's same-store NOI guidance for 2026 is 8-12%.", (
        f"text chunks must pass through unchanged; got: {passage!r}"
    )
    assert "Chart from" not in passage


def test_passage_for_rerank_chart_with_contextualized_text_keeps_prefix() -> None:
    """When contextualized_text is set on a chart chunk, prefix wraps it (not raw text)."""
    from src.retrieval.reranker import _passage_for_rerank

    chunk = _make_chunk(
        chunk_text="raw vision prose",
        contextualized_text="contextualized chart summary",
        content_type="chart_description",
        company="BXP",
        doc_type="investor_presentation",
        page_number=12,
    )

    passage = _passage_for_rerank(_make_rc(chunk))

    assert passage == "Chart from BXP (investor_presentation, p.12): contextualized chart summary"


def test_passage_for_rerank_whitespace_contextualized_text_falls_back() -> None:
    """Whitespace-only contextualized_text must not be used in place of chunk_text."""
    from src.retrieval.reranker import _passage_for_rerank

    chunk = _make_chunk(
        chunk_text="real chunk content",
        contextualized_text="   \n  ",
        content_type="text",
    )

    assert _passage_for_rerank(_make_rc(chunk)) == "real chunk content"
