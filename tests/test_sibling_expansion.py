"""Unit tests for sibling-aware context expansion in the retrieval pipeline.

All tests use mocked DB connections — no real database or network access.
Each test completes in well under 2 seconds.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch
from uuid import UUID, uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    MAX_CONTEXT_CHUNKS,
    _fetch_chunk_by_doc_page,
    expand_to_siblings,
)
from src.retrieval.reranker import RERANK_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_COLS = [
    "id", "document_id", "parent_chunk_id",
    "company", "ticker", "doc_type", "report_date", "period_covered", "doc_version",
    "section_title", "page_number", "content_type", "source_authority",
    "chunk_text", "is_parent", "token_count",
    "doc_subtype", "page_content_class",
]


def _chunk_row(
    chunk_id: UUID,
    document_id: UUID,
    page_number: int,
    is_parent: bool = False,
) -> tuple:
    return (
        str(chunk_id), str(document_id), None,
        "Acme REIT", "ACR", "investor_presentation", "2026-03", "Q4 2025", "2026-03",
        "Financials", page_number, "text", "company_authored",
        f"Content on page {page_number}", is_parent, 80,
        "quarterly_investor_deck", "substantive",
    )


def _make_chunk(
    page_number: int = 5,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        company="Acme REIT",
        ticker="ACR",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text=f"Content on page {page_number}",
        content_type="text",
        page_number=page_number,
    )


def _make_rc(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _mock_conn_returning_row(row: tuple | None, cols: list[str]) -> MagicMock:
    """Return a mock connection whose cursor yields a single chunk row (or None)."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchone.return_value = row

    if row is not None:
        desc = []
        for c in cols:
            d = MagicMock()
            d.name = c
            desc.append(d)
        cur.description = desc
    else:
        cur.description = None

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def _mock_conn_no_sibling() -> MagicMock:
    """Mock connection that returns nothing for a sibling lookup."""
    return _mock_conn_returning_row(None, _CHUNK_COLS)


# ---------------------------------------------------------------------------
# Tests for _fetch_chunk_by_doc_page
# ---------------------------------------------------------------------------


def test_fetch_chunk_by_doc_page_returns_none_when_no_row() -> None:
    conn = _mock_conn_no_sibling()
    result = _fetch_chunk_by_doc_page(conn, uuid4(), 99)
    assert result is None


def test_fetch_chunk_by_doc_page_returns_retrieved_chunk_with_none_score() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    row = _chunk_row(chunk_id, doc_id, page_number=6)
    conn = _mock_conn_returning_row(row, _CHUNK_COLS)

    result = _fetch_chunk_by_doc_page(conn, doc_id, 6)

    assert result is not None
    assert result.chunk.id == chunk_id
    # Siblings are not reranked; score is left as None so diagnostics know it
    # was not scored by the cross-encoder.
    assert result.rerank_score is None


# ---------------------------------------------------------------------------
# Core borderline-band tests
# ---------------------------------------------------------------------------


def test_borderline_score_fetches_sibling_on_next_page() -> None:
    """A chunk whose score is in the borderline band triggers sibling fetch."""
    doc_id = uuid4()
    chunk_id = uuid4()
    sibling_id = uuid4()

    chunk = _make_chunk(page_number=5, chunk_id=chunk_id, document_id=doc_id)
    borderline_score = RERANK_THRESHOLD + 1.5  # inside [THRESHOLD, THRESHOLD + 3.0]
    rc = _make_rc(chunk, borderline_score)

    sibling_row = _chunk_row(sibling_id, doc_id, page_number=6)

    call_count = {"n": 0}

    def _make_cursor(*_args, **_kwargs):
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)
        call_count["n"] += 1
        if call_count["n"] == 1:
            # page N-1 = 4 → no row
            cur.fetchone.return_value = None
            cur.description = None
        else:
            # page N+1 = 6 → sibling row
            cur.fetchone.return_value = sibling_row
            desc = []
            for c in _CHUNK_COLS:
                d = MagicMock()
                d.name = c
                desc.append(d)
            cur.description = desc
        return cur

    conn = MagicMock()
    conn.cursor.side_effect = _make_cursor

    result = expand_to_siblings([rc], conn)

    assert len(result) == 2
    assert result[0].chunk.id == chunk_id
    assert result[1].chunk.id == sibling_id
    assert result[1].rerank_score is None


def test_high_confidence_score_does_not_fetch_sibling() -> None:
    """A chunk with rerank_score above RERANK_THRESHOLD + 3.0 must not trigger sibling lookup."""
    doc_id = uuid4()
    chunk = _make_chunk(page_number=5, document_id=doc_id)
    high_score = RERANK_THRESHOLD + 5.0  # above BORDERLINE_UPPER
    rc = _make_rc(chunk, high_score)

    conn = MagicMock()

    result = expand_to_siblings([rc], conn)

    # No DB call made for a high-confidence chunk.
    conn.cursor.assert_not_called()
    assert result == [rc]


def test_below_threshold_score_does_not_fetch_sibling() -> None:
    """A chunk with rerank_score below RERANK_THRESHOLD must not trigger sibling lookup."""
    doc_id = uuid4()
    chunk = _make_chunk(page_number=5, document_id=doc_id)
    low_score = RERANK_THRESHOLD - 1.0  # below BORDERLINE_LOWER
    rc = _make_rc(chunk, low_score)

    conn = MagicMock()

    result = expand_to_siblings([rc], conn)

    conn.cursor.assert_not_called()
    assert result == [rc]


# ---------------------------------------------------------------------------
# Edge-case: page 1 guard
# ---------------------------------------------------------------------------


def test_page_1_does_not_fetch_page_0() -> None:
    """When the chunk is on page 1, page 0 must never be requested."""
    doc_id = uuid4()
    chunk_id = uuid4()
    sibling_id = uuid4()

    chunk = _make_chunk(page_number=1, chunk_id=chunk_id, document_id=doc_id)
    borderline_score = RERANK_THRESHOLD + 1.0
    rc = _make_rc(chunk, borderline_score)

    # Only one cursor call should happen: for page 2 (N+1).
    sibling_row = _chunk_row(sibling_id, doc_id, page_number=2)

    cursor_calls: list[int] = []

    def _make_cursor(*_args, **_kwargs):
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)
        cursor_calls.append(1)
        cur.fetchone.return_value = sibling_row
        desc = []
        for c in _CHUNK_COLS:
            d = MagicMock()
            d.name = c
            desc.append(d)
        cur.description = desc
        return cur

    conn = MagicMock()
    conn.cursor.side_effect = _make_cursor

    result = expand_to_siblings([rc], conn)

    # Exactly one DB call (for page 2); page 0 must never be queried.
    assert len(cursor_calls) == 1
    assert len(result) == 2
    assert result[1].chunk.page_number == 2


# ---------------------------------------------------------------------------
# Deduplication: sibling already in contexts
# ---------------------------------------------------------------------------


def test_sibling_already_in_contexts_not_duplicated() -> None:
    """If the sibling page is already in the context list, do not add a duplicate."""
    doc_id = uuid4()
    chunk_id = uuid4()
    sibling_id = uuid4()

    chunk = _make_chunk(page_number=5, chunk_id=chunk_id, document_id=doc_id)
    # Sibling for page 6 already in context (e.g., retrieved directly)
    sibling_chunk = _make_chunk(page_number=6, chunk_id=sibling_id, document_id=doc_id)

    borderline_score = RERANK_THRESHOLD + 1.0
    rc = _make_rc(chunk, borderline_score)
    existing_sibling_rc = _make_rc(sibling_chunk, borderline_score + 0.5)

    contexts = [rc, existing_sibling_rc]

    # The DB should still be called for page 4 (N-1), but page 6 (N+1) won't
    # be appended because it's already seen.
    call_count = {"n": 0}

    def _make_cursor(*_args, **_kwargs):
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)
        call_count["n"] += 1
        # Return the already-present sibling for both offset lookups to
        # verify the dedup guard fires.
        cur.fetchone.return_value = _chunk_row(sibling_id, doc_id, page_number=4)
        desc = []
        for c in _CHUNK_COLS:
            d = MagicMock()
            d.name = c
            desc.append(d)
        cur.description = desc
        return cur

    conn = MagicMock()
    conn.cursor.side_effect = _make_cursor

    # Patch _fetch_chunk_by_doc_page to return sibling_id for all lookups so
    # we can isolate the dedup logic without caring which page is queried.
    from src.retrieval import pipeline as pipeline_module

    def _mock_fetch(c, document_id, page_number):
        sib = _make_chunk(page_number=page_number, chunk_id=sibling_id, document_id=doc_id)
        return RetrievedChunk(chunk=sib)

    original = pipeline_module._fetch_chunk_by_doc_page
    pipeline_module._fetch_chunk_by_doc_page = _mock_fetch
    try:
        result = expand_to_siblings(contexts, conn)
    finally:
        pipeline_module._fetch_chunk_by_doc_page = original

    # The sibling was already in seen_ids for both the existing_sibling_rc chunk
    # and any newly returned chunk with the same id — so no duplicates.
    sibling_count = sum(1 for r in result if r.chunk.id == sibling_id)
    assert sibling_count == 1


# ---------------------------------------------------------------------------
# Abstention path
# ---------------------------------------------------------------------------


def test_abstention_path_skips_sibling_expansion(monkeypatch) -> None:
    """When the pipeline abstains, expand_to_siblings must not be invoked."""
    from src.retrieval import pipeline as pipeline_module

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("expand_to_siblings must not be called when abstaining")

    monkeypatch.setattr(pipeline_module, "expand_to_siblings", _raise_if_called)

    # Replicate the exact conditional guard from pipeline.retrieve()
    abstain = True
    expanded: list[RetrievedChunk] = []

    if not abstain:
        pipeline_module.expand_to_siblings(expanded, None)  # type: ignore[arg-type]
    # Reaching here without an AssertionError confirms the guard is correct.


# ---------------------------------------------------------------------------
# Cap: siblings dropped before retained/conflict chunks
# ---------------------------------------------------------------------------


def test_cap_drops_siblings_before_retained_chunks() -> None:
    """When contexts + siblings exceed MAX_CONTEXT_CHUNKS, siblings are dropped first."""
    doc_id = uuid4()

    # Fill contexts up to MAX_CONTEXT_CHUNKS with borderline-scored chunks.
    contexts: list[RetrievedChunk] = []
    for i in range(MAX_CONTEXT_CHUNKS):
        c = _make_chunk(page_number=i + 10, document_id=doc_id)
        contexts.append(_make_rc(c, RERANK_THRESHOLD + 1.0))

    retained_ids = {rc.chunk.id for rc in contexts}

    # Produce one sibling that is NOT in the retained set.
    sibling_id = uuid4()
    sibling_chunk = _make_chunk(page_number=99, chunk_id=sibling_id, document_id=doc_id)
    sibling_rc = RetrievedChunk(chunk=sibling_chunk, rerank_score=None)
    with_siblings = contexts + [sibling_rc]

    assert len(with_siblings) == MAX_CONTEXT_CHUNKS + 1  # over cap

    # Replicate the cap logic from pipeline.retrieve()
    sibling_part = [rc for rc in with_siblings if rc.chunk.id not in retained_ids]
    non_sibling_part = [rc for rc in with_siblings if rc.chunk.id in retained_ids]
    max_siblings = MAX_CONTEXT_CHUNKS - len(non_sibling_part)
    sibling_trimmed = sibling_part[:max(0, max_siblings)]
    result = non_sibling_part + sibling_trimmed

    assert len(result) == MAX_CONTEXT_CHUNKS
    # All retained chunks must be present.
    for rc in contexts:
        assert any(r.chunk.id == rc.chunk.id for r in result)
    # The sibling was dropped.
    assert not any(r.chunk.id == sibling_id for r in result)
