"""Unit tests for table-pair retrieval expansion stage (2c).

Verifies that when a text chunk on page N is in the retained set, any
table/chart_description chunk on the same (document_id, page_number) is
automatically included — even if it didn't clear the rerank top-N.

All tests use mocked DB connections — no real database or network access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import expand_table_pairs


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


def _make_chunk(
    page_number: int = 5,
    content_type: str = "text",
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
        chunk_text=f"Content on page {page_number} ({content_type})",
        content_type=content_type,
        page_number=page_number,
    )


def _make_rc(chunk: Chunk, score: float = 5.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _chunk_row(
    chunk_id: UUID,
    document_id: UUID,
    page_number: int,
    content_type: str = "table",
) -> tuple:
    return (
        str(chunk_id), str(document_id), None,
        "Acme REIT", "ACR", "investor_presentation", "2026-03", "Q4 2025", "2026-03",
        "Financials", page_number, content_type, "company_authored",
        f"Table on page {page_number}", False, 80,
        "quarterly_investor_deck", "substantive",
    )


def _mock_conn_with_single_row(row: tuple | None) -> MagicMock:
    """Return a mock connection whose cursor returns one row (or None)."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = [row] if row is not None else []

    if row is not None:
        desc = []
        for c in _CHUNK_COLS:
            d = MagicMock()
            d.name = c
            desc.append(d)
        cur.description = desc
    else:
        cur.description = None

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def _mock_conn_no_table() -> MagicMock:
    """Mock connection that returns no table chunks."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = []
    cur.description = None
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_table_pair_added_for_text_chunk_on_same_page() -> None:
    """When a text chunk is in top-N and a table chunk exists on same page, include it."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0)

    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_single_row(table_row)

    result = expand_table_pairs([rc], conn)

    chunk_ids = {r.chunk.id for r in result}
    assert text_chunk_id in chunk_ids, "original text chunk must remain"
    assert table_chunk_id in chunk_ids, "table chunk from same page must be added"


def test_chart_description_added_for_text_chunk_on_same_page() -> None:
    """chart_description content type on same page should also be expanded."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    chart_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=7, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=3.0)

    chart_row = _chunk_row(chart_chunk_id, doc_id, page_number=7, content_type="chart_description")
    conn = _mock_conn_with_single_row(chart_row)

    result = expand_table_pairs([rc], conn)

    chunk_ids = {r.chunk.id for r in result}
    assert chart_chunk_id in chunk_ids, "chart_description chunk from same page must be added"


def test_table_already_in_contexts_not_duplicated() -> None:
    """If the table chunk is already in the retained set, do not duplicate it."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    table_chunk = _make_chunk(page_number=10, content_type="table", chunk_id=table_chunk_id, document_id=doc_id)

    rc_text = _make_rc(text_chunk, score=5.0)
    rc_table = _make_rc(table_chunk, score=2.0)

    # Both already in context
    contexts = [rc_text, rc_table]

    # DB would return the table chunk
    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_single_row(table_row)

    result = expand_table_pairs(contexts, conn)

    table_count = sum(1 for r in result if r.chunk.id == table_chunk_id)
    assert table_count == 1, "table chunk must appear exactly once (no duplicate)"


def test_non_text_chunk_does_not_trigger_lookup() -> None:
    """Only text content_type chunks trigger table-pair lookup; tables/charts do not."""
    doc_id = uuid4()
    table_chunk_id = uuid4()

    # A table chunk already in the retained set
    table_chunk = _make_chunk(page_number=5, content_type="table", chunk_id=table_chunk_id, document_id=doc_id)
    rc = _make_rc(table_chunk, score=4.0)

    conn = _mock_conn_no_table()

    result = expand_table_pairs([rc], conn)

    # No DB call should be triggered (table is not a text chunk)
    conn.cursor.assert_not_called()
    assert result == [rc]


def test_text_chunk_without_page_number_skipped() -> None:
    """Text chunks with page_number=None must not trigger table-pair lookup."""
    doc_id = uuid4()
    chunk_id = uuid4()

    chunk = _make_chunk(page_number=1, content_type="text", chunk_id=chunk_id, document_id=doc_id)
    chunk.page_number = None  # simulate missing page
    rc = _make_rc(chunk, score=4.0)

    conn = _mock_conn_no_table()

    result = expand_table_pairs([rc], conn)

    conn.cursor.assert_not_called()
    assert len(result) == 1


def test_no_table_on_page_returns_original_list() -> None:
    """When no table/chart chunk exists for the page, return original list unchanged."""
    doc_id = uuid4()
    text_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=3, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0)

    conn = _mock_conn_no_table()

    result = expand_table_pairs([rc], conn)

    assert len(result) == 1
    assert result[0].chunk.id == text_chunk_id


def test_table_pair_expanded_chunk_has_none_rerank_score() -> None:
    """Table-pair expanded chunks must have rerank_score=None to distinguish from reranked chunks."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0)

    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_single_row(table_row)

    result = expand_table_pairs([rc], conn)

    expanded = [r for r in result if r.chunk.id == table_chunk_id]
    assert len(expanded) == 1
    assert expanded[0].rerank_score is None, "table-pair expanded chunk must have rerank_score=None"


def test_multiple_text_chunks_trigger_multiple_lookups() -> None:
    """Multiple text chunks on different pages each trigger separate table-pair lookups."""
    doc_id = uuid4()
    text1_id = uuid4()
    text2_id = uuid4()
    table1_id = uuid4()
    table2_id = uuid4()

    text1 = _make_chunk(page_number=5, content_type="text", chunk_id=text1_id, document_id=doc_id)
    text2 = _make_chunk(page_number=8, content_type="text", chunk_id=text2_id, document_id=doc_id)

    rc1 = _make_rc(text1, score=5.0)
    rc2 = _make_rc(text2, score=4.0)

    call_count = {"n": 0}

    def _make_cursor(*_args, **_kwargs):
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)
        call_count["n"] += 1
        if call_count["n"] == 1:
            row = _chunk_row(table1_id, doc_id, page_number=5, content_type="table")
        else:
            row = _chunk_row(table2_id, doc_id, page_number=8, content_type="table")
        cur.fetchall.return_value = [row]
        desc = []
        for c in _CHUNK_COLS:
            d = MagicMock()
            d.name = c
            desc.append(d)
        cur.description = desc
        return cur

    conn = MagicMock()
    conn.cursor.side_effect = _make_cursor

    result = expand_table_pairs([rc1, rc2], conn)

    chunk_ids = {r.chunk.id for r in result}
    assert table1_id in chunk_ids
    assert table2_id in chunk_ids
    assert len(result) == 4  # 2 text + 2 table


def test_cap_prefers_table_pair_over_sibling_when_trimming() -> None:
    """When `_apply_post_rerank_pipeline` caps the expanded set, table_pair_expanded
    chunks must survive in preference to sibling_expanded chunks.

    Regression test for the o-basic-1 closure: the table chunks were being
    pulled in correctly by `expand_table_pairs` but dropped by the cap because
    insertion order placed sibling-expanded chunks first in `expanded_part`.
    The fix sorts by stage priority (table-pair > sibling > other).
    """
    from src.retrieval.pipeline import _apply_post_rerank_pipeline
    from unittest.mock import patch

    doc_id = uuid4()
    # Build 7 "core" chunks (rerank + parent + conflict). The cap is 10 so
    # we have room for 3 expansion chunks total. We'll have 3 sibling + 2
    # table-pair → 5 expansion candidates → 3 must be kept. Table-pair (2)
    # MUST be preferred over sibling (3).
    core = [_make_rc(_make_chunk(page_number=p, chunk_id=uuid4(), document_id=doc_id)) for p in range(1, 8)]
    siblings = []
    for p in range(8, 11):
        rc = _make_rc(_make_chunk(page_number=p, chunk_id=uuid4(), document_id=doc_id), score=0.0)
        rc.rerank_score = None
        rc.retrieval_stage = "sibling_expanded"
        siblings.append(rc)
    table_pairs = []
    for p in range(11, 13):
        rc = _make_rc(_make_chunk(page_number=p, content_type="table", chunk_id=uuid4(), document_id=doc_id), score=0.0)
        rc.rerank_score = None
        rc.retrieval_stage = "table_pair_expanded"
        table_pairs.append(rc)

    # Mock out the three expansion functions so the pipeline returns the
    # input expansions exactly.
    with patch("src.retrieval.pipeline.expand_to_parents", side_effect=lambda c, conn: list(c)), \
         patch("src.retrieval.pipeline.expand_to_siblings", side_effect=lambda c, conn: list(c) + siblings), \
         patch("src.retrieval.pipeline.expand_table_pairs", side_effect=lambda c, conn: list(c) + table_pairs), \
         patch("src.retrieval.pipeline.find_conflicting_chunks", side_effect=lambda c, cf, conn: list(c)):
        conn = MagicMock()
        result = _apply_post_rerank_pipeline(
            list(core),
            intent="latest",
            max_chunks=10,
            company_filter=None,
            abstain=False,
            conn=conn,
        )

    surviving_table_pair = [r for r in result if r.retrieval_stage == "table_pair_expanded"]
    surviving_sibling = [r for r in result if r.retrieval_stage == "sibling_expanded"]
    assert len(result) == 10, f"expected cap=10, got {len(result)}"
    assert len(surviving_table_pair) == 2, (
        f"both table-pair chunks should survive cap; got {len(surviving_table_pair)}"
    )
    # 3 expansion slots total - 2 table-pair = 1 sibling survives
    assert len(surviving_sibling) == 1, (
        f"only 1 sibling slot left after table-pair; got {len(surviving_sibling)}"
    )
