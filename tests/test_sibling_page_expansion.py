"""Unit tests for sibling-page retrieval expansion stage.

Verifies that when a reranked chunk on page N is in the retained set, additional
same-page chunks (of any content type) are pulled in as candidates so the LLM
sees the page in cohesion. Addresses the sparse-microchunk failure mode where
the target chunk's text alone has too little keyword overlap to enter BM25 top-K,
and the split-entity mode where evidence lives in a sibling chunk on the same
page.

All tests mock the DB cursor — no real database access.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import expand_sibling_pages


# ---------------------------------------------------------------------------
# Mock helpers — mirror the test_table_pair_expansion pattern
# ---------------------------------------------------------------------------

_CHUNK_COLS = [
    "id", "document_id", "parent_chunk_id",
    "company", "ticker", "doc_type", "report_date", "period_covered", "doc_version",
    "section_title", "page_number", "content_type", "source_authority",
    "chunk_text", "is_parent", "token_count",
    "doc_subtype", "page_content_class",
]


def _make_chunk(
    *,
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
        content_type=content_type,  # type: ignore[arg-type]
        page_number=page_number,
    )


def _make_rc(chunk: Chunk, score: float | None = 5.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _chunk_row(
    chunk_id: UUID,
    document_id: UUID,
    page_number: int,
    content_type: str = "text",
) -> tuple:
    return (
        str(chunk_id), str(document_id), None,
        "Acme REIT", "ACR", "investor_presentation", "2026-03", "Q4 2025", "2026-03",
        "Financials", page_number, content_type, "company_authored",
        f"Sibling on page {page_number} ({content_type})", False, 80,
        "quarterly_investor_deck", "substantive",
    )


def _mock_conn_with_rows(rows: list[tuple]) -> MagicMock:
    """Mock cursor yielding the given list of rows."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = list(rows)

    if rows:
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


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_same_page_sibling_added_for_reranked_chunk() -> None:
    """A reranked chunk on page N triggers pull of OTHER same-page chunks."""
    doc_id = uuid4()
    trigger_id = uuid4()
    sibling_id = uuid4()

    trigger = _make_chunk(page_number=3, chunk_id=trigger_id, document_id=doc_id)
    rc = _make_rc(trigger, score=5.0)

    sibling_row = _chunk_row(sibling_id, doc_id, page_number=3, content_type="text")
    conn = _mock_conn_with_rows([sibling_row])

    result = expand_sibling_pages([rc], conn)

    ids = {r.chunk.id for r in result}
    assert trigger_id in ids, "original chunk must remain"
    assert sibling_id in ids, "same-page sibling must be added"


def test_sibling_page_expansion_caps_at_max_per_page() -> None:
    """When DB returns more siblings than max_per_page, only max_per_page are added."""
    doc_id = uuid4()
    trigger = _make_chunk(page_number=12, document_id=doc_id)
    rc = _make_rc(trigger, score=4.0)

    rows = [
        _chunk_row(uuid4(), doc_id, page_number=12) for _ in range(10)
    ]
    conn = _mock_conn_with_rows(rows)

    result = expand_sibling_pages([rc], conn, max_per_page=4)

    sibling_count = sum(
        1 for r in result if r.retrieval_stage == "sibling_page_expanded"
    )
    assert sibling_count == 4, (
        f"max_per_page=4 must cap additions; got {sibling_count}"
    )


def test_no_op_when_db_returns_no_siblings() -> None:
    """A page with no other chunks returns the input list unchanged."""
    doc_id = uuid4()
    trigger = _make_chunk(page_number=99, document_id=doc_id)
    rc = _make_rc(trigger, score=3.0)
    conn = _mock_conn_with_rows([])

    result = expand_sibling_pages([rc], conn)

    assert len(result) == 1
    assert result[0] is rc


def test_chunks_without_page_number_skipped() -> None:
    """Parent chunks (page_number=None) must not trigger expansion."""
    doc_id = uuid4()
    parent = _make_chunk(page_number=10, document_id=doc_id)
    parent.page_number = None  # type: ignore[assignment]
    rc = _make_rc(parent, score=5.0)

    conn = MagicMock()
    result = expand_sibling_pages([rc], conn)

    assert len(result) == 1
    conn.cursor.assert_not_called()


def test_unscored_expanded_chunks_do_not_re_trigger() -> None:
    """Chunks already from a prior expansion (rerank_score=None) must not
    trigger another lookup, preventing cascade expansion."""
    doc_id = uuid4()
    already_expanded = _make_chunk(page_number=8, document_id=doc_id)
    rc = _make_rc(already_expanded, score=None)  # came from a prior expansion stage

    conn = MagicMock()
    result = expand_sibling_pages([rc], conn)

    assert result == [rc]
    conn.cursor.assert_not_called()


def test_same_page_visited_only_once() -> None:
    """When two reranked chunks share a page, the DB is queried once per
    (document_id, page_number), not per trigger."""
    doc_id = uuid4()
    t1 = _make_rc(_make_chunk(page_number=4, document_id=doc_id), score=5.0)
    t2 = _make_rc(_make_chunk(page_number=4, document_id=doc_id), score=4.5)

    conn = _mock_conn_with_rows([])
    expand_sibling_pages([t1, t2], conn)

    # cursor() called only once for the single (doc, page=4) lookup
    assert conn.cursor.call_count == 1


def test_appended_chunk_carries_retrieval_stage_tag() -> None:
    """Appended siblings have retrieval_stage='sibling_page_expanded' and a
    populated trigger_chunk_id + expansion_reason."""
    doc_id = uuid4()
    trigger_id = uuid4()
    sibling_id = uuid4()

    trigger = _make_chunk(page_number=7, chunk_id=trigger_id, document_id=doc_id)
    rc = _make_rc(trigger, score=4.0)
    row = _chunk_row(sibling_id, doc_id, page_number=7, content_type="table")
    conn = _mock_conn_with_rows([row])

    result = expand_sibling_pages([rc], conn)
    added = [r for r in result if r.chunk.id == sibling_id]
    assert len(added) == 1
    appended = added[0]
    assert appended.retrieval_stage == "sibling_page_expanded"
    assert appended.trigger_chunk_id == str(trigger_id)
    assert "page 7" in (appended.expansion_reason or "")
    assert appended.rerank_score is None


def test_chunks_already_in_retained_set_not_duplicated() -> None:
    """If a sibling chunk is already in the retained set (e.g. surfaced by
    rerank independently), it must not be added again."""
    doc_id = uuid4()
    t_id = uuid4()
    s_id = uuid4()

    t = _make_rc(_make_chunk(page_number=2, chunk_id=t_id, document_id=doc_id), score=5.0)
    s = _make_rc(_make_chunk(page_number=2, chunk_id=s_id, document_id=doc_id), score=4.0)

    # The DB query excludes already-retained ids via SQL filter, so it returns []
    conn = _mock_conn_with_rows([])
    result = expand_sibling_pages([t, s], conn)

    assert len(result) == 2
    ids = [r.chunk.id for r in result]
    assert ids.count(t_id) == 1
    assert ids.count(s_id) == 1
