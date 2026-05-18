"""Unit tests for per-chunk retrieval provenance fields (2g).

Verifies that retrieval_stage, trigger_chunk_id, and expansion_reason are
populated correctly for each expansion type in the pipeline.

All tests use mocked structures — no real database or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    enforce_per_issuer_floor,
    expand_table_pairs,
    expand_to_siblings,
)


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
    company: str = "Acme REIT",
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        company=company,
        ticker="ACR",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text=f"Content on page {page_number} ({content_type})",
        content_type=content_type,
        page_number=page_number,
    )


def _make_rc(
    chunk: Chunk,
    score: float = 5.0,
    stage: str | None = None,
) -> RetrievedChunk:
    rc = RetrievedChunk(chunk=chunk, rerank_score=score)
    rc.retrieval_stage = stage
    return rc


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
        f"Content on page {page_number}", False, 80,
        "quarterly_investor_deck", "substantive",
    )


def _mock_conn_with_row(row: tuple) -> MagicMock:
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = [row]
    desc = []
    for c in _CHUNK_COLS:
        d = MagicMock()
        d.name = c
        desc.append(d)
    cur.description = desc
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


# ---------------------------------------------------------------------------
# Tests: table_pair_expanded provenance
# ---------------------------------------------------------------------------


def test_table_pair_expanded_has_retrieval_stage() -> None:
    """Chunks added by expand_table_pairs must have retrieval_stage='table_pair_expanded'."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0, stage="reranked")

    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_row(table_row)

    result = expand_table_pairs([rc], conn)

    expanded = [r for r in result if r.chunk.id == table_chunk_id]
    assert len(expanded) == 1
    assert expanded[0].retrieval_stage == "table_pair_expanded"


def test_table_pair_expanded_has_trigger_chunk_id() -> None:
    """Chunks added by expand_table_pairs must have trigger_chunk_id set to the text chunk's ID."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0)

    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_row(table_row)

    result = expand_table_pairs([rc], conn)

    expanded = [r for r in result if r.chunk.id == table_chunk_id]
    assert len(expanded) == 1
    assert expanded[0].trigger_chunk_id == str(text_chunk_id)


def test_table_pair_expanded_has_expansion_reason() -> None:
    """Chunks added by expand_table_pairs must have a non-empty expansion_reason."""
    doc_id = uuid4()
    text_chunk_id = uuid4()
    table_chunk_id = uuid4()

    text_chunk = _make_chunk(page_number=10, content_type="text", chunk_id=text_chunk_id, document_id=doc_id)
    rc = _make_rc(text_chunk, score=5.0)

    table_row = _chunk_row(table_chunk_id, doc_id, page_number=10, content_type="table")
    conn = _mock_conn_with_row(table_row)

    result = expand_table_pairs([rc], conn)

    expanded = [r for r in result if r.chunk.id == table_chunk_id]
    assert expanded[0].expansion_reason is not None
    assert len(expanded[0].expansion_reason) > 0


# ---------------------------------------------------------------------------
# Tests: per_issuer_floor provenance
# ---------------------------------------------------------------------------


def test_per_issuer_floor_has_retrieval_stage() -> None:
    """Chunks added by enforce_per_issuer_floor must have retrieval_stage='per_issuer_floor'."""
    companies = ["Alpha REIT", "Beta REIT"]
    retained = [_make_rc(_make_chunk(company="Alpha REIT"), score=5.0, stage="reranked")]
    best = _make_rc(_make_chunk(company="Beta REIT"), score=1.5)
    candidates_by_company = {
        "Alpha REIT": [],
        "Beta REIT": [best],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    floor_chunks = [r for r in result if r.chunk.company == "Beta REIT"]
    assert len(floor_chunks) == 1
    assert floor_chunks[0].retrieval_stage == "per_issuer_floor"


def test_per_issuer_floor_has_expansion_reason() -> None:
    """Chunks added by enforce_per_issuer_floor must have a non-empty expansion_reason."""
    companies = ["Alpha REIT", "Beta REIT"]
    retained = [_make_rc(_make_chunk(company="Alpha REIT"), score=5.0)]
    best = _make_rc(_make_chunk(company="Beta REIT"), score=1.5)
    candidates_by_company = {
        "Alpha REIT": [],
        "Beta REIT": [best],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    floor_chunks = [r for r in result if r.chunk.company == "Beta REIT"]
    assert floor_chunks[0].expansion_reason is not None
    assert "Beta REIT" in floor_chunks[0].expansion_reason


# ---------------------------------------------------------------------------
# Tests: RetrievedChunk dataclass fields exist
# ---------------------------------------------------------------------------


def test_retrieved_chunk_has_provenance_fields() -> None:
    """RetrievedChunk dataclass must have retrieval_stage, trigger_chunk_id, expansion_reason."""
    chunk = _make_chunk()
    rc = RetrievedChunk(chunk=chunk)
    assert hasattr(rc, "retrieval_stage")
    assert hasattr(rc, "trigger_chunk_id")
    assert hasattr(rc, "expansion_reason")
    # Default values are None
    assert rc.retrieval_stage is None
    assert rc.trigger_chunk_id is None
    assert rc.expansion_reason is None


def test_retrieved_chunk_provenance_fields_settable() -> None:
    """Provenance fields must be settable on RetrievedChunk instances."""
    chunk = _make_chunk()
    rc = RetrievedChunk(chunk=chunk)
    rc.retrieval_stage = "reranked"
    rc.trigger_chunk_id = "some-uuid"
    rc.expansion_reason = "test reason"
    assert rc.retrieval_stage == "reranked"
    assert rc.trigger_chunk_id == "some-uuid"
    assert rc.expansion_reason == "test reason"
