"""Unit tests for per-(document_id, page_number) cap on retained chunks.

Verifies that ``_cap_chunks_per_page`` enforces ``MAX_CHUNKS_PER_PAGE`` and
trims excess chunks using the documented stage priority order.
"""

from __future__ import annotations

from uuid import UUID, uuid4


from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    MAX_CHUNKS_PER_PAGE,
    _cap_chunks_per_page,
)


def _make_chunk(
    *,
    page_number: int = 1,
    document_id: UUID | None = None,
    content_type: str = "text",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=document_id or uuid4(),
        company="Acme REIT",
        ticker="ACR",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text=f"Content on page {page_number}",
        content_type=content_type,  # type: ignore[arg-type]
        page_number=page_number,
    )


def _make_rc(chunk: Chunk, stage: str | None, score: float = 5.0) -> RetrievedChunk:
    rc = RetrievedChunk(chunk=chunk, rerank_score=score)
    rc.retrieval_stage = stage
    return rc


def test_cap_constant_is_three() -> None:
    """Documented cap is 3 per (document_id, page_number)."""
    assert MAX_CHUNKS_PER_PAGE == 3


def test_under_cap_pages_untouched() -> None:
    """Pages with fewer than max chunks are passed through unchanged."""
    doc_id = uuid4()
    chunks = [
        _make_rc(_make_chunk(page_number=1, document_id=doc_id), "reranked"),
        _make_rc(_make_chunk(page_number=1, document_id=doc_id), "parent_expanded"),
    ]
    result = _cap_chunks_per_page(chunks)
    assert len(result) == 2


def test_cap_trims_excess_keeping_priority_order() -> None:
    """5 chunks on same page across 3 stages → cap retains top 3 by priority."""
    doc_id = uuid4()
    reranked = _make_rc(_make_chunk(page_number=4, document_id=doc_id), "reranked")
    parent_a = _make_rc(_make_chunk(page_number=4, document_id=doc_id), "parent_expanded")
    table_pair = _make_rc(
        _make_chunk(page_number=4, document_id=doc_id, content_type="table"),
        "table_pair_expanded",
    )
    sibling_page = _make_rc(
        _make_chunk(page_number=4, document_id=doc_id), "sibling_page_expanded"
    )
    sibling_adjacent = _make_rc(
        _make_chunk(page_number=4, document_id=doc_id), "sibling_expanded"
    )

    contexts = [reranked, parent_a, table_pair, sibling_page, sibling_adjacent]
    result = _cap_chunks_per_page(contexts)

    assert len(result) == MAX_CHUNKS_PER_PAGE
    kept_stages = {rc.retrieval_stage for rc in result}
    # Priority order: reranked > entity_anchored > parent_expanded > table_pair_expanded
    # > sibling_page_expanded > sibling_expanded. Top 3 from this set:
    assert kept_stages == {"reranked", "parent_expanded", "table_pair_expanded"}
    # Lower-priority stages must be dropped.
    assert "sibling_page_expanded" not in kept_stages
    assert "sibling_expanded" not in kept_stages


def test_different_pages_capped_independently() -> None:
    """Two pages with 5 chunks each both get trimmed to 3, independently."""
    doc_id = uuid4()
    page1 = [
        _make_rc(_make_chunk(page_number=1, document_id=doc_id), stage)
        for stage in (
            "reranked",
            "parent_expanded",
            "table_pair_expanded",
            "sibling_page_expanded",
            "sibling_expanded",
        )
    ]
    page2 = [
        _make_rc(_make_chunk(page_number=2, document_id=doc_id), stage)
        for stage in (
            "reranked",
            "entity_anchored",
            "table_pair_expanded",
            "sibling_page_expanded",
            "sibling_expanded",
        )
    ]
    result = _cap_chunks_per_page(page1 + page2)
    by_page: dict[int, int] = {}
    for rc in result:
        page = rc.chunk.page_number
        assert page is not None
        by_page[page] = by_page.get(page, 0) + 1
    assert by_page[1] == 3
    assert by_page[2] == 3


def test_chunks_without_page_number_passed_through() -> None:
    """Parent chunks with page_number=None must be kept (not lumped under one cap)."""
    parent_a = _make_chunk(page_number=1)
    parent_a.page_number = None  # type: ignore[assignment]
    parent_b = _make_chunk(page_number=2)
    parent_b.page_number = None  # type: ignore[assignment]
    contexts = [_make_rc(parent_a, "parent_expanded"), _make_rc(parent_b, "parent_expanded")]
    result = _cap_chunks_per_page(contexts)
    assert len(result) == 2


def test_different_documents_same_page_number_not_combined() -> None:
    """Same page_number in different documents are NOT considered the same page."""
    doc_a = uuid4()
    doc_b = uuid4()
    chunks = [
        _make_rc(_make_chunk(page_number=5, document_id=doc_a), "reranked"),
        _make_rc(_make_chunk(page_number=5, document_id=doc_a), "parent_expanded"),
        _make_rc(_make_chunk(page_number=5, document_id=doc_b), "reranked"),
        _make_rc(_make_chunk(page_number=5, document_id=doc_b), "parent_expanded"),
    ]
    result = _cap_chunks_per_page(chunks)
    # Each (doc, page=5) has 2 chunks — both under cap, all preserved.
    assert len(result) == 4
