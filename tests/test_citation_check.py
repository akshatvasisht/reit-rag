"""Tests for check_citations and _normalize_doc_type.

Regression tests for 2b citation-format follow-up: when the LLM emits the
extended `[Company, Doc Type (Subtype), Month Year, p.N]` format, the
matching index built from chunks (which carries doc_type only) must still
recognise the citation as supported.
"""

from __future__ import annotations

from uuid import uuid4

from src.generation.citation_check import (
    _normalize_doc_type,
    check_citations,
)
from src.models import Chunk, RetrievedChunk


def _make_chunk(
    company: str = "Digital Realty",
    doc_type: str = "investor_presentation",
    report_date: str = "2026-03",
    page: int | None = 34,
    doc_subtype: str | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type=doc_type,
        report_date=report_date,
        doc_version=report_date,
        chunk_text="placeholder",
        content_type="text",
        page_number=page,
        doc_subtype=doc_subtype,
    )


def test_normalize_doc_type_strips_subtype() -> None:
    """Parenthetical subtype is stripped so base doc_type comparison works."""
    assert _normalize_doc_type("Investor Presentation (Company Update)") == "investor presentation"
    assert _normalize_doc_type("Investor Presentation") == "investor presentation"
    assert _normalize_doc_type("investor_presentation") == "investor presentation"
    assert _normalize_doc_type("Investor Presentation (Investor Day Session)") == "investor presentation"


def test_check_citations_accepts_subtype_decorated_citation() -> None:
    """Citation with `(Subtype)` matches a chunk whose base doc_type is plain."""
    chunk = _make_chunk(doc_subtype="quarterly_investor_deck")
    answer_text = (
        "Net debt was 4.9x [Digital Realty, Investor Presentation (Company Update), March 2026, p.34]."
    )
    report = check_citations(answer_text, [RetrievedChunk(chunk=chunk)])
    assert report.total == 1
    assert report.supported == 1
    assert report.unsupported == []


def test_check_citations_accepts_plain_citation_after_2b() -> None:
    """Old `[Company, Doc Type, Month Year, p.N]` citations still match."""
    chunk = _make_chunk()
    answer_text = (
        "Net debt was 4.9x [Digital Realty, Investor Presentation, March 2026, p.34]."
    )
    report = check_citations(answer_text, [RetrievedChunk(chunk=chunk)])
    assert report.supported == 1


def test_check_citations_subtype_does_not_make_wrong_page_match() -> None:
    """Citation with different page must remain unsupported even if subtype strips."""
    chunk = _make_chunk(page=34)
    answer_text = (
        "Net debt was 4.9x [Digital Realty, Investor Presentation (Company Update), March 2026, p.99]."
    )
    report = check_citations(answer_text, [RetrievedChunk(chunk=chunk)])
    assert report.supported == 0
    assert len(report.unsupported) == 1


def test_check_citations_collision_disambiguated_both_supported() -> None:
    """Two chunks with same (company, doc_type, date) and different subtypes —
    both subtype-decorated citations are supported."""
    chunk_a = _make_chunk(doc_subtype="investor_day_session", page=10)
    chunk_b = _make_chunk(doc_subtype="quarterly_investor_deck", page=20)
    answer_text = (
        "First [Digital Realty, Investor Presentation (Investor Day Session), March 2026, p.10]. "
        "Second [Digital Realty, Investor Presentation (Quarterly Investor Deck), March 2026, p.20]."
    )
    report = check_citations(
        answer_text,
        [RetrievedChunk(chunk=chunk_a), RetrievedChunk(chunk=chunk_b)],
    )
    assert report.supported == 2
    assert report.unsupported == []
