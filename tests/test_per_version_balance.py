"""Unit tests for per-version top-K rerank balancing.

Ensures that comparison and conflict intents balance retrieval across
version groups instead of letting a single version dominate the rerank
top-N by lexical match on query phrasing.
"""

from __future__ import annotations

from uuid import uuid4

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    PER_VERSION_K_COMPARISON,
    _balance_per_version,
)


def _make_chunk(
    company: str = "Digital Realty",
    doc_subtype: str = "company_update",
    report_date: str = "2026-03",
    page_number: int = 1,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type="investor_presentation",
        report_date=report_date,
        doc_version=report_date,
        chunk_text="body",
        content_type="text",
        page_number=page_number,
        doc_subtype=doc_subtype,
    )


def _make_rc(score: float, **chunk_kwargs) -> RetrievedChunk:
    return RetrievedChunk(chunk=_make_chunk(**chunk_kwargs), rerank_score=score)


def test_balance_no_op_for_latest_intent() -> None:
    rcs = [_make_rc(score=float(i), page_number=i) for i in range(10, 0, -1)]
    out = _balance_per_version(rcs, "latest", total_budget=5)
    assert [rc.chunk.page_number for rc in out] == [10, 9, 8, 7, 6]


def test_balance_no_op_with_single_version_group() -> None:
    rcs = [_make_rc(score=float(i), page_number=i) for i in range(10, 0, -1)]
    out = _balance_per_version(rcs, "comparison", total_budget=5)
    assert [rc.chunk.page_number for rc in out] == [10, 9, 8, 7, 6]


def test_balance_takes_per_version_top_k_when_multiple_versions_present() -> None:
    """Mar deck top-K and Dec deck top-K both surface, even when Dec's scores
    are uniformly lower."""
    mar = [
        _make_rc(score=2.5, doc_subtype="company_update", report_date="2026-03", page_number=34),
        _make_rc(score=1.7, doc_subtype="company_update", report_date="2026-03", page_number=4),
        _make_rc(score=1.3, doc_subtype="company_update", report_date="2026-03", page_number=34),
        _make_rc(score=1.0, doc_subtype="company_update", report_date="2026-03", page_number=33),
        _make_rc(score=0.9, doc_subtype="company_update", report_date="2026-03", page_number=33),
        _make_rc(score=0.5, doc_subtype="company_update", report_date="2026-03", page_number=34),
    ]
    dec = [
        _make_rc(score=0.4, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=3),
        _make_rc(score=-2.1, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=26),
        _make_rc(score=-2.3, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=35),
        _make_rc(score=-3.2, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=36),
    ]
    out = _balance_per_version(mar + dec, "comparison", total_budget=8)
    mar_count = sum(1 for rc in out if rc.chunk.doc_subtype == "company_update")
    dec_count = sum(1 for rc in out if rc.chunk.doc_subtype == "quarterly_investor_deck")
    # With per_version_k=4 and total_budget=8, expect 4 each.
    assert mar_count == PER_VERSION_K_COMPARISON
    assert dec_count == PER_VERSION_K_COMPARISON


def test_balance_trims_to_total_budget() -> None:
    """When per_version_k * n_versions exceeds total_budget, trim by global
    rerank_score descending."""
    v1 = [_make_rc(score=float(s), doc_subtype="company_update", report_date="2026-03", page_number=s) for s in [9, 8, 7, 6]]
    v2 = [_make_rc(score=float(s), doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=s) for s in [5, 4, 3, 2]]
    v3 = [_make_rc(score=float(s), doc_subtype="investor_day_session", report_date="2025-09", page_number=s) for s in [1, 0, -1, -2]]
    out = _balance_per_version(v1 + v2 + v3, "comparison", total_budget=6)
    assert len(out) == 6
    # The 6 highest scores across all 12 are 9, 8, 7, 6, 5, 4 — first two
    # versions dominate but each gets at most per_version_k=4.
    assert {rc.rerank_score for rc in out} == {9.0, 8.0, 7.0, 6.0, 5.0, 4.0}


def test_balance_applies_to_conflict_intent() -> None:
    rcs = [
        _make_rc(score=2.0, doc_subtype="company_update", report_date="2026-03", page_number=1),
        _make_rc(score=1.0, doc_subtype="company_update", report_date="2026-03", page_number=2),
        _make_rc(score=-1.0, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=3),
        _make_rc(score=-2.0, doc_subtype="quarterly_investor_deck", report_date="2025-12", page_number=4),
    ]
    out = _balance_per_version(rcs, "conflict", total_budget=4)
    assert any(rc.chunk.doc_subtype == "quarterly_investor_deck" for rc in out)


def test_balance_empty_input_returns_empty() -> None:
    assert _balance_per_version([], "comparison", total_budget=5) == []
