"""Unit tests for check_numeric_consistency.

All fixtures are plain dataclasses — no network or database access.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.generation.citation_check import check_numeric_consistency
from src.models import Claim, Chunk, RetrievedChunk, StructuredAnswer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Acme REIT",
    report_date: str = "2026-03",
    page_number: int = 5,
    chunk_text: str = "",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="ACM",
        doc_type="investor_presentation",
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=report_date,
        section_title="Financials",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=3.5)


def _make_claim(
    text: str,
    value: str,
    company: str = "Acme REIT",
    date: str = "March 2026",
    page: int = 5,
) -> Claim:
    return Claim(
        text=text,
        value=value,
        citation=f"[{company}, Investor Presentation, {date}, p.{page}]",
        citation_company=company,
        citation_date=date,
        citation_page=page,
    )


def _make_answer(claims: list[Claim], abstain: bool = False) -> StructuredAnswer:
    return StructuredAnswer(
        answer_prose="Some prose.",
        claims=claims,
        abstain=abstain,
        abstain_reason="",
        forward_looking=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_verbatim_match_produces_no_issue() -> None:
    """A claim value that appears verbatim in the chunk text raises no issue."""
    chunk = _make_chunk(chunk_text="Net debt to EBITDA was 4.9x as of year-end.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Net debt to EBITDA was 4.9x.", value="4.9x")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_within_one_percent_tolerance_produces_no_issue() -> None:
    """A claim value within 1% of a chunk value is accepted (rounding tolerance)."""
    # Chunk has 94.1%, claim says 94.0% — difference is ~0.11%, within 1%.
    chunk = _make_chunk(chunk_text="Occupancy rate reached 94.1% this quarter.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Occupancy was approximately 94.0%.", value="94.0%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_missing_value_produces_numeric_mismatch() -> None:
    """A claim value absent from the chunk and outside rounding tolerance is flagged."""
    chunk = _make_chunk(chunk_text="Revenue grew 12% year-over-year.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Revenue grew 25%.", value="25%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["type"] == "numeric_mismatch"
    assert issue["claimed_value"] == "25%"
    assert "12%" in issue["chunk_values_found"]
    assert "chunk_id" in issue


def test_citation_matching_no_chunk_produces_unsupported_citation() -> None:
    """When no retrieved chunk matches the claim's citation fields, record unsupported_citation."""
    chunk = _make_chunk(company="Other Corp", page_number=99, chunk_text="Revenue was $10M.")
    contexts = [_make_rc(chunk)]
    # Claim cites Acme REIT p.5 — no chunk for that
    claim = _make_claim("Acme had revenue of $10M.", value="$10M", company="Acme REIT", page=5)
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert len(issues) == 1
    assert issues[0]["type"] == "unsupported_citation"
    assert issues[0]["value"] == "$10M"


def test_abstaining_answer_skipped() -> None:
    """An abstaining answer returns an empty issue list without inspecting claims."""
    chunk = _make_chunk(chunk_text="Revenue was $5B.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Revenue was $99B.", value="$99B")
    answer = _make_answer([claim], abstain=True)
    issues = check_numeric_consistency(answer, contexts)
    assert issues == []


def test_unit_mismatch_not_accepted() -> None:
    """94% and $94 share digits but differ in unit — must not be treated as matching."""
    chunk = _make_chunk(chunk_text="Occupancy was 94%.")
    contexts = [_make_rc(chunk)]
    # Claim asserts a dollar value of $94; the chunk only has 94%
    claim = _make_claim("Acme earned $94.", value="$94")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["type"] == "numeric_mismatch"
    assert issue["claimed_value"] == "$94"


def test_non_numeric_claims_skipped() -> None:
    """Claims with empty value field are ignored by the numeric check."""
    chunk = _make_chunk(chunk_text="The company is headquartered in San Francisco.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("The company is headquartered in SF.", value="")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_multiple_claims_mixed_outcome() -> None:
    """Multiple claims: one passes, one fails — only the failure is reported."""
    chunk = _make_chunk(
        chunk_text="Net debt to EBITDA was 5.1x. Revenue was $4.9B.",
    )
    contexts = [_make_rc(chunk)]

    good_claim = _make_claim("Leverage was 5.1x.", value="5.1x")
    bad_claim = _make_claim("Revenue was $3.0B.", value="$3.0B")

    issues = check_numeric_consistency(_make_answer([good_claim, bad_claim]), contexts)
    assert len(issues) == 1
    assert issues[0]["claimed_value"] == "$3.0B"


def test_dollar_amount_verbatim_match() -> None:
    """Dollar amounts like $16,774,702 match verbatim in the chunk."""
    chunk = _make_chunk(chunk_text="Total assets: $16,774,702.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Total assets were $16,774,702.", value="$16,774,702")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_basis_points_match() -> None:
    """Basis-point values are matched within the chunk."""
    chunk = _make_chunk(chunk_text="Spread tightened by 25bps during the quarter.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Spread tightened 25bps.", value="25bps")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []
