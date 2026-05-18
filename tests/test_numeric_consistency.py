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


# ---------------------------------------------------------------------------
# Baseline normalization fixes (Unicode dash, approx marker, range parsing)
# Previously labelled "baseline" in the register — implemented here.
# ---------------------------------------------------------------------------


def test_approximation_marker_tilde_matches() -> None:
    """Claim '~5.7%' must match chunk '5.7%' — approx marker stripped."""
    chunk = _make_chunk(chunk_text="Annualized dividend yield was 5.7% as of year-end.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Yield was approximately 5.7%.", value="~5.7%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_approximation_word_approximately_matches() -> None:
    """Claim 'approximately 5.7%' must match chunk '5.7%'."""
    chunk = _make_chunk(chunk_text="Yield was 5.7% at quarter end.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Approximately 5.7% yield.", value="approximately 5.7%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_unicode_en_dash_in_chunk_matches_ascii_hyphen_claim() -> None:
    """Chunk uses U+2013 en-dash ('8–12%'), claim uses ASCII hyphen ('8-12%')."""
    chunk = _make_chunk(chunk_text="2026 same-store NOI guidance: 8–12% growth.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Guidance is 8-12% growth.", value="8-12%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_claim_range_matches_chunk_endpoints_separately() -> None:
    """Range claim '$7.90-$8.00' accepted when both endpoints in chunk values."""
    chunk = _make_chunk(chunk_text="2026 Core FFO guidance: $7.90 to $8.00 per share.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Guidance is $7.90-$8.00.", value="$7.90-$8.00")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


def test_single_value_within_chunk_range_accepted() -> None:
    """Claim '5.2x' is inside chunk's stated range '5.0-5.5x target'."""
    chunk = _make_chunk(chunk_text="Net leverage target is 5.0-5.5x.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Net leverage is 5.2x.", value="5.2x")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == []


# ---------------------------------------------------------------------------
# Composite multi-value claim splitting
# ---------------------------------------------------------------------------


def test_composite_semicolon_separated_values_match() -> None:
    """'~89% end of 2026; 87.25%–88% average' — semicolon split, atoms in chunk."""
    chunk = _make_chunk(chunk_text="Leased 89% at end of 2026, average 87.25%-88%.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("BXP occupancy.", value="~89% end of 2026; 87.25%–88% average")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_composite_percent_with_parenthetical_period_match() -> None:
    """'86.4% (Q2 2025), ~86.9% (Q4 2025)' — comma-parenthetical split."""
    chunk = _make_chunk(chunk_text="Occupancy was 86.4% in Q2 2025 and 86.9% in Q4 2025.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Occupancy.", value="86.4% (Q2 2025), ~86.9% (Q4 2025)")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_composite_slash_separated_values_match() -> None:
    """'~$7Bn / 4.9x' — slash-separated composite."""
    chunk = _make_chunk(chunk_text="DLR had ~$7B in liquidity and leverage of 4.9x.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("DLR liquidity and leverage.", value="~$7Bn / 4.9x")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_composite_dollar_and_percent_match() -> None:
    """'$1,246.2M (39%)' — dollar amount + parenthetical percentage."""
    chunk = _make_chunk(
        chunk_text="Caesars: $1,246.2M annualized cash rent (39% of total)."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Caesars rent.", value="$1,246.2M (39%)")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_composite_geographic_percentages_match() -> None:
    """'North America 52%, Europe 29%, APAC 10%' — multi-percentage composite."""
    chunk = _make_chunk(
        chunk_text="Geographic mix: North America 52%, Europe 29%, APAC 10%."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim(
        "DLR geographic breakdown.",
        value="North America 52%, Europe 29%, APAC 10%",
    )
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Spelled-out scale words
# ---------------------------------------------------------------------------


def test_chk2_spelled_out_billion_matches_B_suffix() -> None:
    """'$10.5 billion' must match chunk '$10.5' (PSA merger psa-merg-basic-1)."""
    chunk = _make_chunk(chunk_text="NSA enterprise value: $10.5 ($ billions).")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("NSA EV was $10.5 billion.", value="$10.5 billion")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk2_spelled_out_billion_matches_B_tagged_chunk() -> None:
    """'$6.2 billion' claim matches chunk '$6.2B' (Realty Income o-adv-6)."""
    chunk = _make_chunk(chunk_text="Annual investment volume approximately $6.2B.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Investment volume was $6.2 billion.", value="$6.2 billion")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Parenthetical negative / sign-stripped values
# ---------------------------------------------------------------------------


def test_chk3_negative_claim_matches_positive_chunk() -> None:
    """'-3.6%' claim matches chunk '3.6%' (sign stripped in chart; psa-cu-adv-5)."""
    chunk = _make_chunk(
        chunk_text="NSA same-store NOI growth: 3.6% (negative bar chart)."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("NSA growth was -3.6%.", value="-3.6%")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk3_parenthetical_negative_matches_positive_chunk() -> None:
    """'($22.3M)' claim (accounting-negative) matches chunk '$22.3' (bxp-morn-adv-7)."""
    chunk = _make_chunk(chunk_text="JV losses: $22.3 million write-down.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("JV net loss was ($22.3M).", value="($22.3M)")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Non-financial units (years, MW, sqft, dates)
# ---------------------------------------------------------------------------


def test_chk4_years_unit_uses_substring_check() -> None:
    """'39.6 years' does not fire mismatch when present in chunk text (VICI xdoc-adv-5)."""
    chunk = _make_chunk(
        chunk_text="Portfolio WALT: 39.6 years (inclusive of all renewal options)."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("VICI WALT is 39.6 years.", value="39.6 years")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk4_kw_unit_uses_substring_check() -> None:
    """'27 kW per rack in 2025' — non-financial unit, substring check (dlr-adv-8)."""
    chunk = _make_chunk(
        chunk_text="Rack density rising: from 7 kW in 2021 to 27 kW per rack in 2025."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Rack density reached 27 kW per rack in 2025.", value="27 kW per rack in 2025")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk4_sqft_unit_uses_substring_check() -> None:
    """'65 million square feet' — non-financial unit, substring check (egp-basic-1)."""
    chunk = _make_chunk(
        chunk_text="65 Million Square Feet Under Ownership by EastGroup Properties."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("EGP owns 65 million square feet.", value="65 million square feet")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk4_date_value_skipped() -> None:
    """'January 27, 2026' as claim value — date string, no numeric mismatch (xdoc-std-6)."""
    chunk = _make_chunk(chunk_text="Board declared dividend payable January 27, 2026.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Payment date is January 27, 2026.", value="January 27, 2026")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Trailing annotation / non-numeric proper noun
# ---------------------------------------------------------------------------


def test_chk5_trailing_annotation_stripped() -> None:
    """'$2.80 per share annualized' — $2.80 is in chunk, trailing annotation skipped."""
    chunk = _make_chunk(chunk_text="Quarterly dividend: $0.70 per share ($2.80 annualized).")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Dividend is $2.80 per share annualized.", value="$2.80 per share annualized")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk5_proper_noun_claim_skipped() -> None:
    """'Norges' as claim value — non-numeric proper noun, no mismatch (bxp-morn-adv-3)."""
    chunk = _make_chunk(chunk_text="Norges Bank Investment Management holds 45% interest.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("The JV partner is Norges.", value="Norges")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# MM (double-M) suffix
# ---------------------------------------------------------------------------


def test_chk6_mm_suffix_matches_m_in_chunk() -> None:
    """'$1,246.2MM' claim matches chunk '$1,246.2M' (vici-adv-3 / xdoc-adv-7)."""
    chunk = _make_chunk(chunk_text="Caesars total annualized cash rent: $1,246.2M.")
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Caesars rent is $1,246.2MM.", value="$1,246.2MM")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


def test_chk6_mm_composite_with_percent() -> None:
    """'$1,246.2MM (39%)' — MM suffix + parenthetical composite (vici-basic-1)."""
    chunk = _make_chunk(
        chunk_text="Caesars: $1,246.2M annualized cash rent (39% of total)."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Caesars rent share.", value="$1,246.2MM (39%)")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Comparison prefix > < ≥ ≤
# ---------------------------------------------------------------------------


def test_chk7_greater_than_prefix_stripped() -> None:
    """'>$15B' claim matches chunk '$15B' (dlr-adv-7 NNM-1 dlr liquidity)."""
    chunk = _make_chunk(
        chunk_text="DLR has more than $15B in JV and fund capital available."
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("JV capital is >$15B.", value=">$15B")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"


# ---------------------------------------------------------------------------
# Dollar-in-thousands table format
# ---------------------------------------------------------------------------


def test_chk8_dollar_thousands_annotation_stripped() -> None:
    """'$28,183 thousand' — strip thousand annotation and check bare integer (xdoc-std-5)."""
    chunk = _make_chunk(
        chunk_text="$ in thousands\nRental revenues\t28,183\t51,665"
    )
    contexts = [_make_rc(chunk)]
    claim = _make_claim("Rental revenues were $28,183 thousand.", value="$28,183 thousand")
    issues = check_numeric_consistency(_make_answer([claim]), contexts)
    assert issues == [], f"Unexpected issues: {issues}"
