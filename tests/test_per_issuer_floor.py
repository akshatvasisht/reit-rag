"""Unit tests for per-issuer floor on all_company_synthesis intent (2c-bis).

Verifies that for all_company_synthesis queries, the final retained set
has at least 1 chunk per named company that has any above-threshold chunk,
even when rerank scores are unbalanced across companies.

All tests use mocked structures — no real database or network access.
"""

from __future__ import annotations

from uuid import UUID, uuid4


from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import enforce_per_issuer_floor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str,
    page_number: int = 1,
    content_type: str = "text",
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    rerank_score: float = 5.0,
) -> RetrievedChunk:
    chunk = Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        company=company,
        ticker=company[:3].upper(),
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text=f"Content for {company} page {page_number}",
        content_type=content_type,
        page_number=page_number,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=rerank_score)


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_all_companies_represented_when_already_balanced() -> None:
    """When all companies already have chunks, floor leaves nothing to add."""
    companies = ["Alpha REIT", "Beta REIT", "Gamma REIT"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
        _make_chunk("Beta REIT", rerank_score=4.0),
        _make_chunk("Gamma REIT", rerank_score=3.0),
    ]
    candidates_by_company = {
        "Alpha REIT": [_make_chunk("Alpha REIT", page_number=2, rerank_score=2.0)],
        "Beta REIT": [_make_chunk("Beta REIT", page_number=2, rerank_score=1.0)],
        "Gamma REIT": [_make_chunk("Gamma REIT", page_number=2, rerank_score=0.5)],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    represented = {rc.chunk.company for rc in result}
    for company in companies:
        assert company in represented


def test_missing_company_gets_best_of_issuer_chunk() -> None:
    """A company missing from retained gets its best-scored candidate added."""
    companies = ["Alpha REIT", "Beta REIT", "Missing Corp"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
        _make_chunk("Beta REIT", rerank_score=4.0),
        # Missing Corp has no chunk in retained
    ]
    best_missing = _make_chunk("Missing Corp", page_number=1, rerank_score=2.0)
    worse_missing = _make_chunk("Missing Corp", page_number=2, rerank_score=0.5)
    candidates_by_company = {
        "Alpha REIT": [],
        "Beta REIT": [],
        "Missing Corp": [best_missing, worse_missing],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    company_names = {rc.chunk.company for rc in result}
    assert "Missing Corp" in company_names

    # Best chunk (rerank_score=2.0) should be chosen
    missing_chunks = [rc for rc in result if rc.chunk.company == "Missing Corp"]
    assert len(missing_chunks) == 1
    assert missing_chunks[0].chunk.id == best_missing.chunk.id


def test_multiple_missing_companies_all_get_floor_chunk() -> None:
    """Multiple missing companies each get their best candidate added."""
    companies = ["Alpha REIT", "Beta REIT", "Gamma REIT", "Delta REIT"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
    ]
    candidates_by_company = {
        "Alpha REIT": [],
        "Beta REIT": [_make_chunk("Beta REIT", rerank_score=1.5)],
        "Gamma REIT": [_make_chunk("Gamma REIT", rerank_score=1.2)],
        "Delta REIT": [_make_chunk("Delta REIT", rerank_score=0.8)],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    represented = {rc.chunk.company for rc in result}
    for company in companies:
        assert company in represented
    assert len(result) == 4


def test_company_with_no_candidates_stays_absent() -> None:
    """If a company has no candidates at all, it cannot be forced into the set."""
    companies = ["Alpha REIT", "Ghost Corp"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
    ]
    candidates_by_company = {
        "Alpha REIT": [],
        "Ghost Corp": [],  # no candidates
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    company_names = {rc.chunk.company for rc in result}
    assert "Alpha REIT" in company_names
    assert "Ghost Corp" not in company_names  # no candidates to add


def test_floor_only_fires_for_all_company_synthesis_intent() -> None:
    """enforce_per_issuer_floor is intent-gated; should be a no-op for non-synthesis queries.

    This test verifies the function itself returns input unchanged for companies=[] (
    the caller should pass companies=[] for non-synthesis intents).
    """
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
    ]
    # No companies specified → no floor enforced
    result = enforce_per_issuer_floor(retained, companies=[], candidates_by_company={})
    assert result == retained


def test_best_candidate_selected_when_multiple_available() -> None:
    """When a missing company has multiple candidates, the highest-scored is chosen."""
    companies = ["Alpha REIT", "Beta REIT"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
    ]
    low_candidate = _make_chunk("Beta REIT", page_number=2, rerank_score=0.3)
    mid_candidate = _make_chunk("Beta REIT", page_number=3, rerank_score=1.8)
    high_candidate = _make_chunk("Beta REIT", page_number=1, rerank_score=2.5)
    candidates_by_company = {
        "Alpha REIT": [],
        "Beta REIT": [low_candidate, mid_candidate, high_candidate],
    }

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    beta_chunks = [rc for rc in result if rc.chunk.company == "Beta REIT"]
    assert len(beta_chunks) == 1
    assert beta_chunks[0].chunk.id == high_candidate.chunk.id


def test_already_present_chunk_not_duplicated() -> None:
    """A company already present in retained must not get a second chunk from floor."""
    companies = ["Alpha REIT"]
    retained = [
        _make_chunk("Alpha REIT", rerank_score=5.0),
    ]
    extra = _make_chunk("Alpha REIT", page_number=99, rerank_score=1.0)
    candidates_by_company = {"Alpha REIT": [extra]}

    result = enforce_per_issuer_floor(retained, companies, candidates_by_company)

    alpha_chunks = [rc for rc in result if rc.chunk.company == "Alpha REIT"]
    assert len(alpha_chunks) == 1
