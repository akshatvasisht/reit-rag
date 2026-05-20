"""Unit tests for per-version-group floor on comparison and conflict intents.

Verifies that for comparison/conflict queries the retained set surfaces at
least one chunk per version group present in the fused candidate pool.
"""

from __future__ import annotations

from uuid import UUID, uuid4


from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import enforce_per_version_floor
from src.versioning.chains import version_group_key


def _make_chunk(
    *,
    company: str = "Acme REIT",
    doc_type: str = "investor_presentation",
    doc_subtype: str = "quarterly_investor_deck",
    doc_version: str = "2026-03",
    page_number: int = 1,
    chunk_id: UUID | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=uuid4(),
        company=company,
        ticker=company[:3].upper(),
        doc_type=doc_type,
        report_date=doc_version,
        doc_version=doc_version,
        chunk_text=f"Content {company} {doc_version} p{page_number}",
        content_type="text",
        page_number=page_number,
        doc_subtype=doc_subtype,
    )


def _make_rc(chunk: Chunk, score: float = 5.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def test_floor_injects_missing_group_when_subtype_differs() -> None:
    """Two distinct version groups (different doc_subtype) — only one in retained.
    The floor must inject the best chunk from the missing group."""
    march = _make_chunk(doc_subtype="quarterly_investor_deck")
    december_other = _make_chunk(doc_subtype="quarterly_investor_deck_dec")
    retained = [_make_rc(march, score=5.0)]

    best_missing = _make_rc(december_other, score=2.5)
    worse_missing = _make_rc(
        _make_chunk(doc_subtype="quarterly_investor_deck_dec", page_number=9),
        score=0.5,
    )
    fused_by_group = {
        version_group_key(march): [_make_rc(march, score=5.0)],
        version_group_key(december_other): [worse_missing, best_missing],
    }

    result = enforce_per_version_floor(retained, fused_by_group, intent="comparison")
    represented_groups = {version_group_key(rc.chunk) for rc in result}
    assert version_group_key(december_other) in represented_groups
    floor_chunks = [rc for rc in result if rc.retrieval_stage == "version_floor"]
    # The best-scored chunk must be injected first (highest priority).
    assert floor_chunks[0].chunk.id == best_missing.chunk.id
    assert floor_chunks[0].expansion_reason is not None
    assert "per-version floor" in floor_chunks[0].expansion_reason


def test_floor_noop_on_latest_intent() -> None:
    """Latest intent → no version-floor injection (deduper already collapses)."""
    march = _make_chunk(doc_subtype="quarterly_investor_deck")
    december = _make_chunk(doc_subtype="quarterly_investor_deck_dec")
    retained = [_make_rc(march, score=5.0)]
    fused_by_group = {
        version_group_key(march): [_make_rc(march, score=5.0)],
        version_group_key(december): [_make_rc(december, score=2.0)],
    }
    result = enforce_per_version_floor(retained, fused_by_group, intent="latest")
    assert result == retained


def test_floor_noop_on_synthesis_intent() -> None:
    """all_company_synthesis uses per-issuer floor, not per-version floor."""
    march = _make_chunk(doc_subtype="quarterly_investor_deck")
    december = _make_chunk(doc_subtype="quarterly_investor_deck_dec")
    retained = [_make_rc(march, score=5.0)]
    fused_by_group = {
        version_group_key(march): [_make_rc(march, score=5.0)],
        version_group_key(december): [_make_rc(december, score=2.0)],
    }
    result = enforce_per_version_floor(
        retained, fused_by_group, intent="all_company_synthesis"
    )
    assert result == retained


def test_floor_noop_when_missing_group_has_no_fused_chunks() -> None:
    """If a version group has zero fused chunks, the floor has nothing to inject."""
    march = _make_chunk(doc_subtype="quarterly_investor_deck")
    retained = [_make_rc(march, score=5.0)]
    fused_by_group = {
        version_group_key(march): [_make_rc(march, score=5.0)],
        ("Acme REIT", "investor_presentation", "ghost_subtype"): [],
    }
    result = enforce_per_version_floor(retained, fused_by_group, intent="comparison")
    assert len(result) == 1
    assert result[0].chunk.id == march.id


def test_floor_fires_on_conflict_intent() -> None:
    """Conflict intent must also trigger the floor."""
    a = _make_chunk(doc_subtype="quarterly_investor_deck")
    b = _make_chunk(doc_subtype="quarterly_investor_deck_dec")
    retained = [_make_rc(a, score=5.0)]
    fused_by_group = {
        version_group_key(a): [_make_rc(a, score=5.0)],
        version_group_key(b): [_make_rc(b, score=1.5)],
    }
    result = enforce_per_version_floor(retained, fused_by_group, intent="conflict")
    represented = {version_group_key(rc.chunk) for rc in result}
    assert version_group_key(b) in represented


def test_floor_caps_chunks_per_group_at_version_floor_max() -> None:
    """A missing group has many fused chunks — at most VERSION_FLOOR_MAX (2) added."""
    from src.retrieval.pipeline import VERSION_FLOOR_MAX

    march = _make_chunk(doc_subtype="quarterly_investor_deck")
    retained = [_make_rc(march, score=5.0)]
    other_subtype = "quarterly_investor_deck_dec"
    fused_by_group = {
        version_group_key(march): [_make_rc(march, score=5.0)],
        ("Acme REIT", "investor_presentation", other_subtype): [
            _make_rc(_make_chunk(doc_subtype=other_subtype, page_number=p), score=float(p))
            for p in (1, 2, 3, 4, 5)
        ],
    }
    result = enforce_per_version_floor(retained, fused_by_group, intent="comparison")
    floor_chunks = [rc for rc in result if rc.retrieval_stage == "version_floor"]
    assert len(floor_chunks) <= VERSION_FLOOR_MAX
