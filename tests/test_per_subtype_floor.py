"""Unit tests for per-doc_subtype floor on the latest intent.

Verifies that for a single company exposing 2+ co-existing doc_subtypes in the
fused candidate pool, the retained set surfaces at least one chunk per subtype —
even when a global rerank top-N collapses onto one subtype. Cross-company
balancing remains the per-issuer floor's job and is never triggered here.
"""

from __future__ import annotations

from uuid import UUID, uuid4


from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import SUBTYPE_FLOOR_MAX, enforce_per_subtype_floor


def _make_chunk(
    *,
    company: str = "BXP",
    doc_type: str = "investor_presentation",
    doc_subtype: str = "quarterly_investor_deck",
    doc_version: str = "2025-12",
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
        chunk_text=f"Content {company} {doc_subtype} p{page_number}",
        content_type="text",
        page_number=page_number,
        doc_subtype=doc_subtype,
    )


def _make_rc(chunk: Chunk, score: float = 5.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _group(chunks: list[RetrievedChunk]) -> dict[tuple[str, str], list[RetrievedChunk]]:
    by_company_subtype: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for rc in chunks:
        by_company_subtype.setdefault((rc.chunk.company, rc.chunk.doc_subtype), []).append(rc)
    return by_company_subtype


def test_floor_injects_starved_subtype_same_company() -> None:
    """Two subtypes for one company, only the deck retained — the floor must
    inject the best investor-day-session chunk from the fused pool."""
    deck = _make_chunk(doc_subtype="quarterly_investor_deck", page_number=7)
    session_best = _make_chunk(doc_subtype="investor_day_session", page_number=18)
    session_worse = _make_chunk(doc_subtype="investor_day_session", page_number=22)
    retained = [_make_rc(deck, score=5.0)]
    fused = _group([
        _make_rc(deck, score=5.0),
        _make_rc(session_worse, score=0.5),
        _make_rc(session_best, score=2.5),
    ])

    result = enforce_per_subtype_floor(retained, fused, intent="latest")
    represented = {(rc.chunk.company, rc.chunk.doc_subtype) for rc in result}
    assert ("BXP", "investor_day_session") in represented
    floor_chunks = [rc for rc in result if rc.retrieval_stage == "subtype_floor"]
    assert len(floor_chunks) == 1
    # The best-scored missing-subtype chunk wins.
    assert floor_chunks[0].chunk.id == session_best.id
    assert floor_chunks[0].expansion_reason is not None
    assert "per-subtype floor" in floor_chunks[0].expansion_reason


def test_floor_noop_single_subtype_company() -> None:
    """A company with only one subtype in the fused pool cannot be starved."""
    deck = _make_chunk(company="EGP", doc_subtype="quarterly_investor_deck", page_number=1)
    other = _make_chunk(company="EGP", doc_subtype="quarterly_investor_deck", page_number=2)
    retained = [_make_rc(deck, score=5.0)]
    fused = _group([_make_rc(deck, score=5.0), _make_rc(other, score=1.0)])

    result = enforce_per_subtype_floor(retained, fused, intent="latest")
    assert result == retained


def test_floor_caps_injection_per_subtype() -> None:
    """A starved subtype with many fused chunks yields at most SUBTYPE_FLOOR_MAX."""
    deck = _make_chunk(doc_subtype="quarterly_investor_deck")
    retained = [_make_rc(deck, score=5.0)]
    fused = _group(
        [_make_rc(deck, score=5.0)]
        + [
            _make_rc(_make_chunk(doc_subtype="investor_day_session", page_number=p), score=float(p))
            for p in (1, 2, 3, 4, 5)
        ]
    )

    result = enforce_per_subtype_floor(retained, fused, intent="latest")
    floor_chunks = [rc for rc in result if rc.retrieval_stage == "subtype_floor"]
    assert len(floor_chunks) <= SUBTYPE_FLOOR_MAX


def test_floor_fires_only_on_latest_intent() -> None:
    """Gating: fires on latest, no-op on the multi-version intents (which get
    the per-version floor instead, so the subtype floor must not double-fire)."""
    deck = _make_chunk(doc_subtype="quarterly_investor_deck")
    session = _make_chunk(doc_subtype="investor_day_session")
    retained = [_make_rc(deck, score=5.0)]
    fused = _group([_make_rc(deck, score=5.0), _make_rc(session, score=2.0)])

    # latest → injects the starved subtype.
    fired = enforce_per_subtype_floor(retained, fused, intent="latest")
    assert any(rc.retrieval_stage == "subtype_floor" for rc in fired)

    # comparison / conflict are owned by the version floor — no-op here.
    for intent in ("comparison", "conflict"):
        result = enforce_per_subtype_floor(
            [_make_rc(deck, score=5.0)], _group([_make_rc(deck, score=5.0), _make_rc(session, score=2.0)]), intent=intent
        )
        assert result == [result[0]]
        assert not any(rc.retrieval_stage == "subtype_floor" for rc in result)
        assert len(result) == 1


def test_floor_does_not_inject_across_companies() -> None:
    """Cross-company isolation: a second company's subtype is never injected to
    satisfy the first company's coverage — that is the per-issuer floor's job.

    Each company here exposes a single subtype, so neither qualifies for the
    per-subtype floor; the missing company's chunk must stay out."""
    bxp_deck = _make_chunk(company="BXP", doc_subtype="quarterly_investor_deck")
    egp_deck = _make_chunk(company="EGP", doc_subtype="quarterly_investor_deck")
    retained = [_make_rc(bxp_deck, score=5.0)]
    fused = _group([_make_rc(bxp_deck, score=5.0), _make_rc(egp_deck, score=4.0)])

    result = enforce_per_subtype_floor(retained, fused, intent="latest")
    assert result == retained
    assert not any(rc.chunk.company == "EGP" for rc in result)


def test_floor_already_present_subtype_not_duplicated() -> None:
    """A subtype already represented in retained gets no second floor chunk."""
    deck = _make_chunk(doc_subtype="quarterly_investor_deck", page_number=7)
    session = _make_chunk(doc_subtype="investor_day_session", page_number=18)
    retained = [_make_rc(deck, score=5.0), _make_rc(session, score=4.0)]
    fused = _group([
        _make_rc(deck, score=5.0),
        _make_rc(session, score=4.0),
        _make_rc(_make_chunk(doc_subtype="investor_day_session", page_number=22), score=1.0),
    ])

    result = enforce_per_subtype_floor(retained, fused, intent="latest")
    assert result == retained
    assert not any(rc.retrieval_stage == "subtype_floor" for rc in result)
