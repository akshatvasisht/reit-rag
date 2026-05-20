"""Unit tests for the entity-anchor boost helpers in src/retrieval/pipeline.py.

Covers anchor extraction from queries and the boost mechanism that promotes
rerank-pool chunks containing a query proper-noun anchor.
"""

from __future__ import annotations

from uuid import uuid4

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    _entity_anchor_boost,
    _extract_query_anchors,
)


def _make_chunk(
    contextualized_text: str | None = None,
    chunk_text: str = "body",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="BXP",
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2025-12",
        doc_version="2025-12",
        chunk_text=chunk_text,
        content_type="text",
        contextualized_text=contextualized_text,
    )


def _make_rc(chunk: Chunk, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


# ---------------------------------------------------------------------------
# _extract_query_anchors
# ---------------------------------------------------------------------------


def test_extract_anchor_with_digit_token() -> None:
    assert "343 Madison Avenue" in _extract_query_anchors(
        "What is the status of BXP's 343 Madison Avenue development?"
    )


def test_extract_anchor_three_token_proper_noun() -> None:
    assert any(
        a == "Empire State Building"
        for a in _extract_query_anchors("Does BXP own Empire State Building?")
    )


def test_extract_anchor_long_token_two_word() -> None:
    """'Digital Realty' qualifies because 'Digital' is ≥7 chars."""
    anchors = _extract_query_anchors("How leveraged is Digital Realty currently?")
    assert "Digital Realty" in anchors


def test_extract_anchor_drops_short_stopword_bigrams() -> None:
    """'What Is' and 'How Did' should not surface as anchors — both tokens
    short, no digit, no triple-token span."""
    anchors = _extract_query_anchors("What Is the deal? How Did the spread move?")
    assert "What Is" not in anchors
    assert "How Did" not in anchors


def test_extract_anchor_empty_query_returns_empty_list() -> None:
    assert _extract_query_anchors("") == []


def test_extract_anchor_rejects_acronym_with_digit_token() -> None:
    """'2025 FFO' is digit + acronym — must NOT be a viable anchor because
    'FFO' appears in nearly every BXP / DLR / RIN chunk and would over-fire
    the boost on generic metric queries."""
    anchors = _extract_query_anchors("What is BXP's full year 2025 FFO per share guidance?")
    assert "2025 FFO" not in anchors


def test_extract_anchor_rejects_all_caps_company_with_digit() -> None:
    """'BXP 2025' has no Title-Cased token — not an anchor."""
    anchors = _extract_query_anchors("How did BXP 2025 figures land?")
    assert "BXP 2025" not in anchors


# ---------------------------------------------------------------------------
# _entity_anchor_boost
# ---------------------------------------------------------------------------


def test_boost_promotes_chunk_mentioning_query_anchor() -> None:
    """A chunk in the rerank pool but not in the reranked top-N must be
    promoted when its contextualized_text contains a query anchor."""
    in_top = _make_rc(_make_chunk(contextualized_text="DLR leverage walkthrough."))
    dropped_match = _make_rc(_make_chunk(
        contextualized_text="343 Madison Avenue project economics table"
    ))
    dropped_no_match = _make_rc(_make_chunk(contextualized_text="something else"))

    fused = [in_top, dropped_match, dropped_no_match]
    reranked = [in_top]

    boosted = _entity_anchor_boost(
        "Tell me about 343 Madison Avenue", fused, reranked
    )
    assert len(boosted) == 1
    assert boosted[0].chunk.id == dropped_match.chunk.id
    assert boosted[0].retrieval_stage == "entity_anchored"


def test_boost_does_not_double_add_chunks_already_in_reranked() -> None:
    rc = _make_rc(_make_chunk(contextualized_text="343 Madison Avenue details"))
    boosted = _entity_anchor_boost("343 Madison Avenue", [rc], [rc])
    assert boosted == []


def test_boost_respects_max_anchored_cap() -> None:
    fused = [
        _make_rc(_make_chunk(contextualized_text=f"343 Madison Avenue note {i}"))
        for i in range(6)
    ]
    boosted = _entity_anchor_boost("343 Madison Avenue", fused, [], max_anchored=2)
    assert len(boosted) == 2


def test_boost_returns_empty_when_query_has_no_anchor() -> None:
    rc = _make_rc(_make_chunk(contextualized_text="343 Madison Avenue details"))
    boosted = _entity_anchor_boost("how is the leverage?", [rc], [])
    assert boosted == []


def test_boost_matches_chunk_text_when_contextualized_text_missing() -> None:
    """Fallback to raw chunk_text when contextualized_text is absent."""
    rc = _make_rc(_make_chunk(
        contextualized_text=None,
        chunk_text="343 Madison Avenue project status update.",
    ))
    boosted = _entity_anchor_boost("343 Madison Avenue", [rc], [])
    assert len(boosted) == 1


def _make_chunk_with_type(content_type: str, contextualized_text: str) -> Chunk:
    c = _make_chunk(contextualized_text=contextualized_text)
    c.content_type = content_type  # type: ignore[assignment]
    return c


def test_boost_prefers_table_chunks_over_text_chunks() -> None:
    """When the cap is tight, data-bearing chunks (table / mixed /
    chart_description) win the slot over narrative text chunks that mention
    the same anchor — the table is usually what the user is after."""
    narrative_first = _make_rc(_make_chunk_with_type(
        "text", "343 Madison Avenue: project overview narrative."
    ))
    narrative_second = _make_rc(_make_chunk_with_type(
        "text", "343 Madison Avenue: leasing update."
    ))
    table_chunk = _make_rc(_make_chunk_with_type(
        "table", "Project economics table including 343 Madison Avenue row."
    ))

    fused = [narrative_first, narrative_second, table_chunk]
    boosted = _entity_anchor_boost("343 Madison Avenue", fused, [], max_anchored=1)
    assert len(boosted) == 1
    assert boosted[0].chunk.id == table_chunk.chunk.id
