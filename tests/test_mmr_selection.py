"""Unit tests for Maximal Marginal Relevance selection in src/retrieval/pipeline.py.

Covers the diversity-aware re-selection that promotes a complementary-facet
chunk into the retained set when the reranked pool is dominated by
near-duplicate high-relevance chunks, while guaranteeing single-fact pools
keep their top-1 chunk and are never displaced.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    MMR_LAMBDA,
    _normalize_relevance,
    _pool_is_redundant,
    mmr_select,
)


def _make_chunk(text: str, chunk_id: UUID | None = None) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=uuid4(),
        company="Realty Income",
        ticker="O",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text=text,
        content_type="text",
        page_number=1,
    )


def _make_rc(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=_make_chunk(text), rerank_score=score)


def _embeddings(
    pool: list[RetrievedChunk], vectors: list[list[float]]
) -> dict[str, list[float]]:
    return {str(rc.chunk.id): vec for rc, vec in zip(pool, vectors)}


# ---------------------------------------------------------------------------
# _normalize_relevance
# ---------------------------------------------------------------------------


def test_normalize_relevance_maps_to_unit_interval() -> None:
    pool = [_make_rc("a", 10.0), _make_rc("b", 5.0), _make_rc("c", 0.0)]
    rel = _normalize_relevance(pool)
    assert rel[pool[0].chunk.id] == 1.0
    assert rel[pool[2].chunk.id] == 0.0
    assert 0.0 < rel[pool[1].chunk.id] < 1.0


def test_normalize_relevance_equal_scores_map_to_one() -> None:
    pool = [_make_rc("a", 3.0), _make_rc("b", 3.0)]
    rel = _normalize_relevance(pool)
    assert all(v == 1.0 for v in rel.values())


# ---------------------------------------------------------------------------
# _pool_is_redundant
# ---------------------------------------------------------------------------


def test_pool_redundant_when_top_chunks_near_identical() -> None:
    pool = [_make_rc("dup1", 9.0), _make_rc("dup2", 8.5), _make_rc("diff", 2.0)]
    embeddings = _embeddings(
        pool,
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0]],
    )
    assert _pool_is_redundant(pool, embeddings) is True


def test_pool_not_redundant_when_diverse() -> None:
    pool = [_make_rc("a", 9.0), _make_rc("b", 8.0), _make_rc("c", 2.0)]
    embeddings = _embeddings(
        pool,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert _pool_is_redundant(pool, embeddings) is False


# ---------------------------------------------------------------------------
# mmr_select — diversity promotion
# ---------------------------------------------------------------------------


def test_mmr_promotes_complementary_chunk_into_retained_set() -> None:
    """MMR keeps top-1 by relevance and promotes a complementary chunk over a near-duplicate in a redundant pool."""
    dup_a = _make_rc("10 countries operations", 9.0)
    dup_b = _make_rc("10 countries footprint", 8.5)
    dup_c = _make_rc("10 countries portfolio", 8.45)
    complement = _make_rc("9 countries scoped figure", 8.4)
    pool = [dup_a, dup_b, dup_c, complement]

    # dup_a/b/c near-identical; complement orthogonal.
    embeddings = _embeddings(
        pool,
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.97, 0.03, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    selected = mmr_select(pool, embeddings, budget=2, lambda_=MMR_LAMBDA)
    selected_ids = {rc.chunk.id for rc in selected}
    assert len(selected) == 2
    # Top-1 by relevance is always retained.
    assert dup_a.chunk.id in selected_ids
    # The complementary chunk is promoted over the near-duplicate dup_b.
    assert complement.chunk.id in selected_ids
    assert dup_b.chunk.id not in selected_ids


def test_mmr_seeds_with_top1_by_relevance() -> None:
    """The first selected chunk is always the highest-rerank candidate."""
    high = _make_rc("primary answer", 9.5)
    others = [_make_rc(f"other {i}", 4.0 - i) for i in range(3)]
    pool = [others[0], high, others[1], others[2]]
    embeddings = _embeddings(
        pool,
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.8, 0.2]],
    )
    selected = mmr_select(pool, embeddings, budget=3, lambda_=MMR_LAMBDA)
    assert selected[0].chunk.id == high.chunk.id


# ---------------------------------------------------------------------------
# mmr_select — single-fact no-regression
# ---------------------------------------------------------------------------


def test_mmr_single_fact_returns_top_chunk_first_no_displacement() -> None:
    """A single-fact pool: one clearly-best chunk, the rest irrelevant and
    mutually dissimilar. MMR must return the top chunk first and not displace
    it for a lower-relevance chunk."""
    best = _make_rc("the exact figure is $1.2B", 11.0)
    irrelevant = [
        _make_rc("unrelated narrative", -1.0),
        _make_rc("boilerplate disclaimer", -2.0),
        _make_rc("table of contents", -3.0),
    ]
    pool = [best, *irrelevant]
    embeddings = _embeddings(
        pool,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.1, 0.9]],
    )
    selected = mmr_select(pool, embeddings, budget=2, lambda_=MMR_LAMBDA)
    assert selected[0].chunk.id == best.chunk.id


def test_mmr_lambda_one_is_plain_relevance_ranking() -> None:
    """lambda=1.0 zeroes the diversity term — MMR must reduce to top-N by
    relevance regardless of similarity structure."""
    pool = [
        _make_rc("a", 9.0),
        _make_rc("b", 8.0),
        _make_rc("c", 7.0),
        _make_rc("d", 1.0),
    ]
    # All near-identical embeddings — would heavily penalize diversity if
    # lambda<1, but lambda=1 ignores similarity entirely.
    embeddings = _embeddings(
        pool,
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]],
    )
    selected = mmr_select(pool, embeddings, budget=3, lambda_=1.0)
    assert [rc.chunk.id for rc in selected] == [
        pool[0].chunk.id,
        pool[1].chunk.id,
        pool[2].chunk.id,
    ]


# ---------------------------------------------------------------------------
# mmr_select — budget and edge cases
# ---------------------------------------------------------------------------


def test_mmr_respects_budget() -> None:
    pool = [_make_rc(f"chunk {i}", float(10 - i)) for i in range(8)]
    embeddings = _embeddings(
        pool, [[float(i), 1.0, 0.0] for i in range(8)]
    )
    selected = mmr_select(pool, embeddings, budget=3)
    assert len(selected) == 3


def test_mmr_pool_smaller_than_budget_returns_all() -> None:
    pool = [_make_rc("a", 9.0), _make_rc("b", 8.0)]
    embeddings = _embeddings(pool, [[1.0, 0.0], [0.0, 1.0]])
    selected = mmr_select(pool, embeddings, budget=5)
    assert len(selected) == 2


def test_mmr_empty_pool_returns_empty() -> None:
    assert mmr_select([], {}, budget=5) == []


def test_mmr_missing_embedding_treated_as_maximally_novel() -> None:
    """A candidate without an embedding incurs a 0.0 similarity penalty, so it
    is never silently suppressed by a missing vector."""
    top = _make_rc("primary", 10.0)
    no_emb = _make_rc("complementary no embedding", 7.0)
    dup = _make_rc("near duplicate of primary", 7.0)
    pool = [top, dup, no_emb]
    # Only top and dup have embeddings (near identical); no_emb is absent.
    embeddings = {
        str(top.chunk.id): [1.0, 0.0],
        str(dup.chunk.id): [0.99, 0.01],
    }
    selected = mmr_select(pool, embeddings, budget=2, lambda_=MMR_LAMBDA)
    selected_ids = {rc.chunk.id for rc in selected}
    assert top.chunk.id in selected_ids
    # no_emb (novel) beats the near-duplicate dup at equal relevance.
    assert no_emb.chunk.id in selected_ids
