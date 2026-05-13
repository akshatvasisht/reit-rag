"""Reciprocal Rank Fusion (RRF) over BM25 and vector retrieval results."""

from __future__ import annotations

import logging
from uuid import UUID

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


def rrf_fuse(
    bm25_results: list[RetrievedChunk],
    vector_results: list[RetrievedChunk],
    k: int = 60,
    top_n: int = 20,
) -> list[RetrievedChunk]:
    """Combine BM25 and vector retrieval results via Reciprocal Rank Fusion.

    For each unique chunk, the fused score is the sum of ``1 / (k + rank)``
    over every ranked list the chunk appears in (rank is 1-indexed).  Chunks
    appearing in both lists receive contributions from both; chunks appearing
    in only one list receive a single contribution.  Ties in ``fused_score``
    are broken by stable sort order (Python's sort is stable, so the relative
    order of tied items mirrors their order in the merged candidate pool —
    BM25 results first, then vector-only results).

    Args:
        bm25_results: Ordered list of RetrievedChunk objects from BM25
            retrieval, highest-scoring first (rank 1 = index 0).
        vector_results: Ordered list of RetrievedChunk objects from dense
            vector retrieval, highest-scoring first.
        k: RRF smoothing constant (default 60).
            Higher values reduce the impact of top-rank differences.
        top_n: Maximum number of results to return (default 20).

    Returns:
        Up to ``top_n`` RetrievedChunk objects sorted by ``fused_score``
        descending, with ``bm25_rank``, ``vector_rank``, and ``fused_score``
        fields populated.
    """
    # Map chunk UUID → RetrievedChunk being built up.
    fused: dict[UUID, RetrievedChunk] = {}

    for rank_0, rc in enumerate(bm25_results):
        rank = rank_0 + 1  # 1-indexed
        chunk_id = rc.chunk.id
        score_contribution = 1.0 / (k + rank)

        if chunk_id not in fused:
            fused[chunk_id] = RetrievedChunk(
                chunk=rc.chunk,
                bm25_rank=rank,
                vector_rank=None,
                fused_score=score_contribution,
                rerank_score=None,
            )
        else:
            # Chunk was already seen (unexpected within a single list);
            # retain this safeguard for robustness.
            existing = fused[chunk_id]
            existing.bm25_rank = rank
            existing.fused_score = (existing.fused_score or 0.0) + score_contribution

    for rank_0, rc in enumerate(vector_results):
        rank = rank_0 + 1  # 1-indexed
        chunk_id = rc.chunk.id
        score_contribution = 1.0 / (k + rank)

        if chunk_id not in fused:
            fused[chunk_id] = RetrievedChunk(
                chunk=rc.chunk,
                bm25_rank=None,
                vector_rank=rank,
                fused_score=score_contribution,
                rerank_score=None,
            )
        else:
            existing = fused[chunk_id]
            existing.vector_rank = rank
            existing.fused_score = (existing.fused_score or 0.0) + score_contribution

    candidates = sorted(
        fused.values(),
        key=lambda rc: rc.fused_score or 0.0,
        reverse=True,
    )

    result = candidates[:top_n]

    logger.debug(
        "RRF fused %d BM25 + %d vector candidates → %d unique → returning top %d",
        len(bm25_results),
        len(vector_results),
        len(fused),
        len(result),
    )

    return result
