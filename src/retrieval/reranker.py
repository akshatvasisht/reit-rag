"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 (sentence-transformers)."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from sentence_transformers import CrossEncoder

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — loaded lazily on first rerank call so importing this
# module does not require network/model availability.
# ---------------------------------------------------------------------------
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Pinned commit SHA — supply-chain mitigation against an upstream model swap.
_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
_cross_encoder: CrossEncoder | None = None

# Chunks whose rerank_score falls below this threshold are considered
# irrelevant and trigger the abstention gate.
#
# ms-marco-MiniLM-L-6-v2 outputs raw logits (not sigmoid-bounded
# probabilities). Relevant passages typically score in the range [0, 12];
# clearly irrelevant pairs often score negative (e.g. -8 to -2).
#
# This threshold is intentionally permissive because raw logits can skew
# negative even for relevant contexts in natural-language query settings.
RERANK_THRESHOLD: float = -5.0


def _get_cross_encoder() -> CrossEncoder:
    """Return the reranker singleton, loading it on first use."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME, revision=_MODEL_REVISION)
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int = 5,
) -> list[RetrievedChunk]:
    """Score and sort retrieval candidates using a cross-encoder reranker.

    Every ``(query, chunk.chunk_text)`` pair is scored by the cross-encoder
    in a single batched ``predict()`` call.  ``rerank_score`` is set on each
    candidate; the returned list contains at most ``top_n`` items sorted by
    ``rerank_score`` descending.

    If ``len(candidates) < top_n``, all candidates are returned (no padding).
    An empty ``candidates`` list returns an empty list without calling the
    model.

    Args:
        query: The user's natural language query string.
        candidates: Ordered list of RetrievedChunk objects, typically the
            top-40 output of RRF fusion (see ``FUSED_TOP_N``).
        top_n: Maximum number of reranked results to return (default 5).

    Returns:
        Up to ``top_n`` RetrievedChunk objects sorted by ``rerank_score``
        descending, with ``rerank_score`` populated on every returned item.
    """
    if not candidates:
        return []

    effective_top_n = min(top_n, len(candidates))
    if len(candidates) < top_n:
        logger.warning(
            "rerank called with only %d candidates but top_n=%d; "
            "returning all %d.",
            len(candidates),
            top_n,
            len(candidates),
        )

    pairs: list[tuple[str, str]] = [(query, rc.chunk.chunk_text) for rc in candidates]

    model = _get_cross_encoder()
    # predict() can return ndarray or tensor depending on backend/config.
    raw_scores = model.predict(cast(Any, pairs))
    if isinstance(raw_scores, np.ndarray):
        scores = raw_scores.tolist()
    elif hasattr(raw_scores, "tolist"):
        scores = raw_scores.tolist()
    else:
        scores = list(raw_scores)

    for rc, score in zip(candidates, scores):
        rc.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)

    result = ranked[:effective_top_n]

    logger.debug(
        "Reranker scored %d candidates; top score=%.3f, bottom score=%.3f; "
        "returning top %d (threshold=%.1f).",
        len(candidates),
        ranked[0].rerank_score or 0.0,
        ranked[-1].rerank_score or 0.0,
        len(result),
        RERANK_THRESHOLD,
    )

    return result
