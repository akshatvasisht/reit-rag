"""Calibrated retrieval-confidence scoring.

Composite of three signals — score gap, score magnitude, entity coverage —
combined with fixed weights to produce a continuous [0, 1] indicator of
post-rerank evidence quality. The score is computed on the pre-augmentation
chunk list so conflict-injected and sibling chunks (added at synthetic
scores) do not skew it.

The three weights and the two band cutoffs below are uncalibrated initial
values chosen to produce intuitive ordering. They should be re-drawn at the
medians of correct / borderline / refused queries once a sufficient
evaluation set has been evaluated.
"""

from __future__ import annotations

import math

from src.models import RetrievedChunk


CONFIDENCE_HIGH_CUTOFF = 0.75   # uncalibrated; pending measurement on evaluation set
CONFIDENCE_LOW_CUTOFF  = 0.45   # uncalibrated; pending measurement on evaluation set
W_GAP, W_MAGNITUDE, W_COVERAGE = 0.50, 0.30, 0.20


def _score_gap_signal(scores: list[float]) -> float:
    """Score the separation between the top-1 and top-2 reranker scores.

    A large positive gap means the best chunk is clearly better than the
    runner-up — a reliable indicator of a clean retrieval hit. Negative gaps
    (top-2 outscores top-1 after sorting) are clamped to 0.
    """
    if len(scores) < 2:
        return 0.0
    top, second = scores[0], scores[1]
    denom = abs(top) if abs(top) > 1e-6 else 1.0
    gap = (top - second) / denom
    return max(0.0, min(1.0, gap))


def _score_magnitude_signal(top_score: float) -> float:
    """Map a cross-encoder logit to [0, 1] via a sigmoid.

    The ms-marco cross-encoder's answered-band logits skew negative
    (threshold is -5.0). The sigmoid is centred at -2.0 with scale 2.0 so
    that top_score=-5 maps to ~0.18, top_score=-2 to 0.50, and
    top_score=+2 to ~0.88.
    """
    return 1.0 / (1.0 + math.exp(-(top_score + 2.0) / 2.0))


def _coverage_signal(contexts: list[RetrievedChunk], company_filter: list[str] | None) -> float:
    """Fraction of returned chunks that belong to an expected company.

    When no entity filter is active the query is corpus-wide and any returned
    chunk is on-target, so coverage is 1.0. With a filter, partial coverage
    suggests the retrieval engine had to pull from off-target companies.
    """
    if not contexts:
        return 0.0
    if not company_filter:
        return 1.0
    expected = {c.lower() for c in company_filter}
    matches = sum(
        1 for rc in contexts
        if rc.chunk.company and rc.chunk.company.lower() in expected
    )
    return matches / len(contexts)


def compute_retrieval_confidence(
    contexts: list[RetrievedChunk],
    company_filter: list[str] | None,
) -> float:
    """Return a composite evidence-quality score in [0, 1].

    Combines three signals — score gap, score magnitude, entity coverage —
    using fixed weights (see module-level constants). Computed on the
    post-rerank, pre-augmentation chunk list so conflict and sibling chunks
    (added at synthetic scores) do not skew the result.
    """
    if not contexts:
        return 0.0
    scores = [rc.rerank_score for rc in contexts if rc.rerank_score is not None]
    if not scores:
        return 0.0
    gap = _score_gap_signal(scores)
    mag = _score_magnitude_signal(scores[0])
    cov = _coverage_signal(contexts, company_filter)
    composite = W_GAP * gap + W_MAGNITUDE * mag + W_COVERAGE * cov
    return max(0.0, min(1.0, composite))


def confidence_band(score: float) -> str:
    """Map a [0, 1] confidence score to a human-readable band label."""
    if score >= CONFIDENCE_HIGH_CUTOFF:
        return "high"
    if score >= CONFIDENCE_LOW_CUTOFF:
        return "medium"
    return "low"
