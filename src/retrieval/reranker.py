"""Cross-encoder reranker using bge-reranker-v2-m3 in raw-logit mode (sentence-transformers)."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Module-level singleton, loaded lazily on first rerank call.
# Configured to emit raw logits (see _configure_raw_logits) instead of the
# default sigmoid, which saturates near 1.0 on this corpus and collapses
# ranking separation.
_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Pinned commit SHA — supply-chain mitigation against an upstream model swap.
_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_cross_encoder: CrossEncoder | None = None

# Registry for loading alternate reranker models by (name, revision), used by
# rerank_with_model. Kept separate from the default singleton so production
# load-time behaviour is unchanged.
_registry: dict[tuple[str, str | None], CrossEncoder] = {}

# Chunks scoring below this threshold are treated as irrelevant and trigger the
# abstention gate. In raw-logit mode bge-reranker-v2-m3 assigns even off-topic
# chunks mildly positive scores on this corpus, so this floor rarely fires: it
# is a backstop, and abstention is driven by the LLM-side guard.
RERANK_THRESHOLD: float = -5.0

# Content types whose `chunk_text` is free-form prose emitted by the vision
# model rather than text extracted from the document. The prose often omits
# the company name and the word "chart" itself, depressing cross-encoder
# scores on queries that use those tokens verbatim.
_VISION_EXTRACTED_CONTENT_TYPES = {"chart_description", "chart_context"}


def _passage_for_rerank(rc: RetrievedChunk) -> str:
    """Return the passage text passed to the cross-encoder for *rc*.

    Prefers ``contextualized_text`` when present and non-whitespace, else
    falls back to ``chunk_text``. For vision-extracted chunks
    (``chart_description`` / ``chart_context``) prepends a short structured
    marker so the cross-encoder has a lexical handle on entity and
    provenance.
    """
    chunk = rc.chunk
    base = (
        chunk.contextualized_text
        if chunk.contextualized_text and chunk.contextualized_text.strip()
        else chunk.chunk_text
    )
    if chunk.content_type in _VISION_EXTRACTED_CONTENT_TYPES:
        page_suffix = f", p.{chunk.page_number}" if chunk.page_number else ""
        prefix = f"Chart from {chunk.company} ({chunk.doc_type}{page_suffix}): "
        return prefix + base
    return base


def _configure_raw_logits(encoder: CrossEncoder) -> CrossEncoder:
    """Force the cross-encoder to emit raw logits instead of its default
    activation.

    bge-reranker-v2-m3 defaults to a sigmoid that saturates near 1.0 on this
    corpus, collapsing the score separation the gate and confidence scoring
    rely on. Setting an identity activation restores the raw-logit ranking
    signal. This is a no-op for models that already emit logits (e.g. the
    ms-marco cross-encoders), so it is safe to apply unconditionally.
    """
    identity = torch.nn.Identity()
    # Prefer the current attribute name; `default_activation_function` is a
    # deprecated alias on newer sentence-transformers.
    if hasattr(encoder, "activation_fn"):
        encoder.activation_fn = identity
    elif hasattr(encoder, "default_activation_function"):
        setattr(encoder, "default_activation_function", identity)
    return encoder


def _select_device() -> str:
    """Pick the device for the cross-encoder and warn loudly when on CPU.

    BGE-reranker-v2-m3 is ~568M params; CPU inference is multi-second per
    pair on this corpus. We pass the device explicitly so a user without a
    GPU sees a clear warning rather than silent multi-minute query latency.
    """
    if torch.cuda.is_available():
        return "cuda"
    logger.warning(
        "No CUDA device detected — BGE reranker will run on CPU and will be "
        "impractically slow (multi-second per pair). README recommends a GPU."
    )
    return "cpu"


def _get_cross_encoder() -> CrossEncoder:
    """Return the reranker singleton, loading it on first use."""
    global _cross_encoder
    if _cross_encoder is None:
        device = _select_device()
        logger.info("Loading reranker %s on device=%s", _MODEL_NAME, device)
        _cross_encoder = _configure_raw_logits(
            CrossEncoder(_MODEL_NAME, revision=_MODEL_REVISION, device=device)
        )
    return _cross_encoder


def _get_registry_encoder(model_name: str, revision: str | None) -> CrossEncoder:
    """Return a CrossEncoder for *model_name*, loading it once and caching it.

    Raises a descriptive exception if the model cannot be loaded; never falls
    back silently to the default model.
    """
    key = (model_name, revision)
    if key not in _registry:
        try:
            kwargs: dict[str, Any] = {"device": _select_device()}
            if revision is not None:
                kwargs["revision"] = revision
            _registry[key] = _configure_raw_logits(CrossEncoder(model_name, **kwargs))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'"
                + (f" (revision={revision})" if revision else "")
                + f": {exc}"
            ) from exc
    return _registry[key]


# Cap the cross-encoder batch so peak VRAM stays within a small shared GPU
# (e.g. a 6 GB laptop card). Attention memory scales with batch_size * seq_len^2,
# and the comparison / synthesis paths rerank the full fused pool more than once
# per query, so an unbounded batch can exhaust VRAM mid-run.
_RERANK_BATCH_SIZE = 8


def _predict_scores(model: CrossEncoder, pairs: list[tuple[str, str]]) -> list[float]:
    """Score *pairs* with the cross-encoder and return raw logits as floats.

    Uses a bounded batch size and releases the CUDA caching allocator after
    scoring so repeated reranks within a single query do not accumulate VRAM on
    a memory-constrained GPU.
    """
    raw_scores = model.predict(cast(Any, pairs), batch_size=_RERANK_BATCH_SIZE)
    if isinstance(raw_scores, np.ndarray):
        scores = raw_scores.tolist()
    elif hasattr(raw_scores, "tolist"):
        scores = raw_scores.tolist()
    else:
        scores = list(raw_scores)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [float(s) for s in scores]


def rerank_with_model(
    query: str,
    candidates: list[RetrievedChunk],
    model_name: str,
    revision: str | None = None,
    top_n: int = 5,
) -> list[RetrievedChunk]:
    """Score and sort *candidates* using an explicitly specified cross-encoder.

    Identical contract to :func:`rerank`, but uses *model_name* (with optional
    *revision*) rather than the module-level default.  The model is loaded
    lazily and cached in the module registry so repeated calls within the same
    process pay the load cost only once.

    Raises:
        RuntimeError: If the requested model cannot be loaded.
    """
    if not candidates:
        return []

    effective_top_n = min(top_n, len(candidates))
    pairs: list[tuple[str, str]] = [(query, _passage_for_rerank(rc)) for rc in candidates]

    model = _get_registry_encoder(model_name, revision)
    scores = _predict_scores(model, pairs)

    for rc, score in zip(candidates, scores):
        rc.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)
    return ranked[:effective_top_n]


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

    pairs: list[tuple[str, str]] = [(query, _passage_for_rerank(rc)) for rc in candidates]

    model = _get_cross_encoder()
    scores = _predict_scores(model, pairs)

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
