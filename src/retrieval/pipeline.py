"""Full retrieval pipeline — orchestrates BM25 + vector + RRF + rerank + version dedup + parent expansion.

This is the single entry point used by the generation layer and the
Streamlit UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.db import connect
from src.models import RetrievedChunk
from src.retrieval.bm25 import bm25_search
from src.retrieval.entity_filter import extract_companies
from src.retrieval.fusion import rrf_fuse
from src.retrieval.reranker import RERANK_THRESHOLD, rerank
from src.retrieval.vector import vector_search
from src.versioning.chains import dedupe_by_chain, expand_to_parents
from src.versioning.classifier import TemporalIntent, classify_intent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable per-call.
# ---------------------------------------------------------------------------

CANDIDATES_PER_RETRIEVER = 50   # top-K from BM25 and from vector
FUSED_TOP_N = 40                # candidates passed to reranker
RERANK_TOP_N = 5                # contexts returned to the LLM
RRF_K = 60                      # standard RRF constant


@dataclass
class RetrievalResult:
    """Output bundle from `retrieve()` — contexts plus diagnostics for the UI."""

    query: str
    intent: TemporalIntent
    contexts: list[RetrievedChunk]
    companies: list[str] = field(default_factory=list)
    abstain: bool = False
    abstain_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    *,
    candidates_per_retriever: int = CANDIDATES_PER_RETRIEVER,
    fused_top_n: int = FUSED_TOP_N,
    rerank_top_n: int = RERANK_TOP_N,
    rerank_threshold: float = RERANK_THRESHOLD,
) -> RetrievalResult:
    """Run the full retrieval pipeline for a single query.

    Order:
        1. classify temporal intent
        2. BM25 and vector search (children only)
        3. RRF fusion → fused_top_n candidates
        4. cross-encoder rerank → rerank_top_n candidates
        5. version-chain dedupe (per intent)
        6. small-to-large expansion (child → parent)

    Args:
        query: Natural-language user query.
        candidates_per_retriever: top-K pulled from each of BM25 and vector.
        fused_top_n: candidates kept after RRF.
        rerank_top_n: contexts returned to the caller.
        rerank_threshold: minimum top reranker score; below this we abstain.

    Returns:
        A `RetrievalResult` whose `contexts` is the parent-expanded list.
        If the reranker top score is below `rerank_threshold`, `abstain=True`
        and contexts is still populated for the UI to display.
    """
    intent = classify_intent(query)
    companies = extract_companies(query)
    logger.info("Query intent: %s | companies=%s | %s", intent, companies, query)

    with connect() as conn:
        bm25_results = bm25_search(
            query, limit=candidates_per_retriever, conn=conn, companies=companies
        )
        vector_results = vector_search(
            query, limit=candidates_per_retriever, conn=conn, companies=companies
        )

        logger.info("  BM25 hits: %d | Vector hits: %d",
                    len(bm25_results), len(vector_results))

        fused = rrf_fuse(bm25_results, vector_results, k=RRF_K, top_n=fused_top_n)
        logger.info("  Fused candidates: %d", len(fused))

        reranked = rerank(query, fused, top_n=rerank_top_n)
        logger.info("  Reranked top score: %.3f",
                    reranked[0].rerank_score if reranked else float("-inf"))

        deduped = dedupe_by_chain(reranked, intent)
        logger.info("  After version dedupe: %d", len(deduped))

        expanded = expand_to_parents(deduped, conn=conn)
        logger.info("  After parent expansion: %d", len(expanded))

    top_rerank_score = (
        reranked[0].rerank_score if reranked and reranked[0].rerank_score is not None
        else float("-inf")
    )
    abstain = top_rerank_score < rerank_threshold
    abstain_reason = (
        f"top reranker score {top_rerank_score:.3f} below threshold {rerank_threshold}"
        if abstain else None
    )

    return RetrievalResult(
        query=query,
        intent=intent,
        contexts=expanded,
        companies=companies,
        abstain=abstain,
        abstain_reason=abstain_reason,
        diagnostics={
            "bm25_hits": len(bm25_results),
            "vector_hits": len(vector_results),
            "fused": len(fused),
            "reranked": len(reranked),
            "after_dedupe": len(deduped),
            "after_expansion": len(expanded),
            "top_rerank_score": top_rerank_score,
            "companies_filter": companies,
        },
    )
