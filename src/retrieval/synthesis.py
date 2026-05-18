"""All-company-synthesis retrieval path.

For queries with intent ``"all_company_synthesis"`` the pipeline runs a
focused retrieval pass per corpus company and merges the results, so every
company gets equal representation rather than competing in a single global
top-K pool. The merged chunk list then flows through the same post-rerank
pipeline (conflict injection, parent and sibling expansion, deterministic
sort) as every other intent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.corpus_registry import CORPUS_REGISTRY
from src.models import RetrievedChunk
from src.retrieval.bm25 import bm25_search
from src.retrieval.confidence import _score_magnitude_signal
from src.retrieval.fusion import rrf_fuse
from src.retrieval.reranker import RERANK_THRESHOLD, rerank
from src.retrieval.vector import vector_search

if TYPE_CHECKING:
    # Avoid a module-load cycle: pipeline.py imports this module at its top.
    from src.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


def _retrieve_single_company(
    query: str,
    company: str,
    conn,
    top_k: int = 3,
) -> "RetrievalResult":
    """Run a focused retrieval pass restricted to a single company.

    Reuses the existing BM25 + vector + RRF + rerank infrastructure with a
    company filter so each corpus company gets equal representation in the
    synthesis context regardless of which company's peer-comparison tables
    happen to score highest in a global pass.

    Args:
        query: The user's natural language query string.
        company: Canonical company name to restrict retrieval to.
        conn: Open psycopg connection shared from the caller.
        top_k: Maximum reranked chunks to keep per company.

    Returns:
        A RetrievalResult for this company; abstain reflects whether the
        company's best chunk cleared the rerank threshold.
    """
    from src.retrieval.pipeline import (  # noqa: PLC0415
        CANDIDATES_PER_RETRIEVER,
        FUSED_TOP_N,
        RRF_K,
        RetrievalResult,
    )

    companies = [company]
    bm25_results = bm25_search(
        query, limit=CANDIDATES_PER_RETRIEVER, conn=conn, companies=companies
    )
    vector_results = vector_search(
        query, limit=CANDIDATES_PER_RETRIEVER, conn=conn, companies=companies
    )
    fused = rrf_fuse(bm25_results, vector_results, k=RRF_K, top_n=FUSED_TOP_N)
    reranked = rerank(query, fused, top_n=top_k)

    top_rerank_score: float = (
        reranked[0].rerank_score
        if reranked and reranked[0].rerank_score is not None
        else float("-inf")
    )
    abstain = top_rerank_score < RERANK_THRESHOLD

    return RetrievalResult(
        query=query,
        intent="all_company_synthesis",
        contexts=reranked,
        companies=companies,
        abstain=abstain,
        abstain_reason=(
            f"top reranker score {top_rerank_score:.3f} below threshold {RERANK_THRESHOLD}"
            if abstain else None
        ),
        diagnostics={
            "top_rerank": top_rerank_score,
        },
    )


def retrieve_all_company_synthesis(
    query: str,
    conn,
    *,
    rerank_top_n: int | None = None,
    rerank_threshold: float | None = None,
) -> "RetrievalResult":
    """Run per-company sub-retrieval and merge results for synthesis queries.

    Fires one focused retrieval pass per corpus company so every company gets
    equal representation — a global top-K pass would over-represent companies
    whose peer-comparison tables happen to score highest and leave others with
    zero coverage.

    The merged chunk list is deduplicated by chunk PK (a chunk incidentally
    mentioning two companies appears only once). The context cap for synthesis
    is ``len(corpus_companies) * 3`` rather than MAX_CONTEXT_CHUNKS because the
    synthesis path intentionally trades a wider context window for cross-company
    coverage.

    Args:
        query: The user's natural language query string.
        conn: Open psycopg connection.

    Returns:
        A RetrievalResult with all per-company chunks merged and deduplicated.
        abstain=True only when no company returned a chunk above the threshold.
    """
    from src.retrieval.pipeline import (  # noqa: PLC0415
        MAX_SYNTHESIS_CONTEXT_CHUNKS,
        RetrievalResult,
        _apply_post_rerank_pipeline,
        enforce_per_issuer_floor,
    )

    per_company_top_k = rerank_top_n if rerank_top_n is not None else 3
    threshold = rerank_threshold if rerank_threshold is not None else RERANK_THRESHOLD

    corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY})
    all_chunks: list[RetrievedChunk] = []
    per_company_diagnostics: dict[str, dict] = {}
    # Retain the per-company candidate lists for the per-issuer floor step.
    candidates_by_company: dict[str, list[RetrievedChunk]] = {}

    for company in corpus_companies:
        company_result = _retrieve_single_company(
            query, company, conn, top_k=per_company_top_k
        )
        per_company_diagnostics[company] = {
            "top_rerank": company_result.diagnostics.get("top_rerank"),
            "chunks_found": len(company_result.contexts),
        }
        all_chunks.extend(company_result.contexts)
        candidates_by_company[company] = list(company_result.contexts)

    # Deduplicate by chunk PK: a chunk mentioning multiple companies in a peer
    # table would otherwise appear once per company sub-query.
    seen: set = set()
    deduped: list[RetrievedChunk] = []
    for rc in all_chunks:
        if rc.chunk.id not in seen:
            seen.add(rc.chunk.id)
            deduped.append(rc)

    any_above_threshold = any(
        d["top_rerank"] is not None and d["top_rerank"] >= threshold
        for d in per_company_diagnostics.values()
        if d["chunks_found"] > 0
    )

    # Per-issuer floor: for synthesis queries, ensure every corpus company that
    # produced at least one candidate is represented in the final retained set.
    # Prevents top-N budget collapse to a single issuer when one company's
    # peer-comparison tables outscore everyone else's chunks globally.
    if any_above_threshold:
        deduped = enforce_per_issuer_floor(
            deduped,
            companies=corpus_companies,
            candidates_by_company=candidates_by_company,
        )

    # For the synthesis path the cross-company score gap is not meaningful
    # (each company was retrieved independently), so confidence is the mean
    # of per-company magnitude signals for companies that produced scored chunks.
    per_company_top_scores = [
        d["top_rerank"]
        for d in per_company_diagnostics.values()
        if d["top_rerank"] is not None and d["chunks_found"] > 0
    ]
    if per_company_top_scores and any_above_threshold:
        synthesis_confidence = max(
            0.0,
            min(
                1.0,
                sum(_score_magnitude_signal(s) for s in per_company_top_scores)
                / len(per_company_top_scores),
            ),
        )
    else:
        synthesis_confidence = 0.0

    # Apply the full post-rerank pipeline to the cross-company merged set.
    # dedupe_by_version_group is a no-op for "all_company_synthesis" (pass-through),
    # but conflict injection, parent expansion, and sibling expansion all run here
    # for the first time on the synthesis path.
    cap = MAX_SYNTHESIS_CONTEXT_CHUNKS or len(corpus_companies) * 3
    final_contexts = _apply_post_rerank_pipeline(
        deduped,
        intent="all_company_synthesis",
        conn=conn,
        company_filter=None,
        max_chunks=cap,
        abstain=not any_above_threshold,
    )

    return RetrievalResult(
        query=query,
        intent="all_company_synthesis",
        contexts=final_contexts,
        companies=corpus_companies,
        abstain=not any_above_threshold,
        abstain_reason=(
            "no company returned evidence above the rerank threshold"
            if not any_above_threshold else None
        ),
        retrieval_confidence=synthesis_confidence,
        diagnostics={
            "intent": "all_company_synthesis",
            "per_company": per_company_diagnostics,
            "total_chunks": len(final_contexts),
        },
    )
