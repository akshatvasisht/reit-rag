"""Full retrieval pipeline — orchestrates BM25 + vector + RRF + rerank + version deduplication + parent expansion.

This is the single entry point used by the generation layer and the
Streamlit UI.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from src.db import connect
from src.corpus_registry import CORPUS_REGISTRY
from src.models import Chunk, RetrievedChunk
from src.retrieval.adaptive import ADAPTIVE_INTENTS, adaptive_retrieve
from src.retrieval.bm25 import bm25_search
from src.retrieval.confidence import (
    CONFIDENCE_HIGH_CUTOFF,
    CONFIDENCE_LOW_CUTOFF,
    _score_magnitude_signal,
    compute_retrieval_confidence,
    confidence_band,
)
from src.retrieval.entity_filter import extract_companies
from src.retrieval.fusion import rrf_fuse
from src.retrieval.reranker import RERANK_THRESHOLD, rerank
from src.retrieval.synthesis import retrieve_all_company_synthesis
from src.retrieval.vector import vector_search
from src.versioning.chains import dedupe_by_version_group, expand_to_parents
from src.versioning.classifier import TemporalIntent, classify_intent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contextual retrieval activation startup check
# ---------------------------------------------------------------------------

_ACTIVATION_CHECK_SQL = """
    SELECT
        COUNT(*) FILTER (WHERE contextualized_embedding IS NOT NULL) AS populated,
        COUNT(*) AS total,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE contextualized_embedding IS NOT NULL)
                / NULLIF(COUNT(*), 0),
            1
        ) AS pct_populated
    FROM chunks
    WHERE content_type = 'text'
"""

_CONTEXTUAL_RETRIEVAL_MIN_PCT = 95.0


def _check_contextual_activation_at_startup() -> None:
    """Warn when contextual retrieval columns are not sufficiently populated.

    Runs once at module import time.  Any failure (missing DB, missing table,
    missing env var) is silently swallowed so the import never raises.
    """
    try:
        from src.db import connect as _connect  # noqa: PLC0415

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(_ACTIVATION_CHECK_SQL)
            row = cur.fetchone()

        if row is None:
            return

        total = row[1]
        if total == 0:
            return

        pct = float(row[2])
        if pct < _CONTEXTUAL_RETRIEVAL_MIN_PCT:
            logger.warning(
                "Contextual retrieval is only %.1f%% populated; retrieval will fall "
                "back to base embedding columns via COALESCE. "
                "Run scripts/contextualize.py --embed to activate.",
                pct,
            )
    except Exception:  # noqa: BLE001
        pass


_check_contextual_activation_at_startup()


# ---------------------------------------------------------------------------
# Defaults — overridable per-call.
# ---------------------------------------------------------------------------

CANDIDATES_PER_RETRIEVER = 50   # top-K from BM25 and from vector
FUSED_TOP_N = 40                # candidates passed to reranker
RERANK_TOP_N = 5                # contexts returned to the LLM
RRF_K = 60                      # standard RRF constant

# Hard cap on total context chunks sent to the LLM.  When conflict chunks are
# injected, the lowest-scoring retained chunks are dropped first (never the
# conflict chunks themselves) until the list fits within this limit.
MAX_CONTEXT_CHUNKS = RERANK_TOP_N * 2

# Context cap for all-company synthesis queries.  Each corpus company is given
# up to 3 chunk slots so no single company dominates while the LLM still
# receives cross-company coverage.
MAX_SYNTHESIS_CONTEXT_CHUNKS: int = len({e["company"] for e in CORPUS_REGISTRY}) * 3

# Synthetic rerank-score nudge applied to conflict-injected chunks so they
# clear the abstention gate and remain distinguishable from legitimately
# high-scoring retained chunks in diagnostics.
CONFLICT_SCORE_OFFSET = 0.1

# Width of the rerank-score band above RERANK_THRESHOLD within which a chunk
# is considered borderline; chunks in this band trigger adjacent-page sibling
# fetches because the answer may live on the immediately preceding or following page.
SIBLING_BORDERLINE_WIDTH = 3.0


# ---------------------------------------------------------------------------
# Conflict-chunk columns — must match the SELECT shape used by bm25/vector.
# ---------------------------------------------------------------------------

_CHUNK_SELECT = """
    id, document_id, parent_chunk_id,
    company, ticker, doc_type, report_date, period_covered, doc_version,
    section_title, page_number, content_type, source_authority,
    chunk_text, is_parent, token_count,
    doc_subtype, page_content_class
""".strip()


def _fetch_chunks_by_ids(conn, ids: list[UUID]) -> list[RetrievedChunk]:
    """Fetch chunks by primary key and hydrate them as RetrievedChunk objects.

    Returns an empty list when ids is empty.  Each returned chunk has
    rerank_score left at None; the caller assigns a score before merging.
    """
    if not ids:
        return []
    str_ids = [str(i) for i in ids]
    sql = f"SELECT {_CHUNK_SELECT} FROM chunks WHERE id = ANY(%s)"
    with conn.cursor() as cur:
        cur.execute(sql, [str_ids])
        if cur.description is None:
            return []
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return [RetrievedChunk(chunk=Chunk.from_row(row, cols)) for row in rows]


def _fetch_chunk_by_doc_page(conn, document_id, page_number: int) -> RetrievedChunk | None:
    """Fetch the best chunk for a given document and page number.

    Prefers the parent chunk (is_parent=True) for that page so the generation
    layer receives the widest available context.  Falls back to the lowest-id
    child chunk when no parent exists.  Returns None when no chunk exists at
    all for that (document_id, page_number) pair.
    """
    sql = f"""
        SELECT {_CHUNK_SELECT}
        FROM chunks
        WHERE document_id = %s
          AND page_number = %s
        ORDER BY is_parent DESC, id ASC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, [str(document_id), page_number])
        if cur.description is None:
            return None
        cols = [d.name for d in cur.description]
        row = cur.fetchone()
    if row is None:
        return None
    # rerank_score is left as None: sibling chunks are added purely for
    # coverage and were not scored by the cross-encoder.
    return RetrievedChunk(chunk=Chunk.from_row(row, cols))


def expand_to_siblings(
    contexts: list[RetrievedChunk],
    conn,
    window: int = 1,
) -> list[RetrievedChunk]:
    """Append adjacent-page sibling chunks for borderline-scored retained chunks.

    When a chunk's rerank score sits in the borderline band
    [RERANK_THRESHOLD, RERANK_THRESHOLD + SIBLING_BORDERLINE_WIDTH], the answer may actually live on
    the page immediately before or after the retrieved one.  This function
    fetches the chunk at page N±window within the same document and appends it
    to the list so the LLM has that additional context.

    Only chunks with a non-None rerank_score in the borderline band trigger
    sibling lookups.  Chunks already present in the context list are never
    duplicated.
    """
    BORDERLINE_LOWER = RERANK_THRESHOLD
    BORDERLINE_UPPER = RERANK_THRESHOLD + SIBLING_BORDERLINE_WIDTH

    seen_ids = {rc.chunk.id for rc in contexts}
    siblings_to_add: list[RetrievedChunk] = []

    for rc in contexts:
        if rc.rerank_score is None or not (BORDERLINE_LOWER <= rc.rerank_score <= BORDERLINE_UPPER):
            continue
        if rc.chunk.page_number is None:
            continue
        for page_offset in (-window, window):
            target_page = rc.chunk.page_number + page_offset
            if target_page < 1:
                continue
            sibling = _fetch_chunk_by_doc_page(conn, rc.chunk.document_id, target_page)
            if sibling and sibling.chunk.id not in seen_ids:
                siblings_to_add.append(sibling)
                seen_ids.add(sibling.chunk.id)

    return contexts + siblings_to_add


def find_conflicting_chunks(
    retained: list[RetrievedChunk],
    company_filter: list[str] | None,
    conn,
) -> list[RetrievedChunk]:
    """Identify and surface chunks whose claim values conflict with retained chunks.

    Looks up the chunk_claims table for each (company, metric) tuple present
    in the retained set.  Any chunk NOT already in the retained set that shares
    the same company + metric but carries a different value is added to the
    output with a rerank_score just above RERANK_THRESHOLD so it is placed
    in context without displacing high-scoring retained chunks.

    Args:
        retained: Chunks kept after version-group deduplication.
        company_filter: Optional list of company names to restrict the lookup;
            None means all companies are in scope.
        conn: Open psycopg connection.

    Returns:
        A new list containing all retained chunks plus any newly discovered
        conflicting chunks.  The caller is responsible for the context cap.
    """
    if not retained:
        return list(retained)

    retained_ids = [str(c.chunk.id) for c in retained]

    # Collect all (company, metric, value) tuples for the retained set.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT company, metric, value, chunk_id
            FROM chunk_claims
            WHERE chunk_id = ANY(%s)
              AND (%s::text[] IS NULL OR company = ANY(%s))
            """,
            [retained_ids, company_filter or None, company_filter or None],
        )
        claim_rows = cur.fetchall()

    if not claim_rows:
        return list(retained)

    # For each (company, metric, value) find chunks outside the retained set
    # that report a different value for the same metric AND belong to the same
    # qualifier class.  Claims from different qualifier classes (e.g. a reported
    # figure vs. a guidance target) are not treated as conflicts because they
    # describe semantically distinct quantities.
    #
    # Qualifier classes:
    #   "reported" — NULL, "approximate", or "estimated"
    #   "forward"  — "guidance" or "target"
    #
    # Two claims conflict only when both belong to the same class and their
    # values differ.
    conflict_ids: set[UUID] = set()
    for company, metric, value, base_chunk_id in claim_rows:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT cc.chunk_id
                FROM chunk_claims cc
                JOIN chunk_claims base
                  ON base.company  = cc.company
                 AND base.metric   = cc.metric
                 AND base.chunk_id = %s
                WHERE cc.company   = %s
                  AND cc.metric    = %s
                  AND cc.value    != base.value
                  AND cc.chunk_id != ALL(%s)
                  AND (
                        (
                          (base.value_qualifier IS NULL
                           OR base.value_qualifier IN ('approximate', 'estimated'))
                          AND
                          (cc.value_qualifier IS NULL
                           OR cc.value_qualifier IN ('approximate', 'estimated'))
                        )
                        OR
                        (
                          base.value_qualifier IN ('guidance', 'target')
                          AND
                          cc.value_qualifier   IN ('guidance', 'target')
                        )
                      )
                """,
                [str(base_chunk_id), company, metric, retained_ids],
            )
            for (cid,) in cur.fetchall():
                conflict_ids.add(cid)

    if not conflict_ids:
        return list(retained)

    conflict_chunks = _fetch_chunks_by_ids(conn, list(conflict_ids))
    # Assign a score that clears the abstention gate so conflict chunks reach
    # the LLM, while remaining distinguishable from legitimately high-scoring
    # retained chunks.
    for rc in conflict_chunks:
        rc.rerank_score = RERANK_THRESHOLD + CONFLICT_SCORE_OFFSET

    logger.info("Conflict detection: injecting %d additional chunk(s)", len(conflict_chunks))
    return list(retained) + conflict_chunks


def _apply_post_rerank_pipeline(
    contexts: list[RetrievedChunk],
    *,
    intent: str,
    conn,
    company_filter: list[str] | None,
    max_chunks: int,
    abstain: bool = False,
) -> list[RetrievedChunk]:
    """Apply the full post-rerank enrichment stages to a list of reranked chunks.

    Runs, in order:
      1. Version-group deduplication (``dedupe_by_version_group``).
      2. Conflict-chunk injection (``find_conflicting_chunks``) — only on the
         answered path, skipped when ``abstain=True`` or when the deduped list
         has no associated claims (the no-op guard lives inside
         find_conflicting_chunks itself).
      3. Context-cap enforcement: when conflict injection pushes the list beyond
         ``max_chunks``, the lowest-scoring retained chunks are dropped first;
         conflict-class chunks are never dropped before retained ones.
      4. Parent expansion (``expand_to_parents``).
      5. Sibling expansion (``expand_to_siblings``) — only on the answered path.
      6. Second-pass context-cap enforcement after sibling expansion (siblings
         are dropped before retained or conflict chunks).

    The deterministic sort ``(company, report_date, page_number)`` is applied
    by the caller so it remains the final step before chunks reach the LLM.

    Args:
        contexts: Reranked chunks from the fusion + cross-encoder stage.
        intent: Temporal intent string; governs dedupe behaviour.
        conn: Open psycopg connection.
        company_filter: Optional list of company names for conflict lookup scope.
        max_chunks: Hard cap on output list length.
        abstain: When True, skips conflict injection and sibling expansion.

    Returns:
        Enriched, capped list of ``RetrievedChunk`` objects.
    """
    # Stage 1 — version-group deduplication.
    deduped = dedupe_by_version_group(contexts, intent)  # type: ignore[arg-type]
    logger.info("  [post-rerank] after version dedup: %d", len(deduped))

    # Stage 2 — conflict injection (answered path only).
    if not abstain:
        with_conflicts = find_conflicting_chunks(deduped, company_filter, conn)
    else:
        with_conflicts = list(deduped)

    # Stage 3 — cap after conflict injection.
    if len(with_conflicts) > max_chunks:
        conflict_chunk_ids = {rc.chunk.id for rc in with_conflicts if rc not in deduped}
        retained_part = [rc for rc in with_conflicts if rc.chunk.id not in conflict_chunk_ids]
        conflict_part = [rc for rc in with_conflicts if rc.chunk.id in conflict_chunk_ids]
        max_retained = max_chunks - len(conflict_part)
        retained_trimmed = sorted(
            retained_part,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:max(0, max_retained)]
        with_conflicts = retained_trimmed + conflict_part
        logger.info(
            "  [post-rerank] cap after conflict: %d retained + %d conflict = %d",
            len(retained_trimmed), len(conflict_part), len(with_conflicts),
        )

    # Stage 4 — parent expansion.
    expanded = expand_to_parents(with_conflicts, conn=conn)
    logger.info("  [post-rerank] after parent expansion: %d", len(expanded))

    # Stage 5 — sibling expansion (answered path only).
    if not abstain:
        with_siblings = expand_to_siblings(expanded, conn)
    else:
        with_siblings = list(expanded)

    # Stage 6 — cap after sibling expansion (siblings are lowest-priority).
    if len(with_siblings) > max_chunks:
        retained_and_conflict_ids = {rc.chunk.id for rc in expanded}
        sibling_part = [rc for rc in with_siblings if rc.chunk.id not in retained_and_conflict_ids]
        non_sibling_part = [rc for rc in with_siblings if rc.chunk.id in retained_and_conflict_ids]
        max_siblings = max_chunks - len(non_sibling_part)
        sibling_trimmed = sibling_part[:max(0, max_siblings)]
        with_siblings = non_sibling_part + sibling_trimmed
        logger.info(
            "  [post-rerank] sibling cap: %d core + %d siblings = %d",
            len(non_sibling_part), len(sibling_trimmed), len(with_siblings),
        )

    logger.info("  [post-rerank] after sibling expansion: %d", len(with_siblings))
    return with_siblings
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
    # Calibrated composite evidence-quality score in [0, 1]; 0.0 when the
    # abstain path fires or no scored chunks were available.
    retrieval_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Internal retrieval core — shared by retrieve() and adaptive_retrieve()
# ---------------------------------------------------------------------------


def _retrieve_core(
    query: str,
    company_filter: list[str] | None,
    conn,
    *,
    candidates_per_retriever: int = CANDIDATES_PER_RETRIEVER,
    fused_top_n: int = FUSED_TOP_N,
    rerank_top_n: int = RERANK_TOP_N,
    rerank_threshold: float = RERANK_THRESHOLD,
    intent: TemporalIntent = "latest",
) -> RetrievalResult:
    """Execute one full retrieval pass without intent classification or adaptive looping.

    Applies BM25 + vector search, RRF fusion, cross-encoder rerank,
    version-group deduplication, conflict injection, parent expansion,
    sibling expansion, and a deterministic sort by document coordinates.
    Used by both the top-level retrieve() and the adaptive hop loop so the
    logic is not duplicated.

    Args:
        query: The sub-question or original query to retrieve for.
        company_filter: Optional list of company names to restrict retrieval.
            None means no filter (corpus-wide).
        conn: Open psycopg connection.
        intent: Temporal intent; governs dedupe behavior.

    Returns:
        A RetrievalResult for this pass.
    """
    companies = company_filter or []

    bm25_results = bm25_search(
        query, limit=candidates_per_retriever, conn=conn,
        companies=companies if companies else None,
    )
    vector_results = vector_search(
        query, limit=candidates_per_retriever, conn=conn,
        companies=companies if companies else None,
    )

    logger.info("  BM25 hits: %d | Vector hits: %d",
                len(bm25_results), len(vector_results))

    fused = rrf_fuse(bm25_results, vector_results, k=RRF_K, top_n=fused_top_n)
    logger.info("  Fused candidates: %d", len(fused))

    reranked = rerank(query, fused, top_n=rerank_top_n)
    logger.info("  Reranked top score: %.3f",
                reranked[0].rerank_score if reranked else float("-inf"))

    top_rerank_score = (
        reranked[0].rerank_score if reranked and reranked[0].rerank_score is not None
        else float("-inf")
    )
    abstain = top_rerank_score < rerank_threshold

    # Confidence is computed on the post-rerank, pre-augmentation chunk list
    # so conflict and sibling chunks (added at synthetic scores) do not skew
    # the magnitude or gap signals.
    retrieval_confidence = (
        0.0 if abstain
        else compute_retrieval_confidence(reranked, company_filter)
    )

    final_contexts = _apply_post_rerank_pipeline(
        reranked,
        intent=intent,
        conn=conn,
        company_filter=companies or None,
        max_chunks=MAX_CONTEXT_CHUNKS,
        abstain=abstain,
    )

    abstain_reason = (
        f"top reranker score {top_rerank_score:.3f} below threshold {rerank_threshold}"
        if abstain else None
    )

    return RetrievalResult(
        query=query,
        intent=intent,
        contexts=final_contexts,
        companies=companies,
        abstain=abstain,
        abstain_reason=abstain_reason,
        retrieval_confidence=retrieval_confidence,
        diagnostics={
            "bm25_hits": len(bm25_results),
            "vector_hits": len(vector_results),
            "fused": len(fused),
            "reranked": len(reranked),
            "after_expansion": len(final_contexts),
            "top_rerank_score": top_rerank_score,
            "companies_filter": companies,
        },
    )



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
        5. version-group deduplication (per intent)
        6. small-to-large expansion (child → parent)
        7. adaptive multi-hop if intent in ADAPTIVE_INTENTS

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

    # All-company synthesis: run a separate retrieval pass per corpus company so
    # every company gets equal representation rather than relying on a global
    # top-K that might be dominated by one company's peer-comparison tables.
    if intent == "all_company_synthesis":
        with connect() as conn:
            synthesis_result = retrieve_all_company_synthesis(query, conn)
            if synthesis_result.abstain:
                return synthesis_result
            final, sub_queries = adaptive_retrieve(query, synthesis_result, conn)
            final.diagnostics["sub_queries"] = sub_queries
            final.diagnostics["retrieval_hops"] = len(sub_queries)
            return final

    with connect() as conn:
        initial = _retrieve_core(
            query,
            company_filter=companies if companies else None,
            conn=conn,
            candidates_per_retriever=candidates_per_retriever,
            fused_top_n=fused_top_n,
            rerank_top_n=rerank_top_n,
            rerank_threshold=rerank_threshold,
            intent=intent,
        )
        if intent in ADAPTIVE_INTENTS and not initial.abstain:
            final, sub_queries = adaptive_retrieve(query, initial, conn)
            final.diagnostics["sub_queries"] = sub_queries
            final.diagnostics["retrieval_hops"] = len(sub_queries)
            return final
        return initial
