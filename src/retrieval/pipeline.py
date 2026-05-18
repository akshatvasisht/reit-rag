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
from src.versioning.classifier import TemporalIntent, classify_intent, classify_intent_full

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


def check_contextual_activation() -> None:
    """Warn when contextual retrieval columns are not sufficiently populated.

    Called explicitly by application entrypoints (Streamlit app, ingestion
    scripts) rather than at module import time so importing this module
    does not require DB availability. Resilient by design: any failure
    (missing DB, missing table, missing env var) is caught so application
    startup never raises from this check — but the exception is now
    logged at WARNING so operators can see when the check is failing.
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Contextual-activation startup check failed (%s: %s); "
            "proceeding without the activation signal.",
            type(exc).__name__, exc,
        )


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


def enforce_per_issuer_floor(
    retained: list[RetrievedChunk],
    companies: list[str],
    candidates_by_company: dict[str, list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    """Ensure every named company has at least one chunk in the retained set.

    For ``all_company_synthesis`` queries the top-N budget may be dominated by
    one issuer, leaving others with zero coverage.  This function enforces a
    per-issuer minimum of 1 chunk: for each named company absent from the
    retained set, it adds the best-scored (highest rerank_score) candidate from
    ``candidates_by_company[company]``.

    Only fires when ``companies`` is non-empty (i.e., for synthesis intents
    where the company list is known).  A no-op for non-synthesis intents
    (pass ``companies=[]``).

    Args:
        retained: Current retained chunks after rerank and initial pipeline stages.
        companies: List of canonical company names that must each be represented.
        candidates_by_company: Mapping from company name to all candidates from
            that company's sub-retrieval pass (already reranked, in any order).

    Returns:
        The retained list extended with best-of-issuer chunks for any company
        that had zero representation.
    """
    if not companies:
        return retained

    represented = {rc.chunk.company for rc in retained}
    to_add: list[RetrievedChunk] = []
    retained_ids = {rc.chunk.id for rc in retained}

    for company in companies:
        if company in represented:
            continue
        candidates = candidates_by_company.get(company, [])
        if not candidates:
            continue
        # Pick the candidate with the highest rerank_score; fall back to first.
        best = max(
            candidates,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
        )
        if best.chunk.id not in retained_ids:
            best.retrieval_stage = "per_issuer_floor"
            best.expansion_reason = (
                f"per-issuer floor: best-of-issuer for {company} "
                f"(score={best.rerank_score})"
            )
            to_add.append(best)
            retained_ids.add(best.chunk.id)
            logger.info(
                "  [per-issuer-floor] added chunk for %s (score=%.3f)",
                company,
                best.rerank_score if best.rerank_score is not None else float("-inf"),
            )

    if to_add:
        logger.info(
            "  [per-issuer-floor] added %d chunk(s) for %d missing company(ies)",
            len(to_add), len(to_add),
        )

    return retained + to_add


def expand_table_pairs(
    contexts: list[RetrievedChunk],
    conn,
) -> list[RetrievedChunk]:
    """Append table/chart_description chunks for text chunks in the retained set.

    When a text chunk on page N is retained in top-N, the cross-encoder may have
    prioritised it over a data-rich table chunk on the same (document_id, page_number).
    This function queries the DB for any chunk with content_type IN ('table',
    'chart_description') on the same page that is NOT already in the retained set,
    and includes it as a "table-pair expanded" chunk.

    Only text chunks trigger lookups; table/chart chunks do not (they are the
    targets of expansion, not the triggers).  Chunks with page_number=None are
    skipped.  Already-present chunks are never duplicated.

    Each appended chunk has rerank_score=None (not scored by the cross-encoder)
    and retrieval_stage="table_pair_expanded" for provenance.

    Args:
        contexts: Retained chunks after rerank (or after sibling expansion).
        conn: Open psycopg connection.

    Returns:
        Original list plus any table/chart_description chunks found on the same
        pages as text chunks.
    """
    seen_ids = {rc.chunk.id for rc in contexts}
    to_add: list[RetrievedChunk] = []

    sql = f"""
        SELECT {_CHUNK_SELECT}
        FROM chunks
        WHERE document_id = %s
          AND page_number = %s
          AND content_type = ANY(%s)
          AND id != ALL(%s)
        ORDER BY content_type, id
    """

    for rc in contexts:
        if rc.chunk.content_type != "text":
            continue
        if rc.chunk.page_number is None:
            continue
        doc_id = str(rc.chunk.document_id)
        page = rc.chunk.page_number
        excluded_ids = [str(i) for i in seen_ids]
        with conn.cursor() as cur:
            cur.execute(sql, [doc_id, page, ["table", "chart_description"], excluded_ids])
            if cur.description is None:
                continue
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        for row in rows:
            chunk = Chunk.from_row(row, cols)
            if chunk.id not in seen_ids:
                new_rc = RetrievedChunk(chunk=chunk)
                new_rc.retrieval_stage = "table_pair_expanded"
                new_rc.trigger_chunk_id = str(rc.chunk.id)
                new_rc.expansion_reason = (
                    f"same-page table/chart on page {page} for text chunk {rc.chunk.id}"
                )
                to_add.append(new_rc)
                seen_ids.add(chunk.id)

    if to_add:
        logger.info(
            "  [table-pair] added %d table/chart chunk(s) for %d text chunk(s)",
            len(to_add),
            sum(1 for rc in contexts if rc.chunk.content_type == "text" and rc.chunk.page_number is not None),
        )

    return contexts + to_add


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
        if rc.retrieval_stage is None:
            rc.retrieval_stage = "conflict_injected"
            rc.expansion_reason = "conflict detection: same-company/metric different-value chunk"

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
      5b. Table-pair expansion (``expand_table_pairs``) — for each text chunk
         in the retained set, include any same-page table or chart_description
         chunk that the reranker may have ranked below the top-N cutoff.
      6. Second-pass context-cap enforcement after sibling/table-pair expansion
         (expanded chunks are dropped before retained or conflict chunks).

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
    pre_parent_ids = {rc.chunk.id for rc in with_conflicts}
    expanded = expand_to_parents(with_conflicts, conn=conn)
    logger.info("  [post-rerank] after parent expansion: %d", len(expanded))
    for rc in expanded:
        if rc.chunk.id not in pre_parent_ids and rc.retrieval_stage is None:
            rc.retrieval_stage = "parent_expanded"
            rc.expansion_reason = "small-to-large parent body substitution"

    # Stage 5 — sibling expansion (answered path only).
    pre_sibling_ids = {rc.chunk.id for rc in expanded}
    if not abstain:
        with_siblings = expand_to_siblings(expanded, conn)
    else:
        with_siblings = list(expanded)
    for rc in with_siblings:
        if rc.chunk.id not in pre_sibling_ids and rc.retrieval_stage is None:
            rc.retrieval_stage = "sibling_expanded"
            rc.expansion_reason = (
                f"adjacent-page sibling on page {rc.chunk.page_number} "
                f"(borderline-band trigger)"
            )

    # Stage 5b — table-pair expansion (answered path only).
    # For each text chunk in the retained set, include same-page table/chart
    # chunks that may have been ranked below the top-N cutoff by the reranker.
    if not abstain:
        with_table_pairs = expand_table_pairs(with_siblings, conn)
    else:
        with_table_pairs = list(with_siblings)

    # Stage 6 — cap after sibling/table-pair expansion. Expanded chunks are
    # lower-priority than core, but table_pair_expanded chunks are higher-
    # priority than sibling_expanded chunks because they were pulled in
    # because a same-page text chunk was already retained (more direct
    # relevance signal). When trimming, keep table-pair before sibling.
    if len(with_table_pairs) > max_chunks:
        # "core" = everything that was present before Stage 5/5b expansions
        core_ids = {rc.chunk.id for rc in expanded}
        expanded_part = [rc for rc in with_table_pairs if rc.chunk.id not in core_ids]
        non_expanded_part = [rc for rc in with_table_pairs if rc.chunk.id in core_ids]
        if len(non_expanded_part) >= max_chunks:
            with_table_pairs = non_expanded_part[:max_chunks]
            expansion_trimmed: list[RetrievedChunk] = []
        else:
            max_expanded = max_chunks - len(non_expanded_part)
            # Priority: table_pair_expanded > sibling_expanded > anything else.
            # Sort key keeps the original within-bucket order.
            _STAGE_PRIORITY = {
                "table_pair_expanded": 0,
                "sibling_expanded": 1,
            }
            expanded_sorted = sorted(
                enumerate(expanded_part),
                key=lambda kv: (_STAGE_PRIORITY.get(kv[1].retrieval_stage or "", 2), kv[0]),
            )
            expansion_trimmed = [rc for _, rc in expanded_sorted[:max(0, max_expanded)]]
            with_table_pairs = non_expanded_part + expansion_trimmed
        logger.info(
            "  [post-rerank] expansion cap: %d core + %d expanded = %d (cap=%d)",
            len(non_expanded_part), len(expansion_trimmed), len(with_table_pairs), max_chunks,
        )

    logger.info("  [post-rerank] after table-pair expansion: %d", len(with_table_pairs))
    return with_table_pairs
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
    # True when the query asks about future/projected/guided values. Computed
    # alongside the intent so abstain stubs and generation paths can use a
    # single source of truth instead of redoing regex detection.
    forward_looking: bool = False


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
    forward_looking: bool = False,
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

    for rc in reranked:
        if rc.retrieval_stage is None:
            rc.retrieval_stage = "reranked"

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
    # Deterministic document-coordinate sort so callers see stable ordering
    # regardless of rerank/conflict/sibling append order.
    final_contexts = sorted(
        final_contexts,
        key=lambda rc: (rc.chunk.company, rc.chunk.report_date, rc.chunk.page_number or 0),
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
        forward_looking=forward_looking,
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
    intent, forward_looking = classify_intent_full(query)
    companies = extract_companies(query)
    logger.info(
        "Query intent: %s | forward_looking=%s | companies=%s | %s",
        intent, forward_looking, companies, query,
    )

    # All-company synthesis: run a separate retrieval pass per corpus company so
    # every company gets equal representation rather than relying on a global
    # top-K that might be dominated by one company's peer-comparison tables.
    # The per-issuer floor (enforce_per_issuer_floor) fires only on this branch.
    # For other intents the floor is intentionally not applied:
    #   - latest / conflict / comparison on a single issuer: extract_companies
    #     returns one company; the companies_filter constrains BM25 and vector
    #     to that issuer's chunks so cross-issuer collapse is not a risk.
    #   - cross-issuer comparison (multiple companies named in the query):
    #     extract_companies returns the named set and companies_filter passes
    #     it to BM25 and vector, constraining the candidate pool. Within that
    #     constrained pool the rerank generally maintains per-issuer
    #     representation. Promote to per-issuer floor here if probes ever show
    #     cross-issuer collapse on this branch.
    if intent == "all_company_synthesis":
        with connect() as conn:
            synthesis_result = retrieve_all_company_synthesis(
                query,
                conn,
                rerank_top_n=rerank_top_n,
                rerank_threshold=rerank_threshold,
            )
            synthesis_result.forward_looking = forward_looking
            if synthesis_result.abstain:
                return synthesis_result
            final, sub_queries = adaptive_retrieve(query, synthesis_result, conn)
            final.forward_looking = forward_looking
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
            forward_looking=forward_looking,
        )
        if intent in ADAPTIVE_INTENTS and not initial.abstain:
            final, sub_queries = adaptive_retrieve(query, initial, conn)
            final.diagnostics["sub_queries"] = sub_queries
            final.diagnostics["retrieval_hops"] = len(sub_queries)
            return final
        return initial
