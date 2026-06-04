"""Full retrieval pipeline — orchestrates BM25 + vector + RRF + rerank + version deduplication + parent expansion.

This is the single entry point used by the generation layer and the
Streamlit UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple
from uuid import UUID

from src.db import connect
from src.corpus_registry import CORPUS_REGISTRY
from src.models import Chunk, RetrievedChunk
from src.retrieval.adaptive import ADAPTIVE_INTENTS, adaptive_retrieve
from src.retrieval.bm25 import _BOILERPLATE_FILTER, bm25_search
from src.retrieval.confidence import (
    compute_retrieval_confidence,
    confidence_band as confidence_band,
)
from src.retrieval.entity_filter import extract_companies
from src.retrieval.fusion import rrf_fuse
from src.retrieval.reranker import RERANK_THRESHOLD, rerank
from src.retrieval.synthesis import retrieve_all_company_synthesis
from src.retrieval.vector import fetch_embeddings_by_ids, vector_search
from src.versioning.chains import dedupe_by_version_group, expand_to_parents, version_group_key
from src.versioning.classifier import TemporalIntent, classify_intent_full

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


class ActivationStatus(NamedTuple):
    """Result of the contextual-retrieval activation check.

    ``state`` is one of:
    - ``"ok"``: at least MIN_PCT of text chunks have a contextualized embedding.
    - ``"empty_db"``: zero text chunks present — ingestion has not run.
    - ``"low_activation"``: chunks exist but contextualize.py has not populated them.
    - ``"check_failed"``: an exception was caught (missing DB, missing table, etc.).
    """

    state: str
    populated: int = 0
    total: int = 0
    pct: float = 0.0


def check_contextual_activation() -> ActivationStatus:
    """Report whether contextual retrieval is activated; warn when it isn't.

    Called explicitly by application entrypoints (Streamlit app, ingestion
    scripts) rather than at module import time so importing this module
    does not require DB availability. Resilient by design: any failure
    (missing DB, missing table, missing env var) is caught and reported as
    ``state="check_failed"`` so application startup never raises from this
    check, but the caller can surface a user-facing banner.
    """
    try:
        from src.db import connect as _connect  # noqa: PLC0415

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(_ACTIVATION_CHECK_SQL)
            row = cur.fetchone()

        if row is None:
            return ActivationStatus(state="check_failed")

        populated = int(row[0] or 0)
        total = int(row[1] or 0)
        if total == 0:
            logger.warning(
                "Activation check: zero text chunks in DB — run scripts/ingest.py "
                "(and the back-fill scripts in the README) before launching the app."
            )
            return ActivationStatus(state="empty_db", populated=0, total=0, pct=0.0)

        pct = float(row[2])
        if pct < _CONTEXTUAL_RETRIEVAL_MIN_PCT:
            logger.warning(
                "Contextual retrieval is only %.1f%% populated; retrieval will fall "
                "back to base embedding columns via COALESCE. "
                "Run scripts/contextualize.py to activate.",
                pct,
            )
            return ActivationStatus(
                state="low_activation", populated=populated, total=total, pct=pct
            )
        return ActivationStatus(state="ok", populated=populated, total=total, pct=pct)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Contextual-activation startup check failed (%s: %s); "
            "proceeding without the activation signal.",
            type(exc).__name__, exc,
        )
        return ActivationStatus(state="check_failed")


# ---------------------------------------------------------------------------
# Defaults — overridable per-call.
# ---------------------------------------------------------------------------

CANDIDATES_PER_RETRIEVER = 50   # top-K from BM25 and from vector
FUSED_TOP_N = 40                # candidates passed to reranker
RERANK_TOP_N = 5                # default contexts returned to the LLM
RRF_K = 60                      # standard RRF constant
ENTITY_ANCHOR_MAX = 3           # cap on chunks promoted by entity-anchor boost
MAX_CHUNKS_PER_PAGE = 3         # per (document_id, page_number) cap after expansion
VERSION_FLOOR_MAX = 2           # max chunks injected per missing version group
SUBTYPE_FLOOR_MAX = 1           # max chunks injected per missing (company, doc_subtype)
PER_VERSION_K_COMPARISON = 4    # top-K per version group for comparison/conflict intents

# Proper-noun-shaped anchor extraction from the query. Each token must be
# either a true Title-Cased word (uppercase initial + at least one lowercase
# letter — "Madison", "Avenue") or a digit-bearing token ("343", "12th").
# All-caps tokens (BXP, FFO, NOI, AI) are deliberately excluded because they
# overlap with reporting acronyms that appear in nearly every chunk and would
# over-fire the boost on generic metric queries like "2025 FFO guidance".
_QUERY_ANCHOR_RE = re.compile(
    r"\b(?:[A-Z][a-z][a-zA-Z]*|\d+\w*)(?:\s+(?:[A-Z][a-z][a-zA-Z]*|\d+\w*))+\b"
)

# Per-intent rerank-budget overrides. Intents that span multiple deck versions
# or multiple issuers need a wider top-K because the answer is built from
# evidence chunks that compete with each other for rerank positions —
# squeezing them through a 5-chunk gate drops legitimate same-page-different-
# version siblings (comparison) or the second-issuer's primary chunk
# (synthesis). Other intents (latest, conflict, the default) use the tight 5.
_RERANK_TOP_N_BY_INTENT: dict[str, int] = {
    "comparison": 8,
    "all_company_synthesis": 8,
    "historical": 7,
}


def _rerank_budget_for_intent(intent: str, forward_looking: bool = False, default: int = RERANK_TOP_N) -> int:
    """Return the rerank-top-N budget appropriate for the given intent.

    Pure function so the dispatch is unit-testable in isolation from the
    retrieval pipeline.  When forward_looking=True and the intent is "latest",
    the budget is widened to 7 so that late-page projection tables are not
    dropped by the tight 5-chunk gate.
    """
    if forward_looking and intent == "latest":
        return 7
    return _RERANK_TOP_N_BY_INTENT.get(intent, default)

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

# Maximal Marginal Relevance — diversity-aware re-selection over the reranked
# candidate pool. MMR trades relevance against novelty so a complementary-facet
# chunk that is present in the pool but outranked by top-relevance near-
# duplicates can still reach the retained set.
#
# Lambda is deliberately relevance-dominant (0.7): most queries are single-fact
# lookups whose best answer IS the cluster of near-identical high-rerank chunks,
# so diversity must only promote a complementary chunk when its relevance is
# comparable to the incumbents. At 0.7 the diversity penalty cannot displace a
# candidate whose normalized relevance lead exceeds 0.3.
MMR_LAMBDA: float = 0.7

# MMR is gated on redundancy: it only re-orders selection when the pool
# actually contains near-duplicate high-rerank chunks. A pool is "redundant"
# when at least two of its top candidates have pairwise cosine similarity above
# this threshold. Below it the pool is already diverse, so MMR would be a
# no-op at best and a relevance-displacing risk at worst — the pipeline falls
# through to plain top-N relevance ranking instead.
MMR_REDUNDANCY_SIM = 0.85

# Number of top-relevance candidates inspected for the redundancy gate. Keeping
# this small focuses the gate on the chunks that would otherwise occupy the
# retained set, not the long tail of the ~FUSED_TOP_N pool.
MMR_REDUNDANCY_TOP_K = 5


# ---------------------------------------------------------------------------
# Conflict-chunk columns — must match the SELECT shape used by bm25/vector.
# ---------------------------------------------------------------------------

_CHUNK_SELECT = """
    id, document_id, parent_chunk_id,
    company, ticker, doc_type, report_date, period_covered, doc_version,
    section_title, page_number, content_type, source_authority,
    chunk_text, is_parent, token_count,
    doc_subtype, page_content_class, contextualized_text
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


def expand_sibling_pages(
    contexts: list[RetrievedChunk],
    conn,
    max_per_page: int = 4,
) -> list[RetrievedChunk]:
    """Append same-page sibling chunks for every page represented by a reranked
    chunk in the retained set.

    Addresses two failure modes that the per-chunk retrieval can't reach on its
    own. (1) Sparse microchunks: a chunk whose ``chunk_text`` is just a tiny
    fragment (e.g. ``"$60 Bn"`` alone) cannot enter top-K via BM25/vector
    because it has no surrounding vocabulary to match a query. (2) Split
    entities: the data lives in one chunk (e.g. a table row) while the
    narrative framing lives in a sibling on the same page. Both cases are
    addressed by pulling the page's other chunks into the candidate set once
    any chunk on that page is already retained.

    Unlike ``expand_table_pairs`` (text-chunk → same-page table/chart only),
    this stage is content-type-agnostic on both ends.

    Only chunks with a non-None ``rerank_score`` trigger expansion — chunks
    that arrived via a prior expansion stage (parent, sibling, table-pair)
    have ``rerank_score=None`` and are skipped, so expansion never cascades.

    Each appended chunk has ``rerank_score=None`` and
    ``retrieval_stage="sibling_page_expanded"`` with the trigger chunk's id
    and a human-readable expansion reason for diagnostics.

    Args:
        contexts: Retained chunks after rerank (+ optional parent/sibling/
            table-pair stages).
        conn: Open psycopg connection.
        max_per_page: Maximum sibling chunks pulled per (document_id,
            page_number). Caps the expansion so the source-panel display is
            not flooded.

    Returns:
        Original list plus up to ``max_per_page`` additional same-page
        siblings per visited page, deduplicated across the whole list.
    """
    seen_ids = {rc.chunk.id for rc in contexts}
    visited_pages: set[tuple] = set()  # (document_id, page_number)
    to_add: list[RetrievedChunk] = []

    # Apply the same page_content_class filter BM25 uses so TOC/cover/legal
    # boilerplate pages do not enter via sibling-page expansion.
    sql = f"""
        SELECT {_CHUNK_SELECT}
        FROM chunks
        WHERE document_id = %s
          AND page_number = %s
          AND id != ALL(%s)
          AND is_parent = FALSE
          {_BOILERPLATE_FILTER}
        ORDER BY id
        LIMIT %s
    """

    for rc in contexts:
        # Skip chunks that arrived via prior expansion — only rerank-scored
        # chunks are valid expansion triggers, preventing cascade.
        if rc.rerank_score is None:
            continue
        if rc.chunk.page_number is None:
            continue
        key = (rc.chunk.document_id, rc.chunk.page_number)
        if key in visited_pages:
            continue
        visited_pages.add(key)

        doc_id = str(rc.chunk.document_id)
        page = rc.chunk.page_number
        excluded_ids = [str(i) for i in seen_ids]
        with conn.cursor() as cur:
            cur.execute(sql, [doc_id, page, excluded_ids, max_per_page])
            if cur.description is None:
                continue
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        added_for_page = 0
        for row in rows:
            if added_for_page >= max_per_page:
                break
            chunk = Chunk.from_row(row, cols)
            if chunk.id in seen_ids:
                continue
            new_rc = RetrievedChunk(chunk=chunk)
            new_rc.retrieval_stage = "sibling_page_expanded"
            new_rc.trigger_chunk_id = str(rc.chunk.id)
            new_rc.expansion_reason = (
                f"same-page sibling on page {page} for trigger chunk {rc.chunk.id}"
            )
            to_add.append(new_rc)
            seen_ids.add(chunk.id)
            added_for_page += 1

    if to_add:
        logger.info(
            "  [sibling-page] added %d sibling(s) across %d page(s)",
            len(to_add),
            len(visited_pages),
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


def _expand_table_and_page_siblings(
    contexts: list[RetrievedChunk],
    conn,
    *,
    abstain: bool,
) -> list[RetrievedChunk]:
    """Run table-pair expansion followed by sibling-page expansion.

    No-op on the abstain path: same-page expansions are only meaningful when
    the LLM will produce an answer.
    """
    if abstain:
        return list(contexts)
    with_table_pairs = expand_table_pairs(contexts, conn)
    return expand_sibling_pages(with_table_pairs, conn)


def _extract_query_anchors(query: str) -> list[str]:
    """Return proper-noun-shaped anchors from *query*.

    Anchors are multi-token Title-Case or digit-bearing sequences. The filter
    requires each anchor to either contain a digit, span ≥3 tokens, or include
    a ≥7-char token — keeps "343 Madison Avenue" / "Empire State Building" /
    "Digital Realty" while rejecting "What Is" / "How Did".
    """
    candidates = _QUERY_ANCHOR_RE.findall(query)
    anchors: list[str] = []
    for c in candidates:
        tokens = c.split()
        has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)
        has_long_token = any(len(t) >= 7 for t in tokens)
        if has_digit or len(tokens) >= 3 or has_long_token:
            anchors.append(c)
    return anchors


_ANCHOR_CONTENT_PRIORITY = {
    "table": 0,
    "mixed": 1,
    "chart_description": 2,
    "text": 3,
    "chart_caption": 4,
    "chart_context": 5,
}


def _entity_anchor_boost(
    query: str,
    fused: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    max_anchored: int = ENTITY_ANCHOR_MAX,
) -> list[RetrievedChunk]:
    """Promote rerank-pool chunks whose contextualized_text mentions a query
    anchor but were dropped by the top-N cutoff.

    Closes the gap where the cross-encoder prefers chunks that exclusively
    discuss an entity over chunks that mention the entity among others.
    Among matching candidates, prefers data-bearing content types
    (``table``, ``mixed``, ``chart_description``) over narrative ``text``:
    when a user names a specific project, the table that lists project
    economics is what they are usually after. RRF order is the within-type
    tiebreaker.
    """
    anchors = _extract_query_anchors(query)
    if not anchors:
        return []

    reranked_ids = {rc.chunk.id for rc in reranked}
    matches: list[tuple[int, int, str, RetrievedChunk]] = []
    for rrf_pos, rc in enumerate(fused):
        if rc.chunk.id in reranked_ids:
            continue
        haystack = (rc.chunk.contextualized_text or "") + " " + (rc.chunk.chunk_text or "")
        matched = next((a for a in anchors if a in haystack), None)
        if matched is None:
            continue
        priority = _ANCHOR_CONTENT_PRIORITY.get(rc.chunk.content_type, 99)
        matches.append((priority, rrf_pos, matched, rc))

    matches.sort(key=lambda m: (m[0], m[1]))
    boosted: list[RetrievedChunk] = []
    for _, _, matched, rc in matches[:max_anchored]:
        rc.retrieval_stage = "entity_anchored"
        rc.expansion_reason = f"query anchor: {matched}"
        boosted.append(rc)
    return boosted


# Stage priority for per-page trimming — most useful first. Stages not listed
# (or None) sort last so unknown provenance never displaces tagged chunks.
_PER_PAGE_STAGE_PRIORITY: dict[str, int] = {
    "reranked": 0,
    "entity_anchored": 1,
    "parent_expanded": 2,
    "table_pair_expanded": 3,
    "sibling_page_expanded": 4,
    "sibling_expanded": 5,
}


def _cap_chunks_per_page(
    contexts: list[RetrievedChunk],
    max_per_page: int = MAX_CHUNKS_PER_PAGE,
) -> list[RetrievedChunk]:
    """Cap chunks sharing a (document_id, page_number) at ``max_per_page``.

    Prevents same-page stacking when parent + sibling-page + table-pair
    expansions all fire on the same page. Pages with fewer than the cap are
    untouched. Within an over-cap page, chunks are kept in stage priority
    order; input order breaks ties so deterministic sort downstream is
    unaffected.
    """
    if not contexts:
        return contexts

    by_page: dict[tuple, list[tuple[int, RetrievedChunk]]] = {}
    no_page: list[tuple[int, RetrievedChunk]] = []
    for idx, rc in enumerate(contexts):
        if rc.chunk.page_number is None:
            no_page.append((idx, rc))
            continue
        key = (rc.chunk.document_id, rc.chunk.page_number)
        by_page.setdefault(key, []).append((idx, rc))

    kept: list[tuple[int, RetrievedChunk]] = list(no_page)
    dropped = 0
    for page_chunks in by_page.values():
        if len(page_chunks) <= max_per_page:
            kept.extend(page_chunks)
            continue
        ranked = sorted(
            page_chunks,
            key=lambda item: (
                _PER_PAGE_STAGE_PRIORITY.get(item[1].retrieval_stage or "", 99),
                item[0],
            ),
        )
        kept.extend(ranked[:max_per_page])
        dropped += len(page_chunks) - max_per_page

    if dropped:
        logger.info("  [per-page-cap] dropped %d chunk(s) above cap=%d", dropped, max_per_page)

    kept.sort(key=lambda item: item[0])
    return [rc for _, rc in kept]


def enforce_per_version_floor(
    retained: list[RetrievedChunk],
    fused_by_version_group: dict[tuple[str, str, str], list[RetrievedChunk]],
    intent: str,
    max_per_group: int = VERSION_FLOOR_MAX,
) -> list[RetrievedChunk]:
    """Ensure every version group present in the fused pool reaches retained.

    Only fires for comparison and conflict intents where balanced
    cross-version representation is the whole point of the query. For each
    version group missing from retained, the highest-rerank-score chunk from
    the fused pool is injected with ``retrieval_stage="version_floor"``.
    """
    if intent not in ("comparison", "conflict"):
        return retained

    represented = {version_group_key(rc.chunk) for rc in retained}
    retained_ids = {rc.chunk.id for rc in retained}
    to_add: list[RetrievedChunk] = []

    for group_key, candidates in fused_by_version_group.items():
        if group_key in represented:
            continue
        if not candidates:
            continue
        scored = sorted(
            candidates,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        added_for_group = 0
        for rc in scored:
            if added_for_group >= max_per_group:
                break
            if rc.chunk.id in retained_ids:
                continue
            rc.retrieval_stage = "version_floor"
            rc.expansion_reason = (
                f"per-version floor: {group_key[0]} / {group_key[1]} / {group_key[2]} "
                f"(score={rc.rerank_score})"
            )
            to_add.append(rc)
            retained_ids.add(rc.chunk.id)
            added_for_group += 1
            logger.info(
                "  [version-floor] added chunk for %s (score=%s)",
                group_key,
                rc.rerank_score,
            )

    if to_add:
        logger.info(
            "  [version-floor] added %d chunk(s) for %d missing group(s)",
            len(to_add), len({version_group_key(rc.chunk) for rc in to_add}),
        )

    return retained + to_add


def enforce_per_subtype_floor(
    retained: list[RetrievedChunk],
    fused_by_company_subtype: dict[tuple[str, str], list[RetrievedChunk]],
    intent: str,
    max_per_subtype: int = SUBTYPE_FLOOR_MAX,
) -> list[RetrievedChunk]:
    """Ensure every co-existing doc_subtype of a company reaches retained.

    Only fires for the ``latest`` intent. When a single company has 2+ distinct
    ``doc_subtype`` values in the fused candidate pool (e.g. an investor-day
    session and a quarterly deck for the same reporting period), a global rerank
    top-N can be consumed entirely by whichever subtype phrases the query best,
    leaving the other subtype with zero coverage even though it carries eligible
    evidence. For each (company, doc_subtype) present in the fused pool but
    absent from retained, the highest-rerank-score chunk from that subtype is
    injected with ``retrieval_stage="subtype_floor"``.

    A company with only one subtype in the fused pool is a no-op: it cannot be
    starved by a competing subtype, and cross-company coverage is the
    per-issuer floor's responsibility, not this one's.
    """
    if intent != "latest":
        return retained

    # Companies that genuinely expose 2+ subtypes in the fused pool; a
    # single-subtype company has nothing to balance against.
    subtypes_by_company: dict[str, set[str]] = {}
    for company, subtype in fused_by_company_subtype:
        subtypes_by_company.setdefault(company, set()).add(subtype)
    multi_subtype_companies = {
        company for company, subtypes in subtypes_by_company.items() if len(subtypes) >= 2
    }
    if not multi_subtype_companies:
        return retained

    represented = {(rc.chunk.company, rc.chunk.doc_subtype) for rc in retained}
    retained_ids = {rc.chunk.id for rc in retained}
    to_add: list[RetrievedChunk] = []

    for (company, subtype), candidates in fused_by_company_subtype.items():
        if company not in multi_subtype_companies:
            continue
        if (company, subtype) in represented:
            continue
        if not candidates:
            continue
        scored = sorted(
            candidates,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        added_for_subtype = 0
        for rc in scored:
            if added_for_subtype >= max_per_subtype:
                break
            if rc.chunk.id in retained_ids:
                continue
            rc.retrieval_stage = "subtype_floor"
            rc.expansion_reason = (
                f"per-subtype floor: {company} / {subtype} "
                f"(score={rc.rerank_score})"
            )
            to_add.append(rc)
            retained_ids.add(rc.chunk.id)
            added_for_subtype += 1
            logger.info(
                "  [subtype-floor] added chunk for %s / %s (score=%s)",
                company, subtype, rc.rerank_score,
            )

    if to_add:
        logger.info(
            "  [subtype-floor] added %d chunk(s) for %d missing subtype(s)",
            len(to_add), len(to_add),
        )

    return retained + to_add


def _balance_per_version(
    reranked_all: list[RetrievedChunk],
    intent: str,
    *,
    total_budget: int,
    per_version_k: int = PER_VERSION_K_COMPARISON,
) -> list[RetrievedChunk]:
    """Take top-K per version group instead of global top-N.

    Comparison and conflict intents need balanced version representation; a
    global top-N pass can skew heavily to whichever version's phrasing best
    matches the query (e.g. a deck whose section dates match the query's
    date range outscores a sibling deck across the board). For other intents
    this is a no-op — caller still gets ``reranked_all[:total_budget]``.

    Returns at most ``total_budget`` chunks, re-sorted by global rerank_score
    descending so the LLM still sees the strongest evidence first.
    """
    if intent not in ("comparison", "conflict") or not reranked_all:
        return reranked_all[:total_budget]

    by_version: dict[tuple[str, str, str], list[RetrievedChunk]] = {}
    for rc in reranked_all:
        by_version.setdefault(version_group_key(rc.chunk), []).append(rc)

    if len(by_version) <= 1:
        return reranked_all[:total_budget]

    balanced: list[RetrievedChunk] = []
    for group_chunks in by_version.values():
        balanced.extend(group_chunks[:per_version_k])

    balanced.sort(
        key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    return balanced[:total_budget]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length embedding vectors.

    Returns 0.0 when either vector has zero magnitude so a degenerate
    embedding never produces a spurious high-similarity penalty. Implemented
    without numpy to keep this hot loop dependency-free; the pool is ~40
    chunks so the pure-Python cost is negligible.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def _normalize_relevance(pool: list[RetrievedChunk]) -> dict[UUID, float]:
    """Min-max-normalize rerank scores across *pool* into ``[0, 1]``.

    The cosine-similarity term in MMR lives in ``[0, 1]``; rerank scores are
    raw cross-encoder logits on an open scale, so they must be rescaled to be
    comparable. When all scores are equal (or only one candidate exists) every
    chunk maps to 1.0 — relevance carries no signal, leaving selection to the
    diversity term and input order. Chunks with no rerank_score map to 0.0.
    """
    scored = [
        rc.rerank_score for rc in pool if rc.rerank_score is not None
    ]
    if not scored:
        return {rc.chunk.id: 0.0 for rc in pool}
    lo, hi = min(scored), max(scored)
    span = hi - lo
    out: dict[UUID, float] = {}
    for rc in pool:
        if rc.rerank_score is None:
            out[rc.chunk.id] = 0.0
        elif span <= 0.0:
            out[rc.chunk.id] = 1.0
        else:
            out[rc.chunk.id] = (rc.rerank_score - lo) / span
    return out


def _pool_is_redundant(
    pool: list[RetrievedChunk],
    embeddings: dict[str, list[float]],
    *,
    top_k: int = MMR_REDUNDANCY_TOP_K,
    sim_threshold: float = MMR_REDUNDANCY_SIM,
) -> bool:
    """Return True when the top-``top_k`` candidates contain near-duplicates.

    The MMR diversity term is only worth applying when the pool actually has
    redundancy to break up: two or more high-rerank candidates whose pairwise
    cosine similarity exceeds ``sim_threshold``. A pool that is already diverse
    falls through to plain relevance ranking, so single-fact queries whose top
    chunks are the genuinely-best (and non-redundant) answers are untouched.
    """
    top = sorted(
        pool,
        key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
        reverse=True,
    )[:top_k]
    vecs = [
        (rc.chunk.id, embeddings[str(rc.chunk.id)])
        for rc in top
        if str(rc.chunk.id) in embeddings
    ]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if _cosine_sim(vecs[i][1], vecs[j][1]) >= sim_threshold:
                return True
    return False


def mmr_select(
    pool: list[RetrievedChunk],
    embeddings: dict[str, list[float]],
    *,
    budget: int,
    lambda_: float = MMR_LAMBDA,
) -> list[RetrievedChunk]:
    """Select up to *budget* chunks from *pool* by Maximal Marginal Relevance.

    Standard MMR: starting from the single highest-relevance candidate, each
    subsequent pick maximizes
    ``lambda_ * rel(c) - (1 - lambda_) * max_sim(c, selected)`` where ``rel`` is
    the min-max-normalized rerank score and ``sim`` is cosine similarity between
    candidate embeddings.

    The top-1-by-relevance chunk is always selected first, so the primary-answer
    chunk for single-fact queries is never displaced; diversity only governs the
    remaining slots. Candidates missing an embedding incur a 0.0 similarity
    penalty (treated as maximally novel) so a missing vector never silently
    suppresses a candidate.

    Returns *pool* truncated to *budget* by relevance when the pool is small
    enough that no selection is needed.
    """
    if not pool:
        return []
    if budget <= 0:
        return []
    if len(pool) <= budget:
        return list(pool)

    rel = _normalize_relevance(pool)

    by_relevance = sorted(
        pool,
        key=lambda rc: rel[rc.chunk.id],
        reverse=True,
    )
    # Seed with the top-1 by relevance so the primary-answer chunk is locked in.
    selected: list[RetrievedChunk] = [by_relevance[0]]
    selected_ids = {by_relevance[0].chunk.id}
    remaining = by_relevance[1:]

    while remaining and len(selected) < budget:
        best_rc: RetrievedChunk | None = None
        best_score = float("-inf")
        for rc in remaining:
            cand_emb = embeddings.get(str(rc.chunk.id))
            if cand_emb is None:
                max_sim = 0.0
            else:
                max_sim = 0.0
                for sel in selected:
                    sel_emb = embeddings.get(str(sel.chunk.id))
                    if sel_emb is None:
                        continue
                    sim = _cosine_sim(cand_emb, sel_emb)
                    if sim > max_sim:
                        max_sim = sim
            mmr_score = lambda_ * rel[rc.chunk.id] - (1.0 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_rc = rc
        if best_rc is None:
            break
        selected.append(best_rc)
        selected_ids.add(best_rc.chunk.id)
        remaining = [rc for rc in remaining if rc.chunk.id not in selected_ids]

    return selected


def _apply_mmr_selection(
    reranked_pool: list[RetrievedChunk],
    conn,
    *,
    budget: int,
) -> list[RetrievedChunk]:
    """Re-select *budget* chunks from the scored rerank pool via gated MMR.

    Hydrates candidate embeddings in one bounded query (the pool is ~FUSED_TOP_N
    chunks), checks the redundancy gate, and applies :func:`mmr_select` only
    when the pool contains near-duplicate high-rerank candidates. Otherwise
    returns the plain relevance-ordered top-``budget`` slice, leaving non-
    redundant single-fact pools exactly as the reranker ordered them.

    The returned list is re-sorted by rerank_score descending so downstream
    stages (which assume relevance order for cap/floor decisions) see the
    strongest evidence first; MMR governs *which* chunks survive, not their
    presentation order.
    """
    if len(reranked_pool) <= budget:
        return reranked_pool

    ids = [str(rc.chunk.id) for rc in reranked_pool]
    embeddings = fetch_embeddings_by_ids(ids, conn)

    if not _pool_is_redundant(reranked_pool, embeddings):
        logger.info("  [mmr] pool not redundant; plain top-%d relevance ranking", budget)
        return sorted(
            reranked_pool,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:budget]

    selected = mmr_select(reranked_pool, embeddings, budget=budget)
    logger.info(
        "  [mmr] redundant pool; MMR-selected %d/%d (lambda=%.2f)",
        len(selected), len(reranked_pool), MMR_LAMBDA,
    )
    return sorted(
        selected,
        key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
        reverse=True,
    )


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

    with_sibling_pages = _expand_table_and_page_siblings(with_siblings, conn, abstain=abstain)

    # Cap chunks-per-page before the global cap so parent + sibling-page +
    # table-pair stages cannot stack many chunks for a single page.
    with_sibling_pages = _cap_chunks_per_page(with_sibling_pages)

    # Stage 6 — cap after all expansion stages. Expanded chunks are
    # lower-priority than core; within expansion, table_pair_expanded has the
    # strongest signal (same-page data-rich chunk pulled because its text
    # sibling was already retained), then sibling_page_expanded (same-page
    # any content type), then sibling_expanded (adjacent-page borderline).
    if len(with_sibling_pages) > max_chunks:
        # "core" = everything that was present before Stage 5/5b/5c expansions
        core_ids = {rc.chunk.id for rc in expanded}
        expanded_part = [rc for rc in with_sibling_pages if rc.chunk.id not in core_ids]
        non_expanded_part = [rc for rc in with_sibling_pages if rc.chunk.id in core_ids]
        if len(non_expanded_part) >= max_chunks:
            with_sibling_pages = non_expanded_part[:max_chunks]
            expansion_trimmed: list[RetrievedChunk] = []
        else:
            max_expanded = max_chunks - len(non_expanded_part)
            # Priority order: see comment above.
            _STAGE_PRIORITY = {
                "table_pair_expanded": 0,
                "sibling_page_expanded": 1,
                "sibling_expanded": 2,
            }
            expanded_sorted = sorted(
                enumerate(expanded_part),
                key=lambda kv: (_STAGE_PRIORITY.get(kv[1].retrieval_stage or "", 3), kv[0]),
            )
            expansion_trimmed = [rc for _, rc in expanded_sorted[:max(0, max_expanded)]]
            with_sibling_pages = non_expanded_part + expansion_trimmed
        logger.info(
            "  [post-rerank] expansion cap: %d core + %d expanded = %d (cap=%d)",
            len(non_expanded_part), len(expansion_trimmed), len(with_sibling_pages), max_chunks,
        )

    logger.info("  [post-rerank] after sibling-page expansion: %d", len(with_sibling_pages))
    return with_sibling_pages
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

    # For comparison/conflict, score the full fused pool so per-version
    # top-K balancing has all candidates to choose from; other intents score
    # the full pool too so MMR diversity re-selection can promote a
    # complementary-facet candidate that the global top-N cutoff would drop.
    # The cross-encoder scores every candidate in either case.
    if intent in ("comparison", "conflict"):
        reranked_all = rerank(query, fused, top_n=len(fused))
        reranked = _balance_per_version(reranked_all, intent, total_budget=rerank_top_n)
    else:
        reranked_all = rerank(query, fused, top_n=len(fused))
        reranked = _apply_mmr_selection(reranked_all, conn, budget=rerank_top_n)
    logger.info("  Reranked top score: %.3f",
                reranked[0].rerank_score if reranked else float("-inf"))

    for rc in reranked:
        if rc.retrieval_stage is None:
            rc.retrieval_stage = "reranked"

    # Entity-anchor boost: promote rerank-pool chunks whose contextualized
    # text mentions a query proper-noun anchor (e.g., "343 Madison Avenue")
    # but were dropped by the top-N cutoff.
    anchored = _entity_anchor_boost(query, fused, reranked)
    if anchored:
        logger.info(
            "  Entity-anchored: %d additional chunk(s) (pages=%s)",
            len(anchored),
            [rc.chunk.page_number for rc in anchored],
        )
        reranked = reranked + anchored

    # Per-version floor: comparison/conflict queries depend on balanced
    # multi-version evidence; inject best chunk from any group absent from
    # the rerank top-N.
    fused_by_version_group: dict[tuple[str, str, str], list[RetrievedChunk]] = {}
    for rc in fused:
        fused_by_version_group.setdefault(version_group_key(rc.chunk), []).append(rc)
    reranked = enforce_per_version_floor(reranked, fused_by_version_group, intent)

    # Per-subtype floor: latest-intent queries can collapse onto whichever of a
    # company's co-existing subtypes (e.g. quarterly deck vs. investor-day
    # session) phrases the query best; inject the best chunk from any subtype
    # starved out of the rerank top-N. Disjoint from the version floor, which
    # fires only on comparison/conflict.
    fused_by_company_subtype: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for rc in fused:
        fused_by_company_subtype.setdefault(
            (rc.chunk.company, rc.chunk.doc_subtype), []
        ).append(rc)
    reranked = enforce_per_subtype_floor(reranked, fused_by_company_subtype, intent)

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
    # Apply per-intent rerank-budget override when the caller did not pass a
    # custom rerank_top_n. Comparison / synthesis / historical queries assemble
    # answers from multiple evidence chunks that compete for rerank positions;
    # the default 5-chunk gate drops legitimate siblings on those intents.
    if rerank_top_n == RERANK_TOP_N:
        rerank_top_n = _rerank_budget_for_intent(intent, forward_looking=forward_looking)
    logger.info(
        "Query intent: %s | forward_looking=%s | companies=%s | rerank_top_n=%d | %s",
        intent, forward_looking, companies, rerank_top_n, query,
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
    #     representation. Promote to a per-issuer floor here if cross-issuer
    #     collapse is observed on this branch.
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
