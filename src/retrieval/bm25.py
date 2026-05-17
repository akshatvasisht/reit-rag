"""BM25 keyword retrieval using PostgreSQL tsvector and ts_rank.

Queries child chunks only (``is_parent = FALSE``). The pipeline expands
to parent chunks via ``parent_chunk_id`` after fusion and reranking.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg

from src.db import connect
from src.models import RetrievedChunk
from src.retrieval._common import _row_to_chunk

logger = logging.getLogger(__name__)

# Columns to SELECT — excludes chunk_text_tsv and embedding to keep rows lean.
_SELECT_COLS = """
    id, document_id, parent_chunk_id,
    company, ticker, doc_type, report_date, period_covered, doc_version,
    section_title, page_number, content_type, source_authority,
    chunk_text, is_parent, token_count,
    doc_subtype, page_content_class
""".strip()

# Exclude known-boilerplate classes from BM25 so legal disclaimers and index
# entries do not pollute keyword recall.  ``unknown`` and ``substantive`` chunks
# are always included; only confidently-classified non-content pages are dropped.
# This filter is intentionally absent from vector_search (dense retrieval handles
# such pages better via embedding distance, and false-positive filtering at that
# stage would hurt recall more than it helps precision).
_BOILERPLATE_FILTER = (
    "AND (page_content_class IS NULL"
    " OR page_content_class NOT IN ('boilerplate_legal', 'index_reference'))"
)

# Use plainto_tsquery for English-aware tokenization/stemming/stop-word removal,
# then rewrite '&' (AND) → '|' (OR) so multi-term natural-language queries
# do not require every token to appear in a chunk. ts_rank_cd rewards chunks
# matching MORE terms, so loose recall + reranker precision is preserved (the
# standard 2-stage retrieval pattern; cross-encoder downstream cleans up).
# COALESCE falls back to chunk_text_tsv when contextualized_text_tsv is NULL,
# ensuring chunks that have not yet been contextualized still rank correctly.
_BM25_SQL = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
        AS tsq
)
SELECT {_SELECT_COLS}
FROM chunks, q
WHERE is_parent = FALSE
  AND COALESCE(contextualized_text_tsv, chunk_text_tsv) @@ tsq
  {_BOILERPLATE_FILTER}
ORDER BY ts_rank_cd(COALESCE(contextualized_text_tsv, chunk_text_tsv), tsq) DESC
LIMIT %s;
"""

# Same as _BM25_SQL but with a company filter — used when the entity extractor
# finds that the query names a known company or ticker.
_BM25_SQL_FILTERED = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
        AS tsq
)
SELECT {_SELECT_COLS}
FROM chunks, q
WHERE is_parent = FALSE
  AND COALESCE(contextualized_text_tsv, chunk_text_tsv) @@ tsq
  AND company = ANY(%s)
  {_BOILERPLATE_FILTER}
ORDER BY ts_rank_cd(COALESCE(contextualized_text_tsv, chunk_text_tsv), tsq) DESC
LIMIT %s;
"""

_BM25_SQL_CT = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
        AS tsq
)
SELECT {_SELECT_COLS}
FROM chunks, q
WHERE is_parent = FALSE
  AND COALESCE(contextualized_text_tsv, chunk_text_tsv) @@ tsq
  AND content_type = ANY(%s)
  {_BOILERPLATE_FILTER}
ORDER BY ts_rank_cd(COALESCE(contextualized_text_tsv, chunk_text_tsv), tsq) DESC
LIMIT %s;
"""

_BM25_SQL_FILTERED_CT = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
        AS tsq
)
SELECT {_SELECT_COLS}
FROM chunks, q
WHERE is_parent = FALSE
  AND COALESCE(contextualized_text_tsv, chunk_text_tsv) @@ tsq
  AND company = ANY(%s)
  AND content_type = ANY(%s)
  {_BOILERPLATE_FILTER}
ORDER BY ts_rank_cd(COALESCE(contextualized_text_tsv, chunk_text_tsv), tsq) DESC
LIMIT %s;
"""


def bm25_search(
    query: str,
    limit: int = 50,
    conn: Optional[psycopg.Connection] = None,
    companies: Optional[list[str]] = None,
    content_types: Optional[list[str]] = None,
) -> list[RetrievedChunk]:
    """Retrieve child chunks ranked by BM25 relevance using tsvector ts_rank.

    Args:
        query: Natural-language query string. Parsed with
            ``plainto_tsquery('english', ...)`` and then rewritten from
            AND-semantics to OR-semantics by replacing ``&`` with ``|``.
            This favors recall; downstream reranking handles precision.
        limit: Maximum number of chunks to return. Defaults to 50.
        conn: Optional existing psycopg connection. When None, a new
            connection is opened and closed within this call.
        companies: Optional list of canonical company names. When non-empty,
            results are restricted to chunks whose ``company`` is in the list
            from query entity extraction. When None or empty, no company
            filter is applied.
        content_types: Optional list of content_type labels. When non-empty,
            restricts results to chunks whose ``content_type`` is in the list.

    Returns:
        List of RetrievedChunk with ``bm25_rank`` set to 1-indexed position,
        ordered from most to least relevant.
    """
    logger.debug(
        "BM25 search: %r (limit=%d, companies=%s, content_types=%s)",
        query,
        limit,
        companies,
        content_types,
    )

    def _run(connection: psycopg.Connection) -> list[RetrievedChunk]:
        with connection.cursor() as cur:
            if companies and content_types:
                cur.execute(_BM25_SQL_FILTERED_CT, (query, companies, content_types, limit))
            elif companies:
                cur.execute(_BM25_SQL_FILTERED, (query, companies, limit))
            elif content_types:
                cur.execute(_BM25_SQL_CT, (query, content_types, limit))
            else:
                cur.execute(_BM25_SQL, (query, limit))
            if cur.description is None:
                return []
            columns = [d.name for d in cur.description]
            rows = cur.fetchall()

        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            chunk = _row_to_chunk(row, columns)
            results.append(RetrievedChunk(chunk=chunk, bm25_rank=rank))

        logger.debug("BM25 returned %d chunks", len(results))
        return results

    if conn is not None:
        return _run(conn)

    with connect() as new_conn:
        return _run(new_conn)
