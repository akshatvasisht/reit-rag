"""Dense vector retrieval using pgvector cosine distance on halfvec(1024) embeddings.

Queries are encoded via Qwen3-Embedding-0.6B (with ``prompt_name="query"`` so
the asymmetric retrieval prefix is applied) and matched against the document
embeddings stored in the ``chunks.embedding halfvec(1024)`` column. The
pipeline only retrieves child chunks (``is_parent = FALSE``) and expands to
parent chunks via ``parent_chunk_id`` after fusion and reranking.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg

from src.db import connect
from src.ingestion.embedder import embed_query
from src.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# ef_search controls how many candidates the HNSW graph explores at query time.
# pgvector defaults to 40; raising it to 200 matches ef_construction and improves
# recall at the cost of a small latency increase — acceptable for batch RAG queries.
HNSW_EF_SEARCH = 200

# Columns to SELECT — excludes chunk_text_tsv and embedding to keep rows lean.
_SELECT_COLS = """
    id, document_id, parent_chunk_id,
    company, ticker, doc_type, report_date, period_covered, doc_version,
    section_title, page_number, content_type, source_authority,
    chunk_text, is_parent, token_count,
    doc_subtype, page_content_class, contextualized_text
""".strip()

# ::halfvec cast picks the halfvec_cosine_ops adapter for <=> so the HNSW
# index is used (psycopg would otherwise send a plain array).
# COALESCE falls back to base embedding for chunks not yet contextualized.
# No boilerplate filter: a false-positive filter here would hurt recall.
_VECTOR_SQL = f"""
SELECT {_SELECT_COLS}
FROM chunks
WHERE is_parent = FALSE
ORDER BY COALESCE(contextualized_embedding, embedding) <=> %s::halfvec
LIMIT %s;
"""

# Same as _VECTOR_SQL but with a company filter — used when the entity
# extractor finds the query names a known company.
_VECTOR_SQL_FILTERED = f"""
SELECT {_SELECT_COLS}
FROM chunks
WHERE is_parent = FALSE
  AND company = ANY(%s)
ORDER BY COALESCE(contextualized_embedding, embedding) <=> %s::halfvec
LIMIT %s;
"""

_VECTOR_SQL_CT = f"""
SELECT {_SELECT_COLS}
FROM chunks
WHERE is_parent = FALSE
  AND content_type = ANY(%s)
ORDER BY COALESCE(contextualized_embedding, embedding) <=> %s::halfvec
LIMIT %s;
"""

_VECTOR_SQL_FILTERED_CT = f"""
SELECT {_SELECT_COLS}
FROM chunks
WHERE is_parent = FALSE
  AND company = ANY(%s)
  AND content_type = ANY(%s)
ORDER BY COALESCE(contextualized_embedding, embedding) <=> %s::halfvec
LIMIT %s;
"""


def fetch_embeddings_by_ids(
    ids: list[str],
    conn: psycopg.Connection,
) -> dict[str, list[float]]:
    """Return ``{chunk_id: embedding}`` for the requested chunk ids.

    Retrieval rows strip the embedding columns for leanness (see
    ``_SELECT_COLS``), so any post-rerank stage that needs vector similarity
    must hydrate the embeddings in a separate bounded query. Uses
    ``COALESCE(contextualized_embedding, embedding)`` so the vector matches the
    one indexed for search, falling back to the base embedding when the
    contextualized column is unpopulated.

    Args:
        ids: Chunk ids (as strings) to hydrate. Empty list returns ``{}``.
        conn: Open psycopg connection with pgvector types registered.

    Returns:
        Mapping from chunk id (string) to its 1024-dim embedding as a list of
        floats. Ids without a stored embedding are omitted from the mapping.
    """
    if not ids:
        return {}
    sql = """
        SELECT id, COALESCE(contextualized_embedding, embedding) AS emb
        FROM chunks
        WHERE id = ANY(%s)
    """
    out: dict[str, list[float]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, [ids])
        for row in cur.fetchall():
            chunk_id, emb = row[0], row[1]
            if emb is None:
                continue
            # pgvector returns halfvec as a HalfVector object, not a plain
            # iterable; .to_list() yields the float components.
            if hasattr(emb, "to_list"):
                emb = emb.to_list()
            out[str(chunk_id)] = [float(x) for x in emb]
    return out


def vector_search(
    query: str,
    limit: int = 50,
    conn: Optional[psycopg.Connection] = None,
    companies: Optional[list[str]] = None,
    content_types: Optional[list[str]] = None,
) -> list[RetrievedChunk]:
    """Retrieve child chunks ranked by cosine similarity to the query embedding.

    Args:
        query: Natural-language query string. Embedded via Qwen3-Embedding-0.6B
            (1024 dims, `prompt_name="query"`) before the database query runs.
        limit: Maximum number of chunks to return. Defaults to 50.
        conn: Optional existing psycopg connection. When None, a new
            connection is opened and closed within this call.
        companies: Optional list of canonical company names. When non-empty,
            results are restricted to chunks whose ``company`` is in the list.
            When None or empty, no company filter is applied.
        content_types: Optional list of content_type labels. When non-empty,
            restricts results to chunks whose ``content_type`` is in the list.

    Returns:
        List of RetrievedChunk with ``vector_rank`` set to 1-indexed position,
        ordered from most to least similar (smallest cosine distance first).
    """
    logger.debug(
        "Embedding query for vector search: %r (companies=%s, content_types=%s)",
        query,
        companies,
        content_types,
    )
    embedding: list[float] = embed_query(query)
    logger.debug("Vector search: %d-dim embedding, limit=%d", len(embedding), limit)

    def _run(connection: psycopg.Connection) -> list[RetrievedChunk]:
        with connection.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
            if companies and content_types:
                cur.execute(_VECTOR_SQL_FILTERED_CT, (companies, content_types, embedding, limit))
            elif companies:
                cur.execute(_VECTOR_SQL_FILTERED, (companies, embedding, limit))
            elif content_types:
                cur.execute(_VECTOR_SQL_CT, (content_types, embedding, limit))
            else:
                cur.execute(_VECTOR_SQL, (embedding, limit))
            if cur.description is None:
                return []
            columns = [d.name for d in cur.description]
            rows = cur.fetchall()

        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            chunk = Chunk.from_row(row, columns)
            results.append(RetrievedChunk(chunk=chunk, vector_rank=rank))

        logger.debug("Vector search returned %d chunks", len(results))
        return results

    if conn is not None:
        return _run(conn)

    with connect() as new_conn:
        return _run(new_conn)
