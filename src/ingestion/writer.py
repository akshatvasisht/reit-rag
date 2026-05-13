"""Batch-write documents and chunks into the postgres database.

The chunker emits chunks in [parent, child, child, …] order, so parents are
inserted before any child references them — no second pass needed.
"""

from __future__ import annotations

import logging
from typing import Iterable
from uuid import UUID

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.types.json import Json

from src.models import Chunk, DocumentMeta

logger = logging.getLogger(__name__)


def write_document(conn: Connection, doc_meta: DocumentMeta) -> UUID:
    """Insert (or upsert) a document row. Returns the document UUID.

    Idempotent on `source_path` — re-running ingestion on the same file returns
    the existing UUID without duplicating the row.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (id, company, ticker, doc_type, report_date, period_covered,
                 doc_version, source_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE
              SET company = EXCLUDED.company,
                  ticker = EXCLUDED.ticker,
                  doc_type = EXCLUDED.doc_type,
                  report_date = EXCLUDED.report_date,
                  period_covered = EXCLUDED.period_covered,
                  doc_version = EXCLUDED.doc_version
            RETURNING id;
            """,
            (
                str(doc_meta.id),
                doc_meta.company,
                doc_meta.ticker,
                doc_meta.doc_type,
                doc_meta.report_date,
                doc_meta.period_covered,
                doc_meta.doc_version,
                doc_meta.source_path,
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("documents insert returned no row")
        return row[0]


def write_chunks(conn: Connection, chunks: list[Chunk], batch_size: int = 100) -> int:
    """Insert chunks in batches; returns the number inserted.

    Chunks must be ordered so every parent precedes its children (the chunker
    guarantees this).
    """
    if not chunks:
        return 0

    register_vector(conn)
    inserted = 0

    with conn.cursor() as cur:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            params = []
            for c in batch:
                embedding = (
                    np.array(c.embedding, dtype=np.float32)
                    if c.embedding is not None
                    else None
                )
                params.append(
                    (
                        str(c.id),
                        str(c.document_id),
                        str(c.parent_chunk_id) if c.parent_chunk_id else None,
                        c.company,
                        c.ticker,
                        c.doc_type,
                        c.report_date,
                        c.period_covered,
                        c.doc_version,
                        c.section_title,
                        c.page_number,
                        c.content_type,
                        c.source_authority,
                        c.chunk_text,
                        embedding,
                        c.is_parent,
                        c.token_count,
                    )
                )
            cur.executemany(
                """
                INSERT INTO chunks
                    (id, document_id, parent_chunk_id, company, ticker,
                     doc_type, report_date, period_covered, doc_version,
                     section_title, page_number, content_type,
                     source_authority, chunk_text, embedding, is_parent,
                     token_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params,
            )
            inserted += len(batch)
            logger.info("  Inserted %d / %d chunks", inserted, len(chunks))

    return inserted


def document_exists(conn: Connection, source_path: str) -> bool:
    """Return True if a document with this `source_path` has already been ingested."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE source_path = %s", (source_path,))
        return cur.fetchone() is not None


def delete_chunks_for_document(conn: Connection, document_id: UUID) -> int:
    """Delete all chunks for a document; returns deleted row count.

    Used by force re-ingest to replace prior chunk rows instead of appending
    duplicates to the same document.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (str(document_id),))
        return cur.rowcount
