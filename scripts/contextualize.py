"""Contextual retrieval orchestration script.

For every chunk in the DB:
  1. Reconstruct the full document text by concatenating parent chunks in order.
  2. Call Claude Haiku 4.5 to generate a 1-line context for the chunk.
  3. Prepend the context to the chunk's text → `contextualized_text`.
  4. Re-embed via Qwen3-Embedding-0.6B → `contextualized_embedding`.
  5. UPDATE chunks SET contextualized_text=?, contextualized_embedding=? WHERE id=?

Original `chunk_text`, `chunk_text_tsv`, and `embedding` are untouched.

Idempotent — by default, only chunks with NULL `contextualized_text` are
processed. Use `--force` to re-process. `--doc <uuid>` limits to one document
(recommended for the first run on a new pipeline change).

The document portion of the prompt uses Claude's prompt caching so the
per-document cost amortizes across all that document's chunks.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgvector.psycopg import register_vector

from src.db import connect
from src.ingestion.contextualizer import generate_context
from src.ingestion.embedder import _get_model as _get_embedder, EMBED_DIMS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("contextualize")

# Proactive RPM throttle: sleep this long between Haiku calls. Anthropic Tier 1
# is ~50 RPM (≈ 1 call every 1.2 s); this default is slightly slower to leave
# headroom for prompt-cache writes (the first call per cache window is heavier).
# Override with HAIKU_THROTTLE_SECONDS if operating on a higher usage tier.
THROTTLE_SECONDS = float(os.environ.get("HAIKU_THROTTLE_SECONDS", "1.5"))


def _list_documents(doc_id: Optional[str]) -> list[dict]:
    """Return documents to process, with their chunk counts."""
    sql = """
        SELECT d.id, d.company, d.source_path,
               (SELECT count(*) FROM chunks c
                  WHERE c.document_id = d.id) AS total_chunks,
               (SELECT count(*) FROM chunks c
                  WHERE c.document_id = d.id
                    AND c.contextualized_text IS NOT NULL) AS done_chunks
        FROM documents d
    """
    params: tuple = ()
    if doc_id:
        sql += " WHERE d.id = %s"
        params = (doc_id,)
    sql += " ORDER BY d.company, d.doc_version"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        assert cur.description is not None  # SELECT always populates description
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]


def _fetch_document_text(conn, doc_id: UUID) -> str:
    """Reconstruct the full document text by concatenating parent chunks in order.

    Parents are emitted by the chunker in document order (section by section), so
    concatenating them approximates the original document. Children are subsets
    of parents, so they are excluded to avoid double-counting text.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE document_id = %s AND is_parent = TRUE
            ORDER BY page_number, created_at
            """,
            (str(doc_id),),
        )
        return "\n\n".join(row[0] for row in cur.fetchall() if row[0])


def _fetch_chunks_to_process(conn, doc_id: UUID, force: bool) -> list[tuple[UUID, str]]:
    """Return (chunk_id, chunk_text) for chunks that need contextualization."""
    with conn.cursor() as cur:
        if force:
            cur.execute(
                "SELECT id, chunk_text FROM chunks WHERE document_id = %s ORDER BY id",
                (str(doc_id),),
            )
        else:
            cur.execute(
                """
                SELECT id, chunk_text FROM chunks
                WHERE document_id = %s AND contextualized_text IS NULL
                ORDER BY id
                """,
                (str(doc_id),),
            )
        return [(row[0], row[1]) for row in cur.fetchall()]


def contextualize_document(doc_row: dict, force: bool) -> int:
    """Contextualize every chunk in one document. Returns count written."""
    doc_id = doc_row["id"]
    company = doc_row["company"]
    total = doc_row["total_chunks"]
    done = doc_row["done_chunks"]

    if done == total and not force:
        logger.info("SKIP (all %d chunks already contextualized): %s", total, company)
        return 0
    if force:
        logger.info("Force re-contextualize: clearing %d chunks for %s", total, company)

    logger.info("==== Contextualizing %s (%d chunks) ====", company, total)

    embedder = _get_embedder()

    with connect() as conn:
        register_vector(conn)
        document_text = _fetch_document_text(conn, doc_id)
        if not document_text:
            logger.error("  Document text is empty for %s — skipping", company)
            return 0
        logger.info("  Document text: %d chars (cached portion)", len(document_text))

        chunks = _fetch_chunks_to_process(conn, doc_id, force)
        logger.info("  Chunks to contextualize: %d", len(chunks))

        written = 0
        for i, (chunk_id, chunk_text) in enumerate(chunks, start=1):
            try:
                context = generate_context(document_text, chunk_text)
            except Exception as e:  # noqa: BLE001 — log and continue
                logger.warning("  Chunk %d/%d (%s): contextualizer failed (%s)",
                               i, len(chunks), chunk_id, type(e).__name__)
                # Sleep extra long after a hard failure so the rate window clears.
                time.sleep(THROTTLE_SECONDS * 4)
                continue

            contextualized = f"{context}\n\n{chunk_text}"
            embedding = embedder.encode(
                contextualized,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if embedding.shape[-1] != EMBED_DIMS:
                logger.error("  Expected %d dims, got %d — skipping chunk %s",
                             EMBED_DIMS, embedding.shape[-1], chunk_id)
                continue

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chunks
                       SET contextualized_text = %s,
                           contextualized_embedding = %s
                     WHERE id = %s
                    """,
                    (
                        contextualized,
                        np.array(embedding, dtype=np.float32),
                        str(chunk_id),
                    ),
                )
            written += 1
            # Commit each successful chunk so interruptions do not drop a batch.
            conn.commit()
            if i % 20 == 0 or i == len(chunks):
                logger.info("  Progress: %d / %d (persisted)", i, len(chunks))
            # Proactive throttle to stay under Anthropic Tier 1 RPM.
            time.sleep(THROTTLE_SECONDS)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", help="UUID of a single document to contextualize")
    parser.add_argument("--force", action="store_true",
                        help="Re-process chunks even when contextualized_text is set")
    args = parser.parse_args()

    docs = _list_documents(doc_id=args.doc)
    if not docs:
        logger.error("No documents found (have you run scripts/ingest.py?)")
        sys.exit(1)

    logger.info("Considering %d document(s)", len(docs))
    total = 0
    for doc_row in docs:
        try:
            total += contextualize_document(doc_row, force=args.force)
        except (FileNotFoundError, ValueError) as e:
            logger.error("FAILED %s: %s", doc_row["source_path"], e)
            continue

    logger.info("Done — contextualized %d chunks total", total)


if __name__ == "__main__":
    main()
