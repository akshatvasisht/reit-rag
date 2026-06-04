"""Contextual retrieval orchestration script — Batch API edition.

For every chunk in the DB:
  1. Reconstruct the full document text by concatenating parent chunks in order.
  2. Submit a Message Batches API request (one request per chunk, up to
     --batch-size chunks per batch submission).
  3. Poll until processing_status == "ended".
  4. Write contextualized_text + re-embed via Qwen3 + update
     contextualized_text_tsv (maintained by a DB trigger).
  5. UPDATE chunks SET contextualized_text=?, contextualized_embedding=? WHERE id=?

Original chunk_text, chunk_text_tsv, and embedding are untouched.

Idempotent — by default, only chunks with NULL contextualized_text are processed.
Use --force to re-process.  --doc <uuid> limits to one document (recommended for
the first run after a pipeline change).

The document portion of the prompt uses Claude's prompt caching so the
per-document cost amortizes across all that document's chunks.
"""

from __future__ import annotations

import argparse
import json
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
from src.ingestion.embedder import _get_model as _get_embedder, EMBED_DIMS


# Haiku 4.5 supports prompt caching and is ~5x cheaper than Sonnet for the
# short-summary task this prompt expects.
CONTEXT_MODEL = "claude-haiku-4-5-20251001"

# Instruction body, verbatim from Anthropic's published Contextual Retrieval example.
_INSTRUCTION = (
    "Here is the chunk to situate within the full document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. Answer "
    "only with the succinct context and nothing else."
)

# Long table chunks get truncated by the 512-token cross-encoder reranker, so
# entity names mid-table are invisible at scoring time. Lead the contextualized
# prefix with explicit entities so they land in the first ~200 chars.
_INSTRUCTION_TABLE = (
    "Here is the chunk to situate within the full document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "This chunk is a table or structured data block. Write a single short "
    "context sentence (80–120 tokens) that LEADS with the specific entities "
    "that appear in the table — project names, company names, property names, "
    "dollar amounts, percentages, dates — then situates the chunk within the "
    "document. The leading entities matter because downstream retrieval scoring "
    "may not read past the first ~200 characters of the chunk. Answer only "
    "with the context sentence — no preamble, no labels, no quotation marks."
)


def _is_table_chunk(chunk_text: str, content_type: str) -> bool:
    """Decide whether to use the table-aware contextualization prompt.

    Fires on:
    - content_type IN ('table', 'mixed') — Docling's explicit table tags
    - long parent chunks (>3000 chars) with multiple pipe-separated rows,
      where truncation would lose mid-table entity names
    """
    if content_type in ("table", "mixed"):
        return True
    if len(chunk_text) > 3000:
        pipe_rows = sum(1 for line in chunk_text.split("\n") if line.count("|") >= 3)
        if pipe_rows >= 3:
            return True
    return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("contextualize")

# Seconds to wait between batch-status polls.  The Batch API typically finishes
# within a few minutes for small batches; 30 s is a reasonable poll interval.
POLL_INTERVAL_SECONDS = 30
# 6 h ceiling; Batch API SLA tops out at 24 h, but stalls beyond 6 h indicate
# a real failure rather than slow processing.
MAX_POLL_SECONDS = 60 * 60 * 6

# Minimum percentage of text chunks that must have a contextualized embedding
# before the script considers contextual retrieval fully activated.
CONTEXTUAL_RETRIEVAL_MIN_PCT = 95.0

# System prompt used for every contextualization request.
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "You are a document analyst. Given a full document and a chunk extracted "
    "from it, write a single short sentence (50–100 tokens) that situates the "
    "chunk within the broader document context to improve search retrieval. "
    "Output only the context sentence — no preamble, no labels, no quotation marks."
)


def _get_client():
    """Return an Anthropic client, raising if the API key is missing."""
    from anthropic import Anthropic  # noqa: PLC0415
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=api_key)


def contextualize_prompt(chunk_text: str, doc_text: str) -> str:
    """Build the user turn for the contextualization request."""
    doc_section = f"<document>\n{doc_text}\n</document>"
    return f"{doc_section}\n\n{_INSTRUCTION.format(chunk=chunk_text)}"


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
        assert cur.description is not None
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]


def _fetch_document_text(conn, doc_id: UUID) -> str:
    """Reconstruct the full document text by concatenating parent chunks in order."""
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


def _fetch_chunks_to_process(
    conn, doc_id: UUID, force: bool
) -> list[tuple[UUID, str, str]]:
    """Return (chunk_id, chunk_text, content_type) for chunks needing contextualization."""
    with conn.cursor() as cur:
        if force:
            cur.execute(
                "SELECT id, chunk_text, content_type FROM chunks WHERE document_id = %s ORDER BY id",
                (str(doc_id),),
            )
        else:
            cur.execute(
                """
                SELECT id, chunk_text, content_type FROM chunks
                WHERE document_id = %s AND contextualized_text IS NULL
                ORDER BY id
                """,
                (str(doc_id),),
            )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def _submit_batch(
    client,
    chunks: list[tuple[UUID, str, str]],
    doc_text: str,
) -> str:
    """Submit one Message Batches API job. Returns the batch id."""
    requests = []
    for chunk_id, chunk_text, content_type in chunks:
        instruction_template = (
            _INSTRUCTION_TABLE
            if _is_table_chunk(chunk_text, content_type)
            else _INSTRUCTION
        )
        requests.append({
            "custom_id": str(chunk_id),
            "params": {
                "model": CONTEXT_MODEL,
                "max_tokens": 200,
                "temperature": 0.0,
                "top_k": 1,
                "system": CONTEXTUALIZE_SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<document>\n{doc_text}\n</document>",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": instruction_template.format(chunk=chunk_text),
                            },
                        ],
                    }
                ],
            },
        })
    batch = client.messages.batches.create(requests=requests)
    return batch.id


# Without persistence, a Ctrl-C between submit and result-write loses the
# batch handle: the next run re-submits (and re-bills) the same chunks while
# the original batch keeps processing.
_STATE_FILE = Path("data/.contextualize_state.json")


def _load_state() -> dict | None:
    """Return the persisted in-flight batch state, or None when none exists."""
    if not _STATE_FILE.exists():
        return None
    try:
        return json.loads(_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s); ignoring", _STATE_FILE, exc)
        return None


def _save_state(doc_id, batch_id: str) -> None:
    """Persist the in-flight batch handle before polling so an interrupt can resume."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps({"doc_id": str(doc_id), "batch_id": batch_id}, indent=2)
    )


def _clear_state() -> None:
    """Remove the state file once results have been written."""
    _STATE_FILE.unlink(missing_ok=True)


def _resume_in_flight_batch(client) -> None:
    """If a previous run was interrupted between submit and write, finish that
    batch first so the user is not re-billed by re-submitting the same chunks.
    Safe to call even when no state file exists — returns immediately.
    """
    state = _load_state()
    if state is None:
        return
    batch_id = state["batch_id"]
    logger.info(
        "Resuming in-flight batch %s from previous run (doc %s)",
        batch_id, state.get("doc_id", "?"),
    )
    embedder = _get_embedder()
    _poll_until_done(client, batch_id)
    with connect() as conn:
        register_vector(conn)
        n = _write_results(conn, client, batch_id, embedder)
    _clear_state()
    logger.info("  Resumed batch wrote %d chunks; state file cleared", n)


def _poll_until_done(client, batch_id: str) -> None:
    """Block until the batch reaches processing_status == 'ended'.

    Raises RuntimeError on a terminal failure status (canceling/canceled/errored)
    so the script does not loop indefinitely on a failed batch. Raises TimeoutError
    when total polling exceeds MAX_POLL_SECONDS so a permanently-stuck batch cannot
    deadlock the contextualization run.
    """
    deadline = time.monotonic() + MAX_POLL_SECONDS
    last_status: str | None = None
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        last_status = status
        logger.info("  Batch %s — status: %s", batch_id, status)
        if status == "ended":
            return
        if status in ("canceling", "canceled", "errored"):
            raise RuntimeError(f"Batch {batch_id} ended in status: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within {MAX_POLL_SECONDS}s "
                f"(last status: {last_status})"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def _write_results(
    conn,
    client,
    batch_id: str,
    embedder,
) -> int:
    """Stream batch results, embed each contextualized text, and UPDATE the DB."""
    register_vector(conn)
    written = 0

    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            logger.warning(
                "  Batch result %s — non-success type: %s",
                result.custom_id,
                result.result.type,
            )
            continue

        message = result.result.message
        context = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not context:
            logger.warning("  Empty context for chunk %s — skipping", result.custom_id)
            continue

        # Fetch the original chunk_text to prepend context.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_text FROM chunks WHERE id = %s",
                (result.custom_id,),
            )
            row = cur.fetchone()
        if row is None:
            logger.warning("  Chunk %s not found in DB — skipping", result.custom_id)
            continue

        contextualized = f"{context}\n\n{row[0]}"

        embedding = embedder.encode(
            contextualized,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if embedding.shape[-1] != EMBED_DIMS:
            logger.error(
                "  Expected %d dims, got %d — skipping chunk %s",
                EMBED_DIMS,
                embedding.shape[-1],
                result.custom_id,
            )
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
                    result.custom_id,
                ),
            )
        conn.commit()
        written += 1

    return written


def contextualize_document(
    doc_row: dict,
    force: bool,
    batch_size: int,
    client,
) -> int:
    """Contextualize every chunk in one document using the Batch API.

    Args:
        doc_row: Document metadata dict from _list_documents.
        force: Re-process even when contextualized_text is already set.
        batch_size: Maximum chunks per Batch API submission.
        client: Anthropic client instance.

    Returns:
        Number of chunks successfully written.
    """
    doc_id = doc_row["id"]
    company = doc_row["company"]
    total = doc_row["total_chunks"]
    done = doc_row["done_chunks"]

    if done == total and not force:
        logger.info(
            "SKIP (all %d chunks already contextualized): %s", total, company
        )
        return 0

    logger.info("==== Contextualizing %s (%d chunks) ====", company, total)

    embedder = _get_embedder()

    with connect() as conn:
        document_text = _fetch_document_text(conn, doc_id)
        if not document_text:
            logger.error(
                "  Document text is empty for %s — skipping", company
            )
            return 0

        chunks = _fetch_chunks_to_process(conn, doc_id, force)
        logger.info(
            "  Chunks to contextualize: %d (doc text: %d chars)",
            len(chunks),
            len(document_text),
        )

    written = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        logger.info(
            "  Submitting batch %d–%d of %d",
            batch_start + 1,
            batch_start + len(batch),
            len(chunks),
        )

        batch_id = _submit_batch(client, batch, document_text)
        _save_state(doc_id, batch_id)
        logger.info("  Batch submitted: %s (state persisted to %s)", batch_id, _STATE_FILE)

        _poll_until_done(client, batch_id)

        with connect() as conn:
            register_vector(conn)
            n = _write_results(conn, client, batch_id, embedder)
        _clear_state()

        written += n
        logger.info("  Batch complete — wrote %d chunks (total so far: %d)", n, written)

    return written


_ACTIVATION_SQL = """
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


def verify_contextual_retrieval_activated() -> None:
    """Check that enough text chunks have been contextualized; hard-error if not.

    Queries the DB for the fraction of text chunks with a non-NULL
    contextualized_embedding.  When the fraction is below
    CONTEXTUAL_RETRIEVAL_MIN_PCT, exits the process with a descriptive error
    message so the operator knows to resume or resubmit the Batch API job.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(_ACTIVATION_SQL)
        row = cur.fetchone()

    if row is None:
        logger.warning("Activation check returned no rows — skipping.")
        return

    populated, total, pct_populated = row[0], row[1], row[2]

    if total == 0:
        logger.warning(
            "No text chunks found in the database; cannot verify contextual retrieval."
        )
        return

    pct = float(pct_populated)
    if pct < CONTEXTUAL_RETRIEVAL_MIN_PCT:
        sys.exit(
            f"CONTEXTUAL RETRIEVAL NOT FULLY ACTIVATED: only {pct}% of text chunks "
            f"have contextualized embeddings. Do not proceed to evaluation until this "
            f"reaches >= 95%. Re-run scripts/contextualize.py (use --force to overwrite) or check Batch "
            f"API job status."
        )

    logger.info(
        "Contextual retrieval activated: %.1f%% populated (%d / %d text chunks)",
        pct,
        populated,
        total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc", help="UUID of a single document to contextualize"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process chunks even when contextualized_text is set",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of chunks per Batch API submission (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N documents",
    )
    args = parser.parse_args()

    client = _get_client()

    # Finish any batch that was submitted but not written by a prior interrupted
    # run; otherwise we'd re-submit (and re-bill) the same chunks below.
    _resume_in_flight_batch(client)

    docs = _list_documents(doc_id=args.doc)
    if not docs:
        logger.error("No documents found (have you run scripts/ingest.py?)")
        sys.exit(1)

    if args.limit is not None:
        docs = docs[: args.limit]

    logger.info("Considering %d document(s)", len(docs))
    total = 0
    for doc_row in docs:
        try:
            total += contextualize_document(
                doc_row,
                force=args.force,
                batch_size=args.batch_size,
                client=client,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error("FAILED %s: %s", doc_row.get("source_path", "?"), e)
            continue

    logger.info("Done — contextualized %d chunks total", total)
    verify_contextual_retrieval_activated()


if __name__ == "__main__":
    main()
