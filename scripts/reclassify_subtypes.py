"""Backfill doc_subtype for documents already present in the database.

For each document row, retrieves the first-page text from the chunks table
and calls the LLM subtype classifier, then updates both the documents and
chunks tables.  Idempotent: running twice produces the same result.

Usage:
    python scripts/reclassify_subtypes.py
    python scripts/reclassify_subtypes.py --dry-run
    python scripts/reclassify_subtypes.py --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect
from src.ingestion.metadata import classify_doc_subtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reclassify_subtypes")

# Fetch all documents along with the first available parent chunk text from
# page 1.  The split_part call extracts just the filename from the full path.
_SELECT_SQL = """
SELECT
    d.id,
    split_part(d.source_path, '/', -1) AS filename,
    d.doc_type,
    (SELECT chunk_text
       FROM chunks
       WHERE document_id = d.id
         AND page_number = 1
         AND is_parent = TRUE
       ORDER BY created_at
       LIMIT 1) AS first_page_text
FROM documents d
ORDER BY d.id;
"""

_UPDATE_DOC_SQL = """
UPDATE documents
   SET doc_subtype = %s
 WHERE id = %s;
"""

_UPDATE_CHUNKS_SQL = """
UPDATE chunks
   SET doc_subtype = %s
 WHERE document_id = %s;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed changes without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N documents (useful for testing).",
    )
    args = parser.parse_args()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_SELECT_SQL)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()

    if args.limit is not None:
        rows = rows[: args.limit]

    logger.info("Processing %d document(s) (dry_run=%s)", len(rows), args.dry_run)

    updates: list[tuple[str, str]] = []  # (doc_subtype, doc_id)
    for row in rows:
        r = dict(zip(cols, row))
        doc_id: str = str(r["id"])
        filename: str = r["filename"] or ""
        doc_type: str = r["doc_type"] or ""
        first_page_text: str = r["first_page_text"] or ""

        subtype = classify_doc_subtype(
            first_page_text=first_page_text,
            filename=filename,
            doc_type=doc_type,
        )
        logger.info("  %s  →  %s  (file: %s)", doc_id, subtype, filename)
        updates.append((subtype, doc_id))

    if args.dry_run:
        logger.info("Dry run complete — no writes performed.")
        return

    with connect() as conn:
        with conn.cursor() as cur:
            for subtype, doc_id in updates:
                cur.execute(_UPDATE_DOC_SQL, (subtype, doc_id))
                cur.execute(_UPDATE_CHUNKS_SQL, (subtype, doc_id))
        conn.commit()

    logger.info("Updated %d document(s) and their chunks.", len(updates))


if __name__ == "__main__":
    main()
