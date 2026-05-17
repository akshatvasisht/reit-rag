"""Backfill page_content_class for chunks where it is still 'unknown'.

Iterates chunks with page_content_class = 'unknown', calls classify_page_content
for each, and UPDATEs the database in batches of 50 with a 1-second pause
between batches to stay within API rate limits.

Usage:
    python scripts/reclassify_page_content.py
    python scripts/reclassify_page_content.py --dry-run
    python scripts/reclassify_page_content.py --limit 200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect
from src.ingestion.chunker import classify_page_content

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reclassify_page_content")

_SELECT_SQL = """
SELECT id, chunk_text, section_title
FROM chunks
WHERE page_content_class = 'unknown'
ORDER BY id
"""

_SELECT_SQL_LIMIT = """
SELECT id, chunk_text, section_title
FROM chunks
WHERE page_content_class = 'unknown'
ORDER BY id
LIMIT %s
"""

_UPDATE_SQL = """
UPDATE chunks
   SET page_content_class = %s
 WHERE id = %s
"""

BATCH_SIZE = 50
INTER_BATCH_SLEEP = 1.0  # seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed classifications without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N chunks (useful for testing).",
    )
    args = parser.parse_args()

    with connect() as conn:
        with conn.cursor() as cur:
            if args.limit is not None:
                cur.execute(_SELECT_SQL_LIMIT, (args.limit,))
            else:
                cur.execute(_SELECT_SQL)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()

    if not rows:
        logger.info("No chunks with page_content_class='unknown' — nothing to do.")
        return

    logger.info(
        "Processing %d chunk(s) (dry_run=%s, batch_size=%d)",
        len(rows),
        args.dry_run,
        BATCH_SIZE,
    )

    updates: list[tuple[str, str]] = []  # (page_content_class, chunk_id)
    for row in rows:
        r = dict(zip(cols, row))
        chunk_id: str = str(r["id"])
        chunk_text: str = r["chunk_text"] or ""
        section_title: str = r["section_title"] or ""

        new_class = classify_page_content(chunk_text, section_title)
        logger.debug("  %s  →  %s", chunk_id, new_class)
        updates.append((new_class, chunk_id))

    if args.dry_run:
        for new_class, chunk_id in updates:
            logger.info("  [dry-run] %s  →  %s", chunk_id, new_class)
        logger.info("Dry run complete — no writes performed.")
        return

    written = 0
    for batch_start in range(0, len(updates), BATCH_SIZE):
        batch = updates[batch_start : batch_start + BATCH_SIZE]
        with connect() as conn:
            with conn.cursor() as cur:
                for new_class, chunk_id in batch:
                    cur.execute(_UPDATE_SQL, (new_class, chunk_id))
            conn.commit()
        written += len(batch)
        logger.info(
            "  Wrote %d / %d chunks", written, len(updates)
        )
        if batch_start + BATCH_SIZE < len(updates):
            time.sleep(INTER_BATCH_SLEEP)

    logger.info("Done — updated %d chunk(s).", written)


if __name__ == "__main__":
    main()
