"""End-to-end ingestion: parse → extract metadata → chunk → embed → write.

Usage:
    python scripts/ingest.py                # process everything in data/pdfs/
    python scripts/ingest.py path/to.pdf    # process one file
    python scripts/ingest.py --force        # re-ingest already-loaded files

Idempotent: by default, skips files whose `source_path` is already in the
documents table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_chunks
from src.ingestion.metadata import extract_document_meta
from src.ingestion.parser import parse_pdf
from src.ingestion.writer import (
    delete_chunks_for_document,
    document_exists,
    write_chunks,
    write_document,
)


PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def ingest_one(pdf_path: Path, force: bool = False) -> None:
    """Run the full pipeline on a single PDF.

    Args:
        pdf_path: Absolute path to a PDF file.
        force: If True, re-ingest even when the file is already in the database.
    """
    source_path = str(pdf_path.resolve())

    with connect() as conn:
        if not force and document_exists(conn, source_path):
            logger.info("SKIP (already ingested): %s", pdf_path.name)
            return

    logger.info("==== Ingesting %s ====", pdf_path.name)

    doc_meta = extract_document_meta(pdf_path)
    doc_meta.source_path = source_path
    logger.info(
        "  Metadata: %s | %s | %s | doc_version=%s",
        doc_meta.company, doc_meta.ticker, doc_meta.doc_type, doc_meta.doc_version,
    )

    docling_doc = parse_pdf(pdf_path)
    chunks = chunk_document(docling_doc, doc_meta)
    parents = sum(1 for c in chunks if c.is_parent)
    children = len(chunks) - parents
    logger.info("  Chunks: %d parents + %d children", parents, children)

    embed_chunks(chunks)

    with connect() as conn:
        doc_id = write_document(conn, doc_meta)
        # Ensure chunk FK uses the canonical row id returned by upsert.
        doc_meta.id = doc_id
        for chunk in chunks:
            chunk.document_id = doc_id
        if force:
            deleted = delete_chunks_for_document(conn, doc_id)
            logger.info("  Force re-ingest: deleted %d prior chunks", deleted)
        n = write_chunks(conn, chunks)
        conn.commit()
        logger.info("  Wrote %d chunks for %s", n, doc_meta.company)


def main() -> None:
    """CLI entrypoint — process one file or every PDF under data/pdfs/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Single PDF path (optional)")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest files already in the database")
    args = parser.parse_args()

    if args.path:
        paths = [Path(args.path)]
    else:
        paths = sorted(PDF_DIR.glob("*.pdf"))

    if not paths:
        logger.error("No PDFs found in %s", PDF_DIR)
        sys.exit(1)

    logger.info("Found %d PDF(s) to process", len(paths))
    for pdf in paths:
        try:
            ingest_one(pdf, force=args.force)
        except (FileNotFoundError, ValueError) as e:
            logger.error("FAILED %s: %s", pdf.name, e)
            continue


if __name__ == "__main__":
    main()
