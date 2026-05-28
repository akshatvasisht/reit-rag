"""Second-pass chart enrichment via Claude Sonnet 4.6 vision.

For every document already in the `documents` table, re-parse the PDF, find
every Docling PictureItem (including the captionless ones the base parser
discards), send the image bytes to Claude vision, and store the structured
description as a new chunk with content_type='chart_description'.

This script is ADDITIVE — it does not touch existing chunks. You do not need
to re-run `scripts/ingest.py`.

Usage:
    python scripts/enrich_charts.py                # process every doc not yet
                                                   # marked as chart-enriched
    python scripts/enrich_charts.py --doc <id>     # one specific document
    python scripts/enrich_charts.py --force        # re-enrich already-enriched
                                                   # documents (deletes their
                                                   # chart_description chunks
                                                   # first)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from docling_core.types.doc.document import PictureItem  # type: ignore

from src.db import connect
from src.ingestion.chart_extractor import build_chart_context_text, extract_chart
from src.ingestion.embedder import embed_chunks
from src.ingestion.parser import parse_pdf
from src.ingestion.writer import write_chunks
from src.models import Chunk, DocumentMeta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enrich")
WRITE_BATCH_SIZE = 8

# Images whose bounding box area (in Docling page-coordinate units) is below
# this threshold are almost certainly logos, icons, or decorative photos rather
# than data charts.  Skipping them avoids unnecessary vision API calls and the
# noise that comes from sending non-chart images to the extractor.
MIN_CHART_AREA_PX = 10_000


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _list_documents(force: bool, doc_id: Optional[str]) -> list[dict]:
    """Return documents that need enrichment.

    Without --force: only docs not marked complete in documents metadata.
    With --force: every doc (existing chart_description chunks will be deleted
    before re-enrichment).
    """
    sql = """
        SELECT d.id, d.company, d.ticker, d.doc_type, d.report_date,
               d.period_covered, d.doc_version, d.source_path,
               d.chart_enrichment_completed_at,
               (SELECT count(*) FROM chunks c
                  WHERE c.document_id = d.id
                    AND c.content_type = 'chart_description') AS existing,
               (SELECT count(*) FROM chunks c
                  WHERE c.document_id = d.id
                    AND c.content_type = 'chart_context') AS chart_context
        FROM documents d
    """
    params: tuple = ()
    if doc_id:
        sql += " WHERE d.id = %s"
        params = (doc_id,)
    elif not force:
        sql += """
            WHERE d.chart_enrichment_completed_at IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM chunks c2
                  WHERE c2.document_id = d.id
                    AND c2.content_type = 'chart_description'
              )
        """
    sql += " ORDER BY d.company, d.doc_version"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        assert cur.description is not None  # SELECT always populates description
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]


def _mark_enrichment_result(
    doc_id: UUID,
) -> None:
    """Persist per-document enrichment completion marker."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET chart_enrichment_completed_at = NOW()
            WHERE id = %s
            """,
            (str(doc_id),),
        )
        conn.commit()


def _clear_enrichment_marker(doc_id: UUID) -> None:
    """Clear completion marker when enrichment run has extraction failures."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET chart_enrichment_completed_at = NULL
            WHERE id = %s
            """,
            (str(doc_id),),
        )
        conn.commit()


def _delete_chart_chunks(doc_id: UUID) -> int:
    """Delete chart_description and chart_context chunks for the given document.

    Both types are cleared together so a forced re-enrichment starts from a
    clean slate regardless of which path previous enrichment took.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE document_id = %s "
            "AND content_type IN ('chart_description', 'chart_context')",
            (str(doc_id),),
        )
        conn.commit()
        return cur.rowcount


def _is_likely_chart(picture_item) -> bool:
    """Return True when the image bbox is large enough to plausibly be a chart.

    Small bounding boxes indicate logos, icons, or decorative elements that
    the vision model cannot extract meaningful data from.  Failing open (True)
    when provenance or bbox is absent preserves the prior behavior.
    """
    provs = getattr(picture_item, "prov", None)
    if not provs:
        return True
    bbox = getattr(provs[0], "bbox", None)
    if bbox is None:
        return True
    # Docling emits PDF bboxes in a BOTTOMLEFT coordinate origin, so b - t is
    # negative; take each side's magnitude so the area is origin-agnostic and a
    # genuine chart isn't discarded as "too small".
    width = abs(bbox.r - bbox.l)
    height = abs(bbox.b - bbox.t)
    area = width * height
    return area >= MIN_CHART_AREA_PX


# ---------------------------------------------------------------------------
# Per-document enrichment
# ---------------------------------------------------------------------------

def _picture_image(item: PictureItem, doc) -> Optional["Image.Image"]:
    """Get a PIL image from a PictureItem; return None if unavailable."""
    # Docling 2.x exposes either `.image.pil_image` or `get_image(doc)`.
    try:
        result = item.get_image(doc)
        return result if isinstance(result, Image.Image) else None
    except (AttributeError, TypeError):
        pass
    try:
        img = item.image
        result = getattr(img, "pil_image", None)
        return result if isinstance(result, Image.Image) else None
    except AttributeError:
        return None


def _page_of(item: PictureItem) -> Optional[int]:
    """1-based page number from item provenance."""
    try:
        provs = getattr(item, "prov", None)
        if provs:
            return int(provs[0].page_no)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _page_image(docling_doc, page: Optional[int]) -> Optional["Image.Image"]:
    """Return the full-page render for *page* (1-based), or None if unavailable.

    Requires the document to have been parsed with page-image generation
    (the image pipeline used by ``parse_pdf(with_images=True)``). Used by the
    fallback path when a full-bleed map/diagram is fragmented across PictureItems
    so no single picture crop yields a confident extraction.
    """
    if page is None:
        return None
    try:
        page_item = docling_doc.pages.get(page)
    except (AttributeError, TypeError):
        return None
    if page_item is None:
        return None
    img_ref = getattr(page_item, "image", None)
    pil = getattr(img_ref, "pil_image", None) if img_ref is not None else None
    return pil if isinstance(pil, Image.Image) else None


def _flush_chunks(conn, pending_chunks: list[Chunk], company: str) -> int:
    """Embed and persist pending chart chunks, then clear the buffer.

    Commits each flush so long-running docs survive interruptions with partial
    progress saved.
    """
    if not pending_chunks:
        return 0
    embed_chunks(pending_chunks)
    written = write_chunks(conn, pending_chunks)
    conn.commit()
    logger.info("  Persisted %d chart chunks for %s", written, company)
    pending_chunks.clear()
    return written


def _collect_page_text(docling_doc, page: Optional[int]) -> list[str]:
    """Collect the text content of non-picture items on a given page.

    Used to populate the sibling-text section of chart_context fallback chunks
    so the page remains findable even when chart extraction fails the confidence
    gate.  Returns an empty list when page is None or no text items exist.
    """
    if page is None:
        return []
    texts: list[str] = []
    try:
        from docling_core.types.doc.document import TextItem  # type: ignore
        for item, _ in docling_doc.iterate_items():
            if isinstance(item, PictureItem):
                continue
            if not isinstance(item, TextItem):
                continue
            provs = getattr(item, "prov", None)
            if not provs:
                continue
            try:
                item_page = int(provs[0].page_no)
            except (AttributeError, TypeError, ValueError):
                continue
            if item_page != page:
                continue
            text = getattr(item, "text", None) or ""
            if text.strip():
                texts.append(text.strip())
    except Exception:  # noqa: BLE001 — sibling text is best-effort
        pass
    return texts


def enrich_document(doc_row: dict, force: bool) -> int:
    """Enrich one document. Returns the number of chart_description chunks written."""
    doc_id = doc_row["id"]
    source_path = doc_row["source_path"]
    company = doc_row["company"]
    existing = doc_row["existing"]
    chart_context = doc_row.get("chart_context", 0)
    completed_at = doc_row.get("chart_enrichment_completed_at")

    if completed_at and not force:
        logger.info("SKIP (already enriched: %d charts): %s", existing, source_path)
        return 0

    if (existing or chart_context) and force:
        _clear_enrichment_marker(UUID(str(doc_id)))
        n = _delete_chart_chunks(doc_id)
        logger.info("Force re-enrich: deleted %d prior chart chunks for %s", n, company)

    logger.info("==== Enriching %s ====", source_path)

    doc_meta = DocumentMeta(
        id=UUID(str(doc_id)),
        company=company,
        ticker=doc_row["ticker"],
        doc_type=doc_row["doc_type"],
        report_date=doc_row["report_date"],
        period_covered=doc_row["period_covered"],
        doc_version=doc_row["doc_version"],
        source_path=source_path,
    )

    docling_doc = parse_pdf(source_path, with_images=True)
    pictures = [
        (item, _page_of(item))
        for item, _ in docling_doc.iterate_items()
        if isinstance(item, PictureItem)
    ]
    logger.info("  Found %d PictureItems in %s", len(pictures), Path(source_path).name)

    pending_chunks: list[Chunk] = []
    written_total = 0
    skipped_not_chart = 0
    skipped_no_image = 0
    skipped_small = 0
    fallback_created = 0
    extraction_errors = 0
    page_recovered_count = 0
    # Pages already attempted / recovered via the full-page fallback, so a page
    # is rendered and scored at most once even when several of its pictures are
    # low-confidence.
    page_fallback_attempted: set[int] = set()
    page_recovered: set[int] = set()

    for i, (item, page) in enumerate(pictures, start=1):
        # Pre-filter: skip images whose bbox is too small to be a data chart.
        if not _is_likely_chart(item):
            skipped_small += 1
            logger.info(
                "  Picture %d/%d on page %s: bbox too small, skipped",
                i, len(pictures), page,
            )
            continue

        image = _picture_image(item, docling_doc)
        if image is None:
            skipped_no_image += 1
            continue
        try:
            description, needs_fallback = extract_chart(image)
        except Exception as e:  # noqa: BLE001 — log and continue processing the document
            extraction_errors += 1
            logger.warning("  Picture %d/%d on page %s: vision call failed (%s)",
                           i, len(pictures), page, type(e).__name__)
            continue

        if description is None and not needs_fallback:
            skipped_not_chart += 1
            logger.info("  Picture %d/%d on page %s: NOT_CHART (skipped)",
                        i, len(pictures), page)
            continue

        if needs_fallback:
            # Per-picture extraction was low-confidence. Docling sometimes
            # fragments a full-bleed map/diagram into pieces no single crop
            # captures, so try the whole page once before giving up.
            if page is not None and page in page_recovered:
                # Page already captured by a full-page extraction; the per-piece
                # crop would only duplicate it.
                continue

            page_desc: Optional[str] = None
            if page is not None and page not in page_fallback_attempted:
                page_fallback_attempted.add(page)
                page_img = _page_image(docling_doc, page)
                if page_img is not None:
                    try:
                        page_desc, _ = extract_chart(page_img)
                    except Exception as e:  # noqa: BLE001 — log and fall through to chart_context
                        logger.warning(
                            "  Page %s full-page fallback call failed (%s)",
                            page, type(e).__name__,
                        )
                        page_desc = None

            if page_desc is not None:
                # Recovered the fragmented visual from the full-page render.
                if page is not None:
                    page_recovered.add(page)
                page_recovered_count += 1
                pending_chunks.append(
                    Chunk(
                        document_id=doc_meta.id,
                        company=doc_meta.company,
                        ticker=doc_meta.ticker,
                        doc_type=doc_meta.doc_type,
                        report_date=doc_meta.report_date,
                        doc_version=doc_meta.doc_version,
                        period_covered=doc_meta.period_covered,
                        source_authority="company_authored",
                        chunk_text=page_desc,
                        content_type="chart_description",
                        section_title=None,
                        page_number=page,
                        is_parent=False,
                        parent_chunk_id=None,
                        token_count=None,
                    )
                )
                logger.info(
                    "  Picture %d/%d on page %s: low-confidence crop — recovered via full-page extraction (%d chars)",
                    i, len(pictures), page, len(page_desc),
                )
            else:
                # Full-page fallback unavailable or also low-confidence: build an
                # additive chart_context chunk so the page stays findable.
                sibling_texts = _collect_page_text(docling_doc, page)
                fallback_text = build_chart_context_text(
                    section_title=None,
                    sibling_texts=sibling_texts,
                )
                pending_chunks.append(
                    Chunk(
                        document_id=doc_meta.id,
                        company=doc_meta.company,
                        ticker=doc_meta.ticker,
                        doc_type=doc_meta.doc_type,
                        report_date=doc_meta.report_date,
                        doc_version=doc_meta.doc_version,
                        period_covered=doc_meta.period_covered,
                        source_authority="company_authored",
                        chunk_text=fallback_text,
                        content_type="chart_context",
                        section_title=None,
                        page_number=page,
                        is_parent=False,
                        parent_chunk_id=None,
                        token_count=None,
                    )
                )
                fallback_created += 1
                logger.info(
                    "  Picture %d/%d on page %s: low-confidence — chart_context fallback created",
                    i, len(pictures), page,
                )
        else:
            # Confident extraction: store as chart_description.
            # is_parent=False so the retrieval layer (which only queries children)
            # picks it up. parent_chunk_id=None so small-to-large expansion
            # treats it as self-contained.
            assert description is not None
            pending_chunks.append(
                Chunk(
                    document_id=doc_meta.id,
                    company=doc_meta.company,
                    ticker=doc_meta.ticker,
                    doc_type=doc_meta.doc_type,
                    report_date=doc_meta.report_date,
                    doc_version=doc_meta.doc_version,
                    period_covered=doc_meta.period_covered,
                    source_authority="company_authored",
                    chunk_text=description,
                    content_type="chart_description",
                    section_title=None,
                    page_number=page,
                    is_parent=False,
                    parent_chunk_id=None,
                    token_count=None,
                )
            )
            logger.info("  Picture %d/%d on page %s: extracted (%d chars)",
                        i, len(pictures), page, len(description))

        if len(pending_chunks) >= WRITE_BATCH_SIZE:
            with connect() as conn:
                written_total += _flush_chunks(conn, pending_chunks, company)

    logger.info(
        "  Vision pass complete: %d charts extracted (%d via full-page fallback), "
        "%d chart_context fallbacks, %d skipped as NOT_CHART, %d too small, %d no image",
        written_total + len(pending_chunks) - fallback_created,
        page_recovered_count,
        fallback_created,
        skipped_not_chart,
        skipped_small,
        skipped_no_image,
    )

    if pending_chunks:
        with connect() as conn:
            written_total += _flush_chunks(conn, pending_chunks, company)
    if extraction_errors > 0:
        _clear_enrichment_marker(UUID(str(doc_id)))
        logger.warning(
            "  Marked enrichment incomplete for %s due to %d extraction failure(s)",
            company,
            extraction_errors,
        )
    else:
        _mark_enrichment_result(UUID(str(doc_id)))
    return written_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", help="UUID of a single document to enrich")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich documents that already have chart_description chunks")
    args = parser.parse_args()

    docs = _list_documents(force=args.force, doc_id=args.doc)
    if not docs:
        logger.info("No documents require chart enrichment")
        return

    logger.info("Considering %d document(s)", len(docs))
    total = 0
    for doc_row in docs:
        try:
            total += enrich_document(doc_row, force=args.force)
        except (FileNotFoundError, ValueError) as e:
            _clear_enrichment_marker(UUID(str(doc_row["id"])))
            logger.error("FAILED %s: %s", doc_row["source_path"], e)
            continue

    logger.info("Done — wrote %d chart_description chunks in total", total)


if __name__ == "__main__":
    main()
