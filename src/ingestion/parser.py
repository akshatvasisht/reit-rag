"""PDF parsing via Docling — returns structured DoclingDocument objects.

Wraps DocumentConverter (expensive to instantiate) as a module-level singleton.
Provides a thin iterator, iter_blocks(), that maps Docling element types onto the
ContentType vocabulary used throughout the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import (
    DocItem,
    SectionHeaderItem,
    TableItem,
    PictureItem,
    TextItem,
    ListItem,
    CodeItem,
    FormulaItem,
)

from src.models import ContentType

logger = logging.getLogger(__name__)

# Build once at import time — loading layout + table models is expensive (~5 s).
# OCR is disabled: these PDFs are text-extractable financial decks. RapidOCR was
# being invoked, returning empty results, and burning ~30 s per document.
_BASE_PIPELINE_OPTIONS = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_BASE_PIPELINE_OPTIONS),
    }
)

# Separate converter for vision-enrichment runs: retains the PIL image bytes
# of every PictureItem so chart_extractor.py can pass them to Claude vision.
# Built lazily because it is not needed for the default ingest path.
_IMAGE_PIPELINE_OPTIONS = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    generate_picture_images=True,
    images_scale=2.0,  # bump resolution; cheap, helps the vision model.
)
_CONVERTER_WITH_IMAGES: Optional[DocumentConverter] = None


def _get_image_converter() -> DocumentConverter:
    """Lazily build the image-retaining DocumentConverter (~5 s init)."""
    global _CONVERTER_WITH_IMAGES
    if _CONVERTER_WITH_IMAGES is None:
        _CONVERTER_WITH_IMAGES = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=_IMAGE_PIPELINE_OPTIONS),
            }
        )
    return _CONVERTER_WITH_IMAGES


@dataclass
class ParsedBlock:
    """A single structural element extracted from a Docling document.

    Consumed by the chunker to build child/parent chunk pairs.
    """

    text: str
    page_number: Optional[int]
    section_title: Optional[str]
    content_type: ContentType


def parse_pdf(pdf_path: str | Path, *, with_images: bool = False) -> DoclingDocument:
    """Parse a PDF file and return the structured Docling document.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        with_images: If True, retain PictureItem image bytes for later vision
            extraction. Defaults to False (saves memory in the base ingest path).

    Returns:
        A DoclingDocument preserving section hierarchy, table cells, captions,
        and page-provenance for every element.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If Docling conversion fails or returns no document.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Parsing PDF: %s (with_images=%s)", path.name, with_images)
    converter = _get_image_converter() if with_images else _CONVERTER
    result = converter.convert(str(path))

    if result is None or result.document is None:
        raise ValueError(f"Docling returned no document for: {path}")

    logger.info("Parsed %s — %d pages", path.name, len(result.document.pages))
    return result.document


def iter_blocks(doc: DoclingDocument) -> Iterator[ParsedBlock]:  # noqa: C901
    """Iterate over structural elements of a DoclingDocument as ParsedBlocks.

    Yields one ParsedBlock per Docling element, skipping elements with no
    usable text.  Section headings encountered in document order are tracked
    and propagated to every subsequent block as section_title until a new
    heading appears.

    Content-type mapping:
        TableItem                      → 'table'
        PictureItem (with caption)     → 'chart_caption'
        PictureItem (no caption)       → skipped (no text to embed)
        SectionHeaderItem              → 'text'  (heading text itself)
        TextItem / ListItem / etc.     → 'text'
        Mixed inline (heading + table) → 'mixed'  (resolved at group level
                                         in the chunker; single items are
                                         classified individually here)

    Args:
        doc: A DoclingDocument returned by parse_pdf().

    Yields:
        ParsedBlock instances in document order.
    """
    current_section: Optional[str] = None

    for item, _ in doc.iterate_items():
        page_number = _page_number(item)

        # --- Section headings: update running context, then emit as 'text' ---
        if isinstance(item, SectionHeaderItem):
            heading_text = item.text.strip()
            if heading_text:
                current_section = heading_text
                yield ParsedBlock(
                    text=heading_text,
                    page_number=page_number,
                    section_title=current_section,
                    content_type="text",
                )
            continue

        # --- Tables --------------------------------------------------------
        if isinstance(item, TableItem):
            table_text = _table_to_text(item)
            if table_text:
                yield ParsedBlock(
                    text=table_text,
                    page_number=page_number,
                    section_title=current_section,
                    content_type="table",
                )
            continue

        # --- Figures / charts ----------------------------------------------
        if isinstance(item, PictureItem):
            caption = _picture_caption(item, doc)
            if caption:
                yield ParsedBlock(
                    text=caption,
                    page_number=page_number,
                    section_title=current_section,
                    content_type="chart_caption",
                )
            # No usable text for a figure without a caption — skip.
            continue

        # --- Plain prose, lists, code, formulae ----------------------------
        if isinstance(item, (TextItem, ListItem, CodeItem, FormulaItem)):
            prose = item.text.strip()
            if prose:
                yield ParsedBlock(
                    text=prose,
                    page_number=page_number,
                    section_title=current_section,
                    content_type="text",
                )
            continue

        # --- Any other DocItem with a .text attribute ----------------------
        if isinstance(item, DocItem):
            raw = getattr(item, "text", None)
            if raw and raw.strip():
                yield ParsedBlock(
                    text=raw.strip(),
                    page_number=page_number,
                    section_title=current_section,
                    content_type="text",
                )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _page_number(item: object) -> Optional[int]:
    """Extract 1-based page number from a Docling item's provenance."""
    try:
        provs = getattr(item, "prov", None)
        if provs:
            # prov is a list of ProvenanceItem; take the first one.
            prov = provs[0]
            return int(prov.page_no)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _table_to_text(item: TableItem) -> str:
    """Render a TableItem as a tab-separated text grid for embedding.

    Uses Docling's built-in export when available; falls back to iterating
    the cell grid directly.

    Args:
        item: A Docling TableItem.

    Returns:
        A multi-line string of tab-separated cell values, or '' if empty.
    """
    try:
        # Docling 2.x — preferred path
        return item.export_to_markdown()
    except AttributeError:
        pass

    # Fallback: iterate the data grid
    try:
        data = item.data
        if data is None:
            return ""
        rows: list[str] = []
        for row in data.grid:
            cells = [cell.text.strip() if cell.text else "" for cell in row]
            rows.append("\t".join(cells))
        return "\n".join(rows)
    except AttributeError:
        return ""


def _picture_caption(item: PictureItem, doc: DoclingDocument) -> str:
    """Extract caption text from a PictureItem via the document graph.

    Uses the Docling-provided caption_text(doc) helper which resolves RefItem
    pointers into their text. Falls back to an empty string if no caption
    is attached.

    Args:
        item: A Docling PictureItem.
        doc: The parent DoclingDocument (needed to dereference captions).

    Returns:
        Caption text, or '' if no caption is present.
    """
    try:
        return (item.caption_text(doc) or "").strip()
    except (AttributeError, TypeError):
        return ""
