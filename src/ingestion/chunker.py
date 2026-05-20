"""Hierarchical small-to-large chunker for REIT investor-presentation PDFs.

Produces parent chunks (~512–1024 tok, one per section/slide) and child chunks
(~256 tok, one per paragraph/table/caption), interleaved so parents always
precede their children for FK-safe DB insertion.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Protocol, cast

import tiktoken

from src.ingestion.page_classifier import classify_page_content
from src.models import Chunk, ContentType, DocumentMeta

if TYPE_CHECKING:
    # Avoid importing Docling at type-check time; the runtime import lives in
    # parser.py.  We only need the structural duck-type here.
    from docling.datamodel.document import DoclingDocument  # noqa: F401

__all__ = ["chunk_document", "classify_page_content", "ParsedBlock"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PARENT_TARGET_TOKENS: int = 768
PARENT_MAX_TOKENS: int = 1024
CHILD_TARGET_TOKENS: int = 256
CHILD_MAX_TOKENS: int = 320

# Sentence-boundary regex: split after ., !, ? followed by whitespace + capital,
# or after a newline sequence. Avoids splitting on common abbreviations by
# requiring the next character to be uppercase or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=\n)\s*(?=\S)")

# Header line that introduces a footnote/endnote/references block. Must match
# the *entire* first non-empty line so generic prose like "Note: ..." is not
# treated as a footnote section.
_FOOTNOTE_HEADER_RE = re.compile(
    r"^(?:footnotes?|notes|references|endnotes)\s*[:.\-]?\s*$",
    re.IGNORECASE,
)

# Leading numeric marker on an already-numbered footnote item, e.g. "(1) ",
# "(1. ", "1. ", "1) ". Used as an idempotency guard.
_NUMBERED_PREFIX_RE = re.compile(r"^\s*(?:\(\d+[.)]?|\d+[.)])\s+")


class ParsedBlock(Protocol):
    """Structural duck-type for blocks produced by ``parser.iter_blocks``."""
    text: str
    page_number: int
    section_title: str
    content_type: ContentType


def _get_encoder() -> tiktoken.Encoding:
    """Return the cl100k_base encoder used for token-count estimates."""
    # Chunk token counting is model-agnostic and only needs stable budgeting;
    # use cl100k_base as a consistent approximation across embedding backends.
    try:
        return tiktoken.encoding_for_model("text-embedding-3-large")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


_ENCODER: tiktoken.Encoding = _get_encoder()


def _count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the embedding encoder.

    Args:
        text: Raw string to measure.

    Returns:
        Integer token count.
    """
    return len(_ENCODER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-level fragments without losing whitespace.

    Args:
        text: Paragraph or multi-sentence string.

    Returns:
        List of sentence strings; never empty (returns ``[text]`` at minimum).
    """
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_into_children(
    text: str,
    content_type: ContentType,
    target: int = CHILD_TARGET_TOKENS,
    max_tokens: int = CHILD_MAX_TOKENS,
) -> list[str]:
    """Split *text* into child-sized fragments respecting sentence boundaries.

    Tables and chart_captions are returned as a single fragment (never split).
    For regular text/mixed content, sentences are accumulated until adding one
    exceeds *max_tokens*, then flush. A sentence that alone exceeds *max_tokens*
    is emitted as its own chunk rather than being truncated.

    Args:
        text: Block text to split.
        content_type: Docling content type tag.
        target: Soft target token count per child chunk.
        max_tokens: Hard cap; flush immediately when exceeded.

    Returns:
        Non-empty list of text fragments.
    """
    # Tables and chart captions must never be split mid-row / mid-caption.
    if content_type in ("table", "chart_caption"):
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    fragments: list[str] = []
    buffer: list[str] = []
    buffer_tokens: int = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)

        # A single sentence that exceeds the hard cap: emit it alone.
        if s_tokens >= max_tokens:
            if buffer:
                fragments.append(" ".join(buffer))
                buffer = []
                buffer_tokens = 0
            fragments.append(sentence)
            continue

        # Adding this sentence would breach the hard cap: flush first.
        if buffer_tokens + s_tokens > max_tokens and buffer:
            fragments.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0

        buffer.append(sentence)
        buffer_tokens += s_tokens

        # Reached the soft target: flush eagerly.
        if buffer_tokens >= target:
            fragments.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0

    if buffer:
        fragments.append(" ".join(buffer))

    return fragments if fragments else [text]


def _group_blocks_by_section(
    blocks: list[ParsedBlock],
) -> list[tuple[str, int, list[ParsedBlock]]]:
    """Group a flat list of parsed blocks by section_title (or page number).

    Consecutive blocks sharing the same section_title are grouped together.
    When section_title is empty/None, the page number is used as a fallback
    key so that unsectioned content is not silently merged across pages.

    Args:
        blocks: Ordered list of ParsedBlock objects from ``iter_blocks``.

    Returns:
        List of ``(group_key, representative_page, blocks_in_group)`` tuples.
    """
    if not blocks:
        return []

    groups: list[tuple[str, int, list[ParsedBlock]]] = []
    current_key: str = ""
    current_page: int = 0
    current_group: list[ParsedBlock] = []

    for block in blocks:
        section = (block.section_title or "").strip()
        key = section if section else f"__page_{block.page_number}__"

        if key != current_key:
            if current_group:
                groups.append((current_key, current_page, current_group))
            current_key = key
            current_page = block.page_number
            current_group = [block]
        else:
            current_group.append(block)

    if current_group:
        groups.append((current_key, current_page, current_group))

    return groups


def _make_chunk(
    *,
    text: str,
    content_type: ContentType,
    section_title: str,
    page_number: int,
    doc_meta: DocumentMeta,
    is_parent: bool,
    parent_chunk_id: uuid.UUID | None,
) -> Chunk:
    """Construct a ``Chunk`` dataclass from block-level fields and doc metadata.

    Args:
        text: Chunk text content.
        content_type: One of ``"text"``, ``"table"``, ``"chart_caption"``, ``"mixed"``.
        section_title: Slide / section heading from Docling hierarchy.
        page_number: Source PDF page number.
        doc_meta: Document-level metadata from ``DocumentMeta``.
        is_parent: ``True`` for parent chunks, ``False`` for children.
        parent_chunk_id: UUID of the parent chunk; ``None`` for parents.

    Returns:
        A fully populated ``Chunk`` dataclass (no embedding yet).
    """
    return Chunk(
        document_id=doc_meta.id,
        company=doc_meta.company,
        ticker=doc_meta.ticker,
        doc_type=doc_meta.doc_type,
        report_date=doc_meta.report_date,
        doc_version=doc_meta.doc_version,
        period_covered=doc_meta.period_covered,
        source_authority="company_authored",
        chunk_text=text,
        content_type=content_type,
        section_title=section_title if section_title and not section_title.startswith("__page_") else None,
        page_number=page_number,
        is_parent=is_parent,
        parent_chunk_id=parent_chunk_id,
        token_count=_count_tokens(text),
    )


def _dominant_content_type(blocks: list[ParsedBlock]) -> ContentType:
    """Return the content type that best describes a group of blocks.

    Mixed-type groups (prose + tables) are tagged ``"mixed"``.  A group with
    only one content type carries that type directly.

    Args:
        blocks: One or more ParsedBlock objects in the same group.

    Returns:
        A ``ContentType`` literal.
    """
    types = {b.content_type for b in blocks}
    if len(types) == 1:
        return next(iter(types))  # type: ignore[return-value]
    return "mixed"


def _restore_footnote_numbering(text: str) -> str:
    """Re-prefix footnote body blocks with ``(1)``, ``(2)``, … markers.

    Docling occasionally strips the leading ``(N)`` markers from Footnotes /
    Notes / References / Endnotes sections during PDF parsing, leaving the
    items as bare paragraphs and destroying inline ``(n)`` cross-references.
    This helper detects that case and re-numbers the body blocks in order.
    """
    if not text:
        return text

    blocks = text.split("\n\n")
    # Locate the first non-empty block; it must be a footnote-section header.
    header_idx = -1
    for i, blk in enumerate(blocks):
        if blk.strip():
            header_idx = i
            break
    if header_idx == -1:
        return text
    if not _FOOTNOTE_HEADER_RE.match(blocks[header_idx].strip()):
        return text

    body = blocks[header_idx + 1:]
    # Idempotency: bail out if any body block already carries a numeric marker.
    for blk in body:
        if blk.strip() and _NUMBERED_PREFIX_RE.match(blk):
            return text

    n = 0
    renumbered: list[str] = []
    for blk in body:
        if blk.strip():
            n += 1
            renumbered.append(f"({n}) {blk.lstrip()}")
        else:
            renumbered.append(blk)

    if n == 0:
        return text

    return "\n\n".join(blocks[: header_idx + 1] + renumbered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: "DoclingDocument",
    doc_meta: DocumentMeta,
) -> list[Chunk]:
    """Convert a parsed Docling document into a flat list of parent + child Chunks.

    Blocks are first grouped by section title (or page number when no title is
    available).  Each group becomes one parent chunk.  If the group text exceeds
    ``PARENT_MAX_TOKENS``, it is split into multiple parents at paragraph
    boundaries to avoid over-large generation contexts.  Each parent is then
    split into child chunks at sentence boundaries, with tables and chart
    captions always kept as atomic children.

    The returned list interleaves ``[parent, child1, child2, …]`` sequences so
    that a DB writer can insert parents before children and resolve foreign keys
    in a single pass.

    Args:
        doc: A ``DoclingDocument`` object returned by the Docling parser.
        doc_meta: Document-level metadata (company, ticker, dates, etc.).

    Returns:
        Ordered list of ``Chunk`` objects — parents immediately followed by
        their children.
    """
    # Import here to avoid a hard dependency at module load time when only
    # type-checking or unit-testing without Docling installed.
    from src.ingestion.parser import iter_blocks  # type: ignore[import]

    blocks = cast(list[ParsedBlock], list(iter_blocks(doc)))
    if not blocks:
        return []

    result: list[Chunk] = []
    groups = _group_blocks_by_section(blocks)

    for group_key, representative_page, group_blocks in groups:
        section_title = (
            group_key
            if not group_key.startswith("__page_")
            else ""
        )

        # ----------------------------------------------------------------
        # Build parent candidates: if the combined group text fits in one
        # parent, emit it as-is.  If it would exceed PARENT_MAX_TOKENS,
        # split the group into sub-groups at block boundaries.
        # ----------------------------------------------------------------
        parent_groups = _split_group_into_parents(group_blocks)

        for parent_blocks in parent_groups:
            parent_text = "\n\n".join(b.text for b in parent_blocks)
            parent_text = _restore_footnote_numbering(parent_text)
            parent_page = parent_blocks[0].page_number
            parent_content_type = _dominant_content_type(parent_blocks)

            parent_chunk = _make_chunk(
                text=parent_text,
                content_type=parent_content_type,
                section_title=section_title,
                page_number=parent_page,
                doc_meta=doc_meta,
                is_parent=True,
                parent_chunk_id=None,
            )
            result.append(parent_chunk)

            # ----------------------------------------------------------------
            # Split each block in the parent into child chunks.  Tables and
            # chart_captions become exactly one child.  Text blocks are split
            # at sentence boundaries.
            # ----------------------------------------------------------------
            children = _make_children(
                parent_blocks=parent_blocks,
                parent_id=parent_chunk.id,
                section_title=section_title,
                doc_meta=doc_meta,
            )
            result.extend(children)

    return result


def _split_group_into_parents(
    blocks: list[ParsedBlock],
) -> list[list[ParsedBlock]]:
    """Partition a section's blocks into parent-sized sub-groups.

    We accumulate blocks until the combined text would exceed
    ``PARENT_MAX_TOKENS``, then flush. A single block that by itself exceeds
    the cap is emitted alone; block text is never truncated.

    Args:
        blocks: All ParsedBlock objects belonging to one section group.

    Returns:
        A list of block-lists, each suitable for a single parent chunk.
    """
    parent_groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_tokens: int = 0

    for block in blocks:
        block_tokens = _count_tokens(block.text)

        # If adding this block breaches the cap and the current group has content,
        # flush the current group first.
        if current_tokens + block_tokens > PARENT_MAX_TOKENS and current:
            parent_groups.append(current)
            current = []
            current_tokens = 0

        current.append(block)
        current_tokens += block_tokens

    if current:
        parent_groups.append(current)

    return parent_groups if parent_groups else [blocks]


def _make_children(
    *,
    parent_blocks: list[ParsedBlock],
    parent_id: uuid.UUID,
    section_title: str,
    doc_meta: DocumentMeta,
) -> list[Chunk]:
    """Generate child chunks for every block within a parent group.

    Each block is split independently (tables/captions become one child; prose
    is sentence-split).  All children share the same ``parent_chunk_id``.

    Args:
        parent_blocks: The blocks that make up the parent chunk.
        parent_id: UUID of the parent chunk for FK linkage.
        section_title: Section heading inherited by all children.
        doc_meta: Document-level metadata.

    Returns:
        Ordered list of child ``Chunk`` objects.
    """
    children: list[Chunk] = []

    for block in parent_blocks:
        fragments = _split_into_children(block.text, block.content_type)

        for fragment in fragments:
            if not fragment.strip():
                continue
            child = _make_chunk(
                text=fragment,
                content_type=block.content_type,
                section_title=section_title,
                page_number=block.page_number,
                doc_meta=doc_meta,
                is_parent=False,
                parent_chunk_id=parent_id,
            )
            children.append(child)

    return children
