"""Tests for the chunker's minimum-child-size coalescing rule.

A bulleted summary/action-plan slide parses into one short ListItem block per
bullet. Emitted individually these become unrankable sub-30-char child chunks.
The coalescing rule merges adjacent short same-page/same-type fragments so a
multi-bullet slide becomes a coherent chunk rather than a dozen microchunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.ingestion.chunker import (
    CHILD_MIN_CHARS,
    _coalesce_small_children,
    _make_children,
)
from src.models import ContentType, DocumentMeta


@dataclass
class _Block:
    """Minimal ParsedBlock stand-in matching the structural duck-type."""

    text: str
    page_number: int
    section_title: str
    content_type: ContentType


def _doc_meta() -> DocumentMeta:
    return DocumentMeta(
        company="BXP",
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2026-01",
        period_covered="Q4 2025",
        doc_version="2026-01",
        source_path=f"test/path/{uuid4()}.pdf",
    )


# Each bullet is a short ListItem block, mirroring how the parser emits a
# "Current Strengths / Action Plan" summary slide one bullet at a time.
_BULLETS = [
    "Premier asset quality",
    "Strong balance sheet",
    "Embedded growth",
    "Aligned management",
    "Diversified tenants",
    "Sustainability profile",
    "Accelerate development",
    "Recycle non-core capital",
    "Grow recurring revenue",
    "Deepen relationships",
    "Disciplined allocation",
    "Long-term targets",
]


def _bulleted_slide_blocks() -> list[_Block]:
    return [
        _Block(
            text=b,
            page_number=50,
            section_title="Summary",
            content_type="text",
        )
        for b in _BULLETS
    ]


def test_microchunks_are_coalesced_above_threshold() -> None:
    """No coalesced child falls below CHILD_MIN_CHARS."""
    children = _make_children(
        parent_blocks=_bulleted_slide_blocks(),
        parent_id=uuid4(),
        section_title="Summary",
        doc_meta=_doc_meta(),
    )

    assert children, "expected at least one coalesced child"
    assert len(children) < len(_BULLETS), (
        "bullets should be merged, not emitted one-per-child"
    )
    for child in children:
        assert len(child.chunk_text.strip()) >= CHILD_MIN_CHARS, (
            f"microchunk survived coalescing: {child.chunk_text!r}"
        )


def test_coalesced_text_preserves_all_bullets() -> None:
    """Coalescing must not drop any bullet text."""
    children = _make_children(
        parent_blocks=_bulleted_slide_blocks(),
        parent_id=uuid4(),
        section_title="Summary",
        doc_meta=_doc_meta(),
    )
    combined = "\n".join(c.chunk_text for c in children)
    for bullet in _BULLETS:
        assert bullet in combined, f"bullet lost during coalescing: {bullet!r}"


def test_recoverable_boundary_marker_between_bullets() -> None:
    """Merged fragments keep a newline boundary so bullets stay recoverable."""
    children = _make_children(
        parent_blocks=_bulleted_slide_blocks(),
        parent_id=uuid4(),
        section_title="Summary",
        doc_meta=_doc_meta(),
    )
    # At least one child must contain a newline joining two former bullets.
    assert any("\n" in c.chunk_text for c in children)


def test_tables_are_not_coalesced_into_prose() -> None:
    """Atomic table fragments must never merge with adjacent text fragments."""
    candidates: list[tuple[str, ContentType, int]] = [
        ("short text", "text", 1),
        ("R1 | R2", "table", 1),
        ("tiny", "text", 1),
    ]
    out = _coalesce_small_children(candidates)
    table_entries = [c for c in out if c[1] == "table"]
    assert len(table_entries) == 1
    assert table_entries[0][0] == "R1 | R2"


def test_long_children_left_intact() -> None:
    """A child already above the threshold is emitted unchanged."""
    long_text = "x" * (CHILD_MIN_CHARS + 20)
    candidates: list[tuple[str, ContentType, int]] = [(long_text, "text", 3)]
    out = _coalesce_small_children(candidates)
    assert out == candidates


def test_short_children_on_different_pages_not_merged() -> None:
    """Short fragments on different pages must not be merged together."""
    candidates: list[tuple[str, ContentType, int]] = [
        ("alpha", "text", 1),
        ("beta", "text", 2),
    ]
    out = _coalesce_small_children(candidates)
    pages = {c[2] for c in out}
    assert pages == {1, 2}
