"""Tests for the chunker's footnote-numbering restoration helper."""
from __future__ import annotations

from src.ingestion.chunker import _restore_footnote_numbering


def test_renumbered_when_markers_missing():
    """Footnotes header + unnumbered body blocks gain (1)..(N) prefixes."""
    text = "Footnotes\n\nFirst note body.\n\nSecond note body.\n\nThird note body."
    out = _restore_footnote_numbering(text)
    expected = (
        "Footnotes\n\n"
        "(1) First note body.\n\n"
        "(2) Second note body.\n\n"
        "(3) Third note body."
    )
    assert out == expected


def test_idempotent_when_already_numbered():
    """Already-numbered footnote bodies are returned unchanged."""
    text = (
        "Footnotes\n\n"
        "(1) First note body.\n\n"
        "(2) Second note body."
    )
    assert _restore_footnote_numbering(text) == text


def test_unchanged_for_non_footnote_text():
    """Normal narrative text (no Footnotes header) is returned unchanged."""
    text = (
        "Operating Highlights\n\n"
        "Same-store NOI grew 4.2% year over year.\n\n"
        "Occupancy ended the quarter at 94.1%."
    )
    assert _restore_footnote_numbering(text) == text
