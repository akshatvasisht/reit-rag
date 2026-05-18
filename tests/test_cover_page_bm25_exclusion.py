"""Tests for cover_page BM25 exclusion (fix 2f).

Verifies that chunks classified as cover_page are excluded from BM25
retrieval results, matching the existing boilerplate_legal / index_reference
exclusion behavior.
"""
from __future__ import annotations

import pytest

from src.retrieval.bm25 import _BOILERPLATE_FILTER


# ---------------------------------------------------------------------------
# 1. cover_page appears in the boilerplate filter string
# ---------------------------------------------------------------------------


def test_cover_page_in_boilerplate_filter():
    """The SQL boilerplate filter must exclude cover_page."""
    assert "cover_page" in _BOILERPLATE_FILTER, (
        f"Expected 'cover_page' in _BOILERPLATE_FILTER but got:\n{_BOILERPLATE_FILTER}"
    )


def test_boilerplate_legal_still_excluded():
    """Regression: boilerplate_legal exclusion must still be present."""
    assert "boilerplate_legal" in _BOILERPLATE_FILTER


def test_index_reference_still_excluded():
    """Regression: index_reference exclusion must still be present."""
    assert "index_reference" in _BOILERPLATE_FILTER


# ---------------------------------------------------------------------------
# 2. The filter uses NOT IN so all three classes are excluded together
# ---------------------------------------------------------------------------


def test_filter_uses_not_in():
    """The filter uses NOT IN, so adding cover_page extends the exclusion set."""
    assert "NOT IN" in _BOILERPLATE_FILTER.upper(), (
        "Expected NOT IN syntax in _BOILERPLATE_FILTER"
    )


# ---------------------------------------------------------------------------
# 3. The filter keeps substantive and unknown classes
# ---------------------------------------------------------------------------


def test_substantive_not_in_filter():
    """substantive must NOT appear in the exclusion list."""
    # The filter is a NOT IN list; 'substantive' should not be one of its members.
    # Simple check: substantive should not appear inside the NOT IN (...) clause.
    import re
    # Extract the contents of the NOT IN (...) clause.
    m = re.search(r"NOT IN \(([^)]+)\)", _BOILERPLATE_FILTER, re.IGNORECASE)
    if m:
        excluded = m.group(1)
        assert "substantive" not in excluded, (
            "'substantive' should not be in the exclusion list"
        )
