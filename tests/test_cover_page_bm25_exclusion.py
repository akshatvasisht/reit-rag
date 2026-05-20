"""Tests for cover_page BM25 exclusion.

Verifies that chunks classified as cover_page are excluded from BM25
retrieval results, matching the existing boilerplate_legal / index_reference
exclusion behavior.
"""
from __future__ import annotations


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
    import re
    m = re.search(r"NOT IN \(([^)]+)\)", _BOILERPLATE_FILTER, re.IGNORECASE)
    assert m is not None, (
        f"Expected a NOT IN (...) clause in _BOILERPLATE_FILTER but found "
        f"none — the test was vacuously passing before this guard was added:\n"
        f"{_BOILERPLATE_FILTER}"
    )
    excluded = m.group(1)
    assert "substantive" not in excluded, (
        "'substantive' should not be in the exclusion list"
    )


# ---------------------------------------------------------------------------
# 4. unknown-class chunks are intentionally included by the current filter
# ---------------------------------------------------------------------------


def test_unknown_class_not_excluded():
    """unknown chunks must pass the filter while the residual is non-trivial.

    Most recent reclassify pass left ~28.6% of chunks at 'unknown'
    (1,283 / 4,489). Excluding 'unknown' would drop a large share of the
    corpus and harm recall, so the filter stays NULL-tolerant / NOT IN(...)
    and excludes only confidently-classified boilerplate classes. Tighten
    to a 'substantive'-only allow-list once the residual drops below ~5%.
    """
    import re
    m = re.search(r"NOT IN \(([^)]+)\)", _BOILERPLATE_FILTER, re.IGNORECASE)
    assert m is not None
    excluded = m.group(1)
    assert "unknown" not in excluded, (
        "'unknown' must not be in the exclusion list while the residual is "
        "non-trivial; tighten only after classifier coverage improves."
    )
    assert "IS NULL" in _BOILERPLATE_FILTER, (
        "Filter must keep NULL-tolerant clause so chunks predating the "
        "classifier still reach BM25 recall."
    )
