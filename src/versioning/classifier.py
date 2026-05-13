"""Temporal-intent classification for version-aware retrieval."""

from __future__ import annotations

import re
from typing import Literal

TemporalIntent = Literal["latest", "historical", "comparison"]

# Comparison triggers that are specific enough to avoid common false positives.
# Bare matches like "between" or "difference" are intentionally avoided because
# they frequently appear in non-temporal questions (e.g., "difference between
# NOI and FFO") and must not force cross-version retrieval.
_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|delta)\b"
    r"|"
    r"\bchanged?\s+(?:between|from|vs\.?|versus)\b"
    r"|"
    r"\bfrom\s+.+?\s+to\s+.+?\b",
    re.IGNORECASE,
)

# Explicit temporal "between ... and/to ..." pattern, for example:
# "between December and March", "between Q4 and Q1".
_TEMPORAL_TERM = (
    r"(?:"
    r"q[1-4]\s*20\d{2}"
    r"|"
    r"20\d{2}-(?:0[1-9]|1[0-2])"
    r"|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept(?:ember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)(?:\s+20\d{2})?"
    r")"
)
_BETWEEN_TEMPORAL_RE = re.compile(
    rf"\bbetween\s+{_TEMPORAL_TERM}\s+(?:and|to)\s+{_TEMPORAL_TERM}\b",
    re.IGNORECASE,
)

# Patterns that signal a specific historical period (when not a comparison).
# Matches month+year forms, quarter+year forms, explicit year references,
# "last year", and YYYY-MM forms.
_HISTORICAL_RE = re.compile(
    r"""
    (?:
        (?:january|february|march|april|may|june|
           july|august|september|october|november|december)
        \s+20\d{2}          # month + year
    )
    |
    (?:q[1-4]\s*20\d{2})    # quarter + year
    |
    (?:\bin\s+20\d{2}\b)    # explicit year phrase
    |
    \blast\s+year\b         # "last year"
    |
    \b20\d{2}\b  # bare year
    |
    20\d{2}-(?:0[1-9]|1[0-2])  # YYYY-MM
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Period mentions that may appear historical but are treated as latest
# reporting periods for this corpus. A query that mentions only these should
# classify as "latest" because it targets the current snapshot rather than a
# backward-looking request.
#
# Derived from CORPUS_REGISTRY so date corrections in one place propagate
# without manual edits here.
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _derive_latest_period_overrides() -> tuple[str, ...]:
    """Build the latest-period set from the corpus registry.

    For every registered document (and every version of multi-version entries)
    collect: ``report_date`` in both ``YYYY-MM`` and ``Month YYYY`` forms, plus
    ``period_covered`` verbatim. All values are lower-cased for matching.
    """
    from src.ingestion.metadata import CORPUS_REGISTRY  # local to avoid cycles

    out: set[str] = set()
    for entry in CORPUS_REGISTRY:
        versions = entry.get("versions")
        sources = list(versions.values()) if versions else [entry]
        for src in sources:
            rd = src.get("report_date")
            if rd:
                out.add(rd.lower())
                try:
                    year, month = rd.split("-")
                    midx = int(month) - 1
                    if 0 <= midx < 12:
                        out.add(f"{_MONTH_NAMES[midx]} {year}")
                except (ValueError, IndexError):
                    pass
            pc = src.get("period_covered")
            if pc:
                out.add(pc.lower())
    return tuple(sorted(out))


_LATEST_PERIOD_OVERRIDES = _derive_latest_period_overrides()


def _derive_max_corpus_year() -> str:
    """Largest 4-digit year present anywhere in the corpus's date strings."""
    years = {
        p[:4]
        for p in _LATEST_PERIOD_OVERRIDES
        if len(p) >= 4 and p[:4].isdigit()
    }
    return max(years) if years else "0000"


_CORPUS_MAX_YEAR = _derive_max_corpus_year()


def classify_intent(query: str) -> TemporalIntent:
    """Classify the temporal intent of a retrieval query.

    Args:
        query: The raw user query string.

    Returns:
        ``"comparison"`` if the query asks to compare versions or periods,
        ``"historical"`` if it targets a specific past period without
        requesting a comparison, or ``"latest"`` (default) otherwise.
        Period mentions that are the corpus's latest reporting period
        (e.g. "Q4 2025") classify as ``"latest"`` even though they
        contain explicit dates.
    """
    lowered = query.lower()

    if _COMPARISON_RE.search(lowered) or _BETWEEN_TEMPORAL_RE.search(lowered):
        return "comparison"

    match = _HISTORICAL_RE.search(lowered)
    if match:
        matched = match.group(0).strip().lower()
        # If the only matched period is a known "latest" period and no other
        # historical period appears, the query is actually about current data.
        if any(latest in matched for latest in _LATEST_PERIOD_OVERRIDES):
            remaining = lowered.replace(matched, "", 1)
            if not _HISTORICAL_RE.search(remaining):
                return "latest"
        # A bare 4-digit year strictly later than any corpus date is a
        # forward-looking projection (e.g. "outlook for 2030"), not a
        # backward-looking query. Treat as latest.
        if matched.isdigit() and len(matched) == 4 and matched > _CORPUS_MAX_YEAR:
            return "latest"
        return "historical"

    return "latest"
