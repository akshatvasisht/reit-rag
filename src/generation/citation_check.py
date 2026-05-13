"""Citation faithfulness check at inference time.

Parses inline citations of the form ``[Company, Doc Type, Month Year, p.N]``
from a generated answer and verifies each one maps to a chunk that was
actually in the retrieved context. Surfaces unsupported citations so the UI
can warn the user — never silently allow them through.

This is *citation-source* faithfulness, not *claim-content* faithfulness.
It verifies "did the LLM cite a real chunk from the context" — it does NOT
verify "does the cited chunk actually support the specific claim." The
latter is measured by claim-level faithfulness metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.prompts import _format_date, _format_doc_type
from src.models import RetrievedChunk


# Regex for `[Company, Doc Type, Month Year, p.N]`.
# - Company: anything except commas and brackets.
# - Doc Type: same.
# - Date: "Month Year" or "Mon Year"; year is 4 digits.
# - Page: `p.N` or `p. N` or `page N`.
# Tolerant of extra whitespace.
_CITATION_RE = re.compile(
    r"\["
    r"\s*(?P<company>[^,\]\[]+?)\s*,"
    r"\s*(?P<doc_type>[^,\]\[]+?)\s*,"
    # `\.?` tolerates an optional trailing period after abbreviated months
    # (for example, "Mar.") if the model varies formatting slightly.
    r"\s*(?P<date>[A-Za-z]+\.?\s+\d{4})\s*,"
    r"\s*(?:p\.?\s*|page\s+)(?P<page>\d+|\?)\s*"
    r"\]",
    re.IGNORECASE,
)


@dataclass
class ParsedCitation:
    """One citation extracted from the answer text."""
    company: str
    doc_type: str
    date: str       # "Month Year" form, normalized
    page: int | None  # None when the LLM emitted "p.?"
    raw: str        # exact substring in the answer


@dataclass
class CitationReport:
    """Result of running `check_citations()` over an answer."""
    total: int
    supported: int
    unsupported: list[ParsedCitation] = field(default_factory=list)

    @property
    def faithfulness_ratio(self) -> float:
        """Supported / total. Returns 1.0 when the answer has zero citations."""
        return 1.0 if self.total == 0 else self.supported / self.total


def parse_citations(answer_text: str) -> list[ParsedCitation]:
    """Extract every `[Company, Doc Type, Month Year, p.N]` from the answer."""
    citations: list[ParsedCitation] = []
    for match in _CITATION_RE.finditer(answer_text):
        page_raw = match.group("page")
        page = None if page_raw == "?" else int(page_raw)
        citations.append(
            ParsedCitation(
                company=match.group("company").strip(),
                doc_type=match.group("doc_type").strip(),
                date=_normalize_date(match.group("date").strip()),
                page=page,
                raw=match.group(0),
            )
        )
    return citations


def check_citations(
    answer_text: str,
    contexts: list[RetrievedChunk],
) -> CitationReport:
    """Verify every citation in *answer_text* matches a chunk in *contexts*.

    Match criteria: company name, doc type, report date, and page number all
    match after normalization. This avoids false support matches when multiple
    versions of the same company document share page numbers.
    """
    citations = parse_citations(answer_text)
    # Build a normalized (company, doc_type, date, page) lookup set.
    index: set[tuple[str, str, str, int | None]] = set()
    for rc in contexts:
        key = (
            rc.chunk.company.strip().lower(),
            _normalize_doc_type(_format_doc_type(rc.chunk.doc_type)),
            _normalize_date(_format_date(rc.chunk.report_date)),
            rc.chunk.page_number,
        )
        index.add(key)

    supported_count = 0
    unsupported: list[ParsedCitation] = []
    for cit in citations:
        key = (
            cit.company.strip().lower(),
            _normalize_doc_type(cit.doc_type),
            _normalize_date(cit.date),
            cit.page,
        )
        if key in index:
            supported_count += 1
        else:
            unsupported.append(cit)

    return CitationReport(
        total=len(citations),
        supported=supported_count,
        unsupported=unsupported,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MONTH_ABBREV = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "sept": "September", "oct": "October",
    "nov": "November", "dec": "December",
}


def _normalize_date(raw: str) -> str:
    """Normalize month-year variants to a canonical "Month YYYY" format.

    Looks up the 3-letter prefix first (covers Jan–Dec), then falls back to the
    4-letter prefix (only relevant for "sept"). The trailing period in "Mar."
    is stripped before lookup.
    """
    parts = raw.split()
    if len(parts) != 2:
        return raw
    month_raw, year = parts
    month_key = month_raw.lower().rstrip(".")
    full = _MONTH_ABBREV.get(month_key[:3]) or _MONTH_ABBREV.get(month_key[:4])
    if full:
        return f"{full} {year}"
    return raw


def _normalize_doc_type(raw: str) -> str:
    """Normalize doc-type strings for robust equality checks."""
    return " ".join(raw.replace("_", " ").split()).strip().lower()
