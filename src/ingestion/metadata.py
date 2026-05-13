"""Document-level metadata extraction for the REIT RAG corpus.

Provides a hardcoded registry for the 10 known PDFs and a filename-based
lookup that tolerates variation in capitalisation, separators, and date
format.  Also provides a lightweight content-type classifier for the
parser's per-chunk heuristics.
"""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

from src.models import ContentType, DocumentMeta


# ---------------------------------------------------------------------------
# Corpus registry
# ---------------------------------------------------------------------------
# Each entry covers one *logical* document group.
# Entries without version multiplicity use "versions": None and carry their
# metadata at the top level.  Entries with version multiplicity (DLR) carry
# a "versions" dict keyed by an internal version id; the top-level company /
# ticker / doc_type fields are shared across all versions.
#
# Schema per entry:
#   keywords          – substrings that must appear (OR) in the stem to match
#                       this company group
#   doc_type          – "investor_presentation" | "thematic_report"
#   company           – display name
#   ticker            – exchange ticker
#   versions          – None  → single version, metadata at top level
#                       dict  → {version_id: {version_keywords, report_date,
#                                              period_covered, doc_version,
#                                              secondary_keywords}}
#   report_date       – "YYYY-MM" (only when versions is None)
#   period_covered    – human period label (only when versions is None)
#   doc_version       – same as report_date (only when versions is None)
#
# Matching rules
#   1. Stem is lower-cased.
#   2. An entry matches if ANY keyword appears as a substring of the stem.
#   3. For multi-version entries, the selected version is the first whose
#      version_keywords all appear in the stem. If no version keyword
#      matches, a ValueError is raised intentionally.
#   4. BXP has two *separate* documents that share the same company but
#      differ in doc_type / period_covered; each is its own top-level entry
#      with a discriminating secondary keyword.
# ---------------------------------------------------------------------------

CORPUS_REGISTRY: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Digital Realty — two versions, version-conflict test case
    # ------------------------------------------------------------------
    {
        "keywords": ["dlr", "digital realty", "digital_realty"],
        "company": "Digital Realty",
        "ticker": "DLR",
        "doc_type": "investor_presentation",
        "versions": {
            "dec-2025": {
                "version_keywords": ["dec", "december", "2025-12", "2025_12"],
                "report_date": "2025-12",
                "period_covered": "Q3 2025",
                "doc_version": "2025-12",
            },
            "mar-2026": {
                "version_keywords": ["mar", "march", "2026-03", "2026_03"],
                "report_date": "2026-03",
                "period_covered": "Q4 2025",
                "doc_version": "2026-03",
            },
        },
    },
    # ------------------------------------------------------------------
    # BXP — morning session deck (thematic/event, same company)
    # ------------------------------------------------------------------
    {
        "keywords": ["bxp"],
        "secondary_keywords": ["morning", "session", "morning_session"],
        "company": "BXP",
        "ticker": "BXP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2025-12",
        "period_covered": "Q4 2025",
        "doc_version": "2025-12",
    },
    # ------------------------------------------------------------------
    # BXP — investor presentation
    # ------------------------------------------------------------------
    {
        "keywords": ["bxp"],
        "secondary_keywords": ["q4", "investor", "presentation", "2025"],
        "company": "BXP",
        "ticker": "BXP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2025-12",
        "period_covered": "Q4 2025",
        "doc_version": "2025-12",
    },
    # ------------------------------------------------------------------
    # PSA (Public Storage) — company update (March 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["psa", "public storage", "public_storage"],
        "secondary_keywords": ["update", "company_update", "company update"],
        "company": "Public Storage",
        "ticker": "PSA",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # PSA (Public Storage) — merger presentation (March 16, 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["psa", "public storage", "public_storage"],
        "secondary_keywords": ["merger", "acquisition", "acq"],
        "company": "Public Storage",
        "ticker": "PSA",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # VICI Properties — investor presentation
    # ------------------------------------------------------------------
    {
        "keywords": ["vici"],
        "company": "VICI Properties",
        "ticker": "VICI",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # Realty Income — investor presentation (February 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["realty income", "realty incom", "realty_income", "rlt", "o_corp"],
        "company": "Realty Income",
        "ticker": "O",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-02",
        "period_covered": "Q4 2025",
        "doc_version": "2026-02",
    },
    # ------------------------------------------------------------------
    # EastGroup Properties — roadshow deck (February 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["eastgroup", "east_group", "egp"],
        "company": "EastGroup Properties",
        "ticker": "EGP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-02",
        "period_covered": "2026",
        "doc_version": "2026-02",
    },
    # ------------------------------------------------------------------
    # Simon Property Group — thematic report (November 2018 colophon)
    # ------------------------------------------------------------------
    {
        "keywords": ["simon", "spg"],
        "company": "Simon Property Group",
        "ticker": "SPG",
        "doc_type": "thematic_report",
        "versions": None,
        "report_date": "2018-11",
        "period_covered": "2017-2018",
        "doc_version": "2018-11",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem(pdf_path: str | PurePath) -> str:
    """Return the lower-cased filename stem (no extension) of *pdf_path*."""
    return PurePath(pdf_path).stem.lower()


def _keywords_hit(stem: str, keywords: list[str]) -> bool:
    """Return True if any keyword in *keywords* appears in *stem*."""
    return any(kw in stem for kw in keywords)


def _secondary_keywords_hit(stem: str, secondary_keywords: list[str]) -> bool:
    """Return True if any secondary keyword appears in *stem*."""
    return any(kw in stem for kw in secondary_keywords)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_document_meta(pdf_path: str | PurePath) -> DocumentMeta:
    """Derive a DocumentMeta for *pdf_path* using the hardcoded corpus registry.

    Args:
        pdf_path: Absolute or relative path to the source PDF.  Only the
                  filename stem is used; the file need not exist yet.

    Returns:
        A fully populated DocumentMeta instance.

    Raises:
        ValueError: If no registry entry matches the filename, or if a
                    multi-version entry is matched but no version keyword
                    disambiguates the specific version.
    """
    stem = _stem(pdf_path)
    source = str(pdf_path)

    # Step 1 — find candidate entries (may be multiple for BXP / PSA which
    # share company keywords across two documents each).
    candidates = [
        entry for entry in CORPUS_REGISTRY
        if _keywords_hit(stem, entry["keywords"])
    ]

    if not candidates:
        raise ValueError(
            f"No corpus registry entry matches filename stem '{stem}'. "
            f"Add an entry to CORPUS_REGISTRY in src/ingestion/metadata.py "
            f"or rename the file to include a recognisable keyword "
            f"(e.g. 'dlr', 'bxp', 'psa', 'vici', 'egp', 'simon')."
        )

    # Step 2 — if multiple candidates share the same company keyword (BXP,
    # PSA), prefer the one whose secondary_keywords also appear in the stem.
    # Fall back to the first candidate if no secondary keyword matches at all
    # (single-entry companies).
    if len(candidates) > 1:
        secondary_matches = [
            entry for entry in candidates
            if "secondary_keywords" in entry
            and _secondary_keywords_hit(stem, entry["secondary_keywords"])
        ]
        entry = secondary_matches[0] if secondary_matches else candidates[0]
    else:
        entry = candidates[0]

    # Step 3 — resolve version for multi-version entries (DLR only).
    if entry["versions"] is not None:
        resolved_version: dict[str, Any] | None = None
        for _version_id, version_data in entry["versions"].items():
            if _keywords_hit(stem, version_data["version_keywords"]):
                resolved_version = version_data
                break

        if resolved_version is None:
            version_hints = {
                vid: vdata["version_keywords"]
                for vid, vdata in entry["versions"].items()
            }
            raise ValueError(
                f"Filename stem '{stem}' matches company '{entry['company']}' "
                f"but no version keyword disambiguates the document version. "
                f"Expected one of: {version_hints}. "
                f"Include a date token such as 'dec2025', '2025-12', 'mar2026', "
                f"or '2026-03' in the filename."
            )

        return DocumentMeta(
            company=entry["company"],
            ticker=entry["ticker"],
            doc_type=entry["doc_type"],
            report_date=resolved_version["report_date"],
            period_covered=resolved_version["period_covered"],
            doc_version=resolved_version["doc_version"],
            source_path=source,
        )

    # Single-version entry — metadata lives at the top level.
    return DocumentMeta(
        company=entry["company"],
        ticker=entry["ticker"],
        doc_type=entry["doc_type"],
        report_date=entry["report_date"],
        period_covered=entry["period_covered"],
        doc_version=entry["doc_version"],
        source_path=source,
    )


# Precompiled marker pattern for classify_content_type
_CHART_MARKER_RE = re.compile(
    r"\b(figure|chart|exhibit|graph|illustration)\b",
    re.IGNORECASE,
)


def classify_content_type(
    text: str,
    has_table: bool,
    has_chart_marker: bool,
) -> ContentType:
    """Heuristically classify a chunk's content type.

    Args:
        text:             Raw extracted text of the chunk.
        has_table:        True if Docling detected a table structure in this
                          chunk.
        has_chart_marker: True if the parser found a chart/figure element
                          (e.g. a Docling Picture item) associated with this
                          chunk.

    Returns:
        One of "text", "table", "chart_caption", or "mixed".
    """
    # Explicit structural signals take priority.
    if has_table and has_chart_marker:
        return "mixed"
    if has_table:
        return "table"
    if has_chart_marker:
        return "chart_caption"

    # Heuristic: high digit density combined with many newlines → tabular data
    # that the parser did not tag as a formal table.
    stripped = text.strip()
    if stripped:
        digit_chars = sum(1 for ch in stripped if ch.isdigit())
        digit_density = digit_chars / len(stripped)
        newline_count = stripped.count("\n")
        if digit_density > 0.15 and newline_count > 4:
            return "table"

    # Heuristic: header tokens for figures/charts without substantial prose →
    # this is a caption with no extractable data values.
    if _CHART_MARKER_RE.search(text):
        # If the non-whitespace content is very short, treat it as a label, not
        # prose that happened to mention a chart.
        non_ws = re.sub(r"\s+", "", text)
        if len(non_ws) < 120:
            return "chart_caption"

    return "text"
