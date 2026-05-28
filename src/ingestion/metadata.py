"""Document-level metadata extraction for the REIT RAG corpus.

Provides a registry for the known PDFs and a filename-based lookup that
tolerates variation in capitalisation, separators, and date format.  Also
provides a lightweight content-type classifier for the parser's per-chunk
heuristics, and an LLM-backed subtype classifier.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import PurePath
from typing import Any, Optional

from src.ingestion.json_response import extract_json_object
from src.models import ContentType, DocumentMeta

logger = logging.getLogger(__name__)

# Re-exported from src.corpus_registry for backward compatibility.
from src.corpus_registry import CORPUS_REGISTRY  # noqa: E402


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
            f"Update CORPUS_REGISTRY in src.corpus_registry or rename the file "
            f"to include a recognisable keyword "
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


# ---------------------------------------------------------------------------
# Document subtype classifier
# ---------------------------------------------------------------------------

# Enum values that the LLM may return.
_VALID_SUBTYPES = {
    "quarterly_investor_deck",
    "investor_day_session",
    "merger_presentation",
    "company_update",
    "annual_supplement",
    "thematic_report",
    "roadshow_deck",
    "unknown",
}

# Minimum LLM confidence to accept a classification; anything below falls
# back to "unknown" rather than propagating a low-confidence guess.
_SUBTYPE_CONFIDENCE_THRESHOLD = 0.75

# JSON schema that constrains the LLM response shape.
SUBTYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_subtype": {
            "type": "string",
            "enum": list(_VALID_SUBTYPES),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["doc_subtype", "confidence"],
    "additionalProperties": False,
}

# One-shot prompt with per-subtype definitions and examples.
_SUBTYPE_PROMPT = """Classify the following REIT document into exactly one subtype.

Subtypes and their definitions:
- "quarterly_investor_deck": Standard quarterly earnings/supplemental investor presentation (e.g., Q4 2025 Investor Presentation)
- "investor_day_session": Investor Day or analyst day session deck (e.g., morning session, afternoon workshop)
- "merger_presentation": Merger, acquisition, or strategic combination deck
- "company_update": General company update or overview not tied to a specific quarter (e.g., "Company Update March 2026")
- "annual_supplement": Annual data supplement, statistical supplement, or full-year data book
- "thematic_report": Thematic, sector, or ESG report not primarily an earnings deck (e.g., "Retail Real Estate Outlook 2018")
- "roadshow_deck": Equity or debt roadshow marketing deck
- "unknown": Cannot be determined from available information

Document details:
  filename: {filename}
  doc_type: {doc_type}
  first page text (truncated to 500 tokens):
  ---
  {first_page_text}
  ---

Return ONLY a JSON object (no markdown fences, no commentary) with:
  "doc_subtype": one of the values above
  "confidence": float in [0.0, 1.0]"""

# See page_classifier for the rationale: ride out burst rate-limiting via the
# SDK's retries, and retry empty/non-JSON bodies at the application level.
_SUBTYPE_CLIENT_MAX_RETRIES = 6
_SUBTYPE_PARSE_ATTEMPTS = 3
_SUBTYPE_PARSE_BACKOFF_S = 0.5

_subtype_client: Optional[Any] = None


def _get_subtype_client() -> Any:
    """Return a module-level Anthropic client, creating it on first use."""
    global _subtype_client
    if _subtype_client is None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _subtype_client = Anthropic(
            api_key=api_key, max_retries=_SUBTYPE_CLIENT_MAX_RETRIES
        )
    return _subtype_client


def classify_doc_subtype(
    first_page_text: str,
    filename: str,
    doc_type: str,
) -> str:
    """Classify a document's granular subtype using Claude Haiku.

    Sends the first ~500 tokens of page-one text, the filename, and the
    coarse doc_type to the model and returns a fine-grained subtype string.
    Returns ``"unknown"`` when confidence is below 0.75 or when the API call
    fails — callers never need to handle exceptions from this function.

    Args:
        first_page_text: Text from the document's first page (may be truncated).
        filename: The source PDF filename, used as a strong hint.
        doc_type: Coarse document type already assigned (e.g. "investor_presentation").

    Returns:
        One of the values in ``_VALID_SUBTYPES``.
    """
    # Truncate first-page text to roughly 500 tokens (≈2000 chars at ~4 chars/token).
    truncated = first_page_text[:2000]

    prompt = _SUBTYPE_PROMPT.format(
        filename=filename,
        doc_type=doc_type,
        first_page_text=truncated,
    )

    last_err: Optional[Exception] = None
    for attempt in range(_SUBTYPE_PARSE_ATTEMPTS):
        try:
            client = _get_subtype_client()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0.0,
                top_k=1,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip()
            if not text:
                raise ValueError("empty subtype response")
            result = extract_json_object(text)
            subtype = result.get("doc_subtype", "unknown")
            confidence = float(result.get("confidence", 0.0))

            if confidence < _SUBTYPE_CONFIDENCE_THRESHOLD or subtype not in _VALID_SUBTYPES:
                return "unknown"
            return subtype

        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < _SUBTYPE_PARSE_ATTEMPTS - 1:
                time.sleep(_SUBTYPE_PARSE_BACKOFF_S * (2 ** attempt))

    logger.debug(
        "classify_doc_subtype failed for %s after retries; returning 'unknown' (%s)",
        filename,
        last_err,
    )
    return "unknown"


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
