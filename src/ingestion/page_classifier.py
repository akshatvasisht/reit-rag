"""Two-tier classifier for chunk page-content type.

A keyword pre-filter handles unambiguous boilerplate-legal and index-reference
pages without an LLM call. Borderline pages fall through to a Claude Haiku 4.5
JSON-schema call. The classifier never blocks ingestion: any failure resolves
to ``"unknown"``.
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING, Optional

from src.ingestion.json_response import extract_json_object

if TYPE_CHECKING:
    import anthropic


# JSON schema constraining the LLM response to a valid class + confidence pair.
PAGE_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "page_content_class": {
            "type": "string",
            "enum": ["substantive", "boilerplate_legal", "index_reference", "cover_page"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["page_content_class", "confidence"],
    "additionalProperties": False,
}

# A line that reads as one item in a list / bullet point: a leading bullet
# glyph or dash marker, or a numbered-list prefix. Several such lines indicate
# a multi-point slide (strengths, priorities, action plan) — substantive
# content, not a one-line index entry.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[•●▪‣⁃∙*\-–—]|\d+[.)])\s+")

# Phrases that strongly indicate SEC/legal boilerplate. Three or more signals
# in a single chunk is a reliable indicator without needing an LLM call.
DEFINITE_BOILERPLATE_SIGNALS = [
    "forward-looking statements",
    "safe harbor",
    "securities act",
    "no assurance",
    "trademark",
    "registered trademark",
    "this presentation does not constitute",
    "not an offer to sell",
]

# Prompt: four one-line class definitions with two concrete examples each.
_PAGE_CONTENT_PROMPT = """\
Classify the following chunk of text from an investor presentation PDF.

Classes:
- substantive: financial data, operational metrics, strategy/priorities, or analytical prose that informs an investment decision. A slide listing multiple substantive bullet points — an action plan, strategy summary, set of priorities, list of initiatives, or strengths/objectives — is substantive even when terse. Examples: "NOI grew 12% YoY to $1.4B in Q4 2025." / "Current Strengths: premier asset quality; strong balance sheet; embedded growth. Action Plan: accelerate development; recycle non-core capital; grow recurring revenue."
- boilerplate_legal: legal disclaimers, forward-looking-statement warnings, safe-harbor notices, or trademark text. Examples: "This presentation contains forward-looking statements subject to risks." / "Nothing herein constitutes an offer to sell securities."
- index_reference: ONLY a genuine table of contents, page index, or cross-reference listing — section names paired with page numbers, or pure navigation labels with no substantive content of their own. A slide that merely has a heading followed by real content is NOT index_reference. Examples: "Overview ..... 3" / "Financials ..... 12   Appendix ..... 27".
- cover_page: title slide or cover page content with company name, date, and logo text but no financial substance. Examples: "Digital Realty Trust — Q4 2025 Investor Presentation" / "March 2026  |  Confidential"

Section title: {section_title}

Chunk text:
{chunk_text}

Return ONLY a JSON object (no markdown fences, no commentary) with "page_content_class" (one of the four values) and "confidence" (0.0–1.0)."""

# Matches a leading ```json or ``` fence (with optional whitespace) and the
# matching trailing ``` fence; tolerant of upper/lowercase language tags.
_FENCE_RE = re.compile(
    r"\A\s*```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)\r?\n?[ \t]*```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _strip_json_fences(text: str) -> str:
    """Strip a surrounding ```json / ``` markdown fence pair if both are present."""
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


# Client-level retry budget for API errors (429 / overloaded / 5xx); higher
# than the SDK default of 2 so a burst of classification calls during a large
# ingest rides out transient rate-limiting via the SDK's exponential backoff.
_LLM_CLIENT_MAX_RETRIES = 6
# Application-level retries for empty / non-JSON response bodies, which the SDK
# does not treat as errors (the HTTP call succeeded; the body was just
# unusable). Observed during large ingest runs as a blank classification that
# would otherwise collapse straight to the 'unknown' fallback.
_LLM_PARSE_ATTEMPTS = 3
_LLM_PARSE_BACKOFF_S = 0.5

# Lazy singleton Anthropic client for page content classification.
_page_content_client: Optional["anthropic.Anthropic"] = None


def _get_page_content_client():  # type: ignore[return]
    """Return a module-level Anthropic client, creating it on first use."""
    global _page_content_client
    if _page_content_client is None:
        try:
            import anthropic  # noqa: PLC0415
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            _page_content_client = anthropic.Anthropic(
                api_key=api_key, max_retries=_LLM_CLIENT_MAX_RETRIES
            )
        except Exception:  # noqa: BLE001
            return None
    return _page_content_client


def _llm_classify_page_content(chunk_text: str, section_title: str) -> dict:
    """Call Claude Haiku 4.5 with a JSON-schema output to classify page content.

    Returns a dict with ``page_content_class`` (str) and ``confidence`` (float).
    Raises on any API or import error so the caller falls back to ``"unknown"``.
    """
    client = _get_page_content_client()
    if client is None:
        raise RuntimeError("Anthropic client unavailable")

    prompt = _PAGE_CONTENT_PROMPT.format(
        section_title=section_title or "(none)",
        chunk_text=chunk_text[:2000],  # cap to avoid exceeding context on very long chunks
    )
    last_err: Optional[Exception] = None
    for attempt in range(_LLM_PARSE_ATTEMPTS):
        try:
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
                raise ValueError("empty classification response")
            return extract_json_object(text)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < _LLM_PARSE_ATTEMPTS - 1:
                time.sleep(_LLM_PARSE_BACKOFF_S * (2 ** attempt))
    assert last_err is not None
    raise last_err


def classify_page_content(chunk_text: str, section_title: str) -> str:
    """Return the content class for a chunk using keyword signals then LLM fallback.

    The two-tier approach avoids unnecessary API calls: obvious boilerplate and
    very short index entries are detected via keyword counting and token length.
    The LLM is invoked only when the keyword pre-filter is inconclusive.

    Args:
        chunk_text: Raw text of the chunk to classify.
        section_title: Section heading associated with the chunk (may be empty).

    Returns:
        One of ``"substantive"``, ``"boilerplate_legal"``, ``"index_reference"``,
        ``"cover_page"``, or ``"unknown"``.  ``"unknown"`` is returned when the
        LLM is unavailable or returns low confidence.
    """
    text_lower = chunk_text.lower()
    total_tokens = len(chunk_text.split())

    signal_count = sum(1 for s in DEFINITE_BOILERPLATE_SIGNALS if s in text_lower)

    # Three or more boilerplate signals in any chunk → boilerplate without LLM.
    if signal_count >= 3:
        return "boilerplate_legal"

    # Two signals in a short chunk (under 150 words) → also reliable enough.
    if signal_count >= 2 and total_tokens < 150:
        return "boilerplate_legal"

    # A multi-point slide (strengths, priorities, action plan) is substantive
    # content, never an index entry, and must reach the LLM tier rather than
    # being short-circuited. Detect it two ways: explicit list glyphs, or simply
    # several non-empty lines — upstream coalescing joins per-bullet fragments
    # with newlines, and the source decks frequently strip the bullet glyphs, so
    # the line count is a more reliable signal than the glyph pattern alone.
    non_empty_lines = [line for line in chunk_text.splitlines() if line.strip()]
    list_item_lines = sum(1 for line in non_empty_lines if _LIST_ITEM_RE.match(line))
    multi_point = list_item_lines >= 2 or len(non_empty_lines) >= 2

    # Very short, single-line chunks with no digits and no list structure are
    # likely table-of-contents entries or navigation labels.
    if (
        total_tokens < 25
        and not multi_point
        and not any(c.isdigit() for c in chunk_text)
    ):
        return "index_reference"

    # LLM tier — only reached when the keyword filter is inconclusive.
    try:
        result = _llm_classify_page_content(chunk_text, section_title)
        if result["confidence"] >= 0.80:
            return result["page_content_class"]
    except Exception:  # noqa: BLE001
        pass

    return "unknown"
