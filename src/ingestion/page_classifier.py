"""Two-tier classifier for chunk page-content type.

A keyword pre-filter handles unambiguous boilerplate-legal and index-reference
pages without an LLM call. Borderline pages fall through to a Claude Haiku 4.5
JSON-schema call. The classifier never blocks ingestion: any failure resolves
to ``"unknown"``.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Optional

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
- substantive: financial data, operational metrics, or analytical prose that informs an investment decision. Examples: "NOI grew 12% YoY to $1.4B in Q4 2025." / "Same-store occupancy reached 95.2% across the North American portfolio."
- boilerplate_legal: legal disclaimers, forward-looking-statement warnings, safe-harbor notices, or trademark text. Examples: "This presentation contains forward-looking statements subject to risks." / "Nothing herein constitutes an offer to sell securities."
- index_reference: table of contents entries, section headers with only a page number, or navigation labels. Examples: "Overview ..... 3" / "Appendix B: Supplemental Data"
- cover_page: title slide or cover page content with company name, date, and logo text but no financial substance. Examples: "Digital Realty Trust — Q4 2025 Investor Presentation" / "March 2026  |  Confidential"

Section title: {section_title}

Chunk text:
{chunk_text}

Return JSON with "page_content_class" (one of the four values) and "confidence" (0.0–1.0)."""

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
            _page_content_client = anthropic.Anthropic(api_key=api_key)
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
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        temperature=0.0,
        top_k=1,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
    return json.loads(_strip_json_fences(text))


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

    # Very short chunks with no digits are likely table-of-contents entries.
    if total_tokens < 25 and not any(c.isdigit() for c in chunk_text):
        return "index_reference"

    # LLM tier — only reached when the keyword filter is inconclusive.
    try:
        result = _llm_classify_page_content(chunk_text, section_title)
        if result["confidence"] >= 0.80:
            return result["page_content_class"]
    except Exception:  # noqa: BLE001
        pass

    return "unknown"
