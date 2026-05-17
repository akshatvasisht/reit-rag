"""Citation faithfulness check at inference time.

Parses inline citations of the form ``[Company, Doc Type, Month Year, p.N]``
from a generated answer and verifies each one maps to a chunk that was
actually in the retrieved context. Surfaces unsupported citations so the UI
can warn the user — never silently allow them through.

This is *citation-source* faithfulness, not *claim-content* faithfulness.
It verifies "did the LLM cite a real chunk from the context" — it does NOT
verify "does the cited chunk actually support the specific claim." The
latter is measured by claim-level faithfulness metrics.

`check_numeric_consistency` extends source-faithfulness with value-level
verification: for every numeric claim, it confirms that the stated value
actually appears (or is within 1% rounding tolerance) in the cited chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.prompts import _format_date, _format_doc_type
from src.models import Claim, RetrievedChunk, StructuredAnswer


# Matches `[Company, Doc Type, Month Year, p.N]` citations; tolerant of an
# optional trailing period after abbreviated months and of extra whitespace.
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
    # Issues from numeric value verification; empty when not yet checked or all clean.
    numeric_mismatches: list[dict] = field(default_factory=list)

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


# Matches financial value patterns only — dollar amounts, multiples, percentages,
# and basis points.  Deliberately excludes bare integers so that page numbers,
# years, and chunk IDs are not treated as financial values.
FINANCIAL_NUMBER_RE = re.compile(
    r"""
    (?:
        \$[\d,]+(?:\.\d+)?[BMK]?     # Dollar amounts: $4.9B, $16,774,702
      | \d+(?:\.\d+)?[x]              # Multiples: 4.9x, 5.1x
      | \d+(?:\.\d+)?\s*%             # Percentages: 94.1%, 6.2%
      | \d+(?:\.\d+)?\s*(?:bps|bp)    # Basis points
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Unit categories used to prevent cross-unit comparison (e.g. 94% vs $94).
_UNIT_DOLLAR = "dollar"
_UNIT_MULTIPLE = "multiple"
_UNIT_PERCENT = "percent"
_UNIT_BPS = "bps"


def _classify_unit(value: str) -> str | None:
    """Return the unit category for a matched financial value string, or None."""
    v = value.strip().lower()
    if v.startswith("$"):
        return _UNIT_DOLLAR
    if v.endswith("x"):
        return _UNIT_MULTIPLE
    if v.endswith("%"):
        return _UNIT_PERCENT
    if v.endswith("bps") or v.endswith("bp"):
        return _UNIT_BPS
    return None


def _parse_numeric(value: str) -> float | None:
    """Strip currency/unit decorators and parse to float; return None on failure."""
    v = value.strip().lower()
    # Remove unit suffixes in order from most specific to least
    for suffix in ("bps", "bp", "%", "x", "b", "m", "k"):
        if v.endswith(suffix):
            v = v[: -len(suffix)]
            break
    # Remove leading $
    v = v.lstrip("$")
    # Remove thousands separators
    v = v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def _normalize_value(value: str) -> str:
    """Strip whitespace and lowercase; remove insignificant trailing zeros."""
    v = value.strip().lower()
    # Collapse internal whitespace between digit and unit (e.g. "94.1 %" → "94.1%")
    v = re.sub(r"(\d)\s+(%|bps|bp|x)", r"\1\2", v)
    return v


def _chunk_key(rc: RetrievedChunk) -> tuple[str, str, int | None]:
    """Canonical (company, date, page) key for a retrieved chunk."""
    return (
        rc.chunk.company.strip().lower(),
        _normalize_date(_format_date(rc.chunk.report_date)),
        rc.chunk.page_number,
    )


def _claim_lookup_key(claim: Claim) -> tuple[str, str, int]:
    """Canonical (company, date, page) key derived from claim citation fields."""
    return (
        claim.citation_company.strip().lower(),
        _normalize_date(claim.citation_date),
        claim.citation_page,
    )


def check_numeric_consistency(
    answer: StructuredAnswer,
    contexts: list[RetrievedChunk],
) -> list[dict]:
    """Verify that numeric values in each claim appear in the cited chunk.

    For each claim with a non-empty ``value``, the function:
    1. Locates the retrieved chunk that matches the claim's citation fields.
    2. Extracts financial values from the chunk text using FINANCIAL_NUMBER_RE.
    3. Checks whether the claimed value is present (exact normalized match) or
       within 1% of any extracted chunk value (rounding tolerance), taking unit
       category into account so that percentages and dollar amounts are never
       treated as equivalent even if their digits are close.

    Returns a list of issue dicts.  An empty list means full numeric consistency.
    Abstaining answers are skipped entirely — the caller should not invoke this
    function for abstentions, but the guard here makes the function safe to call
    unconditionally.
    """
    if answer.abstain:
        return []

    # Index retrieved chunks by (company, date, page) for O(1) lookup.
    chunk_index: dict[tuple[str, str, int | None], RetrievedChunk] = {}
    for rc in contexts:
        key = _chunk_key(rc)
        chunk_index[key] = rc

    issues: list[dict] = []

    for claim in answer.claims:
        if not claim.value:
            continue  # Skip non-numeric claims

        lookup_key = _claim_lookup_key(claim)
        matched_rc = chunk_index.get(lookup_key)

        if matched_rc is None:
            # No matching chunk — already caught by check_citations, but record
            # here too so the numeric report is self-contained.
            issues.append({
                "type": "unsupported_citation",
                "claim": claim.text,
                "value": claim.value,
            })
            continue

        chunk_text = matched_rc.chunk.chunk_text
        raw_matches = FINANCIAL_NUMBER_RE.findall(chunk_text)
        chunk_values = [m.strip() for m in raw_matches]

        claim_norm = _normalize_value(claim.value)
        claim_unit = _classify_unit(claim_norm)
        claim_float = _parse_numeric(claim_norm)

        found = False
        for cv in chunk_values:
            cv_norm = _normalize_value(cv)

            # Exact normalized match
            if claim_norm == cv_norm:
                found = True
                break

            # Rounding tolerance: only compare values of the same unit category
            cv_unit = _classify_unit(cv_norm)
            if claim_unit is not None and claim_unit == cv_unit and claim_float is not None:
                cv_float = _parse_numeric(cv_norm)
                if cv_float is not None and cv_float != 0:
                    if abs(claim_float - cv_float) / abs(cv_float) <= 0.01:
                        found = True
                        break

        if not found:
            issues.append({
                "type": "numeric_mismatch",
                "claim": claim.text,
                "claimed_value": claim.value,
                "chunk_values_found": chunk_values,
                "chunk_id": str(matched_rc.chunk.id),
            })

    return issues


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


# ---------------------------------------------------------------------------
# Out-of-corpus entity attribution guard
# ---------------------------------------------------------------------------


def check_ooc_entity_attribution(
    answer: StructuredAnswer,
    query: str,
    corpus_companies: list[str],
) -> list[dict]:
    """Detect claims that attribute a specific value to an out-of-corpus entity.

    When a query names a company that does not appear as a primary corpus
    subject, a retrieved chunk from a different company may mention that
    entity incidentally (e.g., in a peer comparison table). If the model
    then attributes a specific figure to the out-of-corpus entity while
    citing the in-corpus document, the citation check passes but the answer
    is contextually wrong.

    Uses ``extract_entity_candidates`` rather than ``extract_companies`` so
    that entity names that are NOT in the corpus can be detected.
    ``extract_companies`` only returns canonical corpus names, so
    ``ooc_entities`` would always be empty when using it here.

    Returns a list of issue dicts. An empty list means no such pattern was
    detected. Abstaining answers are skipped entirely.
    """
    # Local import avoids any potential circular-import path through the
    # retrieval → ingestion → generation import chain.
    from src.retrieval.entity_filter import extract_entity_candidates  # noqa: PLC0415

    issues: list[dict] = []
    queried_entities = extract_entity_candidates(query)
    corpus_set = {c.lower() for c in corpus_companies}
    ooc_entities = [e for e in queried_entities if e.lower() not in corpus_set]

    if not ooc_entities or answer.abstain:
        return issues

    for claim in answer.claims:
        if not claim.value:
            continue
        for ooc in ooc_entities:
            if (
                ooc.lower() in claim.text.lower()
                and claim.citation_company.lower() != ooc.lower()
            ):
                issues.append({
                    "type": "ooc_entity_attribution",
                    "ooc_entity": ooc,
                    "claim_text": claim.text,
                    "claimed_value": claim.value,
                    "citing_company": claim.citation_company,
                    "explanation": (
                        f"Answer attributes '{claim.value}' to {ooc} "
                        f"but cites [{claim.citation_company}] as the source. "
                        f"{ooc} is not in the indexed corpus — "
                        f"this figure likely comes from an incidental mention "
                        f"in {claim.citation_company}'s document."
                    ),
                })

    return issues
