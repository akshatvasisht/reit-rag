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
actually appears (or is within 2% rounding tolerance) in the cited chunk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.generation.prompts import _format_date, _format_doc_type
from src.models import Claim, RetrievedChunk, StructuredAnswer


# ---------------------------------------------------------------------------
# Unicode and approximation normalization
# ---------------------------------------------------------------------------

# Unicode dash variants commonly produced by PDF extraction.  All normalised
# to ASCII "-" before parsing so a chunk's "8–12%" and a claim's "8-12%"
# compare equal.
_DASH_VARIANTS = (
    "‐", "‑", "‒", "–", "—", "―",
    "−", "⁃", "﹘", "﹣", "－",
)
_DASH_TRANSLATE = str.maketrans({d: "-" for d in _DASH_VARIANTS})

# Approximation prefixes the LLM or deck may apply to a numeric value.  These
# are stripped before numeric comparison but preserved in qualifier metadata
# so downstream consumers can label the answer.
_APPROX_PREFIXES = ("~", "≈", "≅")
_APPROX_WORDS_RE = re.compile(
    r"^\s*(?:approximately|approx\.?|about|roughly|circa|~)\s+",
    re.IGNORECASE,
)

# Comparison operators that prefix a value (e.g. ">$15B").
_CMP_PREFIX_RE = re.compile(r"^[><=≥≤]+\s*")


def _unicode_normalize(text: str) -> str:
    """NFKC-normalise and translate Unicode dashes to ASCII '-'.

    PDF extraction commonly produces en-dash (U+2013), em-dash (U+2014), or
    minus (U+2212) where the source semantically uses a hyphen-minus.  This
    normalisation makes claim-vs-chunk string comparison robust to that
    extractor variance.
    """
    if not text:
        return text
    return unicodedata.normalize("NFKC", text).translate(_DASH_TRANSLATE)


def _strip_approx(value: str) -> tuple[str, bool]:
    """Strip approximation markers and comparison operators; return (stripped, was_approx).

    Handles:
    - ~, ≈, ≅ prefixes
    - approximation words ("approximately", "about", etc.)
    - comparison operators >, <, >=, <=, ≥, ≤
    """
    v = value.lstrip()
    approx = False
    for prefix in _APPROX_PREFIXES:
        if v.startswith(prefix):
            v = v[len(prefix):].lstrip()
            approx = True
    new_v = _APPROX_WORDS_RE.sub("", v)
    if new_v != v:
        approx = True
        v = new_v
    # Strip comparison operators (>, <, >=, <=, ≥, ≤)
    cmp_match = _CMP_PREFIX_RE.match(v)
    if cmp_match:
        v = v[cmp_match.end():].lstrip()
        approx = True  # treat as approximate for qualifier purposes
    return v, approx


# ---------------------------------------------------------------------------
# Citation regex
# ---------------------------------------------------------------------------

# Matches `[Company, Doc Type, Month Year, p.N]` and also the extended
# `[Company, Doc Type (Subtype), Month Year, p.N]` format.
# The doc_type group is non-greedy and stops at `,` or `]`; the optional
# `( Subtype )` suffix is captured separately so older-format citations still
# parse.
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


# ---------------------------------------------------------------------------
# Numeric value parsing helpers
# ---------------------------------------------------------------------------

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

# Vision-extracted chart values are commonly rounded to one decimal at the
# source while the underlying chunk text carries a slightly more precise
# figure (e.g. "4.9x" labelled on a bar whose backing table reads "4.85x").
# A 2% relative tolerance covers that rounding band without admitting
# genuinely distinct values.
_ROUNDING_TOLERANCE = 0.02

# Non-financial unit words — when present, skip numeric comparison and do
# substring presence check instead.
_NON_FINANCIAL_UNIT_RE = re.compile(
    r"\b(?:years?|year|months?|month|days?|day|"
    r"mw|kw|gw|watts?|megawatts?|kilowatts?|"
    r"sq\.?\s*ft\.?|sqft|square\s+feet|square\s+foot|"
    r"acres?|hectares?|"
    r"january|february|march|april|may|june|july|"
    r"august|september|october|november|december|"
    r"properties|stores|buildings|sites|units)\b",
    re.IGNORECASE,
)

# Date patterns — when claim.value parses as a date, skip numeric comparison.
_DATE_PATTERN_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

# Spelled-out scale words mapped to B/M/K/T multipliers.
_SCALE_WORD_RE = re.compile(
    r"\s+(?P<scale>trillion|billion|million|thousand)\s*$",
    re.IGNORECASE,
)
_SCALE_MAP = {
    "trillion": "t",
    "billion": "b",
    "million": "m",
    "thousand": "k",
}


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
    """Strip currency/unit decorators and parse to float; return None on failure.

    Handles:
    - Standard suffixes: bps, bp, %, x, b/B, m/M, k/K
    - MM (double-M) Wall-Street suffix collapses to a single trailing scale
    - Spelled-out scale words: "billion", "million", "thousand"
    - Parenthetical accounting-negative notation: (N) → -N
    - Trailing annotations after the numeric token
    """
    v = value.strip().lower()

    # Convert accounting-negative parentheticals: (N) or ($N) → -N.
    # Handles unit INSIDE parens: (1.8%) → -1.8%   (4.9x) → -4.9x
    # AND unit OUTSIDE parens:    (1.8)% → -1.8%   (22.3M) → -22.3M
    paren_match = re.match(
        r"^\((\$?[\d,]+(?:\.\d+)?(?:%|x|bps|bp|[bmkt])?)\)(%|x|bps|bp)?$", v
    )
    if paren_match:
        inner = paren_match.group(1)
        outer_unit = paren_match.group(2) or ""
        v = "-" + inner + outer_unit

    # Normalise spelled-out scale words ("$10.5 billion" → "$10.5b").
    scale_match = _SCALE_WORD_RE.search(v)
    if scale_match:
        scale = _SCALE_MAP[scale_match.group("scale")]
        v = v[:scale_match.start()] + scale

    v, _ = _strip_approx(v)

    # Strip repeated trailing M's before other suffixes so "MM" collapses to
    # a single scale before the standard-suffix loop runs.
    for suffix in ("bps", "bp", "%", "x"):
        if v.endswith(suffix):
            v = v[:-len(suffix)]
            break
    else:
        # Handle scale suffixes — strip ALL repeated m's, then one b/k/t
        while v.endswith("m"):
            v = v[:-1]
        for suffix in ("b", "k", "t"):
            if v.endswith(suffix):
                v = v[:-len(suffix)]
                break

    # Remove ALL $ (use replace, not lstrip, so "-$22.3" → "-22.3" works after
    # paren-negative conversion. lstrip only strips from the front, leaving the
    # $ stuck behind a leading "-".)
    v = v.replace("$", "")
    # Remove thousands separators
    v = v.replace(",", "")
    # Handle bare parenthetical negative (after suffix strip)
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    try:
        return float(v)
    except ValueError:
        return None


def _normalize_value(value: str) -> str:
    """Strip whitespace and lowercase; normalize Unicode dashes, approx markers, spacing.

    Applies in order:
      1. Unicode NFKC + dash-variant translation
      2. Approximation-marker stripping (so '~5.7%' compares equal to '5.7%')
      3. Whitespace collapse between digit and unit ('94.1 %' → '94.1%')
    """
    v = _unicode_normalize(value).strip().lower()
    v, _ = _strip_approx(v)
    v = re.sub(r"(\d)\s+(%|bps|bp|x)", r"\1\2", v)
    return v


def _is_non_financial_unit_value(value: str) -> bool:
    """Return True if the value carries non-financial units (years, MW, dates, etc.).

    When True, the checker uses substring presence instead of numeric comparison.
    """
    if _DATE_PATTERN_RE.search(value):
        return True
    if _NON_FINANCIAL_UNIT_RE.search(value):
        return True
    return False


def _is_non_numeric_proper_noun(value: str) -> bool:
    """Return True when the value has no financial-number tokens.

    Such values (proper nouns, descriptive phrases) cannot be numerically
    verified — the checker skips them rather than reporting a mismatch.
    """
    matches = FINANCIAL_NUMBER_RE.findall(value)
    return len(matches) == 0


# ---------------------------------------------------------------------------
# Composite value splitting
# ---------------------------------------------------------------------------

# Splits a composite claim value like "86.4% (Q2 2025), ~86.9% (Q4 2025)"
# or "~$7Bn / 4.9x" into atomic numeric sub-values.
# Only split on commas that are value-level separators — NOT thousands separators.
# A value-level comma is preceded by a non-digit character and followed by a
# new numeric token (digit, ~, $, or approximation prefix).
_COMPOSITE_COMMA_RE = re.compile(
    r"(?<=\D),\s*(?=~?[$]?\d|~\d|[≈≅])",
)


def _split_composite_value(value: str) -> list[str]:
    """Split a composite claim value into atomic sub-values.

    Returns a single-element list when the value is not composite, so callers
    can always iterate the result uniformly.

    Split rules:
    - Semicolons always split
    - ' / ' (space-slash-space) splits when it separates two value tokens
    - Parentheticals "X (Y)" split into ["X", "Y"] (preserving whole-string
      parentheticals like "($22.3M)" which signal accounting negative)
    - Commas with numeric content on both sides split (NOT thousands separators)
    """
    parts: list[str] = [p.strip() for p in value.split(";")]
    slash_parts: list[str] = []
    for p in parts:
        slash_parts.extend(sp.strip() for sp in p.split(" / "))
    # Preserve whole-string parentheticals like "($22.3M)" — they signal
    # accounting negatives, not sub-values to extract.
    paren_parts: list[str] = []
    for p in slash_parts:
        stripped = p.strip()
        if re.fullmatch(r"\([^()]+\)", stripped):
            paren_parts.append(stripped)
            continue
        # Also treat whole-string $(N) as an accounting negative token.
        if re.fullmatch(r"\$\([^()]+\)", stripped):
            # Rewrite to canonical ($N) form so _parse_numeric handles it.
            paren_parts.append("(" + "$" + stripped[2:])
            continue
        # Regex matches $( prefix as well so the leading $ is consumed along
        # with the parenthetical and re-attached inside the inner token.
        paren_matches = list(re.finditer(r"\(([^()]+)\)", stripped))
        if paren_matches:
            # Build inner tokens, re-attaching any immediately-preceding $ that
            # was left stranded when the outer substitution removed "$(N)".
            inner_tokens: list[str] = []
            for m in paren_matches:
                start = m.start()
                dollar_prefix = "$" if start > 0 and stripped[start - 1] == "$" else ""
                inner_val = m.group(1).strip()
                if inner_val:
                    inner_tokens.append(
                        "(" + dollar_prefix + inner_val + ")" if dollar_prefix else inner_val
                    )
            # Remove $(…) and (…) from the outer string. Strip the leading $
            # that belonged to a $(…) group.
            outer = re.sub(r"\$\s*\([^()]+\)\s*|\s*\([^()]+\)\s*", " ", stripped).strip()
            # Clean up any trailing comma or dollar left by the substitution.
            outer = outer.rstrip(",$").strip()
            if outer:
                paren_parts.append(outer)
            paren_parts.extend(inner_tokens)
        else:
            paren_parts.append(stripped)
    comma_parts: list[str] = []
    for p in paren_parts:
        sub = _COMPOSITE_COMMA_RE.split(p)
        comma_parts.extend(s.strip() for s in sub if s.strip())
    # Only split on ", " when every resulting fragment carries a financial
    # number; otherwise "Capital, Inc." would split incorrectly.
    final: list[str] = []
    for p in comma_parts:
        sub = re.split(r",\s+", p)
        if len(sub) > 1 and all(FINANCIAL_NUMBER_RE.search(s) for s in sub):
            final.extend(s.strip() for s in sub if s.strip())
        else:
            final.append(p)
    return final if final else [value]


# ---------------------------------------------------------------------------
# Range parsing
# ---------------------------------------------------------------------------

# Matches a numeric range like "8-12%", "$5-$10M", "5.4-5.5x", AND the
# variant "87.25%-88%" where each endpoint carries the unit (common in
# extracted chart text).  The mid-unit is non-capturing — we use the trailing
# unit as the canonical one.
_RANGE_RE = re.compile(
    r"""
    ^\s*
    (?:\$\s*)?                       # optional leading $
    (?P<lo>\d+(?:\.\d+)?)            # lower endpoint
    (?:%|x|bps|bp|[bmkt])?           # optional unit after lo (non-capturing)
    \s*-\s*                          # ASCII hyphen (Unicode dashes already normalised)
    (?:\$\s*)?                       # optional second $
    (?P<hi>\d+(?:\.\d+)?)            # upper endpoint
    \s*
    (?P<unit>%|x|bps|bp|[bmkt])?     # optional trailing unit (canonical)
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_range(value: str) -> tuple[float, float, str | None] | None:
    """Parse a normalized range string; return (lo, hi, unit) or None."""
    m = _RANGE_RE.match(value)
    if m is None:
        return None
    try:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
    except ValueError:
        return None
    return lo, hi, m.group("unit")


# Matches in-text ranges like "8-12%", "$5-$10M", "5.0-5.5x", "87.25%-88%"
# anywhere in chunk body. Mid-unit on lo is non-capturing.
_INLINE_RANGE_RE = re.compile(
    r"""
    (?<![\d.])
    (?:\$\s*)?
    (?P<lo>\d+(?:\.\d+)?)
    (?:%|x|bps|bp|[bmkt])?           # optional unit after lo (non-capturing)
    \s*-\s*
    (?:\$\s*)?
    (?P<hi>\d+(?:\.\d+)?)
    \s*
    (?P<unit>%|x|bps|bp|[bmkt])?
    (?!\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _find_ranges_in_text(text: str) -> list[tuple[float, float, str | None]]:
    """Return all (lo, hi, unit) numeric ranges found in *text* (Unicode-normalised)."""
    out: list[tuple[float, float, str | None]] = []
    for m in _INLINE_RANGE_RE.finditer(text):
        try:
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
        except ValueError:
            continue
        if hi < lo:
            continue
        out.append((lo, hi, m.group("unit")))
    return out


# ---------------------------------------------------------------------------
# Chunk/claim key helpers
# ---------------------------------------------------------------------------


def _chunk_key(rc: RetrievedChunk) -> tuple[str, str, int | None]:
    """Canonical (company, date, page) key for a retrieved chunk."""
    return (
        rc.chunk.company.strip().lower(),
        _normalize_date(_format_date(rc.chunk.report_date)),
        rc.chunk.page_number,
    )


def _claim_lookup_key(claim: Claim) -> tuple[str, str, int | None]:
    """Canonical (company, date, page) key derived from claim citation fields."""
    return (
        claim.citation_company.strip().lower(),
        _normalize_date(claim.citation_date),
        claim.citation_page,
    )


# ---------------------------------------------------------------------------
# Per-atom numeric verification
# ---------------------------------------------------------------------------


def _extract_numeric_head(value: str) -> str:
    """Extract the numeric token(s) from a value string.

    Strips trailing annotations and surrounding prose so the result is a clean
    financial value, range, or — for accounting-negative wrappers like
    "($22.3M)" — the original parenthesised form.

    Examples:
        "~89% end of 2026"        → "89%"
        "$2.80 per share"         → "$2.80"
        "87.25%-88% average"      → "87.25%-88%"
        "North America 52%"       → "52%"
        "($22.3M)"                → "($22.3M)"
        "27 kW per rack in 2025"  → unchanged (no financial-number tokens)
    """
    v = _normalize_value(value)

    # Whole-string parenthetical is an accounting negative; _parse_numeric's
    # paren_match needs the closing ')' intact so the wrapper is preserved.
    if v.startswith("(") and v.rstrip().endswith(")"):
        return v.rstrip()

    # If the whole string already parses cleanly, return it as-is. Requires
    # both classify AND parse to succeed so prose like "$2.80 per share" is
    # not accepted as a clean value.
    if (_classify_unit(v) is not None and _parse_numeric(v) is not None) or _parse_range(v) is not None:
        return v

    fin_matches = list(FINANCIAL_NUMBER_RE.finditer(v))
    if not fin_matches:
        return v

    # When multiple numbers appear, prefer the first..last span if it is a
    # recognisable range so "87.25%-88% average" keeps both endpoints.
    if len(fin_matches) > 1:
        first_match = fin_matches[0]
        last_match = fin_matches[-1]
        between = v[first_match.start():last_match.end()].strip()
        if _parse_range(between) is not None:
            return between

    last_match = fin_matches[-1]
    return v[last_match.start():last_match.end()].strip()


def _atom_found_in_chunk(
    atom: str,
    chunk_text: str,
    chunk_values: list[str],
) -> bool:
    """Return True if atomic claim value *atom* is supported by *chunk_text*."""
    atom_head = _extract_numeric_head(atom)
    atom_norm = atom_head
    atom_unit = _classify_unit(atom_norm)
    atom_float = _parse_numeric(atom_norm)
    atom_range = _parse_range(atom_norm)

    # Non-financial-unit values (years, MW, sqft, dates) — verify by substring
    # presence in chunk text since numeric comparison is meaningless.
    if _is_non_financial_unit_value(atom):
        bare = re.sub(_NON_FINANCIAL_UNIT_RE, "", atom).strip(" ,.")
        if bare in chunk_text:
            return True
        if atom.lower() in chunk_text.lower():
            return True
        stripped = re.sub(r"\s*thousand\s*", "", atom, flags=re.IGNORECASE).strip()
        if stripped and stripped.lower() in chunk_text.lower():
            return True
        # Approximation markers and trailing prose ("~2.9 GW in-place") leave the
        # whole-string checks above unable to match a chunk that states the value
        # plainly ("approximately 2.9 GW"). Isolate the numeric token and require
        # BOTH it (with digit boundaries) and a unit token from the atom to be
        # present, so the number isn't matched in an unrelated context.
        num_match = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", atom)
        unit_match = _NON_FINANCIAL_UNIT_RE.search(atom)
        if num_match and unit_match:
            num = num_match.group(0)
            unit = unit_match.group(0)
            num_present = re.search(rf"(?<![\d.]){re.escape(num)}(?![\d.])", chunk_text)
            if num_present and unit.lower() in chunk_text.lower():
                return True
        return False

    # Proper nouns and other non-numeric values have nothing to verify.
    if _is_non_numeric_proper_noun(atom):
        return True

    # Range-aware path 1: atom is a range
    if atom_range is not None:
        lo, hi, range_unit = atom_range
        for clo, chi, cu in _find_ranges_in_text(chunk_text):
            if abs(clo - lo) > 0.001 or abs(chi - hi) > 0.001:
                continue
            if range_unit:
                claim_unit_cat = _classify_unit(f"1{range_unit}")
                chunk_unit_cat = _classify_unit(f"1{cu}") if cu else None
                if claim_unit_cat is not None and chunk_unit_cat != claim_unit_cat:
                    continue
            return True
        # Fall back to endpoint-pair check
        lo_seen = hi_seen = False
        for cv in chunk_values:
            cv_norm = _normalize_value(cv)
            cv_float = _parse_numeric(cv_norm)
            cv_unit = _classify_unit(cv_norm)
            if cv_float is None:
                continue
            if range_unit:
                claim_unit_cat = _classify_unit(f"1{range_unit}")
                if claim_unit_cat is not None and cv_unit != claim_unit_cat:
                    continue
            if abs(cv_float - lo) < 0.001 or (lo != 0 and abs(cv_float - lo) / abs(lo) <= _ROUNDING_TOLERANCE):
                lo_seen = True
            if abs(cv_float - hi) < 0.001 or (hi != 0 and abs(cv_float - hi) / abs(hi) <= _ROUNDING_TOLERANCE):
                hi_seen = True
        if lo_seen and hi_seen:
            return True

    # Range-aware path 2: atom is a single value within a chunk-published range
    if atom_float is not None and atom_range is None:
        for clo, chi, cu in _find_ranges_in_text(chunk_text):
            if atom_unit is not None and cu is not None:
                chunk_range_unit_cat = _classify_unit(f"1{cu}")
                if chunk_range_unit_cat != atom_unit:
                    continue
            lo_with_tol = clo * (1 - _ROUNDING_TOLERANCE) if clo > 0 else clo - 0.001
            hi_with_tol = chi * (1 + _ROUNDING_TOLERANCE) if chi > 0 else chi + 0.001
            if lo_with_tol <= atom_float <= hi_with_tol:
                return True

    # Exact normalized match and rounding tolerance
    for cv in chunk_values:
        cv_norm = _normalize_value(cv)

        if atom_norm == cv_norm:
            return True

        cv_unit = _classify_unit(cv_norm)
        if atom_unit is not None and atom_unit == cv_unit and atom_float is not None:
            cv_float = _parse_numeric(cv_norm)
            if cv_float is not None and cv_float != 0:
                if abs(atom_float - cv_float) / abs(cv_float) <= _ROUNDING_TOLERANCE:
                    return True

    # Accept negative claim against positive chunk value when absolute values
    # match — charts often render negatives as bars without a sign on the cell.
    if atom_float is not None and atom_float < 0:
        abs_atom = abs(atom_float)
        for cv in chunk_values:
            cv_norm = _normalize_value(cv)
            cv_unit = _classify_unit(cv_norm)
            if atom_unit is not None and atom_unit != cv_unit:
                continue
            cv_float = _parse_numeric(cv_norm)
            if cv_float is not None and cv_float > 0:
                if abs(abs_atom - cv_float) / cv_float <= _ROUNDING_TOLERANCE:
                    return True

    # "N thousand" annotation: strip and check substring presence for the
    # bare integer (both comma-preserved "28,183" and comma-stripped "28183").
    thousand_match = re.search(r"([\d,]+)\s*thousand", atom, re.IGNORECASE)
    if thousand_match:
        raw_num = thousand_match.group(1)
        if raw_num in chunk_text:
            return True
        bare_num = raw_num.replace(",", "")
        if bare_num in chunk_text:
            return True

    # Tables render multiples and dollar amounts as bare decimals; the regex
    # skips bare decimals to avoid matching page numbers, so substring-search
    # the absolute value with digit-boundary guards.
    looks_like_multiple = atom_unit == _UNIT_MULTIPLE or "x" in atom_norm.lower()
    looks_like_dollar = atom_unit == _UNIT_DOLLAR
    if (looks_like_multiple or looks_like_dollar) and atom_float is not None:
        abs_val = abs(atom_float)
        candidates: set[str] = {f"{abs_val:.2f}", f"{abs_val:.1f}"}
        if abs_val == int(abs_val):
            candidates.add(str(int(abs_val)))
        # Dollar amounts ≥ 1000 are typically comma-rendered in tables; add the
        # comma-formatted form so the substring search matches "18,402,135".
        if looks_like_dollar and abs_val >= 1000:
            candidates.add(f"{int(abs_val):,}")
            if abs_val != int(abs_val):
                candidates.add(f"{abs_val:,.2f}")
        # Reject matches whose bare cell is immediately followed by a non-dollar
        # unit marker so a $-prefixed claim doesn't silently match a percent or
        # multiple cell that shares the same digits (e.g. "$94" vs "94%").
        unit_exclusion = r"(?!\s*(?:%|bps?\b|x\b))" if looks_like_dollar else ""
        for cand in candidates:
            pattern = rf"(?<![\d.]){re.escape(cand)}(?![\d.]){unit_exclusion}"
            if re.search(pattern, chunk_text):
                return True

    return False


# ---------------------------------------------------------------------------
# Main consistency checker
# ---------------------------------------------------------------------------


def check_numeric_consistency(
    answer: StructuredAnswer,
    contexts: list[RetrievedChunk],
) -> list[dict]:
    """Verify that numeric values in each claim appear in the cited chunk.

    For each claim with a non-empty ``value``, the function:
    1. Locates the retrieved chunk that matches the claim's citation fields.
    2. Extracts financial values from the chunk text using FINANCIAL_NUMBER_RE.
    3. Splits composite value strings (e.g. "86.4% (Q2), ~86.9% (Q4)") into
       atomic sub-values and verifies each atom independently.
    4. Checks whether the atomic value is present (exact normalized match,
       within 2% rounding tolerance, range membership, or non-financial unit
       substring presence) in the cited chunk.

    Returns a list of issue dicts.  An empty list means full numeric consistency.
    Abstaining answers are skipped entirely — the caller should not invoke this
    function for abstentions, but the guard here makes the function safe to call
    unconditionally.
    """
    if answer.abstain:
        return []

    # Index retrieved chunks by (company, date, page) for O(1) lookup.
    chunk_index: dict[tuple[str, str, int | None], RetrievedChunk] = {}
    # Page-level index for cross-chunk re-attribution. A claim's value often
    # lives on a co-located chunk (a same-page chart_description or table)
    # rather than the narrative text chunk the LLM cited; the chunk-level
    # check then misses it. Falling back to the union of values across all
    # chunks on the same (document_id, page_number) rescues that case.
    page_index: dict[tuple[object, int | None], list[RetrievedChunk]] = {}
    for rc in contexts:
        chunk_index[_chunk_key(rc)] = rc
        if rc.chunk.page_number is not None:
            page_key = (rc.chunk.document_id, rc.chunk.page_number)
            page_index.setdefault(page_key, []).append(rc)

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

        # Unicode-normalize the chunk text once so dash variants from PDF
        # extraction match an ASCII hyphen in the claim and vice versa.
        chunk_text = _unicode_normalize(matched_rc.chunk.chunk_text or "")
        chunk_values = [m.strip() for m in FINANCIAL_NUMBER_RE.findall(chunk_text)]

        co_located = page_index.get(
            (matched_rc.chunk.document_id, matched_rc.chunk.page_number), []
        )
        page_text = "\n".join(
            _unicode_normalize(rc.chunk.chunk_text or "") for rc in co_located
        )
        page_values = [m.strip() for m in FINANCIAL_NUMBER_RE.findall(page_text)]

        atoms = _split_composite_value(claim.value)

        # A claim passes if every atom that carries financial-numeric meaning
        # is found in the chunk. Pure-annotation atoms are skipped — the
        # non-financial-unit and proper-noun guards inside _atom_found_in_chunk
        # handle them.
        any_atom_has_numeric = False
        all_atoms_found = True
        failing_atom_was_financial = False

        for atom in atoms:
            # Check if this atom has any numeric content to verify
            atom_stripped, _ = _strip_approx(_unicode_normalize(atom))
            atom_is_financial = bool(FINANCIAL_NUMBER_RE.search(atom_stripped))
            # Skip pure annotation atoms (text without financial digits)
            if not atom_is_financial and not _is_non_financial_unit_value(atom):
                # Pure proper-noun / non-numeric atom — skip.
                if _is_non_numeric_proper_noun(atom):
                    continue

            any_atom_has_numeric = True

            # Cited chunk first; on miss, widen the search to the union of
            # all chunks on the same page of the same document.
            found = _atom_found_in_chunk(atom, chunk_text, chunk_values)
            if not found and page_text:
                found = _atom_found_in_chunk(atom, page_text, page_values)

            if not found:
                all_atoms_found = False
                if atom_is_financial:
                    failing_atom_was_financial = True
                break

        if any_atom_has_numeric and not all_atoms_found:
            # The wrong-chunk-class signal requires that NO chunk on the
            # cited page has financial values — otherwise the cited chunk
            # was a narrative companion and the value lives on a sibling.
            issue_type = "numeric_mismatch"
            if failing_atom_was_financial and not chunk_values and not page_values:
                issue_type = "wrong_chunk_no_financial_values"
            issues.append({
                "type": issue_type,
                "claim": claim.text,
                "claimed_value": claim.value,
                "chunk_values_found": chunk_values,
                "chunk_id": str(matched_rc.chunk.id),
            })

    return issues


# ---------------------------------------------------------------------------
# Date and doc-type normalization
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
    """Normalize doc-type strings for robust equality checks.

    Strips the optional "(Subtype)" parenthetical from the citation string so a
    citation like
    "Investor Presentation (Company Update)" matches a chunk whose
    `doc_type='investor_presentation'`. Subtype matching, when needed, is
    handled separately via the doc_subtype column on the chunk; the
    citation-equality check operates on the base doc_type only.
    """
    base = re.sub(r"\s*\([^()]*\)\s*", " ", raw)
    return " ".join(base.replace("_", " ").split()).strip().lower()
