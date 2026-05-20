"""Temporal-intent classification for version-aware retrieval."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)

TemporalIntent = Literal["latest", "historical", "comparison", "conflict", "all_company_synthesis"]

# Minimum LLM confidence required to accept the LLM's classification.
# Below this threshold the deterministic regex fallback is used instead.
INTENT_CONFIDENCE_THRESHOLD = 0.85

# JSON schema that constrains the LLM response to a valid intent + confidence
# pair.  The schema is passed as a tool definition and ``tool_choice`` forces
# the model to invoke it, so the response is always a well-typed ``tool_use``
# block whose ``.input`` contains the validated fields.
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["latest", "historical", "comparison", "conflict", "all_company_synthesis"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}

# System prompt for the LLM classifier.  Explicit examples anchor each intent
# value and the model is instructed to lower confidence when genuinely ambiguous.
INTENT_CLASSIFICATION_PROMPT = """Classify the temporal intent of the following retrieval query into exactly one of these five categories:

- "all_company_synthesis": queries that explicitly request comparison, ranking, or summary across ALL companies in the corpus, such as:
  - "Which company has the highest X?"
  - "Compare X across all companies"
  - "What does each company say about Y?"
  - "Summarise X across the portfolio"
  Examples: "Which REIT has the highest leverage?", "Compare same-store NOI growth across all companies", "What are the capital allocation strategies of each company?"

- "latest": the query asks about the most recent / current state, without targeting a specific past period or requesting a cross-period comparison.
  Examples: "What is the current occupancy rate?", "What was the NOI in Q4 2025?" (when Q4 2025 is the latest reporting period)

- "historical": the query targets a specific past reporting period without asking to compare it against another period.
  Examples: "What was the leverage ratio in Q2 2023?", "Show the FFO for March 2022"

- "comparison": the query explicitly asks to compare, measure change, or track movement across two or more time periods or versions.
  Examples: "How did revenue change from Q3 to Q4?", "Compare DLR leverage between 2024 and 2025", "What is the delta in occupancy versus last year?"

- "conflict": the query asks about contradictions, discrepancies, or disagreements between figures, documents, or sources — regardless of time period.
  Examples: "Where does the company disagree on revenue?", "Why are the NOI figures inconsistent across filings?", "There is a discrepancy in the reported FFO — which is correct?"

Return a JSON object with:
  "intent": one of the five values above (no other values are allowed)
  "confidence": a float in [0.0, 1.0] representing how confident you are

Set confidence < 0.85 when the query is genuinely ambiguous between two or more intents.

Query: {query}"""

# ---------------------------------------------------------------------------
# Regex classifiers — deterministic fallback path
# ---------------------------------------------------------------------------

# Patterns that signal an all-company synthesis request — queries asking for
# rankings, summaries, or comparisons spanning every company in the corpus.
# Checked before the generic comparison regex so "compare all companies" routes
# to synthesis rather than pairwise comparison.
_SYNTHESIS_RE = re.compile(
    r"""
    \b(?:
        all\s+companies|each\s+company|across\s+all|every\s+reit|
        portfolio-wide|compare\s+all|rank(?:ing)?\s+(?:the\s+)?(?:companies|reits)
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Patterns that signal cross-period comparison or data-conflict language.
_COMPARISON_RE = re.compile(
    r"""
    \b(?:
        compar(?:e|ed?|ison)|versus|vs\.?|delta|
        chang(?:e[sd]?|ing)|between|from\s+\w+\s+to\s+\w+|
        conflict(?:ing|s)?|discrepan(?:cy|cies)|
        disagree(?:ment|s)?|inconsisten(?:t|cy|cies)|
        differ(?:ent|ence|s)?(?:\s+(?:figure|number|value|data))?|
        contradict(?:ing|ion|s)?|mismatch(?:es)?
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
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


def _latest_period_overrides() -> tuple[str, ...]:
    """Return the set of latest reporting-period strings derived from the live registry.

    Computed on each call so that newly-ingested documents propagate without a
    module reload — the registry list may be refreshed at runtime and this
    function always reads it fresh.  The result is a sorted tuple of
    lower-cased strings: ``YYYY-MM`` dates, ``Month YYYY`` equivalents, and
    ``period_covered`` values verbatim.

    Cost is O(n) over the registry, which is small (one entry per document
    class) and dwarfed by the regex match itself.
    """
    from src.corpus_registry import CORPUS_REGISTRY  # local to avoid cycles

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


def _derive_max_corpus_year() -> str:
    """Largest 4-digit year present anywhere in the corpus's date strings.

    Computed fresh on every call so that runtime corpus refresh (via
    ``CorpusRegistry.refresh``) immediately propagates without a module reload.
    """
    years = {
        p[:4]
        for p in _latest_period_overrides()
        if len(p) >= 4 and p[:4].isdigit()
    }
    return max(years) if years else "0000"

# ---------------------------------------------------------------------------
# Anthropic client — lazy singleton
# ---------------------------------------------------------------------------

_client: Optional[Anthropic] = None


def _anthropic_client() -> Anthropic:
    """Return a module-level Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

_LOG_PATH = Path("logs/intent_decisions.jsonl")


def _log_intent_decision(
    query: str,
    intent: str,
    method: str,
    confidence: Optional[float] = None,
    llm_intent: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Append a single JSON line to the intent-decisions audit log.

    All file I/O exceptions are swallowed so a log failure never disrupts
    classification.
    """
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "intent": intent,
        "method": method,
    }
    if confidence is not None:
        record["confidence"] = confidence
    if llm_intent is not None:
        record["llm_intent"] = llm_intent
    if error is not None:
        record["error"] = error

    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Deterministic regex classifier
# ---------------------------------------------------------------------------

def _regex_classify_intent(query: str) -> TemporalIntent:
    """Classify query intent using deterministic regex patterns.

    Returns one of ``"all_company_synthesis"``, ``"comparison"``,
    ``"historical"``, or ``"latest"``.  Synthesis patterns are checked before
    generic comparison so cross-corpus queries route to the dedicated synthesis
    path.  Conflict-language keywords resolve to ``"comparison"`` in the regex
    path because the LLM layer is responsible for the finer-grained
    ``"conflict"`` distinction.
    """
    lowered = query.lower()

    # Synthesis check comes first: "compare all companies" is synthesis, not a
    # generic pairwise comparison.
    if _SYNTHESIS_RE.search(lowered):
        return "all_company_synthesis"

    if _COMPARISON_RE.search(lowered) or _BETWEEN_TEMPORAL_RE.search(lowered):
        return "comparison"

    match = _HISTORICAL_RE.search(lowered)
    if match:
        matched = match.group(0).strip().lower()
        # If the only matched period is a known "latest" period and no other
        # historical period appears, the query is actually about current data.
        if any(latest in matched for latest in _latest_period_overrides()):
            remaining = lowered.replace(matched, "", 1)
            if not _HISTORICAL_RE.search(remaining):
                return "latest"
        # A bare 4-digit year strictly later than any corpus date is a
        # forward-looking projection (e.g. "outlook for 2030"), not a
        # backward-looking query. Treat as latest.
        if matched.isdigit() and len(matched) == 4 and matched > _derive_max_corpus_year():
            return "latest"
        return "historical"

    return "latest"


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

_CLASSIFY_INTENT_TOOL = {
    "name": "classify_intent",
    "description": "Return the temporal intent classification for the query.",
    "input_schema": INTENT_SCHEMA,
}


def _llm_classify_intent(query: str) -> dict:
    """Call Claude Haiku with INTENT_SCHEMA enforced via tool-use.

    Forcing ``tool_choice`` on a schema-bound tool guarantees the response is
    a validated dict (no free-form JSON parsing). Raises on any API error or
    if the model fails to return a tool_use block, so the caller falls back to
    regex.
    """
    response = _anthropic_client().messages.create(  # type: ignore[call-overload]
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.0,
        top_k=1,
        tools=[_CLASSIFY_INTENT_TOOL],
        tool_choice={"type": "tool", "name": "classify_intent"},
        messages=[{
            "role": "user",
            "content": INTENT_CLASSIFICATION_PROMPT.format(query=query),
        }],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "classify_intent":
            payload = block.input
            if not isinstance(payload, dict) or "intent" not in payload or "confidence" not in payload:
                raise RuntimeError("classify_intent tool returned invalid payload")
            return payload
    raise RuntimeError("classify_intent did not return a tool_use block")


# ---------------------------------------------------------------------------
# Forward-looking detection (used by classify_intent_full)
# ---------------------------------------------------------------------------

# Conservative set of keywords indicating a forward-looking query. Must stay
# in sync with the equivalent pattern in src/generation/generator.py.
_FORWARD_LOOKING_QUERY_RE = re.compile(
    r"\b("
    r"guidance|guided|guide|"
    r"target|targets|"
    r"outlook|"
    r"projection|projections|projected|"
    r"forecast|forecasted|"
    r"expected\s+(?:to|for)|"
    r"forward[-\s]looking|"
    # Date-anchored future references: "year-end 2027", "by 2028", "in 2029"
    # against a year strictly later than the latest corpus year.
    r"year[-\s]end\s+20\d{2}|"
    # Explicit forward-intent verbs that the deck's narrative uses for
    # planned actions: "plan to deleverage", "intend to refinance",
    # "aim to expand".
    r"plan(?:s|ning|ned)?\s+to|"
    r"intend(?:s|ing|ed)?\s+to|"
    r"aim(?:s|ing|ed)?\s+to"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> TemporalIntent:
    """Classify the temporal intent of a retrieval query.

    Primary path: LLM classification with schema-constrained enum output and
    confidence score.  Falls back to deterministic regex when the LLM returns
    confidence below ``INTENT_CONFIDENCE_THRESHOLD`` or raises an exception.
    Every decision — including fallbacks — is written to the audit log.

    Args:
        query: The raw user query string.

    Returns:
        One of ``"latest"``, ``"historical"``, ``"comparison"``,
        ``"conflict"``, or ``"all_company_synthesis"``.
        ``"all_company_synthesis"`` is returned when the query asks for a
        ranking, summary, or comparison spanning all corpus companies.
        ``"conflict"`` is returned when the query asks about contradictions
        or discrepancies between figures or sources.
        ``"comparison"`` covers cross-period comparisons.  ``"historical"``
        targets a specific past period.  ``"latest"`` is the default.
    """
    try:
        result = _llm_classify_intent(query)
        if result["confidence"] >= INTENT_CONFIDENCE_THRESHOLD:
            _log_intent_decision(
                query, result["intent"], "llm", confidence=result["confidence"]
            )
            return result["intent"]
        else:
            fallback = _regex_classify_intent(query)
            _log_intent_decision(
                query,
                fallback,
                "regex_fallback",
                confidence=result["confidence"],
                llm_intent=result["intent"],
            )
            return fallback
    except Exception as e:  # noqa: BLE001
        fallback = _regex_classify_intent(query)
        _log_intent_decision(
            query, fallback, "regex_error_fallback", error=str(e)
        )
        return fallback


def classify_intent_full(query: str) -> tuple[TemporalIntent, bool]:
    """Classify temporal intent AND detect whether the query asks for forward-looking data.

    Combines ``classify_intent`` with a lightweight forward-looking keyword
    check so callers do not need to duplicate the regex. Both signals are
    derived in a single call to avoid redundant LLM invocations.

    Args:
        query: The raw user query string.

    Returns:
        A 2-tuple ``(intent, forward_looking)`` where ``intent`` is the same
        value as ``classify_intent(query)`` and ``forward_looking`` is True
        when the query contains guidance / target / outlook / projection /
        forecast or similar forward-looking vocabulary.
    """
    intent = classify_intent(query)
    forward_looking = bool(_FORWARD_LOOKING_QUERY_RE.search(query))
    return intent, forward_looking
