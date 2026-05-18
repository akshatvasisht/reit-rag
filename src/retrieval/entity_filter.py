"""Extract company / ticker mentions from a query for metadata pre-filtering.

When a query explicitly names a company in the corpus (e.g. "What is BXP's
FFO?"), that company is passed as a `WHERE company = ANY(...)` filter to
BM25 and vector retrieval — leaving more top-K budget for relevant variants
within the named company rather than spending it on cross-company noise that
the reranker has to filter out.

Empty list = no entity mentioned → corpus-wide retrieval (preserved for
cross-document queries like "Which REIT has the highest debt-to-EBITDA?").

`extract_entity_candidates` is a separate helper for the OOC attribution guard.
It surfaces capitalized name-like phrases from the raw query text, including
names that are NOT in the corpus.  It must not be used for retrieval filtering
because it has lower precision than `extract_companies`.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.corpus_registry import CORPUS_REGISTRY

# ---------------------------------------------------------------------------
# OOC entity candidate extraction — regex-based, corpus-agnostic
# ---------------------------------------------------------------------------

# Matches capitalized proper-noun phrases — either a multi-word sequence
# where every significant word starts with a capital letter (joined by
# optional prepositions of/the/and/de/la), or a single capitalized word
# of at least 5 characters (to exclude common short words like "REIT").
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9&.\-]*(?:\s+(?:[A-Z][A-Za-z0-9&.\-]*|of|the|and|de|la))*\s+[A-Z][A-Za-z0-9&.\-]*"
    r"|"
    r"[A-Z][A-Za-z]{3,}[A-Za-z0-9&.\-]*"
    r")\b"
)

# Matches short all-caps tokens (2–5 uppercase letters) that look like stock
# tickers.  This is intentionally separate from _CAPITALIZED_PHRASE_RE so that
# false positives in the all-caps set can be filtered independently.
_SHORT_CAPS_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Single-word question / instruction words that look capitalized only because
# they open a sentence.  These are never entity names.
_STOPWORDS: frozenset[str] = frozenset({
    "What", "How", "Which", "Where", "When", "Who", "Why", "Does", "Has",
    "Have", "Had", "Is", "Are", "Was", "Were", "Did", "Do", "Can", "Could",
    "Will", "Would", "Should", "Compare", "Rank", "List", "Show", "Give",
    "Tell", "Describe", "Explain", "Summarise", "Summarize", "Find",
    "Report", "Calculate", "Compute", "Identify",
    # Geographic regions extracted as proper-noun candidates but never
    # company names — when the answer references "Sunbelt NOI" or
    # "Northeast portfolio", the OOC guard should not flag the region
    # itself as an out-of-corpus entity (xdoc-adv-8 closure).
    "Sunbelt", "Northeast", "Northwest", "Southeast", "Southwest", "Midwest",
    "Atlantic", "Pacific", "Mountain", "Plains", "Coastal",
    # Common time-period qualifiers (already partially covered by months
    # in upstream entries, but include common multi-word and short forms here).
    "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
    "Approximately", "Across", "Including", "Excluding",
})

# All-caps tokens that are domain vocabulary, not company tickers.  These are
# filtered from the short-ticker pattern only (multi-word phrases and long
# capitalized words are handled by the existing stopword set above).
_ALLCAPS_STOPWORDS: frozenset[str] = frozenset({
    "WHAT", "WHO", "WHEN", "WHERE", "WHY", "HOW", "WHICH",
    "IS", "ARE", "DOES", "DID", "DO",
    "FFO", "AFFO", "NOI", "EBITDA", "REIT", "REITS",
    "CEO", "CFO", "COO", "CTO", "IR",
    "USA", "US",
    "Q1", "Q2", "Q3", "Q4",
    "BPS", "BP",
    "AI", "ML",
    "ESG", "JV", "DCF", "IRR", "NAV", "LTV",
    "AND", "OR", "THE", "OF", "IN", "AT", "BY", "TO",
    "MR", "MS", "DR",
})


def extract_entity_candidates(query: str) -> list[str]:
    """Return capitalized name-like phrases extracted from *query*.

    Unlike ``extract_companies``, this function does NOT restrict results to
    corpus members.  It is used by the OOC attribution guard to detect entities
    that are mentioned in the query but are absent from the indexed corpus —
    a pattern that ``extract_companies`` can never surface because it only
    returns canonical corpus names.

    False positives (capitalized phrases that are not real company names) are
    harmless: they will match the corpus set and be classified as in-corpus,
    leaving the OOC list empty for that phrase.

    Args:
        query: Raw user query string.

    Returns:
        Deduplicated list of capitalized phrase strings.  May be empty.
    """
    if not query:
        return []
    # Strip possessives so "Equinix's" → "Equinix" and "Digital Realty's" →
    # "Digital Realty" match as contiguous capitalized phrases.
    normalized = re.sub(r"'s\b", "", query)
    candidates: list[str] = []
    seen: set[str] = set()
    for m in _CAPITALIZED_PHRASE_RE.finditer(normalized):
        phrase = m.group(1).strip()
        if not phrase:
            continue
        words = phrase.split()
        first_word = words[0]
        if first_word in _STOPWORDS:
            # When the match begins with a stopword (e.g. "Compare Simon
            # Property Group"), discard the stopword and re-examine the tail
            # as a candidate.
            tail = " ".join(words[1:]).strip()
            if not tail:
                continue
            tail_first = tail.split()[0]
            if tail_first in _STOPWORDS or len(tail_first) < 4:
                continue
            phrase = tail
        if phrase and phrase not in seen:
            # Single-word all-caps phrases (e.g. "REIT", "EBITDA") that matched
            # the phrase regex are domain vocabulary, not entity names.
            if len(phrase.split()) == 1 and phrase.isupper() and phrase in _ALLCAPS_STOPWORDS:
                continue
            seen.add(phrase)
            candidates.append(phrase)

    # Additive pass: capture short all-caps tokens that look like stock tickers
    # (e.g. "BXP", "AMT", "PLD").  These are missed by the phrase regex above
    # because it requires >=5 characters for single-word matches.
    for m in _SHORT_CAPS_RE.finditer(normalized):
        token = m.group(1)
        if token in _ALLCAPS_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            candidates.append(token)

    return candidates


@lru_cache(maxsize=1)
def _matchers() -> list[tuple[str, re.Pattern[str], list[str]]]:
    """Build the per-company matcher list on first call.

    Returns a list of `(company, word_boundary_regex, substring_terms)`.
    Short keywords (≤ 3 chars, e.g. tickers like "O") use word-boundary
    regex so "O" does not match every word containing the letter; longer
    keywords use plain substring matching.
    """
    out: list[tuple[str, re.Pattern[str], list[str]]] = []
    for entry in CORPUS_REGISTRY:
        company = entry["company"]
        keywords = list(entry.get("keywords", []))
        wb_terms = [k for k in keywords if len(k) <= 3]
        sub_terms = [k for k in keywords if len(k) > 3]
        wb_re = (
            re.compile(r"\b(?:" + "|".join(re.escape(t) for t in wb_terms) + r")\b", re.IGNORECASE)
            if wb_terms
            else re.compile(r"(?!)")  # never matches
        )
        out.append((company, wb_re, [s.lower() for s in sub_terms]))
    return out


def extract_companies(query: str) -> list[str]:
    """Return canonical company names mentioned in *query* (deduplicated, ordered).

    Args:
        query: Natural-language user query.

    Returns:
        A list of canonical company strings from `CORPUS_REGISTRY` whose
        keywords appear in the query. Empty when none match.
    """
    if not query:
        return []
    q_lower = query.lower()
    seen: set[str] = set()
    matches: list[str] = []
    for company, wb_re, sub_terms in _matchers():
        hit = False
        if wb_re.search(query):
            hit = True
        elif any(term in q_lower for term in sub_terms):
            hit = True
        if hit and company not in seen:
            seen.add(company)
            matches.append(company)
    return matches
