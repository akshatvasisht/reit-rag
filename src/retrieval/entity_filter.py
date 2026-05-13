"""Extract company / ticker mentions from a query for metadata pre-filtering.

When a query explicitly names a company in the corpus (e.g. "What is BXP's
FFO?"), that company is passed as a `WHERE company = ANY(...)` filter to
BM25 and vector retrieval — leaving more top-K budget for relevant variants
within the named company rather than spending it on cross-company noise that
the reranker has to filter out.

Empty list = no entity mentioned → corpus-wide retrieval (preserved for
cross-document queries like "Which REIT has the highest debt-to-EBITDA?").
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.ingestion.metadata import CORPUS_REGISTRY


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
