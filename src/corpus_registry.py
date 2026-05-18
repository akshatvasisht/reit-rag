"""Database-derived corpus registry with hardcoded seed fallback.

The module-level ``CORPUS_REGISTRY`` list is the single shared live view of the
corpus. Consumers that do ``from src.corpus_registry import CORPUS_REGISTRY``
hold a reference to this *same list object*. When ``CorpusRegistry.refresh()``
is called it mutates the list in place (``clear`` + ``extend``), so every prior
import automatically sees the updated contents on its next iteration — no
module reload or re-import is required.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded seed list — the authoritative bootstrap used when the database is
# unreachable and as the single source of truth for unit tests with no live
# DB connection.
# ---------------------------------------------------------------------------
CORPUS_REGISTRY_SEED: list[dict[str, Any]] = [
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
# Shared live list — this object's identity never changes; only its contents
# are replaced by refresh(). Every consumer that imported the name at module
# load time holds a reference to this exact list.
# ---------------------------------------------------------------------------
CORPUS_REGISTRY: list[dict[str, Any]] = list(CORPUS_REGISTRY_SEED)


class CorpusRegistry:
    """Manages the live corpus entry list.

    By default the singleton uses the module-level ``CORPUS_REGISTRY`` list as
    its backing store, so a ``refresh()`` call propagates to every consumer
    that imported that name.

    Pass an explicit ``entries`` list only for test isolation. In that case
    the instance operates on the provided list instead, and
    ``_reset_for_tests()`` will restore the singleton to the module-level list.
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        if entries is not None:
            self._entries: list[dict[str, Any]] = entries
            self._owns_module_list = False
        else:
            self._entries = CORPUS_REGISTRY
            self._owns_module_list = True

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries

    def refresh(self) -> None:
        """Reload entries from the documents table, updating the shared list.

        Mutates ``self._entries`` in place (clear + extend) so that all
        consumers holding a reference to the same list object — including those
        that did ``from src.corpus_registry import CORPUS_REGISTRY`` at import
        time — see the new data on their next iteration.

        Falls back silently: if the database is unreachable the existing
        entries remain unchanged.
        """
        try:
            new_entries = self._load_from_db()
        except Exception as exc:
            logger.warning(
                "corpus_registry DB refresh failed (%s: %s); using seed list",
                type(exc).__name__,
                exc,
            )
            return
        self._entries.clear()
        self._entries.extend(new_entries)

    @staticmethod
    def _load_from_db() -> list[dict[str, Any]]:
        from src.db import connect
        rows: list[dict[str, Any]] = []
        with connect() as conn:
            cur = conn.execute("""
                SELECT DISTINCT
                  company,
                  ticker,
                  doc_type,
                  doc_subtype,
                  report_date,
                  doc_version,
                  period_covered,
                  split_part(source_path, '/', -1) AS source_filename
                FROM documents
                ORDER BY company, report_date DESC
            """)
            description = cur.description or []
            cols = [d.name if hasattr(d, "name") else d[0] for d in description]
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                row.setdefault("keywords", _derive_keywords(row))
                row.setdefault("secondary_keywords", [])
                rows.append(row)
        return rows


def _derive_keywords(row: dict[str, Any]) -> list[str]:
    """Derive a small set of lowercase stem variants from canonical fields."""
    keywords: set[str] = set()
    for field_name in ("company", "ticker", "doc_type", "doc_subtype"):
        v = row.get(field_name)
        if isinstance(v, str) and v:
            keywords.add(v.lower())
            keywords.add(v.lower().replace(" ", "_"))
            for token in v.lower().replace("_", " ").split():
                if len(token) > 2:
                    keywords.add(token)
    return sorted(keywords)


_lock = threading.Lock()
_singleton: CorpusRegistry | None = None


def get_registry() -> CorpusRegistry:
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = CorpusRegistry()
                _singleton.refresh()
    return _singleton


def _reset_for_tests() -> None:
    """Test-only hook: drop the singleton and restore the module-level list to seed.

    After this call ``get_registry()`` will create a fresh ``CorpusRegistry``
    backed by the module-level ``CORPUS_REGISTRY`` list (which is also restored
    to its seed contents), so subsequent tests start from a known state.
    """
    global _singleton
    with _lock:
        _singleton = None
        CORPUS_REGISTRY.clear()
        CORPUS_REGISTRY.extend(CORPUS_REGISTRY_SEED)
