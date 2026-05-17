"""Database-derived corpus registry with hardcoded seed fallback.

The module-level ``CORPUS_REGISTRY`` list is the single shared live view of the
corpus.  Consumers that do ``from src.corpus.registry import CORPUS_REGISTRY``
hold a reference to this *same list object*.  When ``CorpusRegistry.refresh()``
is called it mutates the list in place (``clear`` + ``extend``), so every prior
import automatically sees the updated contents on its next iteration — no
module reload or re-import is required.
"""

from __future__ import annotations

import threading
from typing import Any

from src.ingestion._registry_seed import CORPUS_REGISTRY_SEED

# ---------------------------------------------------------------------------
# Shared live list — this object's identity never changes; only its contents
# are replaced by refresh().  Every consumer that imported the name at module
# load time holds a reference to this exact list.
# ---------------------------------------------------------------------------
CORPUS_REGISTRY: list[dict[str, Any]] = list(CORPUS_REGISTRY_SEED)


class CorpusRegistry:
    """Manages the live corpus entry list.

    By default the singleton uses the module-level ``CORPUS_REGISTRY`` list as
    its backing store, so a ``refresh()`` call propagates to every consumer
    that imported that name.

    Pass an explicit ``entries`` list only for test isolation.  In that case
    the instance operates on the provided list instead, and
    ``_reset_for_tests()`` will restore the singleton to the module-level list.
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        if entries is not None:
            # Caller supplied a private list (test isolation).
            self._entries: list[dict[str, Any]] = entries
            self._owns_module_list = False
        else:
            # Default: use the module-level list so refresh() propagates.
            self._entries = CORPUS_REGISTRY
            self._owns_module_list = True

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries

    def refresh(self) -> None:
        """Reload entries from the documents table, updating the shared list.

        Mutates ``self._entries`` in place (clear + extend) so that all
        consumers holding a reference to the same list object — including those
        that did ``from src.corpus.registry import CORPUS_REGISTRY`` at import
        time — see the new data on their next iteration.

        Falls back silently: if the database is unreachable the existing
        entries remain unchanged.
        """
        try:
            new_entries = self._load_from_db()
        except Exception:
            # Soft failure — leave existing entries in place.
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
    for field in ("company", "ticker", "doc_type", "doc_subtype"):
        v = row.get(field)
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
