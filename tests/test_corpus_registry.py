"""Tests for the CorpusRegistry class and get_registry() singleton."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.corpus_registry import (
    CORPUS_REGISTRY,
    CorpusRegistry,
    _derive_keywords,
    _reset_for_tests,
    get_registry,
)
from src.corpus_registry import CORPUS_REGISTRY_SEED


# ---------------------------------------------------------------------------
# CorpusRegistry construction
# ---------------------------------------------------------------------------


def test_default_construction_matches_seed() -> None:
    """CorpusRegistry() with no args should have entries equal to the seed."""
    reg = CorpusRegistry()
    assert reg.entries == list(CORPUS_REGISTRY_SEED)


def test_explicit_entries_construction() -> None:
    """CorpusRegistry(entries=[...]) should reflect the provided entries."""
    custom: list[dict[str, Any]] = [{"company": "Fake REIT", "ticker": "FKE"}]
    reg = CorpusRegistry(entries=custom)
    assert reg.entries == custom


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------


def test_get_registry_returns_same_instance() -> None:
    """get_registry() must return the same object on repeated calls."""
    _reset_for_tests()
    a = get_registry()
    b = get_registry()
    assert a is b


def test_reset_for_tests_produces_fresh_instance() -> None:
    """After _reset_for_tests(), get_registry() returns a new instance."""
    _reset_for_tests()
    first = get_registry()
    _reset_for_tests()
    second = get_registry()
    assert first is not second


# ---------------------------------------------------------------------------
# refresh() soft-failure behavior
# ---------------------------------------------------------------------------


def test_refresh_with_db_error_leaves_entries_unchanged() -> None:
    """refresh() when _load_from_db raises must leave existing entries intact."""
    reg = CorpusRegistry()
    original = list(reg.entries)
    with patch.object(CorpusRegistry, "_load_from_db", side_effect=RuntimeError("no db")):
        reg.refresh()
    assert reg.entries == original


def test_refresh_db_error_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """refresh() must log a WARNING when _load_from_db raises, and still return seed-shape data."""
    import logging

    reg = CorpusRegistry()
    original = list(reg.entries)
    with caplog.at_level(logging.WARNING, logger="src.corpus_registry"):
        with patch.object(
            CorpusRegistry, "_load_from_db", side_effect=RuntimeError("connection refused")
        ):
            reg.refresh()

    # Fallback behavior: entries unchanged (seed-shape)
    assert reg.entries == original

    # Observability behavior: warning was emitted
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Expected at least one WARNING log record"
    assert "corpus_registry DB refresh failed" in warning_records[0].message
    assert "RuntimeError" in warning_records[0].message
    assert "connection refused" in warning_records[0].message


def test_refresh_with_mocked_db_updates_entries() -> None:
    """refresh() when _load_from_db succeeds must replace entries with new rows."""
    new_rows: list[dict[str, Any]] = [
        {"company": "Mock REIT", "ticker": "MCK", "keywords": ["mock"], "secondary_keywords": []}
    ]
    reg = CorpusRegistry()
    with patch.object(CorpusRegistry, "_load_from_db", return_value=new_rows):
        reg.refresh()
    assert reg.entries == new_rows


# ---------------------------------------------------------------------------
# refresh() propagates to the static-imported module-level symbol
# ---------------------------------------------------------------------------


def test_refresh_propagates_to_statically_imported_corpus_registry() -> None:
    """Calling get_registry().refresh() must update the module-level CORPUS_REGISTRY list.

    Consumers that did ``from src.corpus_registry import CORPUS_REGISTRY`` at
    import time hold a reference to the same list object.  After refresh() the
    list contents must reflect the new rows so those consumers see updated data
    without re-importing.
    """
    _reset_for_tests()
    new_rows: list[dict[str, Any]] = [
        {"company": "Live REIT", "ticker": "LVR", "keywords": ["live"], "secondary_keywords": []}
    ]
    registry = get_registry()
    with patch.object(CorpusRegistry, "_load_from_db", return_value=new_rows):
        registry.refresh()

    # CORPUS_REGISTRY was imported at the top of this module — it must now
    # contain the new rows because refresh() mutates the list in place.
    assert CORPUS_REGISTRY == new_rows


def test_module_level_list_identity_is_preserved_across_refresh() -> None:
    """The list object returned by ``import CORPUS_REGISTRY`` must be the same
    object before and after a refresh — only the contents change."""
    _reset_for_tests()
    import src.corpus_registry as reg_module

    list_id_before = id(reg_module.CORPUS_REGISTRY)

    new_rows: list[dict[str, Any]] = [{"company": "Identity REIT", "ticker": "IDN"}]
    with patch.object(CorpusRegistry, "_load_from_db", return_value=new_rows):
        reg_module.get_registry().refresh()

    assert id(reg_module.CORPUS_REGISTRY) == list_id_before
    assert reg_module.CORPUS_REGISTRY == new_rows


# ---------------------------------------------------------------------------
# _reset_for_tests restores seed without polluting the module-level list
# ---------------------------------------------------------------------------


def test_reset_for_tests_restores_module_level_list_to_seed() -> None:
    """_reset_for_tests() must restore the module-level CORPUS_REGISTRY to the seed."""
    import src.corpus_registry as reg_module

    # Pollute the module-level list.
    reg_module.CORPUS_REGISTRY.clear()
    reg_module.CORPUS_REGISTRY.append({"company": "Polluted REIT"})

    _reset_for_tests()

    assert reg_module.CORPUS_REGISTRY == list(CORPUS_REGISTRY_SEED)


def test_reset_for_tests_new_singleton_uses_same_module_list() -> None:
    """After _reset_for_tests, the fresh singleton's _entries IS the module list."""
    import src.corpus_registry as reg_module

    _reset_for_tests()
    registry = get_registry()

    # The singleton's backing store must be the module-level list (same object).
    assert registry._entries is reg_module.CORPUS_REGISTRY


# ---------------------------------------------------------------------------
# _derive_keywords
# ---------------------------------------------------------------------------


def test_derive_keywords_digital_realty() -> None:
    """_derive_keywords should return lowercase stems for company and ticker."""
    result = _derive_keywords({"company": "Digital Realty", "ticker": "DLR"})
    assert "digital realty" in result
    assert "digital_realty" in result
    assert "digital" in result
    assert "realty" in result
    assert "dlr" in result


# ---------------------------------------------------------------------------
# Backward-compatible module-level CORPUS_REGISTRY
# ---------------------------------------------------------------------------


def test_module_level_corpus_registry_is_list() -> None:
    """CORPUS_REGISTRY exported from src.corpus_registry must be a list."""
    assert isinstance(CORPUS_REGISTRY, list)


def test_module_level_corpus_registry_matches_seed() -> None:
    """CORPUS_REGISTRY at module level must equal the seed after a reset."""
    _reset_for_tests()
    assert CORPUS_REGISTRY == list(CORPUS_REGISTRY_SEED)


# ---------------------------------------------------------------------------
# Cleanup — reset singleton so other tests are not affected
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton_after_each_test():
    yield
    _reset_for_tests()
