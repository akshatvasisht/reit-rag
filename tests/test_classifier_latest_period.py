"""Tests for the lazy _latest_period_overrides() function in the classifier.

Verifies that the function reads the live CORPUS_REGISTRY on every call so
that newly-ingested documents propagate to intent classification without a
module reload.
"""

from __future__ import annotations

from unittest.mock import patch

from src.versioning.classifier import _latest_period_overrides, _regex_classify_intent


# ---------------------------------------------------------------------------
# _latest_period_overrides() reads the live registry on each call
# ---------------------------------------------------------------------------


def test_latest_period_overrides_reflects_registry_contents() -> None:
    """_latest_period_overrides() must include dates from the mocked registry."""
    mock_registry = [
        {"report_date": "2026-06", "period_covered": "Q2 2026"},
    ]
    with patch("src.corpus.registry.CORPUS_REGISTRY", mock_registry):
        result = _latest_period_overrides()

    assert "2026-06" in result
    assert "june 2026" in result
    assert "q2 2026" in result


def test_latest_period_overrides_no_caching() -> None:
    """Calling _latest_period_overrides() twice with different mocks returns
    different results — proving there is no module-level caching."""
    first_registry = [{"report_date": "2025-03", "period_covered": "Q1 2025"}]
    second_registry = [{"report_date": "2027-09", "period_covered": "Q3 2027"}]

    with patch("src.corpus.registry.CORPUS_REGISTRY", first_registry):
        first = _latest_period_overrides()

    with patch("src.corpus.registry.CORPUS_REGISTRY", second_registry):
        second = _latest_period_overrides()

    assert "2025-03" in first
    assert "2025-03" not in second
    assert "2027-09" in second
    assert "2027-09" not in first


def test_latest_period_overrides_handles_versions_dict() -> None:
    """Entries that carry a ``versions`` dict should contribute dates from every
    version, not just the top-level entry."""
    mock_registry = [
        {
            "versions": {
                "v1": {"report_date": "2024-12", "period_covered": "Q4 2024"},
                "v2": {"report_date": "2025-03", "period_covered": "Q1 2025"},
            }
        }
    ]
    with patch("src.corpus.registry.CORPUS_REGISTRY", mock_registry):
        result = _latest_period_overrides()

    assert "2024-12" in result
    assert "december 2024" in result
    assert "2025-03" in result
    assert "march 2025" in result


def test_latest_period_overrides_empty_registry() -> None:
    """An empty registry must return an empty tuple without raising."""
    with patch("src.corpus.registry.CORPUS_REGISTRY", []):
        result = _latest_period_overrides()
    assert result == ()


# ---------------------------------------------------------------------------
# _regex_classify_intent uses the live registry for latest-period detection
# ---------------------------------------------------------------------------


def test_regex_classify_uses_live_registry_for_latest_detection() -> None:
    """A query mentioning a date that is in the mocked registry's latest periods
    should classify as 'latest', not 'historical'."""
    # Inject a registry where 2030-06 is a known latest period.
    mock_registry = [{"report_date": "2030-06", "period_covered": "Q2 2030"}]
    with patch("src.corpus.registry.CORPUS_REGISTRY", mock_registry):
        result = _regex_classify_intent("What is the NOI in June 2030?")
    assert result == "latest"


def test_regex_classify_historical_when_not_in_live_registry() -> None:
    """A query mentioning a date that is NOT in the mocked latest periods should
    classify as 'historical'."""
    mock_registry = [{"report_date": "2030-06", "period_covered": "Q2 2030"}]
    with patch("src.corpus.registry.CORPUS_REGISTRY", mock_registry):
        # 2024-09 is not a latest period in this mock — historical
        result = _regex_classify_intent("What was the leverage in September 2024?")
    assert result == "historical"
