"""Unit tests for the short-ticker detection in extract_entity_candidates."""

from __future__ import annotations

import pytest

from src.retrieval.entity_filter import extract_entity_candidates


# ---------------------------------------------------------------------------
# Short all-caps ticker detection
# ---------------------------------------------------------------------------


def test_bxp_ticker_detected() -> None:
    """Three-letter ticker BXP must be surfaced."""
    result = extract_entity_candidates("What is BXP leverage?")
    assert "BXP" in result


def test_amt_ticker_detected() -> None:
    """Three-letter ticker AMT must be surfaced."""
    result = extract_entity_candidates("What is AMT's portfolio size?")
    assert "AMT" in result


def test_multi_word_phrase_preserved() -> None:
    """Multi-word capitalized company names must still be captured."""
    result = extract_entity_candidates("What is Simon Property Group's foot traffic?")
    assert "Simon Property Group" in result


def test_ffo_stopword_excluded() -> None:
    """FFO is domain vocabulary; it must not appear as a candidate."""
    result = extract_entity_candidates("Compare FFO and AFFO trends.")
    assert "FFO" not in result
    assert "AFFO" not in result


def test_q4_stopword_excluded() -> None:
    """Q4 is a time-period label; it must not appear as a candidate."""
    result = extract_entity_candidates("Q4 2025 results")
    assert "Q4" not in result


def test_reit_stopword_excluded() -> None:
    """REIT is domain vocabulary; it must not appear as a candidate."""
    result = extract_entity_candidates("Which REIT has the highest FFO?")
    assert "REIT" not in result


def test_ebitda_stopword_excluded() -> None:
    """EBITDA is a financial metric abbreviation; it must not appear as a candidate."""
    result = extract_entity_candidates("What is the debt-to-EBITDA ratio?")
    assert "EBITDA" not in result


def test_short_ticker_and_phrase_coexist() -> None:
    """Both a short ticker and a multi-word phrase can be captured in the same query."""
    result = extract_entity_candidates("Compare BXP to Boston Properties leverage.")
    assert "BXP" in result
    # "Boston Properties" should also be captured by the phrase regex.
    assert any("Boston" in c for c in result)


def test_empty_query_returns_empty() -> None:
    result = extract_entity_candidates("")
    assert result == []


def test_no_false_positives_from_common_question_words() -> None:
    """Standard question-opening words must not leak as candidates."""
    result = extract_entity_candidates("What is the latest guidance?")
    # None of these should appear as candidates.
    for word in ("WHAT", "What"):
        assert word not in result


def test_pld_ticker_detected() -> None:
    """Three-letter ticker PLD (Prologis) must be surfaced."""
    result = extract_entity_candidates("What are PLD's development starts?")
    assert "PLD" in result


def test_us_stopword_excluded() -> None:
    """US is on the stopword list; it must not appear as a candidate."""
    result = extract_entity_candidates("How does US exposure affect returns?")
    assert "US" not in result
