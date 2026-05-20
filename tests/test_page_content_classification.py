"""Tests for classify_page_content and retrieval SQL shape."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from src.ingestion.chunker import classify_page_content
from src.retrieval.bm25 import (
    _BM25_SQL,
    _BM25_SQL_FILTERED,
    _BM25_SQL_CT,
    _BM25_SQL_FILTERED_CT,
    _BOILERPLATE_FILTER,
)
from src.retrieval.vector import (
    _VECTOR_SQL,
    _VECTOR_SQL_FILTERED,
    _VECTOR_SQL_CT,
    _VECTOR_SQL_FILTERED_CT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(page_content_class: str, confidence: float) -> MagicMock:
    """Build a minimal Anthropic response mock for classify_page_content."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(
        {"page_content_class": page_content_class, "confidence": confidence}
    )
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Keyword pre-filter: ≥3 boilerplate signals → boilerplate_legal without LLM
# ---------------------------------------------------------------------------


def test_three_boilerplate_signals_no_llm_call() -> None:
    """Three or more boilerplate signals must short-circuit to boilerplate_legal."""
    text = (
        "This presentation contains forward-looking statements under the safe harbor "
        "provisions. This presentation does not constitute an offer to sell securities. "
        "All trademarks are registered trademark of their respective owners."
    )
    with patch("src.ingestion.page_classifier._llm_classify_page_content") as mock_llm:
        result = classify_page_content(text, "Legal Disclaimers")
    # LLM should not have been called at all.
    mock_llm.assert_not_called()
    assert result == "boilerplate_legal"


def test_two_signals_short_chunk_no_llm_call() -> None:
    """Two signals in a very short chunk (<150 words) should also be caught."""
    text = (
        "Forward-looking statements are subject to risks under the safe harbor "
        "provisions of the Securities Act."
    )
    # Under 150 tokens, two signals → boilerplate_legal
    with patch("src.ingestion.page_classifier._llm_classify_page_content") as mock_llm:
        result = classify_page_content(text, "")
    mock_llm.assert_not_called()
    assert result == "boilerplate_legal"


# ---------------------------------------------------------------------------
# Keyword pre-filter: short chunk with no digits → index_reference
# ---------------------------------------------------------------------------


def test_short_no_digit_chunk_is_index_reference() -> None:
    """A chunk under 25 words with no digits should be index_reference."""
    text = "Overview of Portfolio and Strategy"
    with patch("src.ingestion.page_classifier._llm_classify_page_content") as mock_llm:
        result = classify_page_content(text, "Table of Contents")
    mock_llm.assert_not_called()
    assert result == "index_reference"


def test_short_chunk_with_digits_reaches_llm() -> None:
    """A short chunk that contains digits is not caught by the index filter."""
    text = "NOI grew to $1.4B"  # < 25 words but has digits
    mock_resp = _mock_llm_response("substantive", 0.95)

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Financial Highlights")

    assert result == "substantive"


# ---------------------------------------------------------------------------
# LLM path: high-confidence response is returned
# ---------------------------------------------------------------------------


def test_llm_high_confidence_substantive_returned() -> None:
    mock_resp = _mock_llm_response("substantive", 0.92)
    text = (
        "Same-store occupancy reached 95.2% across the North American portfolio "
        "in Q4 2025, up 80 basis points year-over-year, driven by strong demand "
        "in hyperscale and enterprise segments."
    )

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Portfolio Metrics")

    assert result == "substantive"


def test_llm_high_confidence_cover_page_returned() -> None:
    mock_resp = _mock_llm_response("cover_page", 0.88)
    text = "Digital Realty Trust — Q4 2025 Investor Presentation  |  March 2026"

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Cover")

    assert result == "cover_page"


# ---------------------------------------------------------------------------
# LLM exception → unknown
# ---------------------------------------------------------------------------


def test_llm_exception_returns_unknown() -> None:
    text = (
        "Revenue increased significantly year over year across all operating segments, "
        "reflecting positive trends in the REIT sector with strong occupancy rates "
        "and improving net operating income metrics across the portfolio."
    )

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API unavailable")
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Financial Results")

    assert result == "unknown"


def test_llm_client_none_returns_unknown() -> None:
    """When no API key is set, client is None; classify_page_content returns unknown."""
    text = (
        "Portfolio occupancy improved 120 basis points quarter-over-quarter, "
        "driven by hyperscale leasing activity in Northern Virginia."
    )

    with patch("src.ingestion.page_classifier._get_page_content_client", return_value=None):
        result = classify_page_content(text, "Leasing Update")

    assert result == "unknown"


# ---------------------------------------------------------------------------
# LLM low confidence → unknown
# ---------------------------------------------------------------------------


def test_llm_low_confidence_returns_unknown() -> None:
    mock_resp = _mock_llm_response("cover_page", 0.55)
    text = (
        "Financial highlights for the fiscal year ended December 31 2025 are "
        "presented in this supplemental information package alongside the quarterly "
        "earnings release and associated guidance for the upcoming period."
    )

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Supplement Cover")

    assert result == "unknown"


def test_llm_exactly_at_threshold_returns_class() -> None:
    """Confidence exactly 0.80 (the threshold) should be accepted."""
    mock_resp = _mock_llm_response("boilerplate_legal", 0.80)
    text = (
        "No assurance can be given that the projected results described herein will "
        "be achieved. This presentation should not be relied on as legal advice "
        "regarding securities transactions or investment decisions."
    )

    with patch("src.ingestion.page_classifier._get_page_content_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_client_getter.return_value = mock_client

        result = classify_page_content(text, "Disclaimer")

    assert result == "boilerplate_legal"


# ---------------------------------------------------------------------------
# BM25 SQL shape: boilerplate filter is present; unknown not excluded
# ---------------------------------------------------------------------------


def test_bm25_sql_contains_boilerplate_filter() -> None:
    """All BM25 query variants must include the boilerplate exclusion clause."""
    for sql in (_BM25_SQL, _BM25_SQL_FILTERED, _BM25_SQL_CT, _BM25_SQL_FILTERED_CT):
        assert "page_content_class NOT IN" in sql, (
            f"BM25 query missing boilerplate filter:\n{sql}"
        )
        assert "boilerplate_legal" in sql
        assert "index_reference" in sql


def test_bm25_sql_unknown_not_excluded() -> None:
    """The BM25 WHERE clause must NOT exclude 'unknown' chunks.

    The filter uses NOT IN ('boilerplate_legal', 'index_reference') and also
    allows NULL.  Chunks with page_content_class = 'unknown' pass through.
    """
    for sql in (_BM25_SQL, _BM25_SQL_FILTERED, _BM25_SQL_CT, _BM25_SQL_FILTERED_CT):
        # 'unknown' must not appear inside the NOT IN list.
        assert "'unknown'" not in sql, (
            f"BM25 query incorrectly filters 'unknown' chunks:\n{sql}"
        )
        # NULL is covered by the IS NULL guard in the filter string.
        assert "IS NULL" in sql


def test_boilerplate_filter_allows_null() -> None:
    """The filter constant itself must include the IS NULL escape hatch."""
    assert "IS NULL" in _BOILERPLATE_FILTER


def test_bm25_sql_searches_both_tsvectors() -> None:
    """BM25 queries must search the union of chunk_text_tsv and the
    contextualized_text_tsv so entities present in only the original chunk
    text (and dropped by an imperfect contextualizing summary) remain
    reachable. The previous COALESCE-prefer-context form silently lost
    entities like 'Equinix' that the contextualizing pass omitted."""
    for sql in (_BM25_SQL, _BM25_SQL_FILTERED, _BM25_SQL_CT, _BM25_SQL_FILTERED_CT):
        assert "chunk_text_tsv || COALESCE(contextualized_text_tsv" in sql, (
            f"BM25 query must concatenate both tsvectors so entities only "
            f"in chunk_text are searchable:\n{sql}"
        )
        assert "COALESCE(contextualized_text_tsv, chunk_text_tsv)" not in sql, (
            f"Prior prefer-context COALESCE must be replaced:\n{sql}"
        )


# ---------------------------------------------------------------------------
# Vector SQL shape: no boilerplate filter; uses COALESCE embedding
# ---------------------------------------------------------------------------


def test_vector_sql_has_no_boilerplate_filter() -> None:
    """Vector search must NOT contain the boilerplate WHERE exclusion clause.

    The column may appear in the SELECT list for hydrating Chunk objects, but
    it must not appear in a WHERE filter — dense retrieval handles boilerplate
    pages via embedding distance and false-positive filtering would hurt recall.
    """
    for sql in (
        _VECTOR_SQL,
        _VECTOR_SQL_FILTERED,
        _VECTOR_SQL_CT,
        _VECTOR_SQL_FILTERED_CT,
    ):
        assert "page_content_class NOT IN" not in sql, (
            f"Vector query unexpectedly contains boilerplate exclusion filter:\n{sql}"
        )


def test_vector_sql_uses_coalesce_embedding() -> None:
    """Vector queries should use COALESCE(contextualized_embedding, embedding)."""
    for sql in (
        _VECTOR_SQL,
        _VECTOR_SQL_FILTERED,
        _VECTOR_SQL_CT,
        _VECTOR_SQL_FILTERED_CT,
    ):
        assert "COALESCE(contextualized_embedding, embedding)" in sql, (
            f"Vector query not using COALESCE embedding fallback:\n{sql}"
        )


# ---------------------------------------------------------------------------
# Chunk dataclass default
# ---------------------------------------------------------------------------


def test_chunk_default_page_content_class() -> None:
    """Chunk.page_content_class must default to 'unknown'."""
    from uuid import uuid4
    from src.models import Chunk

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="DLR",
        ticker="DLR",
        doc_type="investor_presentation",
        report_date="2025-12",
        doc_version="2025-12",
        content_type="text",
        chunk_text="some text",
    )
    assert chunk.page_content_class == "unknown"
