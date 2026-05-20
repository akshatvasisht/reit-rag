"""Tests for doc_subtype classification and version-group keying."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4


from src.models import Chunk, DocumentMeta, RetrievedChunk
from src.versioning.chains import dedupe_by_version_group, version_group_key
from src.retrieval.bm25 import _SELECT_COLS as _BM25_SELECT
from src.retrieval.vector import _SELECT_COLS as _VECTOR_SELECT
from src.retrieval.pipeline import _CHUNK_SELECT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "BXP",
    doc_type: str = "investor_presentation",
    doc_version: str = "2025-12",
    doc_subtype: str = "quarterly_investor_deck",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="BXP",
        doc_type=doc_type,
        report_date=doc_version,
        doc_version=doc_version,
        content_type="text",
        chunk_text="sample text",
        doc_subtype=doc_subtype,
    )


def _make_rc(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=1.0)


# ---------------------------------------------------------------------------
# DocumentMeta and Chunk default doc_subtype
# ---------------------------------------------------------------------------


def test_document_meta_default_subtype() -> None:
    meta = DocumentMeta(
        company="BXP",
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2025-12",
        period_covered="Q4 2025",
        doc_version="2025-12",
        source_path="/data/bxp.pdf",
    )
    assert meta.doc_subtype == "unknown"


def test_chunk_default_subtype() -> None:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="BXP",
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2025-12",
        doc_version="2025-12",
        content_type="text",
        chunk_text="sample",
    )
    assert chunk.doc_subtype == "unknown"


# ---------------------------------------------------------------------------
# version_group_key — includes doc_subtype
# ---------------------------------------------------------------------------


def test_version_group_key_returns_triple() -> None:
    chunk = _make_chunk(doc_subtype="quarterly_investor_deck")
    key = version_group_key(chunk)
    assert key == ("BXP", "investor_presentation", "quarterly_investor_deck")


def test_version_group_key_unknown_subtype() -> None:
    chunk = _make_chunk(doc_subtype="unknown")
    key = version_group_key(chunk)
    assert key == ("BXP", "investor_presentation", "unknown")


# ---------------------------------------------------------------------------
# Different doc_subtypes → different version groups
# ---------------------------------------------------------------------------


def test_different_subtypes_are_different_groups() -> None:
    """Two chunks with the same (company, doc_type) but different doc_subtype
    must land in different version groups and both survive deduplication."""
    deck = _make_chunk(doc_subtype="quarterly_investor_deck", doc_version="2025-12")
    session = _make_chunk(doc_subtype="investor_day_session", doc_version="2025-12")

    assert version_group_key(deck) != version_group_key(session)

    rc_deck = _make_rc(deck)
    rc_session = _make_rc(session)

    deduped = dedupe_by_version_group([rc_deck, rc_session], intent="latest")
    # Each subtype forms its own group; both survive because each group has
    # exactly one doc_version (no suppression).
    assert rc_deck in deduped
    assert rc_session in deduped


def test_same_subtype_latest_deduplication_suppresses_older() -> None:
    """Within a single (company, doc_type, doc_subtype) group, the older
    doc_version must be suppressed when intent is 'latest'."""
    older = _make_chunk(doc_subtype="quarterly_investor_deck", doc_version="2025-09")
    newer = _make_chunk(doc_subtype="quarterly_investor_deck", doc_version="2025-12")

    rc_older = _make_rc(older)
    rc_newer = _make_rc(newer)

    deduped = dedupe_by_version_group([rc_older, rc_newer], intent="latest")
    assert rc_newer in deduped
    assert rc_older not in deduped


# ---------------------------------------------------------------------------
# classify_doc_subtype — mocked Anthropic client
# ---------------------------------------------------------------------------


def _build_mock_response(subtype: str, confidence: float) -> MagicMock:
    """Build a minimal mock that looks like an Anthropic messages.create response."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps({"doc_subtype": subtype, "confidence": confidence})
    response = MagicMock()
    response.content = [block]
    return response


def test_classify_doc_subtype_high_confidence_returns_llm_value() -> None:
    from src.ingestion.metadata import classify_doc_subtype

    mock_response = _build_mock_response("quarterly_investor_deck", 0.92)

    with patch("src.ingestion.metadata._get_subtype_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_doc_subtype(
            first_page_text="Q4 2025 Investor Presentation BXP",
            filename="bxp_q4_2025.pdf",
            doc_type="investor_presentation",
        )

    assert result == "quarterly_investor_deck"


def test_classify_doc_subtype_low_confidence_returns_unknown() -> None:
    from src.ingestion.metadata import classify_doc_subtype

    mock_response = _build_mock_response("investor_day_session", 0.60)

    with patch("src.ingestion.metadata._get_subtype_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_doc_subtype(
            first_page_text="Investor Day Morning Session",
            filename="bxp_morning_session.pdf",
            doc_type="investor_presentation",
        )

    assert result == "unknown"


def test_classify_doc_subtype_exactly_at_threshold_returns_value() -> None:
    from src.ingestion.metadata import classify_doc_subtype

    mock_response = _build_mock_response("merger_presentation", 0.75)

    with patch("src.ingestion.metadata._get_subtype_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_doc_subtype(
            first_page_text="Merger of Public Storage and ...",
            filename="psa_merger.pdf",
            doc_type="investor_presentation",
        )

    assert result == "merger_presentation"


def test_classify_doc_subtype_exception_returns_unknown() -> None:
    from src.ingestion.metadata import classify_doc_subtype

    with patch("src.ingestion.metadata._get_subtype_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API unavailable")
        mock_get_client.return_value = mock_client

        result = classify_doc_subtype(
            first_page_text="some page text",
            filename="test.pdf",
            doc_type="investor_presentation",
        )

    assert result == "unknown"


def test_classify_doc_subtype_connection_error_returns_unknown() -> None:
    from src.ingestion.metadata import classify_doc_subtype

    with patch("src.ingestion.metadata._get_subtype_client") as mock_get_client:
        mock_get_client.side_effect = RuntimeError("ANTHROPIC_API_KEY is not set")

        result = classify_doc_subtype(
            first_page_text="any text",
            filename="test.pdf",
            doc_type="thematic_report",
        )

    assert result == "unknown"


# ---------------------------------------------------------------------------
# SELECT column lists — doc_subtype must be present so version_group_key works
# ---------------------------------------------------------------------------


def test_bm25_select_includes_doc_subtype() -> None:
    """BM25 SELECT must include doc_subtype so version_group_key receives the real value."""
    assert "doc_subtype" in _BM25_SELECT, (
        f"doc_subtype missing from BM25 SELECT:\n{_BM25_SELECT}"
    )


def test_vector_select_includes_doc_subtype() -> None:
    """Vector SELECT must include doc_subtype so version_group_key receives the real value."""
    assert "doc_subtype" in _VECTOR_SELECT, (
        f"doc_subtype missing from vector SELECT:\n{_VECTOR_SELECT}"
    )


def test_pipeline_chunk_select_includes_doc_subtype() -> None:
    """Pipeline _CHUNK_SELECT must include doc_subtype for conflict and sibling chunks."""
    assert "doc_subtype" in _CHUNK_SELECT, (
        f"doc_subtype missing from pipeline _CHUNK_SELECT:\n{_CHUNK_SELECT}"
    )


def test_expand_to_parents_sql_includes_doc_subtype() -> None:
    """expand_to_parents SQL must include doc_subtype so parent chunks carry the right key."""
    import inspect
    from src.versioning.chains import expand_to_parents
    source = inspect.getsource(expand_to_parents)
    assert "doc_subtype" in source, (
        "doc_subtype missing from expand_to_parents SQL"
    )
