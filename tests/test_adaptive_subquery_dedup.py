"""Tests for the adaptive sub-query repetition guard + Haiku client singleton.

Covers:
  - Duplicate sub-query (exact match, case-insensitive, whitespace-collapsed) is skipped.
  - Near-duplicate (>=90% token overlap) is skipped.
  - Legitimately different sub-query fires normally.
  - _get_haiku_client returns the same object on successive calls.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers (duplicated minimally from test_adaptive_retrieve.py)
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Digital Realty",
    page_number: int = 5,
    chunk_text: str = "Net debt to EBITDA: 4.9x",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type="investor_presentation",
        report_date="2026-03",
        period_covered="Q4 2025",
        doc_version="2026-03",
        section_title="Balance Sheet",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk, score: float = 2.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _make_initial_result(
    query: str = "year-end 2025 net debt to EBITDA leverage ratio",
    contexts: list[RetrievedChunk] | None = None,
) -> RetrievalResult:
    if contexts is None:
        contexts = [_make_rc(_make_chunk())]
    return RetrievalResult(
        query=query,
        intent="historical",  # type: ignore[arg-type]
        contexts=contexts,
        companies=["Digital Realty"],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
    )


def _make_tool_use_response(sub_query: str, company_filter: list[str] = []) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"query": sub_query, "company_filter": company_filter}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_end_turn_response() -> MagicMock:
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    return response


def _make_mock_client(responses: list[MagicMock]) -> MagicMock:
    call_count = {"n": 0}

    def create(**kwargs: Any) -> MagicMock:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[idx]

    messages = MagicMock()
    messages.create = create
    client = MagicMock()
    client.messages = messages
    return client


def _empty_hop_result(query: str = "sub-q") -> RetrievalResult:
    return RetrievalResult(
        query=query,
        intent="historical",  # type: ignore[arg-type]
        contexts=[],
        companies=[],
        abstain=False,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# 2j — identical sub-query is skipped (exact match)
# ---------------------------------------------------------------------------


def test_duplicate_subquery_exact_match_skips_second_hop() -> None:
    """When Haiku proposes an identical sub-query to the original query, the hop is skipped.

    Evidence: bxp-morn-basic-1 and xdoc-basic-1 both show sub_queries repeated identically.
    Probe xdoc-adv-6 shows: sub_queries = ["Simon mall property tax expense",
    "Simon mall property tax expense"].
    """
    from src.retrieval.adaptive import adaptive_retrieve

    original_query = "year-end 2025 net debt to EBITDA leverage ratio"
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    # Haiku proposes the exact same string as the original query on hop 1.
    responses = [
        _make_tool_use_response(original_query),  # duplicate — should be skipped
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return _empty_hop_result(sub_query)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    # The sub-query that duplicates the original must not fire a retrieval pass.
    assert retrieve_core_calls == [], (
        "Duplicate sub-query should not fire _retrieve_core, "
        f"but got calls: {retrieve_core_calls}"
    )
    # The duplicate must not appear in sub_queries either.
    assert sub_queries == [], f"Expected no sub_queries, got: {sub_queries}"


def test_duplicate_subquery_case_insensitive_skips_hop() -> None:
    """Case-insensitive duplicate is rejected (e.g. different casing same words)."""
    from src.retrieval.adaptive import adaptive_retrieve

    original_query = "Simon Property Group mall property tax expense"
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    # Haiku returns the same query in different case.
    duplicate_different_case = "simon property group Mall Property TAX expense"
    responses = [
        _make_tool_use_response(duplicate_different_case),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return _empty_hop_result(sub_query)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    assert retrieve_core_calls == [], f"Case-insensitive duplicate should be skipped: {retrieve_core_calls}"
    assert sub_queries == []


def test_duplicate_subquery_whitespace_collapsed_skips_hop() -> None:
    """Whitespace-collapsed duplicate is rejected."""
    from src.retrieval.adaptive import adaptive_retrieve

    original_query = "BXP total in-service portfolio square footage June 30 2025"
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    # Same words but extra internal whitespace.
    duplicate_extra_spaces = "BXP  total  in-service  portfolio  square  footage  June  30  2025"
    responses = [
        _make_tool_use_response(duplicate_extra_spaces),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return _empty_hop_result(sub_query)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    assert retrieve_core_calls == [], f"Whitespace-collapsed duplicate should be skipped: {retrieve_core_calls}"
    assert sub_queries == []


def test_near_duplicate_high_token_overlap_skips_hop() -> None:
    """>=90% Jaccard token overlap causes the second hop to be skipped.

    e.g. xdoc-adv-6: original="Simon mall property tax expense",
    hop-2 sub_query="Simon mall property tax expense" — exact duplicate.
    Here we test a near-duplicate with 90%+ Jaccard overlap.

    Jaccard = |intersection| / |union|.  With 19 shared tokens and 1 extra,
    union=20, intersection=19, Jaccard = 19/20 = 0.95 >= 0.90 threshold.
    """
    from src.retrieval.adaptive import adaptive_retrieve

    # 19 unique tokens in both; near_dup adds 1 extra → Jaccard = 19/20 = 0.95.
    original_tokens = "a b c d e f g h i j k l m n o p q r s"  # 19 unique tokens
    near_dup_tokens = "a b c d e f g h i j k l m n o p q r s extra"  # 20 unique tokens

    original_query = original_tokens
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    near_dup = near_dup_tokens
    responses = [
        _make_tool_use_response(near_dup),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return _empty_hop_result(sub_query)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    # 19/20 = 0.95 >= 0.90 threshold → should be skipped.
    assert retrieve_core_calls == [], (
        f"Near-duplicate (Jaccard=0.95 >= 0.90) should be skipped: {retrieve_core_calls}"
    )
    assert sub_queries == []


def test_different_subquery_fires_hop() -> None:
    """A meaningfully different sub-query fires the second hop normally."""
    from src.retrieval.adaptive import adaptive_retrieve

    original_query = "year-end 2025 net debt to EBITDA leverage ratio"
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    # A legitimately different sub-query: different vocabulary, different angle.
    different_sub_query = "Public Storage PSA leverage ratio year end 2025"
    responses = [
        _make_tool_use_response(different_sub_query),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    hop_result = RetrievalResult(
        query=different_sub_query,
        intent="historical",  # type: ignore[arg-type]
        contexts=[_make_rc(_make_chunk(company="Public Storage", page_number=15))],
        companies=["Public Storage"],
        abstain=False,
        diagnostics={},
    )

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return hop_result

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    # The different sub-query MUST fire.
    assert retrieve_core_calls == [different_sub_query], (
        f"Different sub-query should fire _retrieve_core, got: {retrieve_core_calls}"
    )
    assert sub_queries == [different_sub_query]


def test_duplicate_among_prior_subqueries_skips_hop() -> None:
    """A sub-query duplicating a prior hop's sub-query is also skipped.

    Multi-hop scenario: hop 1 fires sub_q_1 (different from original),
    hop 2 proposes sub_q_1 again — this second hop must be skipped.
    """
    from src.retrieval.adaptive import adaptive_retrieve, MAX_ADAPTIVE_HOPS

    assert MAX_ADAPTIVE_HOPS >= 2, "This test requires at least 2 adaptive hops"

    original_query = "cross-company leverage comparison 2025"
    initial = _make_initial_result(query=original_query)
    conn = MagicMock()

    sub_q_1 = "Digital Realty net debt EBITDA ratio 2025"  # legitimately different from original
    # hop 2 proposes same as hop 1 — should be skipped
    responses = [
        _make_tool_use_response(sub_q_1),
        _make_tool_use_response(sub_q_1),  # duplicate of hop 1
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    retrieve_core_calls: list[str] = []

    def _mock_retrieve_core(sub_query, **kwargs):
        retrieve_core_calls.append(sub_query)
        return _empty_hop_result(sub_query)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", side_effect=_mock_retrieve_core):
            result, sub_queries = adaptive_retrieve(original_query, initial, conn)

    # sub_q_1 fires on hop 1; hop 2 duplicate of sub_q_1 must not fire.
    assert retrieve_core_calls == [sub_q_1], (
        f"Only one retrieve_core call expected; got: {retrieve_core_calls}"
    )
    assert sub_queries == [sub_q_1]


# ---------------------------------------------------------------------------
# _get_haiku_client returns a singleton (same object identity)
# ---------------------------------------------------------------------------


def test_get_haiku_client_returns_singleton() -> None:
    """Two successive calls to _get_haiku_client must return the same object."""
    import src.retrieval.adaptive as adaptive_mod

    # Reload module to ensure clean state (singleton reset).
    importlib.reload(adaptive_mod)

    fake_instance = MagicMock(name="FakeAnthropic")

    # Patch Anthropic constructor inside the module after reload.
    with patch("src.retrieval.adaptive.Anthropic", return_value=fake_instance) as mock_cls:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client_1 = adaptive_mod._get_haiku_client()
            client_2 = adaptive_mod._get_haiku_client()

    # The constructor should be called exactly once (singleton).
    assert mock_cls.call_count == 1, (
        f"Anthropic() was called {mock_cls.call_count} times; expected 1 (singleton)"
    )
    assert client_1 is client_2, "Two calls must return the same object identity"
