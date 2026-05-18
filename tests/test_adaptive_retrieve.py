"""Tests for adaptive multi-hop retrieval.

All tests mock the Anthropic client and the DB connection — no network or
real database access.  Each test should complete in well under 2 seconds.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, RetrievedChunk, StructuredAnswer, Claim
from src.retrieval.adaptive import (
    ADAPTIVE_INTENTS,
    MAX_ADAPTIVE_HOPS,
    adaptive_retrieve,
)
from src.retrieval.pipeline import RetrievalResult, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Digital Realty",
    doc_version: str = "2026-03",
    report_date: str = "2026-03",
    page_number: int = 5,
    chunk_text: str = "Net debt to EBITDA: 5.2x",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type="investor_presentation",
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=doc_version,
        section_title="Balance Sheet",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk, score: float = 2.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _make_initial_result(
    contexts: list[RetrievedChunk] | None = None,
    intent: str = "historical",
    abstain: bool = False,
) -> RetrievalResult:
    if contexts is None:
        contexts = [_make_rc(_make_chunk())]
    return RetrievalResult(
        query="How does DLR leverage compare to its target?",
        intent=intent,  # type: ignore[arg-type]
        contexts=contexts,
        companies=["Digital Realty"],
        abstain=abstain,
        diagnostics={"top_rerank_score": 2.0},
    )


def _make_tool_use_response(sub_query: str, company_filter: list[str]) -> MagicMock:
    """Build a mock Anthropic response with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"query": sub_query, "company_filter": company_filter}

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_end_turn_response() -> MagicMock:
    """Build a mock Anthropic response with stop_reason=end_turn."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    return response


def _make_mock_client(responses: list[MagicMock]) -> MagicMock:
    """Build a mock Anthropic client whose messages.create() iterates responses."""
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


# ---------------------------------------------------------------------------
# adaptive_retrieve: no tool call → no extra hop
# ---------------------------------------------------------------------------


def test_adaptive_retrieve_end_turn_no_hop() -> None:
    """When Haiku returns end_turn immediately, no extra hop fires."""
    initial = _make_initial_result()
    conn = MagicMock()
    mock_client = _make_mock_client([_make_end_turn_response()])

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        result, sub_queries = adaptive_retrieve(
            "How does DLR leverage compare to its target?",
            initial,
            conn,
        )

    assert sub_queries == []
    assert len(result.contexts) == len(initial.contexts)
    assert {rc.chunk.id for rc in result.contexts} == {rc.chunk.id for rc in initial.contexts}


@pytest.mark.parametrize("fl", [True, False])
def test_adaptive_retrieve_preserves_forward_looking(fl: bool) -> None:
    """merged_result must carry initial_result.forward_looking through the adaptive
    rebuild. Without this propagation, forward-looking-classified queries would
    silently lose their FL signal on the adaptive multi-hop path and the
    forward-looking guard inside the generator would never fire there."""
    initial = RetrievalResult(
        query="What is BXP's 2026 FFO guidance?",
        intent="historical",  # type: ignore[arg-type]
        contexts=[_make_rc(_make_chunk())],
        companies=["BXP"],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
        forward_looking=fl,
    )
    conn = MagicMock()
    mock_client = _make_mock_client([_make_end_turn_response()])

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        result, _ = adaptive_retrieve(initial.query, initial, conn)

    assert result.forward_looking is fl, (
        f"adaptive_retrieve dropped forward_looking; "
        f"expected {fl}, got {result.forward_looking}"
    )


# ---------------------------------------------------------------------------
# adaptive_retrieve: one tool_use then end_turn → one hop, merged dedupe
# ---------------------------------------------------------------------------


def test_adaptive_retrieve_one_hop_merges_new_contexts() -> None:
    """One tool_use → one extra retrieval pass; new chunks merged, dedupe works."""
    initial_chunk = _make_chunk(company="Digital Realty", page_number=5)
    initial_rc = _make_rc(initial_chunk)
    initial = _make_initial_result(contexts=[initial_rc])

    # The hop retrieval returns one new chunk and one duplicate.
    new_chunk = _make_chunk(company="Digital Realty", page_number=10, chunk_text="Covenant: <5.5x")
    dup_chunk = initial_chunk  # same object → same UUID
    hop_contexts = [_make_rc(new_chunk, score=1.5), _make_rc(dup_chunk, score=1.2)]

    hop_result = RetrievalResult(
        query="DLR debt covenant target",
        intent="historical",  # type: ignore[arg-type]
        contexts=hop_contexts,
        companies=["Digital Realty"],
        abstain=False,
        diagnostics={"top_rerank_score": 1.5},
    )

    conn = MagicMock()
    responses = [
        _make_tool_use_response("DLR debt covenant target", ["Digital Realty"]),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", return_value=hop_result):
            result, sub_queries = adaptive_retrieve(
                "How does DLR leverage compare to its target?",
                initial,
                conn,
            )

    assert sub_queries == ["DLR debt covenant target"]
    chunk_ids = {rc.chunk.id for rc in result.contexts}
    assert initial_chunk.id in chunk_ids
    assert new_chunk.id in chunk_ids
    # No duplicate: initial_chunk appears once only.
    assert len([rc for rc in result.contexts if rc.chunk.id == initial_chunk.id]) == 1


# ---------------------------------------------------------------------------
# adaptive_retrieve: cap at MAX_ADAPTIVE_HOPS
# ---------------------------------------------------------------------------


def test_adaptive_retrieve_caps_at_max_hops() -> None:
    """When Haiku calls tool_use repeatedly, we stop after MAX_ADAPTIVE_HOPS hops."""
    initial = _make_initial_result()
    conn = MagicMock()

    # Provide more tool_use responses than the cap allows.
    responses = [
        _make_tool_use_response(f"sub-question {i}", []) for i in range(MAX_ADAPTIVE_HOPS + 3)
    ]
    mock_client = _make_mock_client(responses)

    hop_result = RetrievalResult(
        query="sub-q",
        intent="historical",  # type: ignore[arg-type]
        contexts=[],
        companies=[],
        abstain=False,
        diagnostics={},
    )

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", return_value=hop_result):
            result, sub_queries = adaptive_retrieve(
                "How does DLR leverage compare to its target?",
                initial,
                conn,
            )

    assert len(sub_queries) == MAX_ADAPTIVE_HOPS


# ---------------------------------------------------------------------------
# adaptive_retrieve: exception in orchestrator → break, original contexts
# ---------------------------------------------------------------------------


def test_adaptive_retrieve_exception_returns_original_contexts() -> None:
    """When the Anthropic client raises, original contexts are returned untouched."""
    initial_chunk = _make_chunk()
    initial_rc = _make_rc(initial_chunk)
    initial = _make_initial_result(contexts=[initial_rc])
    conn = MagicMock()

    failing_client = MagicMock()
    failing_client.messages.create.side_effect = RuntimeError("network failure")

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=failing_client):
        result, sub_queries = adaptive_retrieve(
            "How does DLR leverage compare to its target?",
            initial,
            conn,
        )

    assert sub_queries == []
    assert {rc.chunk.id for rc in result.contexts} == {initial_chunk.id}


def test_adaptive_retrieve_client_init_exception_returns_original() -> None:
    """When the client cannot be initialised, original contexts are returned untouched."""
    initial_chunk = _make_chunk()
    initial = _make_initial_result(contexts=[_make_rc(initial_chunk)])
    conn = MagicMock()

    with patch(
        "src.retrieval.adaptive._get_haiku_client",
        side_effect=RuntimeError("no API key"),
    ):
        result, sub_queries = adaptive_retrieve(
            "query",
            initial,
            conn,
        )

    assert sub_queries == []
    assert {rc.chunk.id for rc in result.contexts} == {initial_chunk.id}


# ---------------------------------------------------------------------------
# retrieve() routing: latest skips adaptive, others run adaptive
# ---------------------------------------------------------------------------


def test_retrieve_latest_intent_skips_adaptive(monkeypatch) -> None:
    """'latest' intent must not invoke adaptive_retrieve."""
    adaptive_called: list[bool] = []

    def _mock_adaptive(query, initial, conn):
        adaptive_called.append(True)
        return initial, []

    from src.retrieval import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "classify_intent_full", lambda q: ("latest", False))
    monkeypatch.setattr(_pipeline, "extract_companies", lambda q: [])
    monkeypatch.setattr(_pipeline, "adaptive_retrieve", _mock_adaptive)

    fake_conn = MagicMock()
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(_pipeline, "connect", lambda: fake_conn)
    monkeypatch.setattr(_pipeline, "_retrieve_core", lambda *a, **kw: RetrievalResult(
        query="q", intent="latest", contexts=[], companies=[], abstain=False, diagnostics={},
    ))

    _pipeline.retrieve("What is DLR's current leverage?")
    assert adaptive_called == []


@pytest.mark.parametrize("intent", ["historical", "comparison", "conflict"])
def test_retrieve_adaptive_intents_run_adaptive(monkeypatch, intent: str) -> None:
    """historical / comparison / conflict intents must invoke adaptive_retrieve."""
    adaptive_called: list[str] = []
    initial_result = RetrievalResult(
        query="q", intent=intent, contexts=[], companies=[], abstain=False,  # type: ignore[arg-type]
        diagnostics={"top_rerank_score": 2.0},
    )

    def _mock_adaptive(query, init, conn):
        adaptive_called.append(intent)
        return init, ["sub-q"]

    from src.retrieval import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "classify_intent_full", lambda q: (intent, False))
    monkeypatch.setattr(_pipeline, "extract_companies", lambda q: [])
    monkeypatch.setattr(_pipeline, "adaptive_retrieve", _mock_adaptive)

    fake_conn = MagicMock()
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(_pipeline, "connect", lambda: fake_conn)
    monkeypatch.setattr(_pipeline, "_retrieve_core", lambda *a, **kw: initial_result)

    _pipeline.retrieve("multi-hop query")
    assert adaptive_called == [intent]


def test_retrieve_all_company_synthesis_routes_through_synthesis_then_adaptive(
    monkeypatch,
) -> None:
    """all_company_synthesis must call retrieve_all_company_synthesis first,
    then adaptive_retrieve on the non-abstain result."""
    synthesis_called: list[bool] = []
    adaptive_called: list[bool] = []

    synthesis_result = RetrievalResult(
        query="q",
        intent="all_company_synthesis",
        contexts=[_make_rc(_make_chunk())],
        companies=[],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
    )

    def _mock_synthesis(query, conn, **_kwargs):
        synthesis_called.append(True)
        return synthesis_result

    def _mock_adaptive(query, init, conn):
        adaptive_called.append(True)
        return init, []

    from src.retrieval import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "classify_intent_full", lambda q: ("all_company_synthesis", False))
    monkeypatch.setattr(_pipeline, "extract_companies", lambda q: [])
    monkeypatch.setattr(_pipeline, "retrieve_all_company_synthesis", _mock_synthesis)
    monkeypatch.setattr(_pipeline, "adaptive_retrieve", _mock_adaptive)

    fake_conn = MagicMock()
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(_pipeline, "connect", lambda: fake_conn)

    _pipeline.retrieve("Which REIT has the highest NOI growth?")
    assert synthesis_called == [True]
    assert adaptive_called == [True]


# ---------------------------------------------------------------------------
# StructuredAnswer carries retrieval_hops and sub_queries_fired post-generation
# ---------------------------------------------------------------------------


def test_structured_answer_defaults() -> None:
    """StructuredAnswer must default retrieval_hops=0 and sub_queries_fired=[]."""
    sa = StructuredAnswer(
        answer_prose="test",
        claims=[],
        abstain=False,
        abstain_reason="",
        forward_looking=False,
    )
    assert sa.retrieval_hops == 0
    assert sa.sub_queries_fired == []


def test_answer_structured_populates_hop_diagnostics(monkeypatch) -> None:
    """answer_structured() must copy retrieval_hops and sub_queries_fired from diagnostics."""
    import json
    from src.generation import generator

    hop_diagnostics = {
        "retrieval_hops": 2,
        "sub_queries": ["sub-q1", "sub-q2"],
        "top_rerank_score": 2.5,
    }
    retrieval_result = RetrievalResult(
        query="test query",
        intent="historical",  # type: ignore[arg-type]
        contexts=[_make_rc(_make_chunk())],
        companies=["Digital Realty"],
        abstain=False,
        diagnostics=hop_diagnostics,
    )

    captured: list[StructuredAnswer] = []

    def _mock_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):
        sa = StructuredAnswer(
            answer_prose="DLR leverage is 5.2x.",
            claims=[],
            abstain=False,
            abstain_reason="",
            forward_looking=False,
        )
        captured.append(sa)
        return sa

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval_result)
    monkeypatch.setattr(generator, "_generate_structured", _mock_generate)
    monkeypatch.setattr(generator, "check_citations", lambda t, c: MagicMock(
        supported=0, total=0, faithfulness_ratio=1.0, unsupported=[], numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda s, c: [])

    generator.answer_structured("multi-hop test query")

    assert len(captured) == 1
    sa = captured[0]
    assert sa.retrieval_hops == 2
    assert sa.sub_queries_fired == ["sub-q1", "sub-q2"]


# ---------------------------------------------------------------------------
# Sub-query retrieval produces deterministic context ordering
# ---------------------------------------------------------------------------


def test_adaptive_retrieve_applies_d1_sort() -> None:
    """Merged contexts must be sorted by (company, report_date, page_number)."""
    chunk_a = _make_chunk(company="Zeta REIT", report_date="2026-01", page_number=10)
    chunk_b = _make_chunk(company="Alpha REIT", report_date="2025-12", page_number=5)
    chunk_c = _make_chunk(company="Alpha REIT", report_date="2025-12", page_number=3)

    initial = _make_initial_result(contexts=[_make_rc(chunk_a), _make_rc(chunk_b)])
    conn = MagicMock()

    hop_result = RetrievalResult(
        query="sub-q",
        intent="historical",  # type: ignore[arg-type]
        contexts=[_make_rc(chunk_c)],
        companies=[],
        abstain=False,
        diagnostics={},
    )

    responses = [
        _make_tool_use_response("sub-q", []),
        _make_end_turn_response(),
    ]
    mock_client = _make_mock_client(responses)

    with patch("src.retrieval.adaptive._get_haiku_client", return_value=mock_client):
        with patch("src.retrieval.pipeline._retrieve_core", return_value=hop_result):
            result, _ = adaptive_retrieve("original query", initial, conn)

    companies_order = [rc.chunk.company for rc in result.contexts]
    # Alpha REIT < Zeta REIT alphabetically
    assert companies_order.index("Alpha REIT") < companies_order.index("Zeta REIT")
    # Within Alpha REIT, page 3 before page 5
    alpha_pages = [
        rc.chunk.page_number for rc in result.contexts if rc.chunk.company == "Alpha REIT"
    ]
    assert alpha_pages == sorted(alpha_pages)
