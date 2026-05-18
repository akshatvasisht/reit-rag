"""Unit tests for the per-intent rerank-budget dispatch.

Comparison / synthesis / historical queries need a wider top-K than the
default 5 because their answers are assembled from multiple evidence chunks
that compete for rerank positions. The dispatch helper lives in
src/retrieval/pipeline.py and is consulted by retrieve() before delegating
to the synthesis branch or _retrieve_core.
"""

from __future__ import annotations

from src.retrieval.pipeline import (
    RERANK_TOP_N,
    _RERANK_TOP_N_BY_INTENT,
    _rerank_budget_for_intent,
)


def test_default_intent_uses_default_budget() -> None:
    assert _rerank_budget_for_intent("latest") == RERANK_TOP_N


def test_conflict_intent_uses_default_budget() -> None:
    # conflict is single-issuer; the default budget is sufficient.
    assert _rerank_budget_for_intent("conflict") == RERANK_TOP_N


def test_comparison_intent_uses_wider_budget() -> None:
    assert _rerank_budget_for_intent("comparison") == 8
    assert _rerank_budget_for_intent("comparison") > RERANK_TOP_N


def test_all_company_synthesis_intent_uses_wider_budget() -> None:
    assert _rerank_budget_for_intent("all_company_synthesis") == 8


def test_historical_intent_uses_intermediate_budget() -> None:
    assert _rerank_budget_for_intent("historical") == 7
    assert _rerank_budget_for_intent("historical") > RERANK_TOP_N


def test_unknown_intent_falls_back_to_default() -> None:
    assert _rerank_budget_for_intent("definitely_not_a_real_intent") == RERANK_TOP_N


def test_override_table_does_not_include_latest() -> None:
    """`latest` is the most common intent; widening its budget would dilute
    every single-fact query in the eval. Lock that it's not in the override
    table so a future careless addition is caught."""
    assert "latest" not in _RERANK_TOP_N_BY_INTENT


def test_override_table_widens_only_multi_chunk_intents() -> None:
    """Every override must widen to more than the default; a 'narrower-than-
    default' override would be a bug. Also asserts the override values are
    sane (<= 12, since beyond that the LLM token cost balloons and
    reranker discrimination dies)."""
    for intent, budget in _RERANK_TOP_N_BY_INTENT.items():
        assert budget > RERANK_TOP_N, (
            f"override for {intent!r} ({budget}) is not wider than the "
            f"default {RERANK_TOP_N}; the override would be a no-op or a "
            f"regression"
        )
        assert budget <= 12, (
            f"override for {intent!r} ({budget}) exceeds the sane cap of 12; "
            f"beyond that point the LLM context window dilutes too far"
        )
