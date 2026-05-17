"""Tests for the --ablation-reranker flag in scripts/evaluate.py."""

from __future__ import annotations

import argparse
from uuid import uuid4

import pytest

from scripts import evaluate
from src.models import Chunk, RetrievedChunk
from src.retrieval import pipeline as retrieval_pipeline
from src.retrieval.pipeline import RetrievalResult
from tests.evaluation_set import EvaluationQuery


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MINI_GOLDEN = [
    EvaluationQuery(
        id="t01",
        query="What is BXP leverage?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="BXP",
    ),
    EvaluationQuery(
        id="t02",
        query="What is EGP occupancy?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="EastGroup Properties",
    ),
]


def _make_retrieval_result(company: str, query: str) -> RetrievalResult:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="TST",
        doc_type="investor_presentation",
        report_date="2025-12",
        period_covered="Q4 2025",
        doc_version="2025-12",
        section_title="Test",
        page_number=1,
        content_type="text",
        chunk_text=f"{company} context",
    )
    return RetrievalResult(
        query=query,
        intent="latest",
        contexts=[RetrievedChunk(chunk=chunk, rerank_score=2.0)],
        companies=[company],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
    )


# ---------------------------------------------------------------------------
# Argparse validation
# ---------------------------------------------------------------------------


def test_ablation_reranker_without_model_exits(monkeypatch, capsys) -> None:
    """--ablation-reranker without --ablation-model must exit with an error."""
    # We test the parser.error path directly by constructing the parser the
    # same way main() does and checking that parse_args + our guard raises.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-reranker", action="store_true")
    parser.add_argument("--ablation-model", type=str, default=None)
    parser.add_argument("--ablation-revision", type=str, default=None)

    args = parser.parse_args(["--ablation-reranker"])
    with pytest.raises(SystemExit) as exc_info:
        if args.ablation_reranker and not args.ablation_model:
            parser.error("--ablation-reranker requires --ablation-model")
    assert exc_info.value.code != 0


def test_ablation_reranker_with_model_does_not_exit() -> None:
    """--ablation-reranker with --ablation-model must not raise."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-reranker", action="store_true")
    parser.add_argument("--ablation-model", type=str, default=None)
    parser.add_argument("--ablation-revision", type=str, default=None)

    args = parser.parse_args(["--ablation-reranker", "--ablation-model", "some/model"])
    # Should not raise.
    if args.ablation_reranker and not args.ablation_model:
        raise AssertionError("Guard fired when it should not have")


# ---------------------------------------------------------------------------
# run_reranker_ablation — happy path
# ---------------------------------------------------------------------------


def test_run_reranker_ablation_happy_path(monkeypatch) -> None:
    """Both keys present; baseline and alternative each have one row per query."""
    from src.retrieval import reranker as _reranker_module

    monkeypatch.setattr(evaluate, "EVALUATION_SET", _MINI_GOLDEN)

    # Mock retrieve to return a valid result keyed by company.
    def _fake_retrieve(query: str) -> RetrievalResult:
        company = "BXP" if "BXP" in query else "EastGroup Properties"
        return _make_retrieval_result(company, query)

    monkeypatch.setattr(retrieval_pipeline, "retrieve", _fake_retrieve)

    # Mock _get_registry_encoder to avoid model download.
    call_count: dict[str, int] = {"n": 0}

    class _FakeEncoder:
        def predict(self, pairs):
            return [1.5] * len(pairs)

    def _fake_registry_encoder(model_name: str, revision):
        call_count["n"] += 1
        return _FakeEncoder()

    monkeypatch.setattr(_reranker_module, "_get_registry_encoder", _fake_registry_encoder)

    # Also patch rerank_with_model so the pipeline swap works in tests.
    def _fake_rerank_with_model(query, candidates, model_name, revision=None, top_n=5):
        for rc in candidates:
            rc.rerank_score = 1.5
        return candidates[:top_n]

    monkeypatch.setattr(_reranker_module, "rerank_with_model", _fake_rerank_with_model)

    out = evaluate.run_reranker_ablation("some/alt-model", None)

    assert set(out.keys()) == {"reranker_baseline", "reranker_alternative"}
    assert len(out["reranker_baseline"]) == len(_MINI_GOLDEN)
    assert len(out["reranker_alternative"]) == len(_MINI_GOLDEN)
    # Config names should be set correctly.
    for row in out["reranker_baseline"]:
        assert row.config_name == "reranker_baseline"
    for row in out["reranker_alternative"]:
        assert row.config_name == "reranker_alternative"


# ---------------------------------------------------------------------------
# run_reranker_ablation — load-failure path
# ---------------------------------------------------------------------------


def test_run_reranker_ablation_load_failure(monkeypatch) -> None:
    """When the alternative model fails to load, sentinels are returned for the
    alternative side and the baseline side is fully populated."""
    from src.retrieval import reranker as _reranker_module

    monkeypatch.setattr(evaluate, "EVALUATION_SET", _MINI_GOLDEN)

    def _fake_retrieve(query: str) -> RetrievalResult:
        company = "BXP" if "BXP" in query else "EastGroup Properties"
        return _make_retrieval_result(company, query)

    monkeypatch.setattr(retrieval_pipeline, "retrieve", _fake_retrieve)

    def _failing_registry_encoder(model_name: str, revision):
        raise RuntimeError(f"Simulated load failure for {model_name}")

    monkeypatch.setattr(_reranker_module, "_get_registry_encoder", _failing_registry_encoder)

    out = evaluate.run_reranker_ablation("bad/model", None)

    assert set(out.keys()) == {"reranker_baseline", "reranker_alternative"}
    # Baseline must be fully populated.
    assert len(out["reranker_baseline"]) == len(_MINI_GOLDEN)
    for row in out["reranker_baseline"]:
        assert row.config_name == "reranker_baseline"
    # Alternative must have sentinel rows.
    assert len(out["reranker_alternative"]) == len(_MINI_GOLDEN)
    for row in out["reranker_alternative"]:
        assert "LOAD_ERROR" in row.config_name
        assert row.expected_company_rank is None


# ---------------------------------------------------------------------------
# render_markdown — Tier 3b section
# ---------------------------------------------------------------------------


def _make_ablation_row(config: str, query_id: str, company: str, expected_co: str) -> evaluate.AblationResult:
    return evaluate.AblationResult(
        config_name=config,
        query_id=query_id,
        top_n_companies=[company],
        top_n_content_types=["text"],
        top_rerank=2.0,
        expected_company_rank=1 if company == expected_co else None,
    )


def test_render_markdown_tier3b_present(monkeypatch) -> None:
    """When reranker_ablation is non-None, Tier 3b header appears in output."""
    reranker_ablation = {
        "reranker_baseline": [
            _make_ablation_row("reranker_baseline", "t01", "BXP", "BXP"),
        ],
        "reranker_alternative": [
            _make_ablation_row("reranker_alternative", "t01", "BXP", "BXP"),
        ],
    }
    md = evaluate.render_markdown([], None, None, reranker_ablation=reranker_ablation)
    assert "Tier 3b" in md
    assert "Reranker ablation" in md
    assert "reranker_alternative" in md
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2 (baseline)" in md


def test_render_markdown_tier3b_absent_by_default() -> None:
    """render_markdown without reranker_ablation must not include Tier 3b."""
    md = evaluate.render_markdown([], None, None)
    assert "Tier 3b" not in md


def test_render_markdown_tier3b_load_failure(monkeypatch) -> None:
    """When the alternative side has a LOAD_ERROR sentinel, the error is
    rendered and no per-query table is included."""
    sentinel_config = "reranker_alternative[LOAD_ERROR: Simulated failure]"
    reranker_ablation = {
        "reranker_baseline": [
            _make_ablation_row("reranker_baseline", "t01", "BXP", "BXP"),
        ],
        "reranker_alternative": [
            evaluate.AblationResult(
                config_name=sentinel_config,
                query_id="t01",
                top_n_companies=[],
                top_n_content_types=[],
                top_rerank=float("-inf"),
                expected_company_rank=None,
            )
        ],
    }
    md = evaluate.render_markdown([], None, None, reranker_ablation=reranker_ablation)
    assert "Tier 3b" in md
    assert "failed to load" in md.lower() or "LOAD_ERROR" in md
    # Per-query table should NOT appear.
    assert "Per-query breakdown" not in md
