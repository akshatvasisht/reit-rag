from __future__ import annotations

from uuid import uuid4

from scripts import evaluate
from src.models import Chunk, RetrievedChunk
from src.retrieval import pipeline as retrieval_pipeline
from src.retrieval.pipeline import RetrievalResult
from tests.evaluation_set import EvaluationQuery


def _fake_retrieve(query: str) -> RetrievalResult:
    """Return synthetic retrieval that reflects current entity-filter behavior."""
    companies = retrieval_pipeline.extract_companies(query)
    company = companies[0] if companies else "NO_FILTER"
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2025-12",
        period_covered="Q4 2025",
        doc_version="2025-12",
        section_title="Leverage",
        page_number=1,
        content_type="text",
        chunk_text=f"{company} leverage context",
    )
    return RetrievalResult(
        query=query,
        intent="latest",
        contexts=[RetrievedChunk(chunk=chunk, rerank_score=1.0)],
        companies=companies,
        abstain=False,
        diagnostics={"top_rerank_score": 1.0},
    )


def test_run_ablation_no_entity_filter_changes_behavior(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_pipeline, "retrieve", _fake_retrieve)
    monkeypatch.setattr(
        evaluate,
        "EVALUATION_SET",
        [
            EvaluationQuery(
                id="t01",
                query="What is BXP leverage?",
                category="factual_citation",
                expected_intent="latest",
                expected_company="BXP",
            )
        ],
    )

    out = evaluate.run_ablation()

    baseline = out["baseline"][0]
    no_entity = out["no_entity_filter"][0]

    assert baseline.top_n_companies == ["BXP"]
    assert baseline.expected_company_rank == 1

    assert no_entity.top_n_companies == ["NO_FILTER"]
    assert no_entity.expected_company_rank is None
