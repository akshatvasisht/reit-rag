"""Unit tests for the calibrated retrieval confidence scoring.

All tests use mocked DB connections or plain function calls — no real
database or network access.  Each test completes well under 2 seconds.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.confidence import (
    CONFIDENCE_HIGH_CUTOFF,
    CONFIDENCE_LOW_CUTOFF,
    W_COVERAGE,
    W_GAP,
    W_MAGNITUDE,
    _coverage_signal,
    _score_gap_signal,
    _score_magnitude_signal,
    compute_retrieval_confidence,
    confidence_band,
)
from src.retrieval.pipeline import RetrievalResult
from src.retrieval.reranker import RERANK_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(company: str = "Acme REIT") -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="ACR",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text="Sample text",
        content_type="text",
    )


def _make_rc(score: float | None = 2.0, company: str = "Acme REIT") -> RetrievedChunk:
    return RetrievedChunk(chunk=_make_chunk(company=company), rerank_score=score)


# ---------------------------------------------------------------------------
# _score_gap_signal
# ---------------------------------------------------------------------------


def test_gap_empty_list() -> None:
    assert _score_gap_signal([]) == 0.0


def test_gap_single_score() -> None:
    assert _score_gap_signal([2.0]) == 0.0


def test_gap_positive_gap() -> None:
    # top=2.0, second=1.0 → gap=(2-1)/|2|=0.5
    result = _score_gap_signal([2.0, 1.0])
    assert abs(result - 0.5) < 1e-6


def test_gap_negative_scores_positive_gap() -> None:
    # top=-2.0, second=-3.0 → gap=(-2-(-3))/|-2|=0.5
    result = _score_gap_signal([-2.0, -3.0])
    assert abs(result - 0.5) < 1e-6


def test_gap_negative_gap_clamped_to_zero() -> None:
    # If somehow second > top after sorted (shouldn't happen but guard):
    # gap=(1.0-2.0)/|1.0|=-1 → clamped to 0
    result = _score_gap_signal([1.0, 2.0])
    assert result == 0.0


def test_gap_near_zero_top_score() -> None:
    # denom falls back to 1.0 when |top| < 1e-6
    result = _score_gap_signal([0.0, -1.0])
    # gap = (0 - (-1)) / 1.0 = 1.0, clamped to 1.0
    assert abs(result - 1.0) < 1e-6


def test_gap_large_gap_clamped_to_one() -> None:
    result = _score_gap_signal([10.0, 1.0])
    assert result <= 1.0


# ---------------------------------------------------------------------------
# _score_magnitude_signal
# ---------------------------------------------------------------------------


def test_magnitude_at_threshold() -> None:
    # top_score = -5.0 → sigmoid((-5+2)/2) = sigmoid(-1.5) ≈ 0.182
    result = _score_magnitude_signal(-5.0)
    expected = 1.0 / (1.0 + math.exp(1.5))
    assert abs(result - expected) < 1e-6


def test_magnitude_at_centre() -> None:
    # top_score = -2.0 → sigmoid(0) = 0.5
    result = _score_magnitude_signal(-2.0)
    assert abs(result - 0.5) < 1e-6


def test_magnitude_at_positive() -> None:
    # top_score = 2.0 → sigmoid((2+2)/2) = sigmoid(2) ≈ 0.880
    result = _score_magnitude_signal(2.0)
    expected = 1.0 / (1.0 + math.exp(-2.0))
    assert abs(result - expected) < 1e-6


def test_magnitude_range() -> None:
    for score in [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]:
        r = _score_magnitude_signal(score)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# _coverage_signal
# ---------------------------------------------------------------------------


def test_coverage_empty_contexts() -> None:
    assert _coverage_signal([], ["Acme REIT"]) == 0.0


def test_coverage_no_filter() -> None:
    rcs = [_make_rc(company="Acme REIT"), _make_rc(company="Beta REIT")]
    assert _coverage_signal(rcs, None) == 1.0


def test_coverage_full_match() -> None:
    rcs = [_make_rc(company="Acme REIT"), _make_rc(company="Acme REIT")]
    result = _coverage_signal(rcs, ["Acme REIT"])
    assert abs(result - 1.0) < 1e-6


def test_coverage_partial_match() -> None:
    rcs = [_make_rc(company="Acme REIT"), _make_rc(company="Beta REIT")]
    result = _coverage_signal(rcs, ["Acme REIT"])
    assert abs(result - 0.5) < 1e-6


def test_coverage_no_match() -> None:
    rcs = [_make_rc(company="Beta REIT"), _make_rc(company="Beta REIT")]
    result = _coverage_signal(rcs, ["Acme REIT"])
    assert result == 0.0


def test_coverage_case_insensitive() -> None:
    rcs = [_make_rc(company="Acme REIT")]
    result = _coverage_signal(rcs, ["acme reit"])
    assert abs(result - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_retrieval_confidence
# ---------------------------------------------------------------------------


def test_confidence_empty_contexts() -> None:
    assert compute_retrieval_confidence([], None) == 0.0


def test_confidence_no_scored_chunks() -> None:
    rc = _make_rc(score=None)
    assert compute_retrieval_confidence([rc], None) == 0.0


def test_confidence_range() -> None:
    rcs = [_make_rc(score=2.0, company="Acme REIT"), _make_rc(score=1.0, company="Acme REIT")]
    result = compute_retrieval_confidence(rcs, ["Acme REIT"])
    assert 0.0 <= result <= 1.0


def test_confidence_high_score_full_coverage_close_to_one() -> None:
    # High top score, clear gap, full coverage → high confidence
    rcs = [_make_rc(score=5.0, company="Acme REIT"), _make_rc(score=1.0, company="Acme REIT")]
    result = compute_retrieval_confidence(rcs, ["Acme REIT"])
    assert result > 0.70


def test_confidence_weak_match_low_score() -> None:
    # Near-threshold score, zero gap (single chunk), no filter match
    rc = _make_rc(score=RERANK_THRESHOLD + 0.1, company="Beta REIT")
    result = compute_retrieval_confidence([rc], ["Acme REIT"])
    assert result < 0.40


def test_confidence_weighted_composite_arithmetic() -> None:
    # Verify the composite formula directly with known inputs.
    # scores=[4.0, 2.0], filter=["Acme REIT"], one "Acme REIT" chunk
    rcs = [_make_rc(score=4.0, company="Acme REIT"), _make_rc(score=2.0, company="Beta REIT")]
    result = compute_retrieval_confidence(rcs, ["Acme REIT"])
    # gap=(4-2)/4=0.5, mag=sigmoid((4+2)/2)=sigmoid(3)≈0.953, cov=1/2=0.5
    expected_gap = (4.0 - 2.0) / 4.0
    expected_mag = 1.0 / (1.0 + math.exp(-3.0))
    expected_cov = 0.5
    expected = W_GAP * expected_gap + W_MAGNITUDE * expected_mag + W_COVERAGE * expected_cov
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# confidence_band
# ---------------------------------------------------------------------------


def test_band_high() -> None:
    assert confidence_band(CONFIDENCE_HIGH_CUTOFF) == "high"
    assert confidence_band(0.9) == "high"
    assert confidence_band(1.0) == "high"


def test_band_medium() -> None:
    assert confidence_band(CONFIDENCE_LOW_CUTOFF) == "medium"
    assert confidence_band(0.5) == "medium"
    assert confidence_band(CONFIDENCE_HIGH_CUTOFF - 0.001) == "medium"


def test_band_low() -> None:
    assert confidence_band(0.0) == "low"
    assert confidence_band(0.2) == "low"
    assert confidence_band(CONFIDENCE_LOW_CUTOFF - 0.001) == "low"


# ---------------------------------------------------------------------------
# retrieve() integration — confidence populated post-rerank, pre-augmentation
# ---------------------------------------------------------------------------


def _make_chunk_row(chunk_id, doc_id, company: str = "Acme REIT") -> tuple:
    return (
        str(chunk_id), str(doc_id), None,
        company, "ACR", "investor_presentation", "2026-03", "Q4 2025", "2026-03",
        "Financials", 1, "text", "company_authored",
        "Sample chunk text", False, 80,
        "quarterly_investor_deck", "substantive",
    )


def test_retrieve_core_populates_confidence(monkeypatch) -> None:
    """_retrieve_core attaches retrieval_confidence to the RetrievalResult."""
    from src.retrieval import pipeline as pl

    chunk_id = uuid4()
    doc_id = uuid4()
    rc1 = _make_rc(score=2.0, company="Acme REIT")
    rc1.chunk.id = chunk_id
    rc2 = _make_rc(score=1.0, company="Acme REIT")

    monkeypatch.setattr(pl, "bm25_search", lambda *a, **kw: [rc1])
    monkeypatch.setattr(pl, "vector_search", lambda *a, **kw: [rc1])
    monkeypatch.setattr(pl, "rrf_fuse", lambda *a, **kw: [rc1, rc2])
    monkeypatch.setattr(pl, "rerank", lambda *a, **kw: [rc1, rc2])
    monkeypatch.setattr(pl, "dedupe_by_version_group", lambda chunks, intent: chunks)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda *a, **kw: [rc1, rc2])
    monkeypatch.setattr(pl, "expand_to_parents", lambda chunks, conn: chunks)
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: chunks)

    conn = MagicMock()
    result = pl._retrieve_core(
        "What is NOI?",
        company_filter=["Acme REIT"],
        conn=conn,
        intent="latest",
    )

    assert result.retrieval_confidence > 0.0
    assert 0.0 <= result.retrieval_confidence <= 1.0


def test_retrieve_core_confidence_before_conflict_injection(monkeypatch) -> None:
    """Confidence is computed on the pre-augmentation list; conflict chunks do not shift it."""
    from src.retrieval import pipeline as pl

    rc_main = _make_rc(score=5.0, company="Acme REIT")
    rc_weak = _make_rc(score=RERANK_THRESHOLD + 0.1, company="Other REIT")

    reranked_list = [rc_main]

    conflict_list = [rc_main, rc_weak]

    captured: dict = {}

    def _fake_compute(contexts, company_filter):
        captured["contexts"] = list(contexts)
        return compute_retrieval_confidence(contexts, company_filter)

    monkeypatch.setattr(pl, "bm25_search", lambda *a, **kw: [rc_main])
    monkeypatch.setattr(pl, "vector_search", lambda *a, **kw: [rc_main])
    monkeypatch.setattr(pl, "rrf_fuse", lambda *a, **kw: reranked_list)
    monkeypatch.setattr(pl, "rerank", lambda *a, **kw: reranked_list)
    monkeypatch.setattr(pl, "dedupe_by_version_group", lambda chunks, intent: chunks)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda *a, **kw: conflict_list)
    monkeypatch.setattr(pl, "expand_to_parents", lambda chunks, conn: chunks)
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: chunks)
    monkeypatch.setattr(pl, "compute_retrieval_confidence", _fake_compute)

    conn = MagicMock()
    pl._retrieve_core(
        "What is NOI?",
        company_filter=["Acme REIT"],
        conn=conn,
        intent="latest",
    )

    # The confidence function should have been called with only [rc_main],
    # not the conflict-augmented list [rc_main, rc_weak].
    assert len(captured["contexts"]) == 1
    assert captured["contexts"][0] is rc_main


# ---------------------------------------------------------------------------
# All-company synthesis path — confidence is mean of per-company magnitudes
# ---------------------------------------------------------------------------


def test_synthesis_confidence_mean_of_magnitudes(monkeypatch) -> None:
    """retrieve_all_company_synthesis sets confidence to mean magnitude across companies."""
    from src.retrieval import pipeline as pl

    companies = ["Alpha REIT", "Beta REIT"]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])

    alpha_rc = _make_rc(score=2.0, company="Alpha REIT")
    beta_rc = _make_rc(score=-1.0, company="Beta REIT")

    def _fake_single(query, company, conn, top_k=3):
        rc = alpha_rc if company == "Alpha REIT" else beta_rc
        return pl.RetrievalResult(
            query=query,
            intent="all_company_synthesis",
            contexts=[rc],
            companies=[company],
            abstain=False,
            diagnostics={"top_rerank": rc.rerank_score},
        )

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)

    conn = MagicMock()
    result = pl.retrieve_all_company_synthesis("Compare all companies", conn)

    expected = (_score_magnitude_signal(2.0) + _score_magnitude_signal(-1.0)) / 2.0
    assert abs(result.retrieval_confidence - expected) < 1e-6


def test_synthesis_confidence_zero_when_all_abstain(monkeypatch) -> None:
    from src.retrieval import pipeline as pl

    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": "Alpha REIT"}])

    def _fake_single(query, company, conn, top_k=3):
        return pl.RetrievalResult(
            query=query,
            intent="all_company_synthesis",
            contexts=[],
            companies=[company],
            abstain=True,
            diagnostics={"top_rerank": float("-inf")},
        )

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)

    conn = MagicMock()
    result = pl.retrieve_all_company_synthesis("Compare all", conn)
    assert result.retrieval_confidence == 0.0


# ---------------------------------------------------------------------------
# Generator wires retrieval_confidence onto StructuredAnswer
# ---------------------------------------------------------------------------


def test_generator_wires_retrieval_confidence(monkeypatch) -> None:
    """answer_structured copies retrieval_confidence from RetrievalResult to Answer diagnostics."""
    from src.generation import generator as gen
    from src.models import StructuredAnswer

    rc = _make_rc(score=3.0)
    mock_retrieval = RetrievalResult(
        query="test",
        intent="latest",
        contexts=[rc],
        abstain=False,
        retrieval_confidence=0.82,
        diagnostics={"top_rerank_score": 3.0, "after_expansion": 1, "retrieval_hops": 0, "sub_queries": []},
    )

    def _fake_retrieve(query, **kw):
        return mock_retrieval

    fake_structured = StructuredAnswer(
        answer_prose="The NOI is $500M.",
        claims=[],
        abstain=False,
        abstain_reason="",
        forward_looking=False,
    )

    monkeypatch.setattr(gen, "retrieve", _fake_retrieve)
    monkeypatch.setattr(gen, "_generate_structured", lambda *a, **kw: fake_structured)
    monkeypatch.setattr(gen, "check_citations", lambda *a, **kw: MagicMock(
        supported=0, total=0, faithfulness_ratio=1.0, unsupported=[], numeric_mismatches=[]
    ))
    monkeypatch.setattr(gen, "check_numeric_consistency", lambda *a, **kw: [])

    answer = gen.answer_structured("What is NOI?")
    assert answer.diagnostics.get("retrieval_confidence") == 0.82
    assert fake_structured.retrieval_confidence == 0.82
