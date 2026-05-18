"""Tests for the soft-refusal grading hook in scripts/evaluate.py (fix 2e).

Verifies that when EvaluationQuery.expect_soft_refusal=True, the _grade()
function:
- marks PASS when the answer abstains or uses soft-refusal language
- marks FAIL with "expected_soft_refusal_but_answered" when the answer neither
  abstains nor contains soft-refusal language
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import pytest

from scripts.evaluate import _grade
from src.generation.citation_check import CitationReport
from src.models import Chunk, RetrievedChunk
from tests.evaluation_set import EvaluationQuery


def _make_chunk(company: str = "Digital Realty") -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type="investor_presentation",
        report_date="2026-03",
        period_covered="Q4 2025",
        doc_version="2026-03",
        section_title="Debt Covenants",
        page_number=5,
        content_type="text",
        chunk_text="DLR debt covenant compliance discussion",
    )


# Minimal stub matching the Answer interface used by _grade().
@dataclass
class _FakeAnswer:
    text: str
    abstained: bool
    intent: Optional[str] = "latest"
    contexts: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    citation_report: Optional[CitationReport] = None
    structured: Optional[object] = None


def _query_with_soft_refusal(**kwargs) -> EvaluationQuery:
    defaults = dict(
        id="t-sr",
        query="Does DLR's presentation disclose credit facility changes?",
        category="capital_structure",
        expect_soft_refusal=True,
    )
    defaults.update(kwargs)
    return EvaluationQuery(**defaults)


# ---------------------------------------------------------------------------
# 1. PASS when the answer abstains pre-LLM
# ---------------------------------------------------------------------------


def test_soft_refusal_pass_when_abstained():
    gq = _query_with_soft_refusal()
    a = _FakeAnswer(
        text="I could not find reliable information.",
        abstained=True,
        diagnostics={"top_rerank_score": -2.0, "retrieval_confidence": 0.1},
    )
    result = _grade(gq, a)
    assert "soft_refusal" in result.auto_pass_reasons
    assert "expected_soft_refusal_but_answered" not in result.auto_fail_reasons


# ---------------------------------------------------------------------------
# 2. PASS when the answer is not abstained but contains refusal language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "not disclosed",
    "not provided",
    "do not contain",
    "cannot determine",
    "not updated",
    "not reaffirmed",
    "did not disclose",
])
def test_soft_refusal_pass_on_refusal_language(phrase: str):
    gq = _query_with_soft_refusal()
    a = _FakeAnswer(
        text=f"The information is {phrase} in the source documents.",
        abstained=False,
        diagnostics={"top_rerank_score": 0.5, "retrieval_confidence": 0.4},
        contexts=[RetrievedChunk(chunk=_make_chunk(), rerank_score=0.5)],
    )
    result = _grade(gq, a)
    assert "soft_refusal" in result.auto_pass_reasons
    assert "expected_soft_refusal_but_answered" not in result.auto_fail_reasons


def test_soft_refusal_pass_on_lead_then_adjacent_context():
    """An answer that LEADS with refusal-shaped language and then provides
    adjacent context about what IS disclosed is a textbook soft refusal.
    This is the g23-style case the prior over-strict grader rejected."""
    gq = _query_with_soft_refusal()
    long_context = (
        "For reference, the deck does disclose leverage ratios, debt service "
        "ratios, fixed-charge ratios, and debt composition as of QE 12/31/25 "
        "and QE 12/31/23 — but no narrative discussion of credit facility "
        "amendments, covenant waivers, or compliance certifications."
    )
    a = _FakeAnswer(
        text=(
            "The retrieved chunks from the Investor Presentation do not "
            "contain any explicit disclosure of changes to the credit facility "
            "or debt covenant compliance. " + long_context
        ),
        abstained=False,
        diagnostics={"top_rerank_score": 0.5, "retrieval_confidence": 0.4},
        contexts=[RetrievedChunk(chunk=_make_chunk(), rerank_score=0.5)],
    )
    result = _grade(gq, a)
    assert "soft_refusal" in result.auto_pass_reasons


def test_soft_refusal_fail_when_hedge_buried_at_tail():
    """An answer that asserts substantive content first and then appends a
    refusal hedge at the tail should NOT pass — the prior over-permissive
    grader let these through. The positional first-400-chars check rejects."""
    gq = _query_with_soft_refusal()
    substantive_lead = (
        "DLR has a $4.2B revolving credit facility with a $2.0B accordion "
        "feature and an unused capacity of $1.8B as of Q4 2025. The facility "
        "matures in 2028 and carries an investment-grade-only covenant "
        "package including a maximum leverage ratio of 60% and a minimum "
        "fixed-charge coverage ratio of 1.5x. Recent amendments include an "
        "extension of the accordion feature and a reduction in the unused "
        "fee. "
    )
    a = _FakeAnswer(
        text=substantive_lead + "The credit facility amendments are not disclosed in fine detail.",
        abstained=False,
        diagnostics={"top_rerank_score": 0.9, "retrieval_confidence": 0.8},
        contexts=[RetrievedChunk(chunk=_make_chunk(), rerank_score=0.9)],
    )
    result = _grade(gq, a)
    assert "expected_soft_refusal_but_answered" in result.auto_fail_reasons


# ---------------------------------------------------------------------------
# 3. FAIL when the answer neither abstains nor uses refusal language
# ---------------------------------------------------------------------------


def test_soft_refusal_fail_when_answered_without_refusal():
    gq = _query_with_soft_refusal()
    a = _FakeAnswer(
        text="DLR confirmed compliance with all credit facility covenants as of Q4 2025.",
        abstained=False,
        diagnostics={"top_rerank_score": 0.9, "retrieval_confidence": 0.8},
        contexts=[RetrievedChunk(chunk=_make_chunk(), rerank_score=0.9)],
    )
    result = _grade(gq, a)
    assert "expected_soft_refusal_but_answered" in result.auto_fail_reasons
    assert result.auto_pass is False


# ---------------------------------------------------------------------------
# 4. When expect_soft_refusal=False (default), the check is skipped entirely
# ---------------------------------------------------------------------------


def test_soft_refusal_check_skipped_when_not_expected():
    gq = EvaluationQuery(
        id="t-no-sr",
        query="What is DLR's net debt to EBITDA?",
        category="version_dedup",
        expect_soft_refusal=False,
    )
    a = _FakeAnswer(
        text="DLR's net debt to EBITDA is 5.2x.",
        abstained=False,
        diagnostics={"top_rerank_score": 0.9, "retrieval_confidence": 0.8},
        contexts=[RetrievedChunk(chunk=_make_chunk(), rerank_score=0.9)],
    )
    result = _grade(gq, a)
    # The soft_refusal check must not appear in either pass or fail lists.
    assert "soft_refusal" not in result.auto_pass_reasons
    assert "expected_soft_refusal_but_answered" not in result.auto_fail_reasons
