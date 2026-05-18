"""Tests verifying Answer.structured is populated on all answer paths.

Covers:
- answer() and answer_structured() both attach a real StructuredAnswer.
- Abstain branches carry retrieval_confidence in diagnostics.
- Answered branches carry retrieval_confidence in diagnostics.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.generation.generator import Answer, answer, answer_structured
from src.models import Chunk, RetrievedChunk, StructuredAnswer
from src.retrieval.pipeline import RetrievalResult


def _make_chunk(company: str = "TestCo") -> RetrievedChunk:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="TST",
        doc_type="investor_presentation",
        report_date="2025-12",
        period_covered="Q4 2025",
        doc_version="2025-12",
        section_title="Overview",
        page_number=1,
        content_type="text",
        chunk_text=f"{company} overview text.",
    )
    return RetrievedChunk(chunk=chunk, rerank_score=0.9)


def _make_abstain_result(confidence: float = 0.42) -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        intent="latest",
        contexts=[],
        companies=[],
        abstain=True,
        abstain_reason="no matching corpus entries",
        diagnostics={"top_rerank_score": 0.1},
        retrieval_confidence=confidence,
    )


def _make_answered_result(confidence: float = 0.42) -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        intent="latest",
        contexts=[_make_chunk()],
        companies=["TestCo"],
        abstain=False,
        diagnostics={"top_rerank_score": 0.9},
        retrieval_confidence=confidence,
    )


def _make_structured_answer(forward_looking: bool = False, abstain: bool = False) -> StructuredAnswer:
    return StructuredAnswer(
        answer_prose="TestCo reported strong results.",
        claims=[],
        abstain=abstain,
        abstain_reason="",
        forward_looking=forward_looking,
    )


# ---------------------------------------------------------------------------
# retrieval_confidence in diagnostics — abstain branch
# ---------------------------------------------------------------------------


def test_answer_abstain_carries_retrieval_confidence() -> None:
    """answer() abstain path must write retrieval_confidence into diagnostics."""
    with patch("src.generation.generator.retrieve", return_value=_make_abstain_result(0.42)):
        result = answer("irrelevant query")
    assert result.abstained is True
    assert result.diagnostics.get("retrieval_confidence") == pytest.approx(0.42)


def test_answer_structured_abstain_carries_retrieval_confidence() -> None:
    """answer_structured() abstain path must write retrieval_confidence into diagnostics."""
    with patch("src.generation.generator.retrieve", return_value=_make_abstain_result(0.77)):
        result = answer_structured("irrelevant query")
    assert result.abstained is True
    assert result.diagnostics.get("retrieval_confidence") == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Answer.structured populated on answered path
# ---------------------------------------------------------------------------


def test_answer_attaches_structured_on_answered_path() -> None:
    """answer() must attach a real StructuredAnswer to Answer.structured."""
    sa = _make_structured_answer(forward_looking=True)
    with (
        patch("src.generation.generator.retrieve", return_value=_make_answered_result(0.9)),
        patch("src.generation.generator._generate_structured", return_value=sa),
        patch("src.generation.generator.check_citations") as mock_cit,
        patch("src.generation.generator.check_numeric_consistency", return_value=[]),
    ):
        mock_cit.return_value = MagicMock(supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[])
        result = answer("What is TestCo guidance?")

    assert result.structured is not None
    assert isinstance(result.structured, StructuredAnswer)
    assert result.structured.forward_looking is True


def test_answer_structured_attaches_structured_on_answered_path() -> None:
    """answer_structured() must attach a real StructuredAnswer to Answer.structured."""
    sa = _make_structured_answer(forward_looking=False)
    with (
        patch("src.generation.generator.retrieve", return_value=_make_answered_result(0.85)),
        patch("src.generation.generator._generate_structured", return_value=sa),
        patch("src.generation.generator.check_citations") as mock_cit,
        patch("src.generation.generator.check_numeric_consistency", return_value=[]),
    ):
        mock_cit.return_value = MagicMock(supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[])
        result = answer_structured("What is TestCo leverage?")

    assert result.structured is not None
    assert isinstance(result.structured, StructuredAnswer)
    assert result.structured.forward_looking is False


# ---------------------------------------------------------------------------
# Answer.structured is None on abstain path (no StructuredAnswer produced)
# ---------------------------------------------------------------------------


def test_answer_structured_none_on_abstain_path() -> None:
    """answer() abstain path must not attach a StructuredAnswer."""
    with patch("src.generation.generator.retrieve", return_value=_make_abstain_result(0.1)):
        result = answer("out of corpus query")
    assert result.structured is None


def test_answer_structured_func_none_on_abstain_path() -> None:
    """answer_structured() abstain path must not attach a StructuredAnswer."""
    with patch("src.generation.generator.retrieve", return_value=_make_abstain_result(0.1)):
        result = answer_structured("out of corpus query")
    assert result.structured is None


# ---------------------------------------------------------------------------
# retrieval_confidence in diagnostics — answered branch
# ---------------------------------------------------------------------------


def test_answer_answered_carries_retrieval_confidence() -> None:
    """answer() answered path must write retrieval_confidence into diagnostics."""
    sa = _make_structured_answer()
    with (
        patch("src.generation.generator.retrieve", return_value=_make_answered_result(0.55)),
        patch("src.generation.generator._generate_structured", return_value=sa),
        patch("src.generation.generator.check_citations") as mock_cit,
        patch("src.generation.generator.check_numeric_consistency", return_value=[]),
    ):
        mock_cit.return_value = MagicMock(supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[])
        result = answer("TestCo leverage?")

    assert result.diagnostics.get("retrieval_confidence") == pytest.approx(0.55)
