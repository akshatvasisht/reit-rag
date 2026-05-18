from __future__ import annotations

from uuid import uuid4

import httpx
from anthropic import APIConnectionError

from src.generation import generator
from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import RetrievalResult


def _mk_context() -> list[RetrievedChunk]:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="Digital Realty",
        ticker="DLR",
        doc_type="investor_presentation",
        report_date="2026-03",
        period_covered="Q4 2025",
        doc_version="2026-03",
        section_title="Leverage",
        page_number=14,
        content_type="text",
        chunk_text="Net debt to EBITDA was 5.0x",
    )
    return [RetrievedChunk(chunk=chunk, rerank_score=2.0)]


def _mk_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="test",
        intent="latest",
        contexts=_mk_context(),
        companies=["Digital Realty"],
        abstain=False,
        diagnostics={"top_rerank_score": 2.0},
    )


class _FailingMessages:
    def create(self, **kwargs):  # noqa: ANN003
        raise APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


class _FailingClient:
    def __init__(self) -> None:
        self.messages = _FailingMessages()


def test_answer_returns_controlled_error_on_generation_failure(monkeypatch) -> None:
    monkeypatch.setattr(generator, "retrieve", lambda query: _mk_retrieval())
    monkeypatch.setattr(generator, "_get_client", lambda: _FailingClient())

    result = generator.answer("What is DLR leverage?")
    assert result.abstained is True
    assert "temporary model/API issue" in result.text
    assert result.diagnostics.get("generation_error") == "APIConnectionError"


def test_answer_structured_returns_controlled_error_on_generation_failure(monkeypatch) -> None:
    monkeypatch.setattr(generator, "retrieve", lambda query: _mk_retrieval())
    monkeypatch.setattr(generator, "_get_client", lambda: _FailingClient())

    result = generator.answer_structured("What is DLR leverage?")
    assert result.abstained is True
    assert "temporary model/API issue" in result.text
    assert result.diagnostics.get("generation_error") == "APIConnectionError"


class _UnexpectedBugMessages:
    def create(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("simulated internal defect, not a transient API failure")


class _UnexpectedBugClient:
    def __init__(self) -> None:
        self.messages = _UnexpectedBugMessages()


def test_unexpected_internal_error_propagates_not_silently_abstains(monkeypatch) -> None:
    """A RuntimeError inside generation is not a transient API failure and
    must reach the operator instead of being mapped to an abstain stub."""
    import pytest

    monkeypatch.setattr(generator, "retrieve", lambda query: _mk_retrieval())
    monkeypatch.setattr(generator, "_get_client", lambda: _UnexpectedBugClient())

    with pytest.raises(RuntimeError, match="simulated internal defect"):
        generator.answer("What is DLR leverage?")


def test_typeerror_propagates_rather_than_masquerading_as_abstain(monkeypatch) -> None:
    """TypeError used to be on the transient list, which silently masked
    real bugs in dataclass construction or signature mismatches."""
    import pytest

    class _TypeErrorMessages:
        def create(self, **kwargs):  # noqa: ANN003
            raise TypeError("unexpected keyword argument 'foo' from SDK drift or call-site bug")

    class _TypeErrorClient:
        def __init__(self) -> None:
            self.messages = _TypeErrorMessages()

    monkeypatch.setattr(generator, "retrieve", lambda query: _mk_retrieval())
    monkeypatch.setattr(generator, "_get_client", lambda: _TypeErrorClient())

    with pytest.raises(TypeError):
        generator.answer_structured("What is DLR leverage?")
