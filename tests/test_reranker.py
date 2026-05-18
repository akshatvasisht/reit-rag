"""Unit tests for src/retrieval/reranker.py.

Regression guard for contextualized_text-preferred scoring: the reranker must
score a chunk whose contextualized_text bridges the vocab gap higher than a
chunk that only contains the raw (non-matching) text.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.models import Chunk, RetrievedChunk


def _make_chunk(chunk_text: str, contextualized_text: str | None = None) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="EastGroup Properties",
        ticker="EGP",
        doc_type="investor_presentation",
        report_date="2025-12",
        doc_version="2025-12",
        chunk_text=chunk_text,
        content_type="text",
        contextualized_text=contextualized_text,
    )


def _make_rc(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk)


# ---------------------------------------------------------------------------
# Contextualized-text preference — EGP vocab-gap regression
# ---------------------------------------------------------------------------


def test_contextualized_text_preferred_over_raw_chunk_text(monkeypatch) -> None:
    """Chunk with 'pipeline' in contextualized_text must outscore bare-Program chunk.

    This is a regression guard for the EGP p.26 case where the cross-encoder
    scored (query='pipeline', passage='Current Development and Value-Add Program...')
    at -4.9 and pushed the correct chunk past top-5.  The fix: feed
    contextualized_text (which contains 'pipeline') to the cross-encoder instead
    of raw chunk_text.
    """
    from src.retrieval import reranker as _reranker_module

    # Chunk A: contextualized_text bridges the vocab gap.
    chunk_a = _make_chunk(
        chunk_text="Current Development and Value-Add Program consists of ...",
        contextualized_text="active development and value-add acquisition pipeline",
    )
    # Chunk B: no contextualized_text, only raw "Program" text.
    chunk_b = _make_chunk(
        chunk_text="Current Development and Value-Add Program consists of ...",
        contextualized_text=None,
    )

    rc_a = _make_rc(chunk_a)
    rc_b = _make_rc(chunk_b)

    # Track which passage strings reach the model.
    captured_pairs: list[list[tuple[str, str]]] = []

    class _FakeEncoder:
        def predict(self, pairs: Any) -> list[float]:
            captured_pairs.append(list(pairs))
            # Score by whether the passage contains "pipeline".
            return [5.0 if "pipeline" in p.lower() else -4.9 for _, p in pairs]

    monkeypatch.setattr(_reranker_module, "_cross_encoder", _FakeEncoder())

    query = "What is EastGroup's current development pipeline?"
    results = _reranker_module.rerank(query, [rc_a, rc_b], top_n=2)

    assert len(captured_pairs) == 1, "predict() should be called exactly once"
    passages = [p for _, p in captured_pairs[0]]

    # Chunk A must use contextualized_text (vocab-bridged).
    assert passages[0] == chunk_a.contextualized_text, (
        "reranker should feed contextualized_text for chunk A, not raw chunk_text"
    )
    # Chunk B must fall back to chunk_text (no contextualized_text).
    assert passages[1] == chunk_b.chunk_text, (
        "reranker should fall back to chunk_text when contextualized_text is None"
    )

    # Chunk A (with "pipeline" in contextualized_text) must rank first.
    assert results[0].chunk.id == chunk_a.id, (
        "chunk with pipeline in contextualized_text should outscore bare-Program chunk"
    )
    assert results[1].chunk.id == chunk_b.id


def test_rerank_fallback_when_no_contextualized_text(monkeypatch) -> None:
    """When all chunks lack contextualized_text, raw chunk_text is used unchanged."""
    from src.retrieval import reranker as _reranker_module

    chunk = _make_chunk(chunk_text="some raw text", contextualized_text=None)
    rc = _make_rc(chunk)

    captured: list[list[tuple[str, str]]] = []

    class _FakeEncoder:
        def predict(self, pairs: Any) -> list[float]:
            captured.append(list(pairs))
            return [1.0] * len(pairs)

    monkeypatch.setattr(_reranker_module, "_cross_encoder", _FakeEncoder())

    _reranker_module.rerank("some query", [rc], top_n=1)

    assert captured[0][0][1] == "some raw text", (
        "should fall back to chunk_text when contextualized_text is None"
    )
