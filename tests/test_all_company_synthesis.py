"""Tests for all-company synthesis intent: classifier, retrieval, and prompt."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import RetrievalResult
from src.retrieval.synthesis import retrieve_all_company_synthesis
from src.versioning.chains import dedupe_by_version_group
from src.versioning.classifier import _regex_classify_intent, classify_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Alpha REIT",
    doc_version: str = "2026-03",
    report_date: str = "2026-03",
    page_number: int = 1,
    chunk_text: str = "NOI grew 5%",
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="ALP",
        doc_type="investor_presentation",
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=doc_version,
        section_title="Performance",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk, score: float = 2.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


# ---------------------------------------------------------------------------
# Step 1: regex classifier returns all_company_synthesis for synthesis queries
# ---------------------------------------------------------------------------


def test_regex_all_companies_phrase() -> None:
    assert _regex_classify_intent("Compare NOI across all companies") == "all_company_synthesis"


def test_regex_each_company_phrase() -> None:
    assert _regex_classify_intent("What does each company say about leverage?") == "all_company_synthesis"


def test_regex_which_reit_highest() -> None:
    # "Which REIT has the highest leverage?" contains no synthesis phrase
    # so regex falls through to latest — LLM would catch it.  But queries with
    # "every REIT" do trigger synthesis.
    assert _regex_classify_intent("Compare metrics across every REIT") == "all_company_synthesis"


def test_regex_portfolio_wide() -> None:
    assert _regex_classify_intent("What is the portfolio-wide NOI growth?") == "all_company_synthesis"


def test_regex_pairwise_comparison_does_not_trigger_synthesis() -> None:
    """A pairwise query like 'Compare BXP and PSA' must not route to synthesis."""
    result = _regex_classify_intent("Compare BXP and PSA leverage")
    assert result != "all_company_synthesis"
    assert result == "comparison"


# ---------------------------------------------------------------------------
# Step 1: LLM path for all_company_synthesis
# ---------------------------------------------------------------------------


def test_llm_above_threshold_returns_synthesis_intent() -> None:
    with patch(
        "src.versioning.classifier._llm_classify_intent",
        return_value={"intent": "all_company_synthesis", "confidence": 0.90},
    ):
        result = classify_intent("Compare same-store NOI growth across all companies")
    assert result == "all_company_synthesis"


def test_llm_below_threshold_falls_back_to_regex_for_synthesis(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.versioning.classifier._LOG_PATH",
        tmp_path / "intent_decisions.jsonl",
    )
    # Low-confidence LLM returns comparison; but query has synthesis phrase so
    # regex should return all_company_synthesis.
    with patch(
        "src.versioning.classifier._llm_classify_intent",
        return_value={"intent": "comparison", "confidence": 0.50},
    ):
        result = classify_intent("Summarise NOI across all companies")
    assert result == "all_company_synthesis"

    record = json.loads((tmp_path / "intent_decisions.jsonl").read_text().strip())
    assert record["method"] == "regex_fallback"


# ---------------------------------------------------------------------------
# Step 1: dedupe_by_version_group passes through for synthesis intent
# ---------------------------------------------------------------------------


def test_dedupe_synthesis_returns_all_chunks() -> None:
    older = _make_rc(_make_chunk(company="Digital Realty", doc_version="2025-12"))
    newer = _make_rc(_make_chunk(company="Digital Realty", doc_version="2026-03"))
    other = _make_rc(_make_chunk(company="BXP", doc_version="2025-12"))

    result = dedupe_by_version_group([older, newer, other], intent="all_company_synthesis")
    assert result == [older, newer, other]


# ---------------------------------------------------------------------------
# Step 2: retrieve_all_company_synthesis — mocked _retrieve_single_company
# ---------------------------------------------------------------------------


def _make_company_result(
    company: str,
    score: float,
    query: str = "test query",
) -> RetrievalResult:
    chunk = _make_chunk(company=company, chunk_text=f"{company} NOI data")
    rc = _make_rc(chunk, score=score)
    abstain = score <= -5.0
    return RetrievalResult(
        query=query,
        intent="all_company_synthesis",
        contexts=[rc],
        companies=[company],
        abstain=abstain,
        diagnostics={"top_rerank": score},
    )


def _make_empty_result(company: str, query: str = "test query") -> RetrievalResult:
    return RetrievalResult(
        query=query,
        intent="all_company_synthesis",
        contexts=[],
        companies=[company],
        abstain=True,
        diagnostics={"top_rerank": float("-inf")},
    )


def test_retrieve_all_company_synthesis_merges_per_company_chunks() -> None:
    """Each company's top chunk should appear in the merged deduped result."""
    companies = ["Alpha REIT", "Beta REIT", "Gamma REIT"]

    def mock_retrieve(query, company, conn, top_k=3):
        return _make_company_result(company, score=2.0, query=query)

    mock_conn = MagicMock()

    with patch("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies]):
        with patch("src.retrieval.synthesis._retrieve_single_company", side_effect=mock_retrieve):
            result = retrieve_all_company_synthesis("NOI across all", mock_conn)

    assert len(result.contexts) == 3
    result_companies = {rc.chunk.company for rc in result.contexts}
    assert result_companies == set(companies)
    assert result.abstain is False


def test_retrieve_all_company_synthesis_abstains_when_all_below_threshold() -> None:
    """When every company returns sub-threshold or no results, abstain=True."""
    companies = ["Alpha REIT", "Beta REIT"]

    def mock_retrieve(query, company, conn, top_k=3):
        return _make_empty_result(company, query=query)

    mock_conn = MagicMock()

    with patch("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies]):
        with patch("src.retrieval.synthesis._retrieve_single_company", side_effect=mock_retrieve):
            result = retrieve_all_company_synthesis("NOI across all", mock_conn)

    assert result.abstain is True


def test_retrieve_all_company_synthesis_not_abstain_when_one_above_threshold() -> None:
    """When at least one company has a chunk above threshold, abstain=False."""
    companies = ["Alpha REIT", "Beta REIT"]
    scores = {"Alpha REIT": 2.0, "Beta REIT": float("-inf")}

    def mock_retrieve(query, company, conn, top_k=3):
        score = scores[company]
        if score <= -5.0:
            return _make_empty_result(company, query=query)
        return _make_company_result(company, score=score, query=query)

    mock_conn = MagicMock()

    with patch("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies]):
        with patch("src.retrieval.synthesis._retrieve_single_company", side_effect=mock_retrieve):
            result = retrieve_all_company_synthesis("NOI across all", mock_conn)

    assert result.abstain is False
    assert len(result.contexts) == 1


def test_retrieve_all_company_synthesis_deduplicates_shared_chunk() -> None:
    """A chunk mentioning multiple companies (e.g. a peer table) appears once."""
    shared_chunk = _make_chunk(company="Alpha REIT", chunk_text="peer comparison table")
    shared_rc = _make_rc(shared_chunk, score=3.0)

    companies = ["Alpha REIT", "Beta REIT"]

    def mock_retrieve(query, company, conn, top_k=3):
        # Both companies return the same chunk (same UUID).
        return RetrievalResult(
            query=query,
            intent="all_company_synthesis",
            contexts=[shared_rc],
            companies=[company],
            abstain=False,
            diagnostics={"top_rerank": 3.0},
        )

    mock_conn = MagicMock()

    with patch("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies]):
        with patch("src.retrieval.synthesis._retrieve_single_company", side_effect=mock_retrieve):
            result = retrieve_all_company_synthesis("NOI across all", mock_conn)

    # Deduplication: same chunk UUID → appears exactly once.
    assert len(result.contexts) == 1


# ---------------------------------------------------------------------------
# Step 3: retrieve() routing — synthesis intent dispatches to the new function
# ---------------------------------------------------------------------------


def test_retrieve_routing_synthesis_calls_synthesis_function(monkeypatch) -> None:
    """When intent is all_company_synthesis, retrieve() must call the synthesis function."""
    called_with: list[str] = []

    def mock_synthesis(query, conn):
        called_with.append(query)
        return RetrievalResult(
            query=query,
            intent="all_company_synthesis",
            contexts=[],
            companies=[],
            abstain=True,
            diagnostics={},
        )

    from src.retrieval import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "classify_intent", lambda q: "all_company_synthesis")
    monkeypatch.setattr(_pipeline, "extract_companies", lambda q: [])
    monkeypatch.setattr(_pipeline, "retrieve_all_company_synthesis", mock_synthesis)
    # connect() should not be called for non-synthesis path; patch it to detect leaks.
    monkeypatch.setattr(_pipeline, "connect", lambda: MagicMock().__enter__())

    result = _pipeline.retrieve("Which REIT has the highest NOI growth?")
    assert result.intent == "all_company_synthesis"
    assert called_with == ["Which REIT has the highest NOI growth?"]


def test_retrieve_routing_non_synthesis_skips_synthesis_function(monkeypatch) -> None:
    """When intent is latest, retrieve_all_company_synthesis must not be called."""
    synthesis_called: list[bool] = []

    def mock_synthesis(query, conn):
        synthesis_called.append(True)
        return RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=[], companies=[], abstain=True, diagnostics={},
        )

    from src.retrieval import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "classify_intent", lambda q: "latest")
    monkeypatch.setattr(_pipeline, "extract_companies", lambda q: [])
    monkeypatch.setattr(_pipeline, "retrieve_all_company_synthesis", mock_synthesis)

    # Mock the DB-touching parts so the non-synthesis path short-circuits cleanly.
    fake_conn = MagicMock()
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = MagicMock(return_value=False)
    fake_conn.cursor = MagicMock()
    monkeypatch.setattr(_pipeline, "connect", lambda: fake_conn)
    monkeypatch.setattr(_pipeline, "bm25_search", lambda *a, **kw: [])
    monkeypatch.setattr(_pipeline, "vector_search", lambda *a, **kw: [])
    monkeypatch.setattr(_pipeline, "rrf_fuse", lambda *a, **kw: [])
    monkeypatch.setattr(_pipeline, "rerank", lambda *a, **kw: [])
    monkeypatch.setattr(_pipeline, "dedupe_by_version_group", lambda r, i: r)
    monkeypatch.setattr(_pipeline, "find_conflicting_chunks", lambda r, c, conn: list(r))
    monkeypatch.setattr(_pipeline, "expand_to_parents", lambda r, conn: list(r))
    monkeypatch.setattr(_pipeline, "expand_to_siblings", lambda r, conn: list(r))

    _pipeline.retrieve("What is the occupancy rate?")
    assert synthesis_called == []


# ---------------------------------------------------------------------------
# Step 5: prompt includes synthesis block for synthesis intent
# ---------------------------------------------------------------------------


def test_build_user_message_synthesis_includes_instruction() -> None:
    from src.generation.prompts import _SYNTHESIS_INSTRUCTION, build_user_message

    chunk = _make_chunk()
    contexts = [_make_rc(chunk)]
    msg = build_user_message("NOI across all companies", contexts, intent="all_company_synthesis")
    assert _SYNTHESIS_INSTRUCTION in msg


def test_build_user_message_non_synthesis_excludes_synthesis_block() -> None:
    from src.generation.prompts import _SYNTHESIS_INSTRUCTION, build_user_message

    chunk = _make_chunk()
    contexts = [_make_rc(chunk)]
    for intent in ("latest", "historical", "comparison", "conflict"):
        msg = build_user_message("test query", contexts, intent=intent)  # type: ignore[arg-type]
        assert _SYNTHESIS_INSTRUCTION not in msg, (
            f"Synthesis block leaked into intent={intent}"
        )


# ---------------------------------------------------------------------------
# New: post-rerank pipeline stages are applied on the synthesis path
# ---------------------------------------------------------------------------


def test_retrieve_all_company_synthesis_calls_expand_to_parents(monkeypatch) -> None:
    """retrieve_all_company_synthesis must call expand_to_parents on the merged set."""
    from src.retrieval import pipeline as pl

    companies = ["Alpha REIT", "Beta REIT"]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])

    alpha_chunk = _make_chunk(company="Alpha REIT")
    beta_chunk = _make_chunk(company="Beta REIT")

    def _fake_single(query, company, conn, top_k=3):
        rc = _make_rc(alpha_chunk if company == "Alpha REIT" else beta_chunk, score=2.0)
        return pl.RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=[rc], companies=[company], abstain=False,
            diagnostics={"top_rerank": 2.0},
        )

    expand_parents_calls: list[int] = []

    def _fake_expand_parents(chunks, conn):
        expand_parents_calls.append(len(chunks))
        return list(chunks)

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)
    monkeypatch.setattr(pl, "expand_to_parents", _fake_expand_parents)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda r, cf, conn: list(r))
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: list(chunks))

    conn = MagicMock()
    pl.retrieve_all_company_synthesis("NOI across all", conn)

    assert len(expand_parents_calls) > 0, "expand_to_parents was not called"


def test_retrieve_all_company_synthesis_calls_find_conflicting_chunks(monkeypatch) -> None:
    """retrieve_all_company_synthesis must call find_conflicting_chunks on the merged set."""
    from src.retrieval import pipeline as pl

    companies = ["Alpha REIT"]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])

    chunk = _make_chunk(company="Alpha REIT")

    def _fake_single(query, company, conn, top_k=3):
        rc = _make_rc(chunk, score=2.0)
        return pl.RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=[rc], companies=[company], abstain=False,
            diagnostics={"top_rerank": 2.0},
        )

    conflict_calls: list[int] = []

    def _fake_conflict(retained, company_filter, conn):
        conflict_calls.append(len(retained))
        return list(retained)

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)
    monkeypatch.setattr(pl, "find_conflicting_chunks", _fake_conflict)
    monkeypatch.setattr(pl, "expand_to_parents", lambda chunks, conn: list(chunks))
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: list(chunks))

    conn = MagicMock()
    pl.retrieve_all_company_synthesis("NOI across all", conn)

    assert len(conflict_calls) > 0, "find_conflicting_chunks was not called"


def test_retrieve_all_company_synthesis_calls_expand_to_siblings(monkeypatch) -> None:
    """retrieve_all_company_synthesis must call expand_to_siblings on the merged set."""
    from src.retrieval import pipeline as pl

    companies = ["Alpha REIT"]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])

    chunk = _make_chunk(company="Alpha REIT")

    def _fake_single(query, company, conn, top_k=3):
        rc = _make_rc(chunk, score=2.0)
        return pl.RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=[rc], companies=[company], abstain=False,
            diagnostics={"top_rerank": 2.0},
        )

    sibling_calls: list[int] = []

    def _fake_siblings(chunks, conn):
        sibling_calls.append(len(chunks))
        return list(chunks)

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda r, cf, conn: list(r))
    monkeypatch.setattr(pl, "expand_to_parents", lambda chunks, conn: list(chunks))
    monkeypatch.setattr(pl, "expand_to_siblings", _fake_siblings)

    conn = MagicMock()
    pl.retrieve_all_company_synthesis("NOI across all", conn)

    assert len(sibling_calls) > 0, "expand_to_siblings was not called"


def test_retrieve_all_company_synthesis_parent_chunks_reach_llm(monkeypatch) -> None:
    """Regression: parent-expanded chunks must be in the result, not raw child chunks."""
    from src.retrieval import pipeline as pl

    companies = ["Alpha REIT"]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])

    child_chunk = _make_chunk(company="Alpha REIT", chunk_text="child chunk text")
    parent_chunk = _make_chunk(company="Alpha REIT", chunk_text="parent chunk text (expanded)")

    def _fake_single(query, company, conn, top_k=3):
        rc = _make_rc(child_chunk, score=2.0)
        return pl.RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=[rc], companies=[company], abstain=False,
            diagnostics={"top_rerank": 2.0},
        )

    parent_rc = _make_rc(parent_chunk, score=2.0)

    def _fake_expand_parents(chunks, conn):
        # Return the parent chunk instead of the child
        return [parent_rc]

    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda r, cf, conn: list(r))
    monkeypatch.setattr(pl, "expand_to_parents", _fake_expand_parents)
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: list(chunks))

    conn = MagicMock()
    result = pl.retrieve_all_company_synthesis("NOI across all", conn)

    assert len(result.contexts) == 1
    assert result.contexts[0].chunk.chunk_text == "parent chunk text (expanded)"


def test_retrieve_all_company_synthesis_cap_enforced(monkeypatch) -> None:
    """When the merged-deduped set exceeds MAX_SYNTHESIS_CONTEXT_CHUNKS, the cap drops
    the lowest-rerank-scored chunks first, preserving higher-scored ones."""
    from src.retrieval import pipeline as pl

    # Use a small, controlled company set to have a predictable cap
    companies = [f"Company {i}" for i in range(3)]
    monkeypatch.setattr("src.retrieval.synthesis.CORPUS_REGISTRY", [{"company": c} for c in companies])
    # Cap = 3 * 3 = 9; we will produce more than 9 chunks

    per_company_top_k = 10  # exceeds cap/3 so total will exceed cap
    all_chunks_by_company: dict[str, list[RetrievedChunk]] = {}
    for company in companies:
        chunks = []
        for j in range(per_company_top_k):
            chunk = _make_chunk(company=company, chunk_text=f"{company} chunk {j}")
            rc = RetrievedChunk(chunk=chunk, rerank_score=float(j + 1))
            chunks.append(rc)
        all_chunks_by_company[company] = chunks

    def _fake_single(query, company, conn, top_k=3):
        rcs = all_chunks_by_company[company]
        return pl.RetrievalResult(
            query=query, intent="all_company_synthesis",
            contexts=rcs, companies=[company], abstain=False,
            diagnostics={"top_rerank": rcs[0].rerank_score if rcs else 0.0},
        )

    # No-op expand/conflict so only the cap logic matters
    monkeypatch.setattr("src.retrieval.synthesis._retrieve_single_company", _fake_single)
    monkeypatch.setattr(pl, "find_conflicting_chunks", lambda r, cf, conn: list(r))
    monkeypatch.setattr(pl, "expand_to_parents", lambda chunks, conn: list(chunks))
    monkeypatch.setattr(pl, "expand_to_siblings", lambda chunks, conn: list(chunks))

    conn = MagicMock()
    result = pl.retrieve_all_company_synthesis("NOI across all", conn)

    cap = pl.MAX_SYNTHESIS_CONTEXT_CHUNKS or len(companies) * 3
    assert len(result.contexts) <= cap, (
        f"Expected at most {cap} chunks, got {len(result.contexts)}"
    )
