from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from src.models import Chunk, RetrievedChunk
from src.versioning.chains import dedupe_by_version_group
from src.versioning.classifier import (
    INTENT_CONFIDENCE_THRESHOLD,
    _regex_classify_intent,
    classify_intent,
)


def _mk_retrieved(company: str, doc_version: str, page: int) -> RetrievedChunk:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="DLR",
        doc_type="investor_presentation",
        report_date=doc_version,
        period_covered="Q4 2025",
        doc_version=doc_version,
        section_title="Leverage",
        page_number=page,
        content_type="text",
        chunk_text=f"{company} leverage in {doc_version}",
    )
    return RetrievedChunk(chunk=chunk, rerank_score=1.0)


def test_classify_intent_non_temporal_between_query_is_latest() -> None:
    query = "What is the difference between NOI and FFO for Digital Realty?"
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "latest", "confidence": 0.95}):
        assert classify_intent(query) == "latest"


def test_classify_intent_between_temporal_periods_is_comparison() -> None:
    query = "How did DLR leverage move between December 2025 and March 2026?"
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "comparison", "confidence": 0.97}):
        assert classify_intent(query) == "comparison"


def test_classify_intent_from_to_temporal_query_is_comparison() -> None:
    query = "How did DLR leverage change from December 2025 to March 2026?"
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "comparison", "confidence": 0.96}):
        assert classify_intent(query) == "comparison"


def test_dedupe_latest_only_when_non_temporal_between_query() -> None:
    # Same version group, two versions. For a non-temporal "difference
    # between X and Y" query, the classifier should remain on latest and
    # deduplication should keep one version.
    retrieved = [
        _mk_retrieved("Digital Realty", "2025-12", page=14),
        _mk_retrieved("Digital Realty", "2026-03", page=15),
    ]
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "latest", "confidence": 0.95}):
        intent = classify_intent("What is the difference between NOI and FFO for DLR?")
    deduped = dedupe_by_version_group(retrieved, intent)
    assert intent == "latest"
    assert len(deduped) == 1
    assert deduped[0].chunk.doc_version == "2026-03"


def test_regex_conflict_word_classifies_as_comparison() -> None:
    # The regex path maps conflict-language to "comparison"; the LLM layer
    # is responsible for the finer-grained "conflict" distinction.
    assert _regex_classify_intent("Why is there a discrepancy in the reported FFO?") == "comparison"


def test_regex_disagreement_classifies_as_comparison() -> None:
    assert _regex_classify_intent("There is a disagreement in the revenue figures.") == "comparison"


def test_regex_inconsistent_classifies_as_comparison() -> None:
    assert _regex_classify_intent("The numbers look inconsistent across filings.") == "comparison"


def test_regex_conflicting_classifies_as_comparison() -> None:
    assert _regex_classify_intent("Conflicting revenue data between the two reports.") == "comparison"


def test_regex_mismatch_classifies_as_comparison() -> None:
    assert _regex_classify_intent("There is a mismatch in the Q3 occupancy numbers.") == "comparison"


def test_llm_above_threshold_returns_llm_intent() -> None:
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "conflict", "confidence": 0.92}):
        result = classify_intent("Why does the company disagree on revenue figures?")
    assert result == "conflict"


def test_llm_returns_historical_above_threshold() -> None:
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "historical", "confidence": 0.90}):
        result = classify_intent("What was the FFO in Q2 2023?")
    assert result == "historical"


def test_llm_below_threshold_falls_back_to_regex(tmp_path, monkeypatch) -> None:
    # Route the audit log to a temp file so the test is hermetic.
    monkeypatch.setattr(
        "src.versioning.classifier._LOG_PATH",
        tmp_path / "intent_decisions.jsonl",
    )
    # LLM returns low confidence; regex should take over.
    # The query has "discrepancy" so regex returns "comparison".
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "conflict", "confidence": 0.50}):
        result = classify_intent("Is there a discrepancy in the numbers?")

    assert result == "comparison"

    # Verify audit log contains method=regex_fallback.
    log_file = tmp_path / "intent_decisions.jsonl"
    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record["method"] == "regex_fallback"
    assert record["llm_intent"] == "conflict"
    assert record["confidence"] == 0.50


def test_llm_exactly_at_threshold_uses_llm_intent() -> None:
    # Confidence exactly equal to the threshold should pass (>= comparison).
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "comparison", "confidence": INTENT_CONFIDENCE_THRESHOLD}):
        result = classify_intent("Compare revenue from Q3 to Q4.")
    assert result == "comparison"


def test_llm_exception_falls_back_to_regex_and_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.versioning.classifier._LOG_PATH",
        tmp_path / "intent_decisions.jsonl",
    )
    with patch("src.versioning.classifier._llm_classify_intent",
               side_effect=RuntimeError("API down")):
        # Query has no comparison/historical markers → regex returns "latest"
        result = classify_intent("What is the current occupancy rate?")

    assert result == "latest"

    log_file = tmp_path / "intent_decisions.jsonl"
    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record["method"] == "regex_error_fallback"
    assert "API down" in record["error"]


def test_llm_exception_conflict_query_falls_back_to_regex(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.versioning.classifier._LOG_PATH",
        tmp_path / "intent_decisions.jsonl",
    )
    with patch("src.versioning.classifier._llm_classify_intent",
               side_effect=ConnectionError("timeout")):
        result = classify_intent("There is a discrepancy in the reported NOI.")

    # Regex catches "discrepancy" → "comparison"
    assert result == "comparison"


def test_conflict_intent_bypasses_dedupe() -> None:
    retrieved = [
        _mk_retrieved("Digital Realty", "2025-06", page=1),
        _mk_retrieved("Digital Realty", "2025-12", page=2),
        _mk_retrieved("Digital Realty", "2026-03", page=3),
    ]
    # conflict intent must NOT drop older versions
    deduped = dedupe_by_version_group(retrieved, "conflict")
    assert len(deduped) == 3


def test_conflict_intent_classify_end_to_end_bypasses_dedupe() -> None:
    retrieved = [
        _mk_retrieved("Digital Realty", "2025-06", page=1),
        _mk_retrieved("Digital Realty", "2026-03", page=2),
    ]
    with patch("src.versioning.classifier._llm_classify_intent",
               return_value={"intent": "conflict", "confidence": 0.93}):
        intent = classify_intent("Why are the revenue figures inconsistent across filings?")

    assert intent == "conflict"
    deduped = dedupe_by_version_group(retrieved, intent)
    assert len(deduped) == 2  # all versions preserved
