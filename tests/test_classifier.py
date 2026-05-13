from __future__ import annotations

from uuid import uuid4

from src.models import Chunk, RetrievedChunk
from src.versioning.chains import dedupe_by_chain
from src.versioning.classifier import classify_intent


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
    assert classify_intent(query) == "latest"


def test_classify_intent_between_temporal_periods_is_comparison() -> None:
    query = "How did DLR leverage move between December 2025 and March 2026?"
    assert classify_intent(query) == "comparison"


def test_classify_intent_from_to_temporal_query_is_comparison() -> None:
    query = "How did DLR leverage change from December 2025 to March 2026?"
    assert classify_intent(query) == "comparison"


def test_dedupe_latest_only_when_non_temporal_between_query() -> None:
    # Same chain, two versions. For a non-temporal "difference between X and Y"
    # query, the classifier should remain on latest and dedupe should keep one version.
    retrieved = [
        _mk_retrieved("Digital Realty", "2025-12", page=14),
        _mk_retrieved("Digital Realty", "2026-03", page=15),
    ]
    intent = classify_intent("What is the difference between NOI and FFO for DLR?")
    deduped = dedupe_by_chain(retrieved, intent)
    assert intent == "latest"
    assert len(deduped) == 1
    assert deduped[0].chunk.doc_version == "2026-03"
