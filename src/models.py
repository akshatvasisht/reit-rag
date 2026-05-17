"""Shared dataclasses passed through the ingestion and retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import UUID, uuid4


# Granular document-subtype vocabulary.  ``"unknown"`` is a valid steady-state
# value meaning the classifier could not confidently identify the subtype —
# it is treated as a distinct key for version-group deduplication.
DOC_SUBTYPES = Literal[
    "quarterly_investor_deck",
    "investor_day_session",
    "merger_presentation",
    "company_update",
    "annual_supplement",
    "thematic_report",
    "roadshow_deck",
    "unknown",
]


@dataclass
class Claim:
    """One attributed factual claim in a generated answer."""
    text: str              # The claim sentence as written
    value: str             # The specific numerical or factual value asserted, empty string if non-numerical
    citation: str          # Full citation string: "Company, Doc Type, Month Year, p.N"
    citation_company: str  # Parsed company name from citation
    citation_date: str     # Parsed date from citation e.g. "March 2026"
    citation_page: int     # Parsed page number from citation


@dataclass
class StructuredAnswer:
    """Schema-validated answer replacing free-form text generation."""
    answer_prose: str
    claims: list[Claim]
    abstain: bool
    abstain_reason: str
    forward_looking: bool
    # Per-claim numeric consistency issues surfaced after generation; empty list
    # means every cited value was verified in its source chunk.
    numeric_consistency_report: list[dict] = field(default_factory=list)
    # Out-of-corpus entity attribution issues; empty list means no such pattern
    # was detected in this answer.
    ooc_attribution_issues: list[dict] = field(default_factory=list)
    # Number of additional retrieval passes performed beyond the initial one.
    retrieval_hops: int = 0
    # Sub-questions that triggered additional retrieval passes, in order fired.
    sub_queries_fired: list[str] = field(default_factory=list)
    # Calibrated composite confidence score in [0, 1] derived from post-rerank
    # evidence quality; 0.0 when not populated (e.g. abstained path).
    retrieval_confidence: float = 0.0


ContentType = Literal["text", "table", "chart_caption", "chart_description", "chart_context", "mixed"]


@dataclass
class DocumentMeta:
    """Document-level metadata derived once per file at ingestion."""
    company: str
    ticker: str
    doc_type: str            # e.g. "investor_presentation" | "thematic_report"
    report_date: str         # "YYYY-MM"
    period_covered: Optional[str]  # e.g. "Quarter period label"
    doc_version: str         # "YYYY-MM"; used for chain ordering
    source_path: str
    doc_subtype: str = "unknown"  # granular subtype within doc_type
    id: UUID = field(default_factory=uuid4)


@dataclass
class Chunk:
    """A retrieval or generation unit, pre-embedding.

    `parent_chunk_id` links a small (retrieval) chunk to its large (generation)
    parent for small-to-large expansion. Parents have `is_parent=True` and
    `parent_chunk_id=None`.
    """
    document_id: UUID
    company: str
    ticker: str
    doc_type: str
    report_date: str
    doc_version: str
    chunk_text: str
    content_type: ContentType
    period_covered: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    source_authority: str = "company_authored"
    parent_chunk_id: Optional[UUID] = None
    is_parent: bool = False
    token_count: Optional[int] = None
    embedding: Optional[list[float]] = None
    doc_subtype: str = "unknown"  # granular subtype within doc_type; flows from DocumentMeta
    page_content_class: str = "unknown"  # page-level content classification for boilerplate filtering
    id: UUID = field(default_factory=uuid4)


@dataclass
class RetrievedChunk:
    """A chunk plus retrieval scores, returned by the retrieval pipeline."""
    chunk: Chunk
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
