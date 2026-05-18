"""Shared dataclasses passed through the ingestion and retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence
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
    citation_page: int | None  # Parsed page number from citation; None when the cited chunk lacks a page


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

    @classmethod
    def from_row(cls, row: Sequence[Any], columns: list[str]) -> "Chunk":
        """Hydrate a Chunk from a psycopg result row plus its column names."""
        r: dict[str, Any] = dict(zip(columns, row))

        def _uuid(val: Any) -> Optional[UUID]:
            if val is None:
                return None
            return val if isinstance(val, UUID) else UUID(str(val))

        return cls(
            id=_uuid(r["id"]),  # type: ignore[arg-type]
            document_id=_uuid(r["document_id"]),  # type: ignore[arg-type]
            parent_chunk_id=_uuid(r["parent_chunk_id"]),
            company=r["company"],
            ticker=r["ticker"],
            doc_type=r["doc_type"],
            report_date=r["report_date"],
            period_covered=r.get("period_covered"),
            doc_version=r["doc_version"],
            section_title=r.get("section_title"),
            page_number=r.get("page_number"),
            content_type=r["content_type"],
            source_authority=r["source_authority"],
            chunk_text=r["chunk_text"],
            is_parent=r["is_parent"],
            token_count=r.get("token_count"),
            doc_subtype=r.get("doc_subtype", "unknown") or "unknown",
            page_content_class=r.get("page_content_class", "unknown") or "unknown",
            embedding=None,  # never returned from retrieval queries for efficiency
        )


@dataclass
class RetrievedChunk:
    """A chunk plus retrieval scores, returned by the retrieval pipeline."""
    chunk: Chunk
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # Provenance fields (2g) — indicate how this chunk arrived in the context.
    # Values: "reranked" | "parent_expanded" | "sibling_expanded" |
    #         "conflict_injected" | "table_pair_expanded"
    retrieval_stage: Optional[str] = None
    # ID (as str) of the chunk that triggered this expansion, when applicable.
    trigger_chunk_id: Optional[str] = None
    # Human-readable description of why this chunk was included.
    expansion_reason: Optional[str] = None
