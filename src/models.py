"""Shared dataclasses passed through the ingestion and retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import UUID, uuid4


ContentType = Literal["text", "table", "chart_caption", "chart_description", "mixed"]


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
    id: UUID = field(default_factory=uuid4)


@dataclass
class RetrievedChunk:
    """A chunk plus retrieval scores, returned by the retrieval pipeline."""
    chunk: Chunk
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
