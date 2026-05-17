"""Shared helpers for BM25 and vector retrieval modules."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from src.models import Chunk, ContentType


def _row_to_chunk(row: Sequence[Any], columns: list[str]) -> Chunk:
    """Hydrate a Chunk dataclass from a psycopg result row.

    Args:
        row: A single result row from a psycopg cursor.
        columns: List of column names in the same order as the row values,
            obtained from ``[d.name for d in cursor.description]``.

    Returns:
        A fully populated Chunk instance.
    """
    r: dict[str, Any] = dict(zip(columns, row))

    def _uuid(val: Any) -> UUID | None:
        if val is None:
            return None
        return val if isinstance(val, UUID) else UUID(str(val))

    return Chunk(
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
        content_type=r["content_type"],  # type: ignore[arg-type]
        source_authority=r["source_authority"],
        chunk_text=r["chunk_text"],
        is_parent=r["is_parent"],
        token_count=r.get("token_count"),
        doc_subtype=r.get("doc_subtype", "unknown") or "unknown",
        page_content_class=r.get("page_content_class", "unknown") or "unknown",
        embedding=None,  # never returned from retrieval queries for efficiency
    )
