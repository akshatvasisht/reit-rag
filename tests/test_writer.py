"""Tests for src/ingestion/writer.py — inserted-count accuracy and idempotency.

Each test opens a real DB connection and rolls back at the end so no test data
persists.  Tests are skipped when DATABASE_URL is not configured.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.models import Chunk, DocumentMeta

# Skip all tests in this module when the database is not reachable.
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping writer integration tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_meta(
    source_path: str | None = None,
    company: str = "Test Corp",
    ticker: str = "TST",
    doc_type: str = "investor_presentation",
    report_date: str = "2026-01",
    doc_version: str = "2026-01",
    doc_subtype: str = "unknown",
) -> DocumentMeta:
    return DocumentMeta(
        company=company,
        ticker=ticker,
        doc_type=doc_type,
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=doc_version,
        source_path=source_path or f"test/path/{uuid4()}.pdf",
        doc_subtype=doc_subtype,
    )


def _make_chunk(document_id, page: int = 1) -> Chunk:
    return Chunk(
        document_id=document_id,
        company="Test Corp",
        ticker="TST",
        doc_type="investor_presentation",
        report_date="2026-01",
        doc_version="2026-01",
        chunk_text=f"Test content on page {page}",
        content_type="text",
        page_number=page,
    )


# ---------------------------------------------------------------------------
# Sub-step A: inserted-count reflects actual rows affected
# ---------------------------------------------------------------------------


class TestWriteChunksInsertedCount:
    """write_chunks must return the number of rows actually inserted, not batch size."""

    def test_first_insert_returns_full_count(self, real_conn):
        from src.ingestion.writer import write_chunks, write_document

        doc = _make_doc_meta()
        doc_id = write_document(real_conn, doc)
        chunks = [_make_chunk(doc_id, page=i) for i in range(1, 4)]

        count = write_chunks(real_conn, chunks)
        assert count == 3, f"Expected 3 inserts, got {count}"

    def test_duplicate_insert_returns_zero(self, real_conn):
        """Re-inserting the same chunk IDs must report 0 actual inserts."""
        from src.ingestion.writer import write_chunks, write_document

        doc = _make_doc_meta()
        doc_id = write_document(real_conn, doc)
        chunks = [_make_chunk(doc_id, page=i) for i in range(1, 4)]

        first = write_chunks(real_conn, chunks)
        assert first == 3

        second = write_chunks(real_conn, chunks)
        assert second == 0, (
            f"Expected 0 on duplicate insert, got {second}; "
            "write_chunks is using len(batch) instead of cur.rowcount"
        )

    def test_empty_input_returns_zero(self, real_conn):
        from src.ingestion.writer import write_chunks

        count = write_chunks(real_conn, [])
        assert count == 0

    def test_partial_duplicate_returns_new_only(self, real_conn):
        """Only the genuinely new chunks in a mixed batch are counted."""
        from src.ingestion.writer import write_chunks, write_document

        doc = _make_doc_meta()
        doc_id = write_document(real_conn, doc)
        old_chunks = [_make_chunk(doc_id, page=i) for i in range(1, 3)]
        write_chunks(real_conn, old_chunks)  # insert 2

        new_chunk = _make_chunk(doc_id, page=99)
        mixed = old_chunks + [new_chunk]
        count = write_chunks(real_conn, mixed)
        assert count == 1, f"Expected 1 new insert in mixed batch, got {count}"


# ---------------------------------------------------------------------------
# Sub-step B: write_document idempotency on source_path
# ---------------------------------------------------------------------------


class TestWriteDocumentIdempotency:
    """write_document must return the original document_id on duplicate source_path."""

    def test_first_insert_returns_uuid(self, real_conn):
        from src.ingestion.writer import write_document

        doc = _make_doc_meta()
        doc_id = write_document(real_conn, doc)
        assert doc_id is not None

    def test_duplicate_source_path_returns_same_id(self, real_conn):
        """Inserting the same source_path twice must not raise and must return original id."""
        from src.ingestion.writer import write_document

        doc = _make_doc_meta()
        first_id = write_document(real_conn, doc)

        doc2 = _make_doc_meta(source_path=doc.source_path)
        second_id = write_document(real_conn, doc2)

        assert second_id == first_id, (
            f"Expected same document_id on duplicate source_path; "
            f"got first={first_id}, second={second_id}"
        )

    def test_no_duplicate_rows_after_double_insert(self, real_conn):
        """The documents table must contain exactly one row for the source_path."""
        from src.ingestion.writer import write_document

        doc = _make_doc_meta()
        write_document(real_conn, doc)
        write_document(real_conn, doc)

        with real_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE source_path = %s",
                (doc.source_path,),
            )
            count = cur.fetchone()[0]  # type: ignore[index]
        assert count == 1, f"Expected exactly 1 document row, found {count}"

    def test_duplicate_source_path_preserves_original_metadata(self, real_conn):
        """A second write_document with the same source_path but different metadata
        must NOT overwrite the original row's fields. ON CONFLICT DO NOTHING leaves
        the existing row untouched; the function only returns its id."""
        from src.ingestion.writer import write_document

        original = _make_doc_meta(
            company="Test Corp",
            ticker="TST",
            doc_type="investor_presentation",
            doc_version="2026-01",
            doc_subtype="quarterly_investor_deck",
        )
        write_document(real_conn, original)

        # Same source_path, different everything else — simulates a re-ingest
        # where the metadata classifier changed its mind.
        clobber = _make_doc_meta(
            source_path=original.source_path,
            company="Other Corp",
            ticker="OTH",
            doc_type="thematic_report",
            doc_version="2025-99",
            doc_subtype="thematic_report",
        )
        write_document(real_conn, clobber)

        with real_conn.cursor() as cur:
            cur.execute(
                "SELECT company, ticker, doc_type, doc_version, doc_subtype "
                "FROM documents WHERE source_path = %s",
                (original.source_path,),
            )
            row = cur.fetchone()
        assert row is not None
        company, ticker, doc_type, doc_version, doc_subtype = row
        assert company == original.company
        assert ticker == original.ticker
        assert doc_type == original.doc_type
        assert doc_version == original.doc_version
        assert doc_subtype == original.doc_subtype


# ---------------------------------------------------------------------------
# Fixture: real DB connection that rolls back after each test
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_conn():
    """Yield a psycopg connection; roll back after the test so data is not persisted."""
    import psycopg
    from pgvector.psycopg import register_vector

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, autocommit=False) as conn:
        register_vector(conn)
        yield conn
        conn.rollback()
