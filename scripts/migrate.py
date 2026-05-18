"""Create the documents and chunks tables plus indexes.

Idempotent: safe to re-run; uses IF NOT EXISTS everywhere.

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration steps — each entry is (step_name, sql).
# Steps that are inherently idempotent (IF NOT EXISTS / CREATE OR REPLACE /
# DROP … IF EXISTS) need no special handling.  The one exception is the
# content_type CHECK constraint, where we use a DO-block that only silences
# the narrow "duplicate object" SQLSTATE family (42P07 / 42P06 / 42710 /
# 42701).  Every other exception is re-raised so broken DDL fails loud.
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, str]] = [
    (
        "enable_vector_extension",
        "CREATE EXTENSION IF NOT EXISTS vector;",
    ),
    (
        "create_table_documents",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            company         TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            doc_type        TEXT NOT NULL,
            report_date     TEXT NOT NULL,
            period_covered  TEXT,
            doc_version     TEXT NOT NULL,
            source_path     TEXT NOT NULL UNIQUE,
            chart_enrichment_completed_at TIMESTAMPTZ,
            ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    ),
    (
        "alter_documents_add_chart_enrichment_completed_at",
        """
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS chart_enrichment_completed_at TIMESTAMPTZ;
        """,
    ),
    (
        "alter_documents_add_doc_subtype",
        """
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS doc_subtype TEXT NOT NULL DEFAULT 'unknown';
        """,
    ),
    (
        "create_index_documents_chain",
        """
        CREATE INDEX IF NOT EXISTS documents_chain_idx
            ON documents (company, doc_type, doc_version);
        """,
    ),
    (
        "create_index_documents_chart_enrichment_completed",
        """
        CREATE INDEX IF NOT EXISTS documents_chart_enrichment_completed_idx
            ON documents (chart_enrichment_completed_at);
        """,
    ),
    (
        "create_table_chunks",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            parent_chunk_id   UUID NULL REFERENCES chunks(id) ON DELETE CASCADE,

            -- Metadata (mirrored from documents for query-time filtering speed)
            company           TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            doc_type          TEXT NOT NULL,
            report_date       TEXT NOT NULL,
            period_covered    TEXT,
            doc_version       TEXT NOT NULL,
            section_title     TEXT,
            page_number       INTEGER,
            content_type      TEXT NOT NULL CHECK (content_type IN ('text', 'table', 'chart_caption', 'chart_description', 'chart_context', 'mixed')),
            source_authority  TEXT NOT NULL DEFAULT 'company_authored',

            -- Content
            chunk_text        TEXT NOT NULL,
            chunk_text_tsv    tsvector,
            embedding         halfvec(1024),  -- Qwen/Qwen3-Embedding-0.6B

            -- Contextual retrieval columns — additive: chunk_text /
            -- chunk_text_tsv / embedding above are preserved; new retrieval queries use
            -- the contextualized columns populated by scripts/contextualize.py.
            contextualized_text       TEXT,
            contextualized_text_tsv   tsvector,
            contextualized_embedding  halfvec(1024),

            -- Granular document subtype; populated after initial ingestion via
            -- scripts/reclassify_subtypes.py or during re-ingestion.
            doc_subtype       TEXT NOT NULL DEFAULT 'unknown',

            -- Role
            is_parent         BOOLEAN NOT NULL DEFAULT FALSE,
            token_count       INTEGER,

            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    ),
    (
        "create_or_replace_fn_chunks_tsv_update",
        """
        CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.chunk_text_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
    ),
    (
        "recreate_trigger_chunks_tsv_update_trg",
        """
        DROP TRIGGER IF EXISTS chunks_tsv_update_trg ON chunks;
        CREATE TRIGGER chunks_tsv_update_trg
            BEFORE INSERT OR UPDATE OF chunk_text ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();
        """,
    ),
    (
        "create_or_replace_fn_chunks_ctx_tsv_update",
        """
        CREATE OR REPLACE FUNCTION chunks_ctx_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.contextualized_text_tsv := to_tsvector('english', COALESCE(NEW.contextualized_text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
    ),
    (
        "recreate_trigger_chunks_ctx_tsv_update_trg",
        """
        DROP TRIGGER IF EXISTS chunks_ctx_tsv_update_trg ON chunks;
        CREATE TRIGGER chunks_ctx_tsv_update_trg
            BEFORE INSERT OR UPDATE OF contextualized_text ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_ctx_tsv_update();
        """,
    ),
    (
        "create_index_chunks_tsv",
        """
        CREATE INDEX IF NOT EXISTS chunks_tsv_idx
            ON chunks USING GIN (chunk_text_tsv);
        """,
    ),
    (
        "recreate_index_chunks_embedding_hnsw",
        """
        DROP INDEX IF EXISTS chunks_embedding_hnsw;
        CREATE INDEX chunks_embedding_hnsw
            ON chunks USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """,
    ),
    (
        "create_index_chunks_company_version",
        """
        CREATE INDEX IF NOT EXISTS chunks_company_version_idx
            ON chunks (company, doc_version);
        """,
    ),
    (
        "create_index_chunks_content_type",
        """
        CREATE INDEX IF NOT EXISTS chunks_content_type_idx
            ON chunks (content_type);
        """,
    ),
    (
        "create_index_chunks_parent",
        """
        CREATE INDEX IF NOT EXISTS chunks_parent_idx
            ON chunks (parent_chunk_id);
        """,
    ),
    (
        "create_index_chunks_document",
        """
        CREATE INDEX IF NOT EXISTS chunks_document_idx
            ON chunks (document_id);
        """,
    ),
    (
        "create_index_chunks_ctx_tsv",
        """
        CREATE INDEX IF NOT EXISTS chunks_ctx_tsv_idx
            ON chunks USING GIN (contextualized_text_tsv);
        """,
    ),
    (
        "recreate_index_chunks_ctx_embed_hnsw",
        """
        DROP INDEX IF EXISTS chunks_ctx_embed_hnsw;
        CREATE INDEX chunks_ctx_embed_hnsw
            ON chunks USING hnsw (contextualized_embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """,
    ),
    (
        "alter_chunks_add_doc_subtype",
        """
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS doc_subtype TEXT NOT NULL DEFAULT 'unknown';
        """,
    ),
    (
        "alter_chunks_add_page_content_class",
        """
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS page_content_class TEXT NOT NULL DEFAULT 'unknown';
        """,
    ),
    (
        "create_table_chunk_claims",
        """
        CREATE TABLE IF NOT EXISTS chunk_claims (
            claim_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id     UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            company      TEXT NOT NULL,
            doc_version  TEXT NOT NULL,
            metric       TEXT NOT NULL,
            value        TEXT NOT NULL,
            period       TEXT,
            page_number  INTEGER,
            extracted_at TIMESTAMPTZ DEFAULT now()
        );
        """,
    ),
    (
        "alter_chunk_claims_add_value_qualifier",
        """
        ALTER TABLE chunk_claims ADD COLUMN IF NOT EXISTS value_qualifier TEXT;
        """,
    ),
    (
        "create_index_chunk_claims_company_metric",
        """
        CREATE INDEX IF NOT EXISTS chunk_claims_company_metric
            ON chunk_claims (company, metric);
        """,
    ),
    # The CHECK constraint drop-and-recreate is not fully idempotent via plain
    # DDL because the constraint may already exist under a different name on
    # older deployments.  We keep a DO-block but narrow the exception handler
    # to only the "duplicate object" SQLSTATE family so real DDL errors are
    # never silently swallowed.
    (
        "recreate_chunks_content_type_check",
        """
        DO $$
        BEGIN
            ALTER TABLE chunks
                DROP CONSTRAINT IF EXISTS chunks_content_type_check;
            ALTER TABLE chunks
                ADD CONSTRAINT chunks_content_type_check
                CHECK (content_type IN (
                    'text', 'table', 'chart_caption',
                    'chart_description', 'chart_context', 'mixed'
                ));
        EXCEPTION
            -- 42P07 duplicate_table  42P06 duplicate_schema
            -- 42710 duplicate_object  42701 duplicate_column
            WHEN duplicate_table OR duplicate_object
              OR duplicate_column OR sqlstate '42P06'
            THEN
                NULL;  -- constraint already exists under this name — safe to skip
            WHEN OTHERS THEN
                RAISE;  -- surface every other DDL failure
        END;
        $$;
        """,
    ),
]

# Duplicate-object SQLSTATEs that Python-side should treat as idempotent.
_IDEMPOTENT_SQLSTATES = {"42P07", "42P06", "42710", "42701"}


def _run_step(cur: object, name: str, sql: str) -> None:  # type: ignore[type-arg]
    """Execute one migration step and log the outcome."""
    import psycopg

    try:
        cur.execute(sql)  # type: ignore[attr-defined]
        log.info("step %-55s  OK", name)
    except psycopg.Error as exc:
        sqlstate = getattr(exc, "sqlstate", None) or getattr(getattr(exc, "diag", None), "sqlstate", None)
        if sqlstate in _IDEMPOTENT_SQLSTATES:
            log.info("step %-55s  skipped (idempotent, sqlstate=%s)", name, sqlstate)
        else:
            log.error("step %-55s  FAILED — sqlstate=%s: %s", name, sqlstate, exc)
            raise


def main() -> None:
    with connect() as conn:
        for name, sql in STEPS:
            with conn.cursor() as cur:
                _run_step(cur, name, sql)
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name;"
            )
            tables = [r[0] for r in cur.fetchall()]

    log.info("Migration complete. Tables: %s", tables)
    print(f"Migration complete. Tables: {tables}")


if __name__ == "__main__":
    main()
