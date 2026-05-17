"""Create the documents and chunks tables plus indexes.

Idempotent: safe to re-run; uses IF NOT EXISTS everywhere.

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

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

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS chart_enrichment_completed_at TIMESTAMPTZ;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS doc_subtype TEXT NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS documents_chain_idx
    ON documents (company, doc_type, doc_version);

CREATE INDEX IF NOT EXISTS documents_chart_enrichment_completed_idx
    ON documents (chart_enrichment_completed_at);

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

-- Trigger to keep chunk_text_tsv in sync with chunk_text
CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.chunk_text_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_update_trg ON chunks;
CREATE TRIGGER chunks_tsv_update_trg
    BEFORE INSERT OR UPDATE OF chunk_text ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();

-- Sibling trigger for the contextualized_text column.
CREATE OR REPLACE FUNCTION chunks_ctx_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.contextualized_text_tsv := to_tsvector('english', COALESCE(NEW.contextualized_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_ctx_tsv_update_trg ON chunks;
CREATE TRIGGER chunks_ctx_tsv_update_trg
    BEFORE INSERT OR UPDATE OF contextualized_text ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_ctx_tsv_update();

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING GIN (chunk_text_tsv);

-- HNSW on halfvec(1024); halfvec supports high-dimensional vectors for this model.
-- ef_construction=200 yields higher recall at index-build time for production RAG workloads.
DROP INDEX IF EXISTS chunks_embedding_hnsw;
CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS chunks_company_version_idx
    ON chunks (company, doc_version);

CREATE INDEX IF NOT EXISTS chunks_content_type_idx
    ON chunks (content_type);

CREATE INDEX IF NOT EXISTS chunks_parent_idx
    ON chunks (parent_chunk_id);

CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON chunks (document_id);

-- Contextual retrieval indexes — GIN + HNSW over contextualized columns.
CREATE INDEX IF NOT EXISTS chunks_ctx_tsv_idx
    ON chunks USING GIN (contextualized_text_tsv);

-- ef_construction=200 matches the primary HNSW index for consistent recall.
DROP INDEX IF EXISTS chunks_ctx_embed_hnsw;
CREATE INDEX chunks_ctx_embed_hnsw
    ON chunks USING hnsw (contextualized_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Additive: doc_subtype column for granular version-group keying.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS doc_subtype TEXT NOT NULL DEFAULT 'unknown';

-- Additive: page-level content classification for boilerplate filtering.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS page_content_class TEXT NOT NULL DEFAULT 'unknown';

-- Atomic numerical claims extracted from each chunk at ingestion time.
-- Indexed by (company, metric) to support conflict-detection joins at
-- query time without a full-table scan.
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

ALTER TABLE chunk_claims ADD COLUMN IF NOT EXISTS value_qualifier TEXT;

CREATE INDEX IF NOT EXISTS chunk_claims_company_metric
    ON chunk_claims (company, metric);

-- Extend the content_type CHECK constraint to include chart_context.
-- Drop-and-recreate is idempotent: the IF NOT EXISTS on the table above means
-- new deployments get the constraint inline; existing deployments get it here.
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
    WHEN others THEN
        NULL;  -- tolerate if constraint name differs across environments
END;
$$;
"""


def main() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        )
        tables = [r[0] for r in cur.fetchall()]
        print(f"Migration complete. Tables: {tables}")


if __name__ == "__main__":
    main()
