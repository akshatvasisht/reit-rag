-- Anthropic Contextual Retrieval — schema for prepended chunk-context.
-- Additive: original chunk_text / chunk_text_tsv / embedding are preserved.
-- Currently scaffolded but not populated; see README "What I Would Improve".

ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS contextualized_text TEXT,
  ADD COLUMN IF NOT EXISTS contextualized_text_tsv tsvector,
  ADD COLUMN IF NOT EXISTS contextualized_embedding halfvec(1024);

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

CREATE INDEX IF NOT EXISTS chunks_ctx_tsv_idx
    ON chunks USING GIN (contextualized_text_tsv);

CREATE INDEX IF NOT EXISTS chunks_ctx_embed_hnsw
    ON chunks USING hnsw (contextualized_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
