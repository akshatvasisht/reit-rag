# Corporate RAG for REIT Investor Presentations

Retrieval-augmented question answering over 10 REIT documents (investor presentations plus one thematic report). The system is designed for analyst-style financial queries with explicit evidence, source citations, and refusal behavior when support is insufficient.

## Features

- Hybrid retrieval (BM25 + dense) with reranking.
- Version-aware handling for companies with multiple presentation versions.
- Chart enrichment pipeline that indexes vision-extracted chart descriptions.
- Inline citation format with company, document type, date, and page.
- Streamlit interface with source expanders and confidence summary metrics.

## How It Works

1. Parse and chunk PDFs with metadata per chunk.
2. Index lexical and vector representations in PostgreSQL/pgvector.
3. Retrieve and rerank evidence for each query.
4. Assemble context and generate a cited answer.
5. Validate citation support against retrieved contexts before display.

## Database

PostgreSQL with the pgvector extension, run locally via Docker. Each chunk row carries dense (`halfvec(1024)`) and lexical (`tsvector`) representations in the same table; HNSW index on the embedding column, GIN index on the tsvector. Storing both signals in one database removes the need for a separate vector store and keeps version-aware filtering as a plain SQL `WHERE`. Production path is Snowflake Cortex Search — the metadata schema and retrieval logic carry over unchanged.

## Chunking Strategy

Hierarchical small-to-large at structural boundaries from Docling output. Child chunks (~256 tokens, one per paragraph / table row / chart caption) carry the retrieval signal; parent chunks (~512–1024 tokens, one per slide or section) provide generation context. Children are linked to their parent via a self-FK; at retrieval time, matched children are expanded to their parents before the LLM sees them. Fixed-size chunking and slide-level-only chunking were rejected — the former routinely splits table headers from data rows; the latter dilutes the retrieval vector across mixed content.

## Retrieval Approach

Hybrid: BM25 (PostgreSQL tsvector) and dense (Qwen3-Embedding-0.6B + pgvector HNSW) fire in parallel, fuse via Reciprocal Rank Fusion (k=60), then a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the top candidates. Hybrid is non-negotiable for financial documents — exact-term matches (tickers, FFO, NOI, Q4 2025) need lexical retrieval, semantic queries ("how leveraged is DLR") need dense. An entity pre-filter narrows the candidate pool when the query names a known company; abstention fires when the top rerank score falls below a calibrated threshold or the LLM's own system prompt cannot ground an answer in the retrieved context.

## Versioning

Documents are grouped into version chains by `(company, doc_type)`, ordered by `report_date`. An intent classifier reads the query for `latest` (default), `historical`, or `comparison` triggers. For `latest` intent, version-chain dedup suppresses older versions before the LLM sees context, so the model never silently averages between (for example) DLR's December 2025 and March 2026 figures. For `comparison` intent ("how did DLR change between Dec and Mar"), both versions are retained and the answer surfaces each with its date in the citation.

## Conflicting Information

When the retrieved context contains figures that disagree across sources, the system surfaces each value with its own inline citation `[Company, Doc Type, Date, p.N]` rather than merging or averaging. The system prompt instructs the LLM to label management guidance as "Management guided…" rather than as reported fact. A post-generation citation faithfulness check parses every citation out of the answer and verifies it maps to a chunk that was actually in the retrieved context; unsupported citations surface as a UI warning rather than being silently allowed through.

## Charts and Tables

Tables are extracted cell-by-cell by Docling's TableFormer and indexed as `content_type='table'` chunks. Charts are handled by a second-pass enrichment script: each Docling-detected `PictureItem` is sent to Claude vision with a structured extraction prompt covering chart type, title, axes, data series, key values, and a one-line insight. Decorative images are filtered out via a `NOT_CHART` sentinel response. Chart-derived chunks are tagged `content_type='chart_description'` so the UI can surface that the value came from vision extraction rather than text. When the vision model hedges or refuses, the system falls back to honest abstention with the page cited.

## Known Limitations

- **Lexical mismatch on chart-derived queries.** Vision-extracted chart descriptions use the model's own phrasing; a query word can fail to match a near-synonym in the description and miss retrieval. Wider RRF candidate pool mitigates but does not eliminate this.
- **No conversation memory.** The Streamlit UI is intentionally single-turn; follow-up queries ("what about for VICI?") do not carry context from the previous answer.
- **Citation faithfulness is source-level, not claim-level.** The check verifies that every cited chunk was in the retrieved context, not that the chunk's content actually supports the specific claim attached. Claim-level verification is RAGAS territory and runs offline.
- **Multi-tenant access control is conceptual.** The metadata schema is RLS-ready but row-level security is not configured; documents are treated as a single tenant.
- **Vision extraction can be conservative.** When a chart value isn't fully readable, the model marks it "approximately" — which is correct behavior, but means some answers carry hedged numbers.
- **Corpus knowledge is hardcoded.** `src/ingestion/metadata.py` maps filename keywords to document metadata; entity filtering and the "latest-period" classifier read from this same registry. Production path is LLM-based metadata extraction at ingestion (RAGFlow / Haystack pattern), with entity vocabulary derived from `SELECT DISTINCT company FROM documents`.

## What I Would Improve With More Time

- **Run Contextual Retrieval at scale.** The schema and script for Anthropic-style chunk-context prepending are in place (`scripts/contextualize.py`) but the corpus is not yet contextualized — Tier 1 ITPM rate-limited the one-shot run. Production path is the Anthropic Batch API.
- **Fine-tune the reranker on financial-domain pairs.** The cross-encoder's logit distribution skews negative for natural-language questions against passage-style chunks; a domain-tuned reranker would let the abstention threshold tighten without re-introducing false abstentions.
- **Forward-looking-statement classifier at ingestion.** Currently the LLM is prompted to label "Management guided…"; a FinBERT-class classifier at ingestion would tag chunks with `statement_type: reported | guidance | risk_factor` and remove the inference burden from generation.
- **Multi-pass chart faithfulness check.** A second vision call comparing extracted values against the chart image, flagging discrepancies before display.
- **Multi-tenant access control.** Add a `tenant_id` column plus PostgreSQL row-level security policies; pattern is identical whether the storage layer is Docker or Snowflake.

## Run

```bash
# Requires Python 3.10+ (docling, sentence-transformers minimums)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python scripts/migrate.py
mkdir -p data/pdfs && cp /path/to/provided/*.pdf data/pdfs/   # 10 corpus PDFs
python scripts/ingest.py
python scripts/enrich_charts.py
streamlit run app.py
```

`scripts/enrich_charts.py` runs Claude vision over every detected chart picture; expect roughly $2–5 in API cost for the full corpus on first run, idempotent on re-runs.

Required environment variables:

- `ANTHROPIC_API_KEY`
- `DATABASE_URL` (default local value is documented in `.env.example`)

## Validate

```bash
python -m pytest tests/ -v
python -m mypy src/ --ignore-missing-imports
python scripts/evaluate.py --out evaluation_report.md
```
