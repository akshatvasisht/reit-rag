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

PostgreSQL with the pgvector extension, run locally via Docker. Each chunk row carries dense (`halfvec(1024)`) and lexical (`tsvector`) representations in the same table; HNSW index on the embedding column, GIN index on the tsvector. Storing both signals in one database removes the need for a separate vector store and keeps version-aware filtering as a plain SQL `WHERE`. The metadata schema and retrieval logic are storage-layer-agnostic, so a managed vector store can replace pgvector without changes upstream.

## Chunking Strategy

Hierarchical small-to-large at structural boundaries from Docling output. Child chunks (~256 tokens, one per paragraph / table row / chart caption) carry the retrieval signal; parent chunks (~512–1024 tokens, one per slide or section) provide generation context. Children are linked to their parent via a self-FK; at retrieval time, matched children are expanded to their parents before the LLM sees them. Fixed-size chunking and slide-level-only chunking were rejected — the former routinely splits table headers from data rows; the latter dilutes the retrieval vector across mixed content.

## Retrieval Approach

Hybrid: BM25 (PostgreSQL tsvector) and dense (Qwen3-Embedding-0.6B + pgvector HNSW) fire in parallel, fuse via Reciprocal Rank Fusion (k=60), then a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the top candidates. The cross-encoder scores `(query, contextualized_text or chunk_text)` pairs, so the vocab-bridging signal from contextualization flows through every retrieval stage. Hybrid is non-negotiable for financial documents — exact-term matches (tickers, FFO, NOI, Q4 2025) need lexical retrieval, semantic queries ("how leveraged is DLR") need dense. An entity pre-filter narrows the candidate pool when the query names a known company; abstention fires when the top rerank score falls below a calibrated threshold or the LLM's own system prompt cannot ground an answer in the retrieved context.

Contextualization runs via the Anthropic Batch API (`scripts/contextualize.py`) and populates a `contextualized_text` column per chunk — a one-line situating prefix derived from neighboring chunks. BM25, dense retrieval, and the reranker all consume this signal: BM25 unions the `chunk_text` and `contextualized_text` tsvectors at query time, dense retrieval re-embeds against the contextualized form, and the cross-encoder scores against it. The Batch API is the live ingestion path, not a deferred plan.

A page-content classifier (`src/ingestion/page_classifier.py`, Claude Haiku with a keyword pre-filter and LLM fallback) tags each chunk with `page_content_class` ∈ {`substantive`, `boilerplate_legal`, `index_reference`, `cover_page`}. BM25 retrieval excludes the three non-substantive classes via a SQL filter in `src/retrieval/bm25.py`; dense retrieval is left unfiltered (embedding distance handles such pages without the precision cost at recall stage). This prevents, e.g., a Digital Realty trademark/disclaimer page from outranking the substantive p.4–7 strategy content on a broad strategy query.

## Versioning

Documents are grouped into version groups by `(company, doc_type, doc_subtype)`, ordered by `report_date`. An intent classifier reads the query for `latest` (default), `historical`, `comparison`, `conflict`, or `all_company_synthesis`. For `latest` intent, version-group deduplication suppresses older versions before the LLM sees context, so the model never silently averages between (for example) DLR's December 2025 and March 2026 figures. For `comparison` intent ("how did DLR change between Dec and Mar"), both versions are retained and the answer surfaces each with its date in the citation.

## Conflicting Information

When the retrieved context contains figures that disagree across sources, the system surfaces each value with its own inline citation `[Company, Doc Type, Date, p.N]` rather than merging or averaging. When two chunks share `(company, doc_type, report_date)` — e.g. the BXP Investor Day Session vs Quarterly Investor Deck, both December 2025 — the citation is automatically extended to `[Company, Doc Type (Subtype), Month Year, p.N]` to keep references unambiguous. The system prompt instructs the LLM to label management guidance as "Management guided…" rather than as reported fact. When a query carries forward-looking vocabulary (guidance / target / outlook / projection / forecast) but no retrieved chunk contains forward-looking language, the system prompt is augmented (in `src/generation/generator.py`) to force a soft refusal rather than letting the model present historical actuals as guidance. A post-generation citation faithfulness check parses every citation out of the answer and verifies it maps to a chunk that was actually in the retrieved context; unsupported citations surface as a UI warning rather than being silently allowed through. A complementary numeric-consistency check (`src/generation/citation_check.py:check_numeric_consistency`) verifies every numeric claim against the cited chunk text — composite values are split atomically, parenthetical negatives are handled, and scale words and unit categories are normalized before comparison.

## Charts and Tables

Tables are extracted cell-by-cell by Docling's TableFormer and indexed as `content_type='table'` chunks. Charts are handled by a second-pass enrichment script: each Docling-detected `PictureItem` is sent to Claude vision with a structured extraction prompt covering chart type, title, axes, data series, key values, and a one-line insight. Decorative images are filtered out via a `NOT_CHART` sentinel response. Chart-derived chunks are tagged `content_type='chart_description'` so the UI can surface that the value came from vision extraction rather than text. When the vision model hedges or refuses, the system falls back to abstention with the page cited.

## Retrieval Confidence

After reranking, the pipeline computes a composite evidence-quality score in [0, 1] that quantifies how trustworthy the retrieved context is. This score is distinct from the binary abstain gate: abstain fires when the top rerank score falls below `RERANK_THRESHOLD`; confidence operates only on the answerable band above that threshold.

### Three-signal composite

| Signal | Weight | Description |
|---|---|---|
| Score gap | 0.50 | Normalised difference between the top-1 and top-2 reranker scores. A large gap indicates the best chunk is clearly better than the runner-up. |
| Score magnitude | 0.30 | Sigmoid-mapped top-1 reranker logit, centred at −2.0. Maps the −5→+2 logit range to roughly 0.18→0.88. |
| Entity coverage | 0.20 | Fraction of returned chunks belonging to an expected company. 1.0 when no entity filter is active (corpus-wide query). |

The weighted sum is clamped to [0, 1] and mapped to three display bands:

| Band | Score range | Display |
|---|---|---|
| High | ≥ 0.75 | Green — "High confidence" |
| Medium | 0.45 – 0.75 | Amber — "Medium confidence" with a verification caveat |
| Low | < 0.45 | Red — "Low confidence" with a manual-review caveat |

**Calibration note.** The 0.75 / 0.45 cutoffs and the 0.50 / 0.30 / 0.20 weights are uncalibrated initial values. To calibrate: run `scripts/evaluate.py --out report.md`, collect the per-query confidence values from the report, then redraw the cutoffs at the medians of correct / borderline / refused query groups. Refit weights by minimising band-label error on the evaluation set.

For the all-company synthesis path (one sub-retrieval per corpus company), the cross-company score gap is not meaningful — each company was retrieved independently. Confidence is instead the mean of per-company magnitude signals for companies that returned scored chunks.

Confidence is computed on the post-rerank chunk list, before conflict injection and expansion stages (parent/sibling/table-pair/same-page expansions), so synthetic or unscored expanded chunks do not skew the result.

## Known Limitations

- **Lexical mismatch on chart-derived queries.** Vision-extracted chart descriptions use the model's own phrasing; a query word can fail to match a near-synonym in the description and miss retrieval. Wider RRF candidate pool mitigates but does not eliminate this.
- **No conversation memory.** The Streamlit UI is intentionally single-turn; follow-up queries ("what about for VICI?") do not carry context from the previous answer.
- **Citation faithfulness is source-level, not claim-level.** The check verifies that every cited chunk was in the retrieved context, not that the chunk's content actually supports the specific claim attached. Claim-level verification is RAGAS territory and runs offline.
- **Multi-tenant access control is conceptual.** The metadata schema is RLS-ready but row-level security is not configured; documents are treated as a single tenant.
- **Vision extraction can be conservative.** When a chart value isn't fully readable, the model marks it "approximately" — which is correct behavior, but means some answers carry hedged numbers.
- **Corpus knowledge is hardcoded.** `src/corpus_registry.py` is the canonical registry mapping filename keywords to document metadata (re-exported from `src/ingestion/metadata.py` for backward compatibility); entity filtering and the "latest-period" classifier read from this same registry. Production path is LLM-based metadata extraction at ingestion (RAGFlow / Haystack pattern), with entity vocabulary derived from `SELECT DISTINCT company FROM documents`.

## What I Would Improve With More Time

- **Fine-tune the reranker on financial-domain pairs.** The cross-encoder's logit distribution skews negative for natural-language questions against passage-style chunks; a domain-tuned reranker would let the abstention threshold tighten without re-introducing false abstentions.
- **Ingestion-time forward-looking-statement tagger.** The runtime prompt-level guard at generation already prevents historical actuals from being presented as guidance; complementing it with a FinBERT-class tagger that classifies chunks as `reported | guidance | risk_factor` would move the signal from query-time inference into the index itself.
- **Multi-pass chart faithfulness check.** A second vision call comparing extracted values against the chart image, flagging discrepancies before display.
- **Multi-tenant access control.** Add a `tenant_id` column plus PostgreSQL row-level security policies; the pattern transfers cleanly to a managed vector store should the storage layer change later.

## Highlighted Changes

### 1. Intent classifier on conflict-language queries

**Found.** The deterministic regex did not match conflict, discrepancy, or disagreement keywords; the example query ("Are there conflicting data points across documents on Digital Realty leverage?") routed to `latest` and version-group dedup hid the December DLR deck.

**Changed.** `_COMPARISON_RE` now matches `conflict|discrepancy|disagree|inconsistent|contradict|mismatch`, and the LLM-side classifier carries an explicit `conflict` intent. The existing `find_conflicting_chunks` stage queries the `chunk_claims` table for same-`(company, metric)` pairs with differing values and injects mismatches into the LLM context.

**Decided not to change.** A separate retrieval path that forces all versions on `conflict` intent. `comparison` already retains all versions and `find_conflicting_chunks` already surfaces cross-version disagreements; a parallel path would duplicate behavior.

### 2. Metadata registry collisions on same-date documents

**Found.** Both BXP files resolved to `report_date: 2025-12` and both PSA files to `2026-03`; version-group dedup could not distinguish them and downstream behavior was order-dependent.

**Changed.** A `doc_subtype` column now sits in the version-group key (see Versioning) and is surfaced in the citation when retrieval collisions occur (see Conflicting Information).

**Decided not to change.** Making `doc_type` itself more granular as suggested. That would conflate document type with subtype and break legitimate version chains within the same subtype — a Q4 2025 quarterly deck → Q1 2026 quarterly deck is a version chain worth honoring. The separate column preserves those chains.

### 3. Retrieval ranks boilerplate above substantive on broad strategy queries

**Found.** Broad strategy queries surfaced trademark / disclaimer / forward-looking-statement pages because BM25 token density on those pages was dominated by the company name.

**Changed.** The `page_content_class` system (see Retrieval Approach) excludes the three non-substantive classes from BM25 while leaving dense retrieval unfiltered.

**Decided not to change.** Skipping boilerplate chunks at ingestion entirely. Retention preserves the option to answer queries about disclosure wording or document structure; retrieval-stage filtering is the right place to enforce relevance.

## Overall Changes

Three review issues addressed: conflict-language queries route to `comparison`/`conflict` and retain all versions (DLR December deck no longer suppressed); same-date deck collisions (BXP Investor Day vs Quarterly, PSA Company Update vs Merger) disambiguated via `doc_subtype` and surfaced in citations; ingestion classifies each chunk as `substantive`/`boilerplate_legal`/`index_reference`/`cover_page`, with BM25 excluding the three non-substantive. Beyond review: contextualization threaded through the cross-encoder reranker, forward-looking integrity guard at generation, numeric-consistency check hardened, corpus-membership framing added.

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

# Optional back-fills (needed for full feature set; see agentcontext/DEFERRED.md
# for cost and ordering):
python scripts/reclassify_subtypes.py        # populate doc_subtype
python scripts/reclassify_page_content.py    # populate page_content_class
python scripts/contextualize.py              # Batches API: contextualize + Qwen3 re-embed; activation gate
python scripts/extract_claims.py             # populate chunk_claims (M4 verification pass)

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
