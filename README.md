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

An entity-anchor boost (`src/retrieval/pipeline.py:_entity_anchor_boost`) promotes rerank-pool chunks whose `contextualized_text` contains a proper-noun phrase from the query but were dropped by the top-N cutoff. Closes the gap where the cross-encoder prefers chunks exclusively about an entity over broader chunks that mention the entity among others — anchor case is a project-economics table that lists six projects competing against single-project narrative pages. Among matching candidates, data-bearing content types (`table`, `mixed`, `chart_description`) are preferred over narrative `text`, with RRF order as the within-type tiebreaker. Capped at 3 promoted chunks per query. The anchor extractor requires each token in a proper-noun phrase to be a Title-Cased word with lowercase letters or a digit-bearing token, which excludes all-caps acronyms (FFO, BXP, NOI) that would over-fire on generic metric queries.

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

**Calibration note.** The 0.75 / 0.45 cutoffs are anchored against the 28-query evaluation set. Empirically the answered-path confidence distribution has median 0.54 (min 0.11, max 0.93); the 0.75 cutoff isolates the two highest-trust queries (a confident soft-refusal and a conflict-detection case with both versions retained), and the 0.45 cutoff catches the broad-synthesis tail where the cross-company score gap is naturally narrow. Hard-pass and soft-pass groups did not separate by confidence (medians 0.540 vs 0.527), so the cutoffs are head/tail discriminators on this corpus rather than pass/fail predictors. The 0.50 / 0.30 / 0.20 signal weights remain uncalibrated; rerun this analysis if the corpus or eval set materially changes.

For the all-company synthesis path (one sub-retrieval per corpus company), the cross-company score gap is not meaningful — each company was retrieved independently. Confidence is instead the mean of per-company magnitude signals for companies that returned scored chunks.

Confidence is computed on the post-rerank chunk list, before conflict injection and expansion stages (parent/sibling/table-pair/same-page expansions), so synthetic or unscored expanded chunks do not skew the result.

## Known Limitations

- **Lexical mismatch on chart-derived queries.** Vision-extracted chart descriptions use the model's own phrasing; a query word can fail to match a near-synonym in the description and miss retrieval. Mitigated by a wider RRF candidate pool and by the reranker prepending a structured provenance marker (`Chart from {company} ({doc_type}, p.N): `) to chart_description/chart_context chunks so the cross-encoder has a lexical handle on entity and document type (`src/retrieval/reranker.py:_passage_for_rerank`). Not eliminated — synonym gaps inside the vision prose itself still depress recall.
- **No conversation memory.** The Streamlit UI is intentionally single-turn; follow-up queries ("what about for VICI?") do not carry context from the previous answer.
- **Citation faithfulness is partially claim-level.** Every numeric claim is verified against the cited chunk at runtime — `src/generation/citation_check.py:check_numeric_consistency` splits composite values, normalizes scale words and units, handles parenthetical accounting negatives, and flags mismatches before the answer is returned. Non-numeric (qualitative) claim-level verification — does the chunk's text actually support the specific qualitative assertion attached — remains source-level and would require a RAGAS-style offline pass.
- **Multi-tenant access control is conceptual.** The metadata schema is RLS-ready but row-level security is not configured; documents are treated as a single tenant.
- **Vision extraction can be conservative.** When a chart value isn't fully readable, the model marks it "approximately" — which is correct behavior, but means some answers carry hedged numbers.
- **Corpus knowledge is seed-plus-DB.** `src/corpus_registry.py` ships a hardcoded seed list as bootstrap; on first access `CorpusRegistry.refresh()` reloads entries from the `documents` table so the live registry reflects whatever has actually been ingested, with the seed retained as a fallback on DB unavailability. Fully LLM-extracted metadata at ingestion (RAGFlow / Haystack pattern) is still the next step; the registry-driven entity vocabulary and "latest-period" classifier read from the unified registry.
- **Cross-encoder reranker context truncation.** The local cross-encoder (ms-marco-MiniLM-L-6-v2) truncates passages at ~512 tokens. Mid-table entity mentions on long table chunks (e.g., a $3.6B project-economics table) are invisible to the reranker, so specific-entity queries against deep table content can miss top-5 even when the chunk is in the candidate pool. The post-rerank entity-anchor boost (`src/retrieval/pipeline.py:_entity_anchor_boost`) compensates by promoting broader chunks whose contextualized text contains the query's proper-noun phrase, but the substring match is case- and form-sensitive (canonical vs possessive forms aren't unified), so the boost narrows the gap without closing it. Fix path is either targeted re-contextualization with entity-enumerating prompts, a normalization-aware anchor matcher, or a swap to a longer-context reranker.

## What I Would Improve With More Time

- **Swap to a longer-context reranker.** Move from ms-marco-MiniLM-L-6-v2 (512-token window) to BGE Reranker v2 (8K window) or Cohere Rerank v4 (32K window). The gap on long table chunks is context truncation, not domain mismatch — mid-table entity names fall outside the visible window for cross-encoder scoring. A longer-context model removes the ceiling without retraining.
- **Ingestion-time forward-looking-statement tagger.** The runtime prompt-level guard at generation already prevents historical actuals from being presented as guidance; complementing it with a FinBERT-class tagger that classifies chunks as `reported | guidance | risk_factor` would move the signal from query-time inference into the index itself.
- **Surface `chunk_claims` as structured LLM context.** The pre-extracted `(company, metric, value, qualifier)` tuples in the database are consumed only by retrieval-side conflict detection and the post-generation numeric checker. Surfacing them as a separate prompt section alongside the chunk excerpts would give the LLM canonical numbers for the queried entity instead of leaving it to re-read the chunk text under truncation pressure. Requires a query-conditioned filter (only `(company, metric)` rows matching the query) plus a prompt-section schema so structured facts and chunk text don't double-count.
- **Multi-pass chart faithfulness check.** A second vision call comparing extracted values against the chart image, flagging discrepancies before display.
- **Multi-tenant access control.** Add a `tenant_id` column plus PostgreSQL row-level security policies; the pattern transfers cleanly to a managed vector store should the storage layer change later.
- **Section-aware chunker.** Replace page-boundary chunking with topic/section-aware splitting that coalesces sub-threshold fragments, restores footnote markers at parse-time, and emits `mixed` chunks where a text block and an adjacent table describe one logical figure. Eliminates several retrieval-time noise filters currently compensating for fragment-grade chunks.
- **Ingestion-time structured-metadata pass.** Extract `data_as_of_date`, financial-value lists, and an entity index per chunk at ingestion rather than re-running regex per query. Lets retrieval gain precision via metadata filters — historical reconciliation tables would stop surfacing alongside current-period ones.
- **Stronger ingestion classifiers.** Rework the page-content classifier to push the residual `unknown` rate below 5% (currently ~28%) so the BM25 filter can be tightened to an explicit `substantive`-only allow-list; same principle for chart enrichment, where company / doc-type provenance should be baked into the stored `chunk_text` rather than injected by the reranker at score time.
- **Chunk-id-level citation faithfulness.** The post-generation check matches at `(company, doc_type, date, page)`, so every chunk on a cited page registers as cited. Adding a stable chunk-coordinate token to the citation format and verifying against it gives per-claim faithfulness reporting and accurate source-panel attribution.

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

# One-time back-fills (needed for full feature set; idempotent on re-run):
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
