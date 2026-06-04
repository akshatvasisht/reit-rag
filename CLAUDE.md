# Corporate RAG Assessment

Agent-facing rules for this repository. Keep this file short and essential.

## Read First

- `README.md` — `## Run` (setup), `## Validate` (test / typecheck / lint), architecture, known limitations.
- `.env.example` — required environment variables and optional tuning.
- `Makefile` — one-command shortcuts (`make all`, `make app`, `make test|typecheck|lint`); run `make help` to list every target.

## Non-Negotiable Invariants

- **Version safety:** keep version boundaries explicit; do not silently merge conflicting versions in answers.
- **Citations:** preserve citation-grounded generation flow; do not switch to uncited answer paths.
- **Failure behavior:** refusal on insufficient evidence is valid behavior; do not replace it with speculative synthesis.
- **Chart handling:** chart-derived extraction remains additive (`content_type="chart_description"`), not a replacement for base ingestion chunks.
- **Contextualization path:** `scripts/contextualize.py` is Batch API-based; keep docs and implementation wording consistent with that path.
- **Generation checks:** runtime post-generation checks are citation + numeric consistency; corpus-membership protection is prompt-level framing.

## Editing Discipline

- Prefer minimal, targeted changes over broad rewrites.
- Keep `README.md` aligned with runtime behavior.
- When adding/changing evaluation logic, ensure reported labels match actual behavior.
- Add tests for any behavior changes in ingestion, retrieval, or evaluation paths.
