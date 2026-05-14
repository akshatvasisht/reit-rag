# Corporate RAG Assessment

Agent-facing rules for this repository. Keep this file short and load-bearing.

## Read First

- `README.md` for setup, run, and validation commands.
- `.env.example` for required environment variables.

## Non-Negotiable Invariants

- **Version safety:** keep version boundaries explicit; do not silently merge conflicting versions in answers.
- **Citations:** preserve citation-grounded generation flow; do not switch to uncited answer paths.
- **Failure behavior:** refusal on insufficient evidence is valid behavior; do not replace it with speculative synthesis.
- **Chart handling:** chart-derived extraction remains additive (`content_type="chart_description"`), not a replacement for base ingestion chunks.

## Editing Discipline

- Prefer minimal, targeted changes over broad rewrites.
- Keep `README.md` aligned with runtime behavior.
- When adding/changing evaluation logic, ensure reported labels match actual behavior.
- Add tests for any behavior changes in ingestion, retrieval, or evaluation paths.

## Quick Commands

```bash
python -m pytest tests/ -v
python -m mypy src/ --ignore-missing-imports
python scripts/evaluate.py --out evaluation_report.md
```
