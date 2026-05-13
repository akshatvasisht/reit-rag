"""Evaluation harness.

Runs the manual golden set (`tests/golden_set.py`) end-to-end through the
production answer pipeline, scores each query against its pass criteria, and
emits per-query plus aggregate results. Optional flags add a retrieval-only
ablation across recent improvements. The `--ragas` flag is currently a stub
that records non-executed status.

Usage:
    python scripts/evaluate.py                          # golden only (~$0.50, ~3 min)
    python scripts/evaluate.py --ablation               # + retrieval ablation (~$1, ~5 min)
    python scripts/evaluate.py --ragas                  # include RAGAS status stub in report
    python scripts/evaluate.py --ablation --ragas       # full report with ablation + RAGAS status
    python scripts/evaluate.py --out report             # write markdown report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.generator import Answer, answer
from src.retrieval.pipeline import RetrievalResult, retrieve
from tests.golden_set import GOLDEN_SET, GoldenQuery


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GoldenResult:
    """One graded golden-set query."""
    id: str
    query: str
    category: str
    # Auto checks
    intent_match: bool | None
    company_filter_match: bool | None
    abstain_match: bool | None
    chart_in_context: bool | None
    both_dlr_versions: bool | None
    citation_supported_ratio: float | None
    # Raw signals for human review
    top_rerank: float
    abstained: bool
    intent_detected: str
    companies_filter_applied: list[str]
    context_companies: list[str]
    context_content_types: list[str]
    answer_text: str
    # Aggregate
    auto_pass: bool  # True if all auto checks pass
    auto_pass_reasons: list[str] = field(default_factory=list)
    auto_fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Golden runner
# ---------------------------------------------------------------------------


def _grade(gq: GoldenQuery, a: Answer) -> GoldenResult:
    """Score one query's answer against its golden expectations."""
    diag = a.diagnostics
    context_companies = sorted({rc.chunk.company for rc in a.contexts})
    context_content_types: list[str] = sorted({str(rc.chunk.content_type) for rc in a.contexts})
    companies_filter = diag.get("companies_filter") or []
    top_rerank = float(diag.get("top_rerank_score", float("-inf")))

    # Auto-check each pass dimension where the GoldenQuery declares it.
    intent_match = None
    if gq.expected_intent is not None:
        intent_match = a.intent == gq.expected_intent

    company_filter_match = None
    if gq.expected_company is not None:
        # Either the entity extractor pre-filtered to the expected company,
        # OR the retrieved contexts came primarily from it.
        company_filter_match = (
            gq.expected_company in companies_filter
            or gq.expected_company in context_companies
        )

    abstain_match = None
    if gq.expect_hard_abstain:
        # Accept EITHER a pre-LLM hard-abstain OR a clean LLM-driven refusal.
        # After the rerank-threshold calibration, borderline out-of-corpus
        # queries fall through the gate; the LLM then refuses via the system
        # prompt instruction about insufficient context evidence. The LLM
        # phrases this refusal naturally (we do not enforce exact wording),
        # so we accept any of several semantically-equivalent patterns.
        refusal_phrases = (
            "couldn't find reliable information",
            "could not find reliable information",
            "do not contain sufficient information",
            "does not contain sufficient information",
            "do not contain information",
            "do not include",
            "do not have information",
            "do not provide",
            "no information",
            "insufficient evidence",
            "insufficient information",
            "not contain enough information",
        )
        text_lower = a.text.lower()
        llm_refused = any(p in text_lower for p in refusal_phrases)
        abstain_match = bool(a.abstained) or llm_refused

    chart_in_context = None
    if gq.expect_chart_chunk_in_context:
        chart_in_context = "chart_description" in context_content_types

    both_dlr_versions = None
    if gq.expect_both_dlr_versions:
        dlr_versions = {
            rc.chunk.doc_version
            for rc in a.contexts
            if rc.chunk.company == "Digital Realty"
        }
        both_dlr_versions = {"2025-12", "2026-03"}.issubset(dlr_versions)

    cit_ratio = None
    if a.citation_report is not None and a.citation_report.total > 0:
        cit_ratio = a.citation_report.faithfulness_ratio

    # Aggregate pass/fail.
    reasons_pass: list[str] = []
    reasons_fail: list[str] = []
    auto_checks = [
        ("intent", intent_match),
        ("company_filter", company_filter_match),
        ("abstain", abstain_match),
        ("chart_in_context", chart_in_context),
        ("both_dlr_versions", both_dlr_versions),
    ]
    for name, val in auto_checks:
        if val is True:
            reasons_pass.append(name)
        elif val is False:
            reasons_fail.append(name)
    # Citation faithfulness: any unsupported citation is a fail.
    if cit_ratio is not None:
        if cit_ratio == 1.0:
            reasons_pass.append("citations_supported")
        elif cit_ratio < 1.0:
            reasons_fail.append(f"unsupported_citations({cit_ratio:.0%})")

    return GoldenResult(
        id=gq.id,
        query=gq.query,
        category=gq.category,
        intent_match=intent_match,
        company_filter_match=company_filter_match,
        abstain_match=abstain_match,
        chart_in_context=chart_in_context,
        both_dlr_versions=both_dlr_versions,
        citation_supported_ratio=cit_ratio,
        top_rerank=top_rerank,
        abstained=a.abstained,
        intent_detected=a.intent or "?",
        companies_filter_applied=companies_filter,
        context_companies=context_companies,
        context_content_types=context_content_types,
        answer_text=a.text,
        auto_pass=len(reasons_fail) == 0,
        auto_pass_reasons=reasons_pass,
        auto_fail_reasons=reasons_fail,
    )


def run_golden() -> list[GoldenResult]:
    """Run every golden query end-to-end and grade it."""
    results: list[GoldenResult] = []
    for gq in GOLDEN_SET:
        logger.warning("Running %s: %s", gq.id, gq.query)
        t0 = time.monotonic()
        a = answer(gq.query)
        dt = time.monotonic() - t0
        logger.warning("  done in %.1fs — abstain=%s top_rerank=%.2f",
                       dt, a.abstained, a.diagnostics.get("top_rerank_score", float("-inf")))
        results.append(_grade(gq, a))
    return results


# ---------------------------------------------------------------------------
# Retrieval-only ablation
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """One ablation config × one golden query."""
    config_name: str
    query_id: str
    top_n_companies: list[str]
    top_n_content_types: list[str]
    top_rerank: float
    expected_company_in_top5: bool | None


def run_ablation() -> dict[str, list[AblationResult]]:
    """Compare retrieval candidate quality across our recent improvements.

    Three configs:
    - baseline: current pipeline (BM25 OR + entity filter + chart chunks)
    - no_entity_filter: monkey-patch extract_companies to return []
    - no_chart_chunks: filter content_type != 'chart_description' in
      the pipeline result (excluding in SQL would require duplicating search
      functions, so this trim is applied post-retrieval).
    """
    from src.retrieval import pipeline as retrieval_pipeline
    from src.retrieval.pipeline import retrieve as _retrieve

    out: dict[str, list[AblationResult]] = {"baseline": [], "no_entity_filter": [], "no_chart_chunks": []}

    # --- baseline ---
    logger.warning("Ablation: baseline")
    for gq in GOLDEN_SET:
        r = _retrieve(gq.query)
        out["baseline"].append(_ablation_grade("baseline", gq, r))

    # --- no_entity_filter ---
    logger.warning("Ablation: no_entity_filter")
    original_extract = retrieval_pipeline.extract_companies
    retrieval_pipeline.extract_companies = lambda q: []  # type: ignore[assignment]
    try:
        for gq in GOLDEN_SET:
            r = _retrieve(gq.query)
            out["no_entity_filter"].append(_ablation_grade("no_entity_filter", gq, r))
    finally:
        retrieval_pipeline.extract_companies = original_extract  # type: ignore[assignment]

    # --- no_chart_chunks ---
    logger.warning("Ablation: no_chart_chunks")
    for gq in GOLDEN_SET:
        r = _retrieve(gq.query)
        # Trim chart_descriptions from contexts.
        r.contexts = [rc for rc in r.contexts if rc.chunk.content_type != "chart_description"]
        out["no_chart_chunks"].append(_ablation_grade("no_chart_chunks", gq, r))

    return out


def _ablation_grade(config: str, gq: GoldenQuery, r: RetrievalResult) -> AblationResult:
    top_n = r.contexts[:5]
    top_n_companies = sorted({rc.chunk.company for rc in top_n})
    top_n_content_types: list[str] = sorted({str(rc.chunk.content_type) for rc in top_n})
    top_rerank = float(r.diagnostics.get("top_rerank_score", float("-inf")))
    expected = None
    if gq.expected_company is not None:
        expected = gq.expected_company in top_n_companies
    return AblationResult(
        config_name=config,
        query_id=gq.id,
        top_n_companies=top_n_companies,
        top_n_content_types=top_n_content_types,
        top_rerank=top_rerank,
        expected_company_in_top5=expected,
    )


# ---------------------------------------------------------------------------
# RAGAS (optional, disabled by default)
# ---------------------------------------------------------------------------


def run_ragas() -> Optional[dict]:
    """Stub for the `--ragas` flag.

    Returns a `not_run` status when invoked. The full RAGAS TestsetGenerator
    integration is intentionally not wired up here — manual golden set plus
    retrieval ablation are the in-scope evaluation tiers. Kept as a hook for
    future RAGAS work; the import probe also surfaces an actionable error
    message if `ragas` is uninstalled.
    """
    try:
        from ragas import evaluate as ragas_evaluate  # noqa: F401
        from ragas.testset import TestsetGenerator  # noqa: F401
    except ImportError as e:
        logger.error("ragas not importable: %s — skipping", e)
        return None
    logger.warning("RAGAS path is not executed in this harness.")
    return {"status": "not_run", "note": "RAGAS integration is not executed in this harness."}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def render_markdown(
    golden: list[GoldenResult],
    ablation: Optional[dict[str, list[AblationResult]]],
    ragas: Optional[dict],
) -> str:
    """Write a markdown evaluation report."""
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Generated by `scripts/evaluate.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}.")
    lines.append("")

    # --- Tier 1: golden set ---
    pass_n = sum(1 for r in golden if r.auto_pass)
    total = len(golden)
    lines.append(f"## Tier 1 — Manual golden set ({pass_n}/{total} pass)")
    lines.append("")
    lines.append("Auto-checks: intent match, entity-filter match, abstention/refusal behavior where expected, chart_description in context where expected, both DLR versions where expected, 100% citation support.")
    lines.append("")
    lines.append("| ID | Category | Query | Auto-pass | Pass signals | Fail signals | Top rerank |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in golden:
        mark = "✅" if r.auto_pass else "❌"
        passes = ",".join(r.auto_pass_reasons) or "—"
        fails = ",".join(r.auto_fail_reasons) or "—"
        q = r.query[:60] + ("…" if len(r.query) > 60 else "")
        lines.append(f"| {r.id} | {r.category} | {q} | {mark} | {passes} | {fails} | {r.top_rerank:.2f} |")
    lines.append("")

    # Category breakdown
    lines.append("### By category")
    lines.append("")
    by_cat: dict[str, tuple[int, int]] = {}
    for r in golden:
        n_pass, n_tot = by_cat.get(r.category, (0, 0))
        by_cat[r.category] = (n_pass + (1 if r.auto_pass else 0), n_tot + 1)
    lines.append("| Category | Pass / Total |")
    lines.append("|---|---|")
    for cat, (p, t) in sorted(by_cat.items()):
        lines.append(f"| {cat} | {p}/{t} |")
    lines.append("")

    # --- Tier 2: RAGAS (or its absence) ---
    if ragas:
        lines.append("## Tier 2 — RAGAS synthetic")
        lines.append("")
        lines.append(f"Status: {ragas.get('status', 'unknown')}")
        if ragas.get("note"):
            lines.append("")
            lines.append(f"_{ragas['note']}_")
        lines.append("")

    # --- Tier 3: ablation ---
    if ablation:
        lines.append("## Tier 3 — Retrieval ablation")
        lines.append("")
        lines.append("Per-config: how often does the expected company appear in the top-5 retrieved (parent-expanded) contexts?")
        lines.append("")
        lines.append("| Config | Recall@5 (expected company) | Top rerank (mean) |")
        lines.append("|---|---|---|")
        for cfg, rows in ablation.items():
            scored = [r for r in rows if r.expected_company_in_top5 is not None]
            recall = sum(1 for r in scored if r.expected_company_in_top5) / max(1, len(scored))
            mean_rr = sum(r.top_rerank for r in rows) / max(1, len(rows))
            lines.append(f"| {cfg} | {recall:.0%} ({sum(1 for r in scored if r.expected_company_in_top5)}/{len(scored)}) | {mean_rr:.2f} |")
        lines.append("")

    # Per-query detail
    lines.append("## Appendix — per-query detail")
    lines.append("")
    for r in golden:
        lines.append(f"### [{r.id}] {r.query}")
        lines.append("")
        lines.append(f"- Category: `{r.category}`")
        lines.append(f"- Intent: detected `{r.intent_detected}` (auto-check: {_fmt_check(r.intent_match)})")
        lines.append(f"- Entity filter: applied `{r.companies_filter_applied}` (auto-check: {_fmt_check(r.company_filter_match)})")
        lines.append(f"- Abstained: {r.abstained} (auto-check: {_fmt_check(r.abstain_match)})")
        lines.append(f"- Top rerank: `{r.top_rerank:.2f}`")
        lines.append(f"- Context companies: {r.context_companies}")
        lines.append(f"- Context content types: {r.context_content_types}")
        if r.chart_in_context is not None:
            lines.append(f"- Chart chunk in context: {r.chart_in_context}")
        if r.both_dlr_versions is not None:
            lines.append(f"- Both DLR versions retrieved: {r.both_dlr_versions}")
        if r.citation_supported_ratio is not None:
            lines.append(f"- Citation faithfulness: {r.citation_supported_ratio:.0%}")
        lines.append("")
        lines.append("```")
        lines.append(r.answer_text[:800] + ("…" if len(r.answer_text) > 800 else ""))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _fmt_check(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "✅" if v else "❌"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true",
                        help="Run retrieval-only ablation across recent improvements.")
    parser.add_argument("--ragas", action="store_true",
                        help="Run RAGAS synthetic evaluation.")
    parser.add_argument("--out", type=str, default=None,
                        help="Write markdown report to this path (default: stdout summary only).")
    args = parser.parse_args()

    logger.warning("== Tier 1: golden set ==")
    golden = run_golden()
    pass_n = sum(1 for r in golden if r.auto_pass)
    logger.warning("Golden: %d / %d passed auto-checks", pass_n, len(golden))

    ablation = None
    if args.ablation:
        logger.warning("== Tier 3: retrieval ablation ==")
        ablation = run_ablation()

    ragas = None
    if args.ragas:
        logger.warning("== Tier 2: RAGAS synthetic ==")
        ragas = run_ragas()

    if args.out:
        report = render_markdown(golden, ablation, ragas)
        Path(args.out).write_text(report)
        logger.warning("Wrote report to %s (%d bytes)", args.out, len(report))
    else:
        # Concise stdout summary
        print()
        print(f"GOLDEN: {pass_n}/{len(golden)} auto-pass")
        print()
        for r in golden:
            mark = "✅" if r.auto_pass else "❌"
            print(f"  {mark} [{r.id}] {r.category:24s} {r.query[:60]}")
            if not r.auto_pass:
                print(f"         fails: {','.join(r.auto_fail_reasons)}")


if __name__ == "__main__":
    main()
