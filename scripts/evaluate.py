"""Evaluation harness.

Runs the manual evaluation set (`tests/evaluation_set.py`) end-to-end through the
production answer pipeline, scores each query against its pass criteria, and
emits per-query plus aggregate results. Optional flags add a retrieval-only
ablation across recent improvements. The `--ragas` flag is currently a stub
that records non-executed status.

Usage:
    python scripts/evaluate.py                          # evaluation only (~$0.50, ~3 min)
    python scripts/evaluate.py --ablation               # + retrieval ablation (~$1, ~5 min)
    python scripts/evaluate.py --ragas                  # include RAGAS status stub in report
    python scripts/evaluate.py --ablation --ragas       # full report with ablation + RAGAS status
    python scripts/evaluate.py --out report             # write markdown report
    python scripts/evaluate.py --ablation-reranker --ablation-model cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.generator import Answer, answer
from src.corpus_registry import CORPUS_REGISTRY
from src.retrieval.pipeline import RetrievalResult, confidence_band, retrieve
from tests.evaluation_set import EVALUATION_SET, EvaluationQuery


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
class EvaluationResult:
    """One graded evaluation-set query."""
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
    min_companies_check: bool | None
    forward_looking_check: bool | None
    soft_refusal_check: bool | None
    # Raw signals for human review
    top_rerank: float
    retrieval_confidence: float
    retrieval_confidence_band: str
    abstained: bool
    intent_detected: str
    companies_filter_applied: list[str]
    context_companies: list[str]
    context_content_types: list[str]
    answer_text: str
    # Aggregate
    auto_pass: bool  # True if all hard auto checks pass (soft fails do not gate)
    auto_pass_reasons: list[str] = field(default_factory=list)
    auto_fail_reasons: list[str] = field(default_factory=list)
    # Soft fails: signals worth surfacing in the report but not gating auto_pass
    # (e.g. intent mismatch on a query flagged expected_intent_strict=False).
    auto_soft_fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def _grade(gq: EvaluationQuery, a: Answer) -> EvaluationResult:
    """Score one query's answer against its evaluation expectations."""
    diag = a.diagnostics
    context_companies = sorted({rc.chunk.company for rc in a.contexts})
    context_content_types: list[str] = sorted({str(rc.chunk.content_type) for rc in a.contexts})
    companies_filter = diag.get("companies_filter") or []
    top_rerank = float(diag.get("top_rerank_score", float("-inf")))
    retrieval_conf = float(diag.get("retrieval_confidence", 0.0))
    retrieval_conf_band = confidence_band(retrieval_conf) if not a.abstained else "n/a"

    # Auto-check each pass dimension where the EvaluationQuery declares it.
    # expected_intent may be a single value OR a list of acceptable values.
    intent_match = None
    if gq.expected_intent is not None:
        if isinstance(gq.expected_intent, list):
            intent_match = a.intent in gq.expected_intent
        else:
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

    min_companies_check = None
    if gq.expect_min_companies_in_context is not None:
        distinct_companies = len({rc.chunk.company for rc in a.contexts})
        min_companies_check = distinct_companies >= gq.expect_min_companies_in_context

    forward_looking_check = None
    if gq.expect_forward_looking:
        # Only meaningful when the model produced an answer (not an abstention).
        if a.structured is not None:
            forward_looking_check = bool(a.structured.forward_looking) and not a.abstained
        else:
            forward_looking_check = None

    # Soft-refusal check: when expect_soft_refusal=True the answer must either
    # abstain or be PRIMARILY refusal language. An answer that includes a
    # refusal phrase as a hedge over otherwise substantive content does not
    # count — stripping the refusal phrases must leave essentially no
    # remaining substance.
    soft_refusal_check = None
    if gq.expect_soft_refusal:
        _soft_refusal_phrases = (
            "not disclosed",
            "not provided",
            "do not contain",
            "does not contain",
            "cannot determine",
            "not found",
            "no information",
            "not available",
        )
        if a.abstained:
            soft_refusal_check = True
        else:
            text_lower = a.text.lower()
            has_refusal_phrase = any(p in text_lower for p in _soft_refusal_phrases)
            if not has_refusal_phrase:
                soft_refusal_check = False
            else:
                # Strip refusal phrases and surrounding punctuation; the
                # remaining alphanumeric content must be small enough to
                # represent only scaffolding ("the information is", "in the
                # source documents") rather than substantive claims that
                # happen to embed a refusal phrase as a hedge.
                text_norm = re.sub(r"[^a-z0-9\s]", " ", text_lower)
                for phrase in _soft_refusal_phrases:
                    text_norm = text_norm.replace(phrase, " ")
                remaining_alnum = re.sub(r"\s+", "", text_norm)
                soft_refusal_check = len(remaining_alnum) <= 50

    # Aggregate pass/fail.
    reasons_pass: list[str] = []
    reasons_fail: list[str] = []
    # Soft checks contribute a signal name to pass/fail but do NOT gate
    # auto_pass when failing — used for debatable signals like ambiguous-intent
    # queries flagged with expected_intent_strict=False.
    reasons_soft_fail: list[str] = []
    auto_checks = [
        ("intent", intent_match),
        ("company_filter", company_filter_match),
        ("abstain", abstain_match),
        ("chart_in_context", chart_in_context),
        ("both_dlr_versions", both_dlr_versions),
        ("min_companies_check", min_companies_check),
        ("forward_looking_check", forward_looking_check),
        ("soft_refusal", soft_refusal_check),
    ]
    for name, val in auto_checks:
        if val is True:
            reasons_pass.append(name)
        elif val is False:
            if name == "intent" and not gq.expected_intent_strict:
                reasons_soft_fail.append("intent_soft")
            elif name == "soft_refusal":
                reasons_fail.append("expected_soft_refusal_but_answered")
            else:
                reasons_fail.append(name)
    # Citation faithfulness: any unsupported citation is a fail.
    if cit_ratio is not None:
        if cit_ratio == 1.0:
            reasons_pass.append("citations_supported")
        elif cit_ratio < 1.0:
            reasons_fail.append(f"unsupported_citations({cit_ratio:.0%})")

    return EvaluationResult(
        id=gq.id,
        query=gq.query,
        category=gq.category,
        intent_match=intent_match,
        company_filter_match=company_filter_match,
        abstain_match=abstain_match,
        chart_in_context=chart_in_context,
        both_dlr_versions=both_dlr_versions,
        citation_supported_ratio=cit_ratio,
        min_companies_check=min_companies_check,
        forward_looking_check=forward_looking_check,
        soft_refusal_check=soft_refusal_check,
        top_rerank=top_rerank,
        retrieval_confidence=retrieval_conf,
        retrieval_confidence_band=retrieval_conf_band,
        abstained=a.abstained,
        intent_detected=a.intent or "?",
        companies_filter_applied=companies_filter,
        context_companies=context_companies,
        context_content_types=context_content_types,
        answer_text=a.text,
        auto_pass=len(reasons_fail) == 0,
        auto_pass_reasons=reasons_pass,
        auto_fail_reasons=reasons_fail,
        auto_soft_fail_reasons=reasons_soft_fail,
    )


def run_evaluation() -> list[EvaluationResult]:
    """Run every evaluation query end-to-end and grade it."""
    results: list[EvaluationResult] = []
    for gq in EVALUATION_SET:
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
    """One ablation config × one evaluation query."""
    config_name: str
    query_id: str
    top_n_companies: list[str]
    top_n_content_types: list[str]
    top_rerank: float
    # 1-indexed rank of the expected company in the retrieved candidate list;
    # None when no expected company is defined or it falls outside the top-k window.
    expected_company_rank: int | None


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
    for gq in EVALUATION_SET:
        r = _retrieve(gq.query)
        out["baseline"].append(_ablation_grade("baseline", gq, r))

    # --- no_entity_filter ---
    logger.warning("Ablation: no_entity_filter")
    original_extract = retrieval_pipeline.extract_companies
    retrieval_pipeline.extract_companies = lambda q: []  # type: ignore[assignment]
    try:
        for gq in EVALUATION_SET:
            r = _retrieve(gq.query)
            out["no_entity_filter"].append(_ablation_grade("no_entity_filter", gq, r))
    finally:
        retrieval_pipeline.extract_companies = original_extract  # type: ignore[assignment]

    # --- no_chart_chunks ---
    logger.warning("Ablation: no_chart_chunks")
    for gq in EVALUATION_SET:
        r = _retrieve(gq.query)
        # Trim chart_descriptions from contexts.
        r.contexts = [rc for rc in r.contexts if rc.chunk.content_type != "chart_description"]
        out["no_chart_chunks"].append(_ablation_grade("no_chart_chunks", gq, r))

    return out


def _ablation_grade(config: str, gq: EvaluationQuery, r: RetrievalResult) -> AblationResult:
    top_n = r.contexts[:5]
    top_n_companies = sorted({rc.chunk.company for rc in top_n})
    top_n_content_types: list[str] = sorted({str(rc.chunk.content_type) for rc in top_n})
    top_rerank = float(r.diagnostics.get("top_rerank_score", float("-inf")))
    # Compute the 1-indexed rank of the expected company in the full context list.
    # Using the full list (not just top_n) so callers can choose their own k.
    expected_rank: int | None = None
    if gq.expected_company is not None:
        for rank_idx, rc in enumerate(r.contexts, start=1):
            if rc.chunk.company == gq.expected_company:
                expected_rank = rank_idx
                break
    return AblationResult(
        config_name=config,
        query_id=gq.id,
        top_n_companies=top_n_companies,
        top_n_content_types=top_n_content_types,
        top_rerank=top_rerank,
        expected_company_rank=expected_rank,
    )


# ---------------------------------------------------------------------------
# Reranker model ablation
# ---------------------------------------------------------------------------


def _compute_recall_at_k(results: list[AblationResult], k: int = 5) -> float:
    """Fraction of queries where the expected company appears in the top-k."""
    scored = [r for r in results if r.expected_company_rank is not None]
    if not scored:
        return 0.0
    return sum(1 for r in scored if r.expected_company_rank <= k) / len(scored)


def _compute_mrr_at_k(results: list[AblationResult], k: int = 3) -> float:
    """Mean Reciprocal Rank at k.

    Conventional MRR: for each query, the reciprocal rank is 1/rank when the
    expected company appears at position `rank` within the top-k window, and 0
    otherwise. The mean is taken over ALL queries that have an expected company
    defined (not just hits), matching the standard IR definition.
    """
    scored = [r for r in results if r.expected_company_rank is not None]
    if not scored:
        return 0.0
    return sum(
        (1.0 / r.expected_company_rank)
        for r in scored
        if r.expected_company_rank <= k
    ) / len(scored)


def run_reranker_ablation(
    alt_model: str,
    alt_revision: str | None,
) -> dict[str, list[AblationResult]]:
    """Run the evaluation set twice: once with the default reranker, once with *alt_model*.

    Returns a dict with keys ``'reranker_baseline'`` and
    ``'reranker_alternative'``, each holding a ``list[AblationResult]`` aligned
    to EVALUATION_SET.

    If the alternative model fails to load, every entry in
    ``'reranker_alternative'`` is a sentinel AblationResult whose
    ``config_name`` encodes the load error; the baseline side is always
    populated.
    """
    from src.retrieval import pipeline as retrieval_pipeline
    from src.retrieval.pipeline import retrieve as _retrieve
    from src.retrieval import reranker as _reranker_module

    out: dict[str, list[AblationResult]] = {
        "reranker_baseline": [],
        "reranker_alternative": [],
    }

    # --- baseline (default reranker) ---
    logger.warning("Reranker ablation: baseline")
    for gq in EVALUATION_SET:
        r = _retrieve(gq.query)
        out["reranker_baseline"].append(_ablation_grade("reranker_baseline", gq, r))

    # --- alternative reranker ---
    logger.warning("Reranker ablation: alternative (%s)", alt_model)
    alt_load_error: str | None = None
    try:
        # Pre-load the alternative model once; raises RuntimeError on failure.
        _reranker_module._get_registry_encoder(alt_model, alt_revision)
    except Exception as exc:
        alt_load_error = str(exc)
        logger.error("Alternative reranker failed to load: %s", alt_load_error)

    if alt_load_error is not None:
        # Populate every alternative slot with a sentinel that carries the error.
        sentinel_config = f"reranker_alternative[LOAD_ERROR: {alt_load_error}]"
        for gq in EVALUATION_SET:
            out["reranker_alternative"].append(
                AblationResult(
                    config_name=sentinel_config,
                    query_id=gq.id,
                    top_n_companies=[],
                    top_n_content_types=[],
                    top_rerank=float("-inf"),
                    expected_company_rank=None,
                )
            )
    else:
        # Monkey-patch the rerank function used by the pipeline so that all
        # retrieval calls in this block use the alternative model.
        original_rerank = retrieval_pipeline.rerank  # type: ignore[attr-defined]

        def _alt_rerank(query: str, candidates, top_n: int = 5):  # type: ignore[no-untyped-def]
            return _reranker_module.rerank_with_model(
                query, candidates, alt_model, alt_revision, top_n
            )

        retrieval_pipeline.rerank = _alt_rerank  # type: ignore[attr-defined]
        try:
            for gq in EVALUATION_SET:
                r = _retrieve(gq.query)
                out["reranker_alternative"].append(
                    _ablation_grade("reranker_alternative", gq, r)
                )
        finally:
            retrieval_pipeline.rerank = original_rerank  # type: ignore[attr-defined]

    return out


# ---------------------------------------------------------------------------
# RAGAS (optional, disabled by default)
# ---------------------------------------------------------------------------


def run_ragas() -> Optional[dict]:
    """Stub for the `--ragas` flag.

    Returns a `not_run` status when invoked. The full RAGAS TestsetGenerator
    integration is intentionally not wired up here — manual evaluation set plus
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
    evaluation: list[EvaluationResult],
    ablation: Optional[dict[str, list[AblationResult]]],
    ragas: Optional[dict],
    *,
    reranker_ablation: Optional[dict[str, list[AblationResult]]] = None,
) -> str:
    """Write a markdown evaluation report."""
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Generated by `scripts/evaluate.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}.")
    lines.append("")

    # --- Tier 1: evaluation set ---
    pass_n = sum(1 for r in evaluation if r.auto_pass)
    total = len(evaluation)
    lines.append(f"## Tier 1 — Manual evaluation set ({pass_n}/{total} pass)")
    lines.append("")
    lines.append("Auto-checks: intent match, entity-filter match, abstention/refusal behavior where expected, chart_description in context where expected, both DLR versions where expected, 100% citation support, OOC entity attribution where expected.")
    lines.append("")
    lines.append("| ID | Category | Query | Auto-pass | Pass signals | Fail signals | Top rerank | Confidence | Band |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in evaluation:
        mark = "✅" if r.auto_pass else "❌"
        passes = ",".join(r.auto_pass_reasons) or "—"
        fail_parts = list(r.auto_fail_reasons)
        if r.auto_soft_fail_reasons:
            fail_parts.extend(f"~{s}" for s in r.auto_soft_fail_reasons)
        fails = ",".join(fail_parts) or "—"
        q = r.query[:60] + ("…" if len(r.query) > 60 else "")
        lines.append(
            f"| {r.id} | {r.category} | {q} | {mark} | {passes} | {fails} "
            f"| {r.top_rerank:.2f} | {r.retrieval_confidence:.3f} | {r.retrieval_confidence_band} |"
        )
    lines.append("")

    # Category breakdown
    lines.append("### By category")
    lines.append("")
    by_cat: dict[str, tuple[int, int]] = {}
    for r in evaluation:
        n_pass, n_tot = by_cat.get(r.category, (0, 0))
        by_cat[r.category] = (n_pass + (1 if r.auto_pass else 0), n_tot + 1)
    lines.append("| Category | Pass / Total |")
    lines.append("|---|---|")
    for cat, (p, t) in sorted(by_cat.items()):
        lines.append(f"| {cat} | {p}/{t} |")
    lines.append("")

    # Confidence distribution summary
    answered = [r for r in evaluation if not r.abstained]
    if answered:
        import statistics as _stats
        conf_values = [r.retrieval_confidence for r in answered]
        conf_mean = _stats.mean(conf_values)
        conf_stdev = _stats.stdev(conf_values) if len(conf_values) > 1 else 0.0
        band_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for r in answered:
            band_counts[r.retrieval_confidence_band] = band_counts.get(r.retrieval_confidence_band, 0) + 1
        lines.append("### Retrieval confidence summary (answered queries only)")
        lines.append("")
        lines.append(f"- Mean ± stdev: **{conf_mean:.3f} ± {conf_stdev:.3f}**")
        lines.append(f"- Band distribution: high={band_counts['high']}, medium={band_counts['medium']}, low={band_counts['low']}")
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
            scored = [r for r in rows if r.expected_company_rank is not None]
            recall = sum(1 for r in scored if r.expected_company_rank <= 5) / max(1, len(scored))
            mean_rr = sum(r.top_rerank for r in rows) / max(1, len(rows))
            lines.append(f"| {cfg} | {recall:.0%} ({sum(1 for r in scored if r.expected_company_rank <= 5)}/{len(scored)}) | {mean_rr:.2f} |")
        lines.append("")

    # --- Tier 3b: reranker ablation ---
    if reranker_ablation:
        lines.append("## Tier 3b — Reranker ablation")
        lines.append("")
        baseline_rows = reranker_ablation.get("reranker_baseline", [])
        alt_rows = reranker_ablation.get("reranker_alternative", [])

        # Detect load failure from sentinel config name.
        alt_error: str | None = None
        if alt_rows and alt_rows[0].config_name.startswith("reranker_alternative[LOAD_ERROR:"):
            # Extract error message from sentinel config_name.
            alt_error = alt_rows[0].config_name[len("reranker_alternative[LOAD_ERROR: "):-1]

        # Derive the alternative model name from the first non-error row, or
        # from the sentinel if loading failed.
        if alt_error is None and alt_rows:
            alt_label = alt_rows[0].config_name
        elif alt_rows:
            # sentinel — display as unknown
            alt_label = "alternative (failed)"
        else:
            alt_label = "alternative"

        # Summary table.
        baseline_recall = _compute_recall_at_k(baseline_rows)
        baseline_mrr = _compute_mrr_at_k(baseline_rows)
        lines.append("| Reranker | Recall@5 | MRR@3 |")
        lines.append("|---|---|---|")
        lines.append(
            f"| cross-encoder/ms-marco-MiniLM-L-6-v2 (baseline) "
            f"| {baseline_recall:.3f} | {baseline_mrr:.3f} |"
        )
        if alt_error is not None:
            lines.append(
                f"| {alt_label} | — | — |"
            )
            lines.append("")
            lines.append(f"**Alternative reranker failed to load:** {alt_error}")
        else:
            alt_recall = _compute_recall_at_k(alt_rows)
            alt_mrr = _compute_mrr_at_k(alt_rows)
            lines.append(
                f"| {alt_label} (alternative) "
                f"| {alt_recall:.3f} | {alt_mrr:.3f} |"
            )
        lines.append("")

        if alt_error is None and baseline_rows and alt_rows:
            # Per-query breakdown.
            lines.append("### Per-query breakdown")
            lines.append("")
            lines.append("| Query ID | Baseline Recall@5 | Baseline MRR@3 | Alt Recall@5 | Alt MRR@3 |")
            lines.append("|---|---|---|---|---|")
            baseline_by_id = {r.query_id: r for r in baseline_rows}
            alt_by_id = {r.query_id: r for r in alt_rows}
            for qid in baseline_by_id:
                b = baseline_by_id[qid]
                a = alt_by_id.get(qid)
                b_in_top5 = b.expected_company_rank is not None and b.expected_company_rank <= 5
                b_in_top3 = b.expected_company_rank is not None and b.expected_company_rank <= 3
                b_rec = "1.000" if b_in_top5 else ("0.000" if b.expected_company_rank is not None else "n/a")
                b_mrr = f"{1.0/b.expected_company_rank:.3f}" if b_in_top3 else ("0.000" if b.expected_company_rank is not None else "n/a")
                if a is not None:
                    a_in_top5 = a.expected_company_rank is not None and a.expected_company_rank <= 5
                    a_in_top3 = a.expected_company_rank is not None and a.expected_company_rank <= 3
                    a_rec = "1.000" if a_in_top5 else ("0.000" if a.expected_company_rank is not None else "n/a")
                    a_mrr = f"{1.0/a.expected_company_rank:.3f}" if a_in_top3 else ("0.000" if a.expected_company_rank is not None else "n/a")
                else:
                    a_rec = a_mrr = "n/a"
                lines.append(f"| {qid} | {b_rec} | {b_mrr} | {a_rec} | {a_mrr} |")
            lines.append("")

    # Per-query detail
    lines.append("## Appendix — per-query detail")
    lines.append("")
    for r in evaluation:
        lines.append(f"### [{r.id}] {r.query}")
        lines.append("")
        lines.append(f"- Category: `{r.category}`")
        lines.append(f"- Intent: detected `{r.intent_detected}` (auto-check: {_fmt_check(r.intent_match)})")
        lines.append(f"- Entity filter: applied `{r.companies_filter_applied}` (auto-check: {_fmt_check(r.company_filter_match)})")
        lines.append(f"- Abstained: {r.abstained} (auto-check: {_fmt_check(r.abstain_match)})")
        lines.append(f"- Top rerank: `{r.top_rerank:.2f}`")
        lines.append(f"- Retrieval confidence: `{r.retrieval_confidence:.3f}` ({r.retrieval_confidence_band})")
        lines.append(f"- Context companies: {r.context_companies}")
        lines.append(f"- Context content types: {r.context_content_types}")
        if r.chart_in_context is not None:
            lines.append(f"- Chart chunk in context: {r.chart_in_context}")
        if r.both_dlr_versions is not None:
            lines.append(f"- Both DLR versions retrieved: {r.both_dlr_versions}")
        if r.citation_supported_ratio is not None:
            lines.append(f"- Citation faithfulness: {r.citation_supported_ratio:.0%}")
        if r.min_companies_check is not None:
            lines.append(f"- Min companies in context: {_fmt_check(r.min_companies_check)}")
        if r.forward_looking_check is not None:
            lines.append(f"- Forward-looking label check: {_fmt_check(r.forward_looking_check)}")
        if r.soft_refusal_check is not None:
            lines.append(f"- Soft-refusal check: {_fmt_check(r.soft_refusal_check)}")
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
    parser.add_argument("--ablation-reranker", action="store_true",
                        help="Run reranker ablation: evaluation set scored with the default reranker and an alternative side-by-side.")
    parser.add_argument("--ablation-model", type=str, default=None,
                        help="HuggingFace model identifier for the alternative reranker (used with --ablation-reranker).")
    parser.add_argument("--ablation-revision", type=str, default=None,
                        help="Optional revision hash for the alternative reranker.")
    args = parser.parse_args()

    if args.ablation_reranker and not args.ablation_model:
        parser.error("--ablation-reranker requires --ablation-model")

    logger.warning("== Tier 1: evaluation set ==")
    evaluation = run_evaluation()
    pass_n = sum(1 for r in evaluation if r.auto_pass)
    logger.warning("Evaluation: %d / %d passed auto-checks", pass_n, len(evaluation))

    ablation = None
    if args.ablation:
        logger.warning("== Tier 3: retrieval ablation ==")
        ablation = run_ablation()

    reranker_ablation = None
    if args.ablation_reranker:
        logger.warning("== Tier 3b: reranker ablation (%s) ==", args.ablation_model)
        reranker_ablation = run_reranker_ablation(args.ablation_model, args.ablation_revision)

    ragas = None
    if args.ragas:
        logger.warning("== Tier 2: RAGAS synthetic ==")
        ragas = run_ragas()

    if args.out:
        report = render_markdown(evaluation, ablation, ragas, reranker_ablation=reranker_ablation)
        Path(args.out).write_text(report)
        logger.warning("Wrote report to %s (%d bytes)", args.out, len(report))
    else:
        # Concise stdout summary
        print()
        print(f"EVAL: {pass_n}/{len(evaluation)} auto-pass")
        print()
        for r in evaluation:
            mark = "✅" if r.auto_pass else "❌"
            print(f"  {mark} [{r.id}] {r.category:24s} {r.query[:60]}")
            if not r.auto_pass:
                print(f"         fails: {','.join(r.auto_fail_reasons)}")


if __name__ == "__main__":
    main()
