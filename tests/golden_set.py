"""Manual query set for retrieval-and-generation evaluation.

Curated queries spanning all pass categories plus chart-targeted
probes that exercise the 179 chart_description chunks produced by the
vision-extraction enrichment.

Each entry pairs:
- Automatable expectations (intent, abstention, expected company, source
  content_type) the evaluation harness can check programmatically.
- A short human-eval description for what a correct natural-language answer
  needs to demonstrate — graded manually because no ground-truth answers
  exist for this corpus.

The harness in `scripts/evaluate.py` consumes this list and produces per-query
pass/fail plus aggregate Recall@K and MRR over the expected retrieval signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PassCategory = Literal[
    "factual_citation",       # answer cites a chunk from the expected company
    "numerical_specific",     # answer reports a specific number the deck contains
    "version_dedup",          # latest-only intent dedupes to the most recent
    "version_comparison",     # comparison intent surfaces both versions with dates
    "abstention_oo_corpus",   # query is out-of-corpus → must abstain cleanly
    "abstention_chart_only",  # data exists only in a chart → was unanswerable before chart enrichment, may now succeed
    "chart_extraction",       # answer cites a chart_description chunk
    "cross_doc_synthesis",    # answer integrates 2+ companies
    "forward_looking_label",  # answer flags guidance vs reported
    "thematic",               # query targets the thematic report
]

ExpectedIntent = Literal["latest", "historical", "comparison"]


@dataclass
class GoldenQuery:
    """One curated test query with pass criteria the harness can evaluate."""
    id: str
    query: str
    category: PassCategory
    expected_intent: ExpectedIntent | None = None
    # Pre-filter expectation: company that the entity extractor should detect.
    expected_company: str | None = None
    # Whether the abstention gate is expected to fire pre-LLM.
    expect_hard_abstain: bool = False
    # Whether the model is expected to refuse in text even if the gate did not fire.
    expect_soft_refusal: bool = False
    # If True, the harness should find at least one `chart_description` chunk
    # in the retrieved context.
    expect_chart_chunk_in_context: bool = False
    # If True, the answer must surface both DLR versions in context.
    expect_both_dlr_versions: bool = False
    # Free-text description of what a correct answer must demonstrate.
    human_eval_rubric: str = ""
    # Optional notes for the manual reviewer.
    notes: str = ""


GOLDEN_SET: list[GoldenQuery] = [
    # ---------- Core hard queries ----------
    GoldenQuery(
        id="g01",
        query="What is DLR's net debt to EBITDA in the most recent presentation?",
        category="version_dedup",
        expected_intent="latest",
        expected_company="Digital Realty",
        human_eval_rubric=(
            "Answer must come from the Mar 2026 DLR deck (not Dec 2025). "
            "Citation must include 'March 2026'. Specific numeric value expected."
        ),
    ),
    GoldenQuery(
        id="g02",
        query="How did DLR's leverage change between December 2025 and March 2026?",
        category="version_comparison",
        expected_intent="comparison",
        expected_company="Digital Realty",
        expect_both_dlr_versions=True,
        human_eval_rubric=(
            "Both DLR docs must be surfaced with dates. Answer must report the "
            "delta — not a hedged average. If retrieval can't find both, "
            "explicit abstention with both docs listed is acceptable."
        ),
        notes="Version-conflict stress case.",
    ),
    GoldenQuery(
        id="g03",
        query="What is VICI's total revenue for Q4 2025?",
        category="numerical_specific",
        expected_intent="latest",
        expected_company="VICI Properties",
        human_eval_rubric="Specific numeric value from VICI's Mar 2026 deck.",
    ),
    GoldenQuery(
        id="g04",
        query="Compare same-store NOI growth across all companies",
        category="cross_doc_synthesis",
        expected_intent="comparison",
        expected_company=None,
        human_eval_rubric=(
            "Multi-company answer. Should include at least 3 REITs with "
            "explicit NOI growth figures and citations to each."
        ),
    ),
    GoldenQuery(
        id="g05",
        query="What does PSA say about the merger impact on FFO?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="Public Storage",
        human_eval_rubric=(
            "Citation must be to PSA Merger-Presentation. FFO impact discussed."
        ),
    ),
    GoldenQuery(
        id="g06",
        query="What is EGP's occupancy trend over the past 4 quarters?",
        category="abstention_chart_only",
        expected_intent="latest",
        expected_company="EastGroup Properties",
        human_eval_rubric=(
            "Pre-chart-enrichment: should abstain. Post-chart-enrichment: "
            "may now produce a cited answer from a chart_description on the "
            "occupancy slide. EITHER is acceptable as long as no hallucinated "
            "numbers appear."
        ),
        notes="Abstention behavior check.",
    ),
    GoldenQuery(
        id="g07",
        query="Which company has the highest leverage?",
        category="cross_doc_synthesis",
        expected_intent="latest",
        expected_company=None,
        human_eval_rubric=(
            "Cross-company ranking. Should compare actual leverage figures "
            "across at least 4-5 companies and pick a winner with citation."
        ),
    ),
    GoldenQuery(
        id="g08",
        query="What is Realty Income's dividend yield guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expected_company="Realty Income",
        human_eval_rubric=(
            "If the answer mentions a guided figure, it must be labeled as "
            "'Management guided...' or equivalent — never as a reported fact."
        ),
    ),
    GoldenQuery(
        id="g09",
        query="What does Simon Property's report say about foot traffic trends?",
        category="thematic",
        expected_intent="latest",
        expected_company="Simon Property Group",
        human_eval_rubric=(
            "Citation must be to Simon's thematic_report (not "
            "investor_presentation). Should discuss brick-and-mortar trends "
            "qualitatively."
        ),
    ),
    GoldenQuery(
        id="g10",
        query="What was BXP's leasing activity in Q4 2025?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="BXP",
        human_eval_rubric=(
            "Citation must be to one of the BXP decks. Specific leasing volume "
            "figure or activity description expected."
        ),
    ),

    # ---------- Chart-targeted queries (leverage chart_descriptions) ----------
    GoldenQuery(
        id="g11",
        query="What is DLR's projected AI inference capacity in GW for 2030?",
        category="chart_extraction",
        expected_intent="latest",
        expected_company="Digital Realty",
        expect_chart_chunk_in_context=True,
        human_eval_rubric=(
            "Answer must reference the Mar 2026 deck's p.10 chart with "
            "Non-AI / AI Training / AI Inference stacked-bar data. Specific "
            "GW value for 2030 expected (or 'approximately X GW' if hedged)."
        ),
    ),
    GoldenQuery(
        id="g12",
        query="What are Realty Income's top tenants by % of annualized base rent?",
        category="chart_extraction",
        expected_intent="latest",
        expected_company="Realty Income",
        expect_chart_chunk_in_context=True,
        human_eval_rubric=(
            "Answer should name at least 3-5 specific tenants with their % of "
            "ABR (e.g., 7-Eleven, Dollar General, Walgreens at 3.x% each), "
            "cited to the Realty Income deck."
        ),
    ),
    GoldenQuery(
        id="g13",
        query="What is Digital Realty's future development capacity in GW?",
        category="chart_extraction",
        expected_intent="latest",
        expected_company="Digital Realty",
        expect_chart_chunk_in_context=True,
        human_eval_rubric=(
            "Answer should report ~5 GW total with the capacity-block "
            "breakdown (>100MW = 53%, 25-100MW = 35%, <25MW = 12%) from "
            "the Mar 2026 deck's map+pie chart."
        ),
    ),

    # ---------- Abstention probes ----------
    GoldenQuery(
        id="g14",
        query="What is BXP CEO's favorite color?",
        category="abstention_oo_corpus",
        expected_intent="latest",
        expected_company="BXP",
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Must abstain. Refusal text should mention BXP documents were "
            "searched. No invented answer."
        ),
    ),
    GoldenQuery(
        id="g15",
        query="What is Equinix's portfolio size?",
        category="abstention_oo_corpus",
        expected_intent="latest",
        expected_company=None,
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Equinix is not in our corpus. Must abstain (no Equinix sources "
            "to retrieve). Refusal text should make clear which companies "
            "ARE in scope."
        ),
    ),
]


def iter_golden() -> list[GoldenQuery]:
    """Return the curated golden set."""
    return list(GOLDEN_SET)


def print_summary() -> None:
    """Smoke entrypoint — print a one-line summary per query."""
    by_cat: dict[str, int] = {}
    for q in GOLDEN_SET:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    print(f"Golden set: {len(GOLDEN_SET)} queries")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:30s} {n}")
    print()
    for q in GOLDEN_SET:
        flag = "🔴" if q.expect_hard_abstain else ("📊" if q.expect_chart_chunk_in_context else "  ")
        print(f"  {flag} [{q.id}] {q.category:24s} {q.query}")


if __name__ == "__main__":
    print_summary()
