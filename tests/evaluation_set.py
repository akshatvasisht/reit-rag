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
    "conflict_detection",     # answer surfaces contradicting data points across documents
    "all_company_synthesis",  # answer synthesises data across all corpus companies
    "metric_precision",       # answer reports specific metric components and the adjustments between them
    "historical",             # answer draws on a historical document version
    "capital_structure",      # answer addresses debt maturity, covenants, or interest-rate exposure
]

ExpectedIntent = Literal["latest", "historical", "comparison", "conflict", "all_company_synthesis"]


@dataclass
class EvaluationQuery:
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
    # When set, the harness counts distinct companies in retrieved context and
    # fails the query if fewer than this many companies are represented.
    expect_min_companies_in_context: int | None = None
    # When True, the harness verifies the structured answer sets forward_looking=True.
    expect_forward_looking: bool = False
    # When True, intent mismatch is a hard FAIL. When False (default), an
    # intent mismatch downgrades to soft_fail — useful for queries that could
    # legitimately route as either "latest" or "comparison".
    expected_intent_strict: bool = False


EVALUATION_SET: list[EvaluationQuery] = [
    # ---------- Core hard queries ----------
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
        id="g04",
        query="Compare same-store NOI growth across all companies",
        category="all_company_synthesis",
        expected_intent="all_company_synthesis",
        expected_company=None,
        expect_min_companies_in_context=4,
        human_eval_rubric=(
            "Multi-company answer. Should include at least 4 REITs with "
            "explicit NOI growth figures and citations to each."
        ),
        notes="Must retrieve data from at least 4 of 6 corpus companies.",
    ),
    EvaluationQuery(
        id="g05",
        query="What does PSA say about the merger impact on FFO?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="Public Storage",
        human_eval_rubric=(
            "Citation must be to PSA Merger-Presentation. FFO impact discussed."
        ),
    ),
    EvaluationQuery(
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
    EvaluationQuery(
        id="g07",
        query="Rank all companies by Net Debt to EBITDA and identify which has the most conservative balance sheet",
        category="all_company_synthesis",
        expected_intent="all_company_synthesis",
        expected_company=None,
        expect_min_companies_in_context=4,
        human_eval_rubric=(
            "Full cross-company ranking with specific Net Debt/EBITDA figures and source "
            "citations. The most conservative issuer must be identified with supporting data."
        ),
    ),
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
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
    EvaluationQuery(
        id="g14",
        query="What is Simon Property Group's same-store sales per square foot for Q3 2025?",
        category="abstention_oo_corpus",
        expected_intent="latest",
        expected_company="Simon Property Group",
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Must abstain. Simon's same-store sales per square foot is not reported "
            "in the corpus documents. No invented figure; refusal text should make "
            "clear the specific metric was not found."
        ),
    ),
    EvaluationQuery(
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
        notes=(
            "Must abstain AND must not attribute any financial figure to Equinix. "
            "Previous version passed abstention check but hallucinated a market cap figure "
            "sourced from an incidental mention in VICI Properties' peer comparison table."
        ),
    ),

    # ---------- Analyst-level queries ----------
    EvaluationQuery(
        id="g16",
        query="Are there conflicting data points across documents on Digital Realty leverage?",
        category="conflict_detection",
        expected_intent="conflict",
        expected_company="Digital Realty",
        expect_both_dlr_versions=True,
        human_eval_rubric=(
            "Answer must surface the Dec 2025 and Mar 2026 DLR leverage figures "
            "side-by-side and explicitly note the discrepancy with dates. "
            "Hedged averages or single-version answers are failures."
        ),
        notes="Conflict-detection stress case; requires both DLR version-groups in context.",
    ),
    EvaluationQuery(
        id="g17",
        query="What are the capital allocation priorities across the portfolio — which companies are prioritising development versus dividends?",
        category="all_company_synthesis",
        expected_intent="all_company_synthesis",
        expected_company=None,
        expect_min_companies_in_context=3,
        human_eval_rubric=(
            "Answer must contrast at least 3 companies on development-spend versus "
            "dividend policy with cited evidence, not generic descriptions."
        ),
    ),
    EvaluationQuery(
        id="g18",
        query="What is the difference between Realty Income's FFO and AFFO per share, and what adjustments drive the gap?",
        category="metric_precision",
        expected_intent="latest",
        expected_company="Realty Income",
        expect_soft_refusal=True,
        human_eval_rubric=(
            "Answer must report both FFO and AFFO per share values and name the "
            "specific line-item adjustments (e.g. straight-line rent, acquisition costs) "
            "that reconcile them. Soft refusal is acceptable if the deck omits one metric."
        ),
        notes="Soft refusal is acceptable if FFO/AFFO reconciliation is absent from corpus.",
    ),
    EvaluationQuery(
        id="g19",
        query="What is PSA's AFFO per share guidance for 2026, and how does it compare to 2025 actuals?",
        category="abstention_oo_corpus",
        expected_intent="latest",
        expected_company="Public Storage",
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Must abstain. The PSA Company Update (March 2026) and Merger Presentation "
            "provide only Core FFO per share guidance for 2026 ($16.35–$17.00); no AFFO "
            "per share guidance or AFFO per share actuals appear in either document."
        ),
        notes=(
            "Verified PSA Company Update p.15 contains 2026 Core FFO/share guidance "
            "($16.35–$17.00) but no AFFO/share figure in any section of either PSA PDF; "
            "correct behavior is abstention on the AFFO-specific question."
        ),
    ),
    EvaluationQuery(
        id="g20",
        query="How has BXP's occupancy rate trended over the reported quarters in the Investor Day deck?",
        category="abstention_oo_corpus",
        expected_intent="historical",
        expected_company="BXP",
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Must abstain. The BXP 2025 Investor Day deck contains occupancy data for "
            "Q2 2025 (86.4% on p.79) and a projected year-end 2026 figure (89.0%), but "
            "no multi-quarter historical occupancy rate series across consecutive quarters."
        ),
        notes=(
            "Verified BXP Morning Session Deck p.79 shows Q2 2025 occupancy of 86.4% "
            "and projected year-end 2026 occupancy of 89.0%, but no quarterly series "
            "of reported occupancy rates is present; single-point data does not satisfy "
            "the trend query."
        ),
    ),
    EvaluationQuery(
        id="g21",
        query="What has been EGP's leasing spread trend over the periods covered in its roadshow deck?",
        category="abstention_oo_corpus",
        expected_intent="historical",
        expected_company="EastGroup Properties",
        expect_hard_abstain=True,
        human_eval_rubric=(
            "Must abstain. The EGP February 2026 roadshow deck (40 pages) contains no "
            "leasing spreads, rent spreads, mark-to-market figures, or multi-period "
            "leasing economics data of any kind."
        ),
        notes=(
            "Verified EGP February 2026 roadshow (all 40 pages): no leasing spread, "
            "rent spread, or mark-to-market series is present anywhere in the document; "
            "correct behavior is abstention on the spread-trend question."
        ),
    ),
    EvaluationQuery(
        id="g22",
        query="What is VICI's weighted average debt maturity and interest rate exposure — fixed vs floating?",
        category="capital_structure",
        expected_intent="latest",
        expected_company="VICI Properties",
        human_eval_rubric=(
            "Answer must state weighted average debt maturity in years and the "
            "fixed/floating split as percentages or absolute amounts, with citation "
            "to VICI's deck."
        ),
    ),
    EvaluationQuery(
        id="g23",
        query="Does DLR's March 2026 presentation disclose any changes to its credit facility or debt covenant compliance?",
        category="capital_structure",
        expected_intent="latest",
        expected_company="Digital Realty",
        expect_soft_refusal=True,
        human_eval_rubric=(
            "Answer must cite the Mar 2026 DLR deck. If covenant or credit-facility "
            "language is present, it must be quoted or paraphrased accurately. "
            "Soft refusal is acceptable if the deck contains no such disclosure."
        ),
        notes="Soft refusal acceptable if no covenant language is present in the Mar 2026 deck.",
    ),
    EvaluationQuery(
        id="g24",
        query="What is Digital Realty's management guidance on data center demand drivers for the next 12-18 months?",
        category="forward_looking_label",
        expected_intent="latest",
        expected_company="Digital Realty",
        expect_forward_looking=True,
        human_eval_rubric=(
            "Answer must label management's demand projections as guidance or forward-looking "
            "statements — not reported facts. Specific demand drivers (AI workloads, cloud "
            "migration, etc.) should be named with citation."
        ),
    ),
    EvaluationQuery(
        id="g25",
        query="What macro tailwinds does Digital Realty cite as drivers of data centre demand, and are these claims corroborated by Simon's brick-and-mortar analysis?",
        category="thematic",
        expected_intent="all_company_synthesis",
        expected_company=None,
        human_eval_rubric=(
            "Answer must draw on DLR deck for demand-driver claims and Simon's thematic "
            "report for corroboration or contrast. Cross-document synthesis with explicit "
            "citations to both sources required."
        ),
        notes="Requires retrieval from both DLR investor presentation and Simon thematic report.",
    ),
    # Regression closure cases (g26–g29)
    EvaluationQuery(
        id="g26",
        query="What was Realty Income's full year 2025 AFFO per share?",
        category="factual_citation",
        expected_intent="latest",
        expected_company="Realty Income",
        human_eval_rubric=(
            "Answer must report $4.28 per diluted share with citation to RI Q4 2025 "
            "deck page 40 reconciliation. The full-page reconciliation table lives "
            "on the same page as a section-title text chunk, so retrieval must "
            "pull the table chunk alongside any text chunk it surfaces. Faithful "
            "citation required."
        ),
        notes="Regression guard: same-page table chunk must reach the LLM when a text chunk on the same page is retained.",
    ),
    EvaluationQuery(
        id="g27",
        query="What is BXP's full year 2025 FFO per share guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expected_company="BXP",
        expect_soft_refusal=True,
        human_eval_rubric=(
            "BXP Investor Day session (December 2025) explicitly DID NOT reaffirm "
            "the 2025 FFO guidance issued in the July 29, 2025 earnings release. "
            "Answer must soft-refuse with explicit framing that the December 2025 "
            "decks do not carry reaffirmed 2025 guidance. Substituting the eventual "
            "actual ($7.10) as the headline answer is a forward-looking integrity "
            "violation."
        ),
        notes="Regression guard: forward-looking-intent queries must soft-refuse when retrieved chunks lack forward-looking language.",
    ),
    EvaluationQuery(
        id="g28",
        query="Compare BXP's portfolio occupancy as reported in the December 2025 Investor Day session versus the Q4 2025 quarterly investor deck.",
        category="version_comparison",
        expected_intent="comparison",
        expected_company="BXP",
        human_eval_rubric=(
            "Both decks carry BXP occupancy figures. Citations MUST distinguish "
            "the two BXP December 2025 presentations via subtype in the citation "
            "header — i.e., [BXP, Investor Presentation (Investor Day Session), "
            "December 2025, p.X] vs [BXP, Investor Presentation (Quarterly "
            "Investor Deck), December 2025, p.Y]."
        ),
        notes="Regression guard: citations must include doc_subtype when (company, doc_type, report_date) collides across retrieved chunks.",
    ),
    EvaluationQuery(
        id="g29",
        query="Across all REITs in the corpus, which has the strongest exposure to Sunbelt markets?",
        category="all_company_synthesis",
        expected_intent="all_company_synthesis",
        expected_company=None,
        expect_min_companies_in_context=4,
        human_eval_rubric=(
            "Answer must surface EastGroup Properties as the corpus's Sunbelt-"
            "industrial primary, with state-level concentration (TX/FL/CA/AZ/NC). "
            "Synthesis retrieval must reach every named issuer; collapsing to a "
            "subset and silently excluding EGP is a regression."
        ),
        notes="Regression guard: all_company_synthesis must surface at least one chunk per named issuer in the retained set.",
    ),
]


def iter_evaluation_set() -> list[EvaluationQuery]:
    """Return the curated evaluation set."""
    return list(EVALUATION_SET)


def print_summary() -> None:
    """Smoke entrypoint — print a one-line summary per query."""
    by_cat: dict[str, int] = {}
    for q in EVALUATION_SET:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    print(f"Evaluation set: {len(EVALUATION_SET)} queries")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:30s} {n}")
    print()
    for q in EVALUATION_SET:
        flag = "🔴" if q.expect_hard_abstain else ("📊" if q.expect_chart_chunk_in_context else "  ")
        print(f"  {flag} [{q.id}] {q.category:24s} {q.query}")


if __name__ == "__main__":
    print_summary()
