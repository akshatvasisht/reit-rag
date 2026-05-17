"""Claude generation with citation enforcement and abstention.

Entry point: `answer(query)` runs the full retrieval+generation pipeline and
returns a structured `Answer`. `answer_structured(query)` returns a
`StructuredAnswer` with schema-validated fields, suitable for typed display.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from src.generation.citation_check import (
    CitationReport,
    check_citations,
    check_numeric_consistency,
    check_ooc_entity_attribution,
)
from src.generation.prompts import (
    ANSWER_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
    render_abstention,
)
from src.models import Claim, RetrievedChunk, StructuredAnswer
from src.retrieval.pipeline import RetrievalResult, retrieve
from src.versioning.classifier import TemporalIntent

logger = logging.getLogger(__name__)

# Keep the model identifier centralized for straightforward updates.
MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.0


_client: Anthropic | None = None

# Errors that indicate a transient API-layer failure and should trigger the
# controlled error path rather than propagating an unhandled exception.
# TypeError is included so that SDK version mismatches (unexpected kwargs, etc.)
# degrade gracefully instead of crashing the user request.
_GENERATION_ERRORS = (APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, TypeError)

# Tool definition that instructs the model to return its answer as a structured
# object matching ANSWER_JSON_SCHEMA.  Tool-use is used in place of a native
# JSON-output mode because it is supported across all SDK versions in active use.
_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final answer as a structured object. "
        "Call this tool to return your response — do not emit free-form text."
    ),
    "input_schema": ANSWER_JSON_SCHEMA,
}


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=api_key)
    return _client


@dataclass
class Answer:
    """Final answer bundle returned to the UI / evaluator."""

    query: str
    text: str
    abstained: bool
    contexts: list[RetrievedChunk] = field(default_factory=list)
    intent: str | None = None
    diagnostics: dict = field(default_factory=dict)
    citation_report: CitationReport | None = None
    ooc_attribution_issues: list[dict] = field(default_factory=list)
    # The typed StructuredAnswer produced during generation, if available.
    # Present on every answered path so callers can inspect schema-validated
    # fields (e.g. forward_looking) without resorting to getattr hacks.
    structured: StructuredAnswer | None = None


# ---------------------------------------------------------------------------
# Non-streaming entry point
# ---------------------------------------------------------------------------


def answer(query: str) -> Answer:
    """Retrieve, optionally abstain, otherwise generate a cited answer."""
    retrieval = retrieve(query)

    if retrieval.abstain or not retrieval.contexts:
        logger.info("Abstaining: %s", retrieval.abstain_reason or "no contexts")
        abstain_diagnostics = dict(retrieval.diagnostics) if retrieval.diagnostics else {}
        abstain_diagnostics["retrieval_confidence"] = retrieval.retrieval_confidence
        return Answer(
            query=query,
            text=render_abstention(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=abstain_diagnostics,
        )

    # Sort contexts by document coordinates before prompt assembly so that
    # floating-point score ties in reranker output don't produce different
    # orderings across runs.  The original list is kept for diagnostic display.
    contexts_sorted = sorted(
        retrieval.contexts,
        key=lambda c: (c.chunk.company, c.chunk.report_date, c.chunk.page_number or 0),
    )

    try:
        structured = _generate_structured(query, contexts_sorted, intent=retrieval.intent)
    except _GENERATION_ERRORS as e:
        logger.exception("Generation call failed: %s", type(e).__name__)
        diagnostics = dict(retrieval.diagnostics)
        diagnostics["generation_error"] = type(e).__name__
        diagnostics["retrieval_confidence"] = retrieval.retrieval_confidence
        return Answer(
            query=query,
            text=_render_generation_error(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=diagnostics,
        )

    text = structured.answer_prose
    report = check_citations(text, retrieval.contexts)
    logger.info("Citation faithfulness: %d/%d supported (%.0f%%)",
                report.supported, report.total, report.faithfulness_ratio * 100)

    numeric_issues: list[dict] = []
    if not structured.abstain:
        numeric_issues = check_numeric_consistency(structured, retrieval.contexts)
        if numeric_issues:
            logger.info("Numeric consistency issues: %d", len(numeric_issues))
    report.numeric_mismatches = numeric_issues
    structured.numeric_consistency_report = numeric_issues

    ooc_issues: list[dict] = []
    if not structured.abstain:
        from src.corpus_registry import CORPUS_REGISTRY  # noqa: PLC0415
        corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY})
        ooc_issues = check_ooc_entity_attribution(structured, query, corpus_companies)
        structured.ooc_attribution_issues = ooc_issues
        if ooc_issues:
            logger.info("OOC entity attribution issues: %d", len(ooc_issues))

    structured.retrieval_hops = retrieval.diagnostics.get("retrieval_hops", 0)
    structured.sub_queries_fired = list(retrieval.diagnostics.get("sub_queries", []))
    structured.retrieval_confidence = retrieval.retrieval_confidence

    diagnostics = dict(retrieval.diagnostics)
    diagnostics["retrieval_confidence"] = retrieval.retrieval_confidence

    return Answer(
        query=query,
        text=text,
        abstained=structured.abstain,
        contexts=retrieval.contexts,
        intent=retrieval.intent,
        diagnostics=diagnostics,
        citation_report=report,
        ooc_attribution_issues=ooc_issues,
        structured=structured,
    )


def _generate_structured(
    query: str,
    contexts: list[RetrievedChunk],
    intent: TemporalIntent = "latest",
) -> StructuredAnswer:
    """Call Claude via tool-use and return a typed StructuredAnswer.

    Uses the tool-use pattern (tools + tool_choice) to guarantee that the
    response conforms to ANSWER_JSON_SCHEMA without relying on any
    output_config or native JSON-mode extension.  The model is forced to call
    the submit_answer tool, whose input is already a parsed dict.
    """
    try:
        client = _get_client()
        response = client.messages.create(  # type: ignore[call-overload]
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_k=1,  # greedy decoding — temperature=0 alone is not deterministic on Claude
            system=build_system_prompt(),
            messages=[{"role": "user", "content": build_user_message(query, contexts, intent=intent)}],
            tools=[_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
        )
        tool_use_block = next(
            (
                b for b in response.content
                if getattr(b, "type", None) == "tool_use" and b.name == "submit_answer"
            ),
            None,
        )
        if tool_use_block is None:
            raise RuntimeError("Model did not return a submit_answer tool_use block")
        raw: dict = tool_use_block.input
        return StructuredAnswer(
            answer_prose=raw["answer_prose"],
            claims=[Claim(**c) for c in raw["claims"]],
            abstain=raw["abstain"],
            abstain_reason=raw["abstain_reason"],
            forward_looking=raw["forward_looking"],
        )
    except _GENERATION_ERRORS:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in structured generation: %s", type(exc).__name__)
        return StructuredAnswer(
            answer_prose="",
            claims=[],
            abstain=True,
            abstain_reason="Generation failed; verify the source documents directly.",
            forward_looking=False,
        )


# ---------------------------------------------------------------------------
# Structured entry point — used by the Streamlit UI
# ---------------------------------------------------------------------------


def answer_structured(query: str) -> Answer:
    """Retrieve and generate a schema-validated answer; return an Answer for display.

    Uses structured JSON output to guarantee citation and field consistency.
    The returned Answer.contexts carries the original reranker-ordered chunks
    for source display.
    """
    retrieval = retrieve(query)

    if retrieval.abstain or not retrieval.contexts:
        logger.info("Abstaining: %s", retrieval.abstain_reason or "no contexts")
        abstain_diagnostics = dict(retrieval.diagnostics) if retrieval.diagnostics else {}
        abstain_diagnostics["retrieval_confidence"] = retrieval.retrieval_confidence
        return Answer(
            query=query,
            text=render_abstention(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=abstain_diagnostics,
        )

    # Sort contexts by document coordinates before prompt assembly so that
    # floating-point score ties in reranker output don't produce different
    # orderings across runs.  The original list is kept for diagnostic display.
    contexts_sorted = sorted(
        retrieval.contexts,
        key=lambda c: (c.chunk.company, c.chunk.report_date, c.chunk.page_number or 0),
    )

    try:
        structured = _generate_structured(query, contexts_sorted, intent=retrieval.intent)
    except _GENERATION_ERRORS as e:
        logger.exception("Structured generation failed: %s", type(e).__name__)
        diagnostics = dict(retrieval.diagnostics)
        diagnostics["generation_error"] = type(e).__name__
        diagnostics["retrieval_confidence"] = retrieval.retrieval_confidence
        return Answer(
            query=query,
            text=_render_generation_error(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=diagnostics,
        )

    text = structured.answer_prose
    report = check_citations(text, retrieval.contexts)
    logger.info("Citation faithfulness: %d/%d supported (%.0f%%)",
                report.supported, report.total, report.faithfulness_ratio * 100)

    numeric_issues: list[dict] = []
    if not structured.abstain:
        numeric_issues = check_numeric_consistency(structured, retrieval.contexts)
        if numeric_issues:
            logger.info("Numeric consistency issues: %d", len(numeric_issues))
    report.numeric_mismatches = numeric_issues
    structured.numeric_consistency_report = numeric_issues

    ooc_issues_structured: list[dict] = []
    if not structured.abstain:
        from src.corpus_registry import CORPUS_REGISTRY  # noqa: PLC0415
        corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY})
        ooc_issues_structured = check_ooc_entity_attribution(structured, query, corpus_companies)
        structured.ooc_attribution_issues = ooc_issues_structured
        if ooc_issues_structured:
            logger.info("OOC entity attribution issues: %d", len(ooc_issues_structured))

    structured.retrieval_hops = retrieval.diagnostics.get("retrieval_hops", 0)
    structured.sub_queries_fired = list(retrieval.diagnostics.get("sub_queries", []))
    structured.retrieval_confidence = retrieval.retrieval_confidence

    diagnostics_structured = dict(retrieval.diagnostics)
    diagnostics_structured["retrieval_confidence"] = retrieval.retrieval_confidence

    return Answer(
        query=query,
        text=text,
        abstained=structured.abstain,
        contexts=retrieval.contexts,
        intent=retrieval.intent,
        diagnostics=diagnostics_structured,
        citation_report=report,
        ooc_attribution_issues=ooc_issues_structured,
        structured=structured,
    )


def _render_generation_error(contexts: list[RetrievedChunk]) -> str:
    """Build a user-facing message for transient generation failures."""
    docs = {
        (rc.chunk.company, rc.chunk.doc_type, rc.chunk.report_date)
        for rc in contexts
    }
    if docs:
        docs_str = "; ".join(f"{c} ({dt}, {d})" for c, dt, d in sorted(docs))
    else:
        docs_str = "(no documents matched the query)"
    return (
        "I couldn't generate a reliable answer right now due to a temporary "
        "model/API issue. Please retry this query.\n\n"
        f"Documents retrieved before failure: {docs_str}"
    )
