"""Claude generation with citation enforcement and abstention.

Entry point: `answer(query)` runs the full retrieval+generation pipeline and
returns a structured `Answer`. `answer_structured(query)` returns a
`StructuredAnswer` with schema-validated fields, suitable for typed display.
"""

from __future__ import annotations

import logging
import os
import re
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
)
from src.generation.prompts import (
    ANSWER_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
    render_abstention,
)
from src.models import Claim, RetrievedChunk, StructuredAnswer
from src.retrieval.pipeline import retrieve
from src.versioning.classifier import TemporalIntent

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-4-6"
# Synthesis queries (multi-company tables) routinely exceeded the 1000-token
# ceiling, causing the model to stop_reason=max_tokens before emitting
# `claims`/`abstain`/`forward_looking` — silent KeyError downstream even with
# strict tool use, because grammar-constrained sampling can't conjure tokens
# that don't exist. 4096 leaves comfortable headroom for the largest answers.
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.0


_client: Anthropic | None = None

# Errors that indicate a transient API-layer failure and should trigger the
# controlled error path rather than propagating an unhandled exception.
# Scope is intentionally narrow: only transient network/API conditions and
# rate-limiting. Other exception classes (TypeError from a dataclass
# construction bug, KeyError from a malformed tool-use payload, etc.)
# propagate so operators can distinguish a transient outage from a code
# defect — both used to produce identical abstain-shaped output.
_GENERATION_ERRORS = (APIConnectionError, APITimeoutError, APIStatusError, RateLimitError)

# Tool definition that instructs the model to return its answer as a structured
# object matching ANSWER_JSON_SCHEMA.  Tool-use is used in place of a native
# JSON-output mode because it is supported across all SDK versions in active use.
_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final answer as a structured object. "
        "Call this tool to return your response — do not emit free-form text."
    ),
    # strict=True invokes grammar-constrained sampling; without it the model
    # can silently omit required fields. Sonnet 4.5+ supports strict.
    "strict": True,
    "input_schema": ANSWER_JSON_SCHEMA,
}


# ---------------------------------------------------------------------------
# Forward-looking integrity guard
# ---------------------------------------------------------------------------

# Keywords in the query that indicate a forward-looking ask (guidance /
# targets / projections). Conservative set to avoid over-refusal on
# non-forward-looking queries.
_FL_QUERY_PATTERN = re.compile(
    r"\b("
    r"guidance|guided|guide|"
    r"target|targets|"
    r"outlook|"
    r"projection|projections|projected|"
    r"forecast|forecasted|"
    r"expected\s+(?:to|for)|"  # "expected to" or "expected for" — avoids "what was expected"
    r"forward[-\s]looking|"
    r"year[-\s]end\s+20\d{2}|"
    r"plan(?:s|ning|ned)?\s+to|"
    r"intend(?:s|ing|ed)?\s+to|"
    r"aim(?:s|ing|ed)?\s+to"
    r")\b",
    re.IGNORECASE,
)

# Tokens in retrieved CHUNK TEXT that indicate forward-looking language for
# a metric. Presence of at least one of these in any retrieved chunk is
# sufficient to clear the guard.
_FL_CHUNK_PATTERN = re.compile(
    r"\b("
    r"guidance|guided|"
    r"target|targets|targeted|"
    r"expect|expected|expects|"
    r"project(?:ed|s)?|"
    r"forecast(?:ed|s)?|"
    r"range of|midpoint|"
    r"management guided|we guide|we expect|we target"
    r")\b",
    re.IGNORECASE,
)

# Instruction appended to the system prompt when FL intent is detected but
# no forward-looking language is found in retrieved chunks.
_FL_ABSENT_INSTRUCTION = """

FORWARD-LOOKING GUARD — MANDATORY:
The query asks for forward-looking information (guidance / target / outlook /
projection / forecast). However, none of the retrieved document excerpts
contain explicit forward-looking language (such as "guidance", "target",
"expected", "projected", "forecast", or "range of") for the queried metric.

You MUST:
1. State explicitly that this deck / document does NOT disclose guidance for \
the queried metric.
2. If the deck reports a historical or current ACTUAL value for this metric, \
you MAY mention it, but MUST clearly label it as a REPORTED/ACTUAL figure, \
NOT as guidance.
3. Do NOT present any reported/historical value as if it were forward-looking \
guidance.
4. Use framing such as:
   "The [Company] deck does not disclose [METRIC] guidance. The most recent \
reported figure is [VALUE] [citation], which is a historical/actual figure, \
not a forward projection."
5. Set abstain=false only if you can provide a meaningful caveat answer as \
above; otherwise set abstain=true with abstain_reason explaining that no \
guidance was found."""

# ---------------------------------------------------------------------------
# Corpus-membership framing
# ---------------------------------------------------------------------------

# Appended to the system prompt so the LLM can distinguish
# "I did not retrieve a chunk from issuer X" from "X is absent from corpus".
_CORPUS_MEMBERSHIP_INSTRUCTION_TEMPLATE = """

CORPUS MEMBERSHIP — CRITICAL FRAMING RULE:
The following companies are confirmed members of the indexed document corpus:
{company_list}

When your retrieved excerpts do NOT include a chunk from one of the above
companies, you MUST say:
  "I did not retrieve a chunk from [Company] for this query."
You MUST NOT say:
  "X is not in the corpus", "X is not represented in the indexed corpus",
  or "X does not appear in the retrieved documents" in a way that implies
  corpus absence.

A retrieval miss is NOT the same as corpus absence. Only assert corpus absence
for companies NOT in the list above."""


def _is_forward_looking_query(query: str) -> bool:
    """Return True if the query asks for forward-looking information.

    Uses a conservative keyword set to avoid false positives on historical
    or reported-figure queries.
    """
    return bool(_FL_QUERY_PATTERN.search(query))


def _chunks_have_forward_looking_support(chunks: list[RetrievedChunk]) -> bool:
    """Return True if any retrieved chunk carries forward-looking language.

    A single match in any chunk clears the guard.
    """
    for rc in chunks:
        if _FL_CHUNK_PATTERN.search(rc.chunk.chunk_text or ""):
            return True
    return False


def _build_system_prompt_with_corpus(corpus_companies: list[str]) -> str:
    """Return the system prompt augmented with the corpus-company list.

    Always called by answer() paths so every generation runs against the
    corpus-aware prompt, preventing false "not in corpus" assertions when
    retrieval simply missed an in-corpus issuer.
    """
    company_list = "\n".join(f"  - {c}" for c in sorted(corpus_companies))
    corpus_instruction = _CORPUS_MEMBERSHIP_INSTRUCTION_TEMPLATE.format(
        company_list=company_list,
    )
    return SYSTEM_PROMPT + corpus_instruction


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
    # The typed StructuredAnswer produced during generation, if available.
    # Present on every answered path so callers can inspect schema-validated
    # fields (e.g. forward_looking) without resorting to getattr hacks.
    structured: StructuredAnswer | None = None


# ---------------------------------------------------------------------------
# Non-streaming entry point
# ---------------------------------------------------------------------------


def answer(query: str) -> Answer:
    """Retrieve, optionally abstain, otherwise generate a cited answer.

    Thin wrapper around answer_structured. The two paths used to diverge
    only in log strings and local variable names; consolidating them
    removes the drift surface where each shared-behavior change had to
    be applied in two places.
    """
    return answer_structured(query)


def _generate_structured(
    query: str,
    contexts: list[RetrievedChunk],
    intent: TemporalIntent = "latest",
    forward_looking_hint: bool = False,
    system_prompt_override: str | None = None,
) -> StructuredAnswer:
    """Call Claude via tool-use and return a typed StructuredAnswer.

    Uses the tool-use pattern (tools + tool_choice) to guarantee that the
    response conforms to ANSWER_JSON_SCHEMA without relying on any
    output_config or native JSON-mode extension.  The model is forced to call
    the submit_answer tool, whose input is already a parsed dict.

    ``system_prompt_override``: when provided, replaces the default system prompt.
    The answer() / answer_structured() callers always build the corpus-aware prompt
    and pass it here.
    """
    try:
        client = _get_client()
        # Use the caller-supplied system prompt when provided; fall back to the
        # plain build_system_prompt() for direct _generate_structured() callers
        # that bypass the answer() path (e.g. some unit tests).
        system = system_prompt_override if system_prompt_override is not None else build_system_prompt()
        response = client.messages.create(  # type: ignore[call-overload]
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_k=1,  # greedy decoding — temperature=0 alone is not deterministic on Claude
            system=system,
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
        # strict=True guarantees field presence WHEN the model completes the
        # tool input. If max_tokens cuts the response off mid-stream, later
        # fields are silently missing. Default them so callers don't crash
        # on a truncation; the eval still surfaces this as forward_looking=False.
        return StructuredAnswer(
            answer_prose=raw.get("answer_prose", ""),
            claims=[Claim(**c) for c in raw.get("claims", [])],
            abstain=bool(raw.get("abstain", False)),
            abstain_reason=raw.get("abstain_reason", ""),
            forward_looking=bool(raw.get("forward_looking", False)),
        )
    except _GENERATION_ERRORS:
        raise


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

    from src.corpus_registry import CORPUS_REGISTRY  # noqa: PLC0415
    corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY})
    system_prompt = _build_system_prompt_with_corpus(corpus_companies)

    if retrieval.forward_looking and not _chunks_have_forward_looking_support(contexts_sorted):
        logger.info(
            "Forward-looking guard fired: classifier marked the query as "
            "forward-looking but no retrieved chunk carries FL language"
        )
        system_prompt = system_prompt + _FL_ABSENT_INSTRUCTION

    try:
        structured = _generate_structured(
            query, contexts_sorted,
            intent=retrieval.intent,
            forward_looking_hint=retrieval.forward_looking,
            system_prompt_override=system_prompt,
        )
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
