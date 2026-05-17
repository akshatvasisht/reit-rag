"""Adaptive multi-hop retrieval orchestrated by Claude Haiku.

After the initial retrieval pass, a Haiku model with a ``retrieve_more`` tool
decides whether the current evidence is sufficient to answer the query. If
not, it proposes a focused sub-question and the pipeline runs another
retrieval pass, merging contexts by chunk ID dedupe. Capped at
``MAX_ADAPTIVE_HOPS`` additional hops. Any Haiku error returns the initial
result unchanged so the user query is never blocked by a failed hop call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from typing import TYPE_CHECKING

from src.models import RetrievedChunk

if TYPE_CHECKING:
    # Avoid a module-load cycle: pipeline.py imports this module at its top.
    from src.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


# Intents that may benefit from follow-up retrieval passes when initial evidence
# is judged insufficient.  "latest" is excluded because single-company snapshot
# queries do not require cross-document chaining.
ADAPTIVE_INTENTS: frozenset[str] = frozenset(
    {"historical", "comparison", "conflict", "all_company_synthesis"}
)

# Maximum number of additional retrieval passes beyond the initial one.
MAX_ADAPTIVE_HOPS = 2

# Path for hop-orchestration audit records.
_HOP_LOG_PATH = Path("logs/adaptive_hops.jsonl")

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Tool schema presented to Haiku when it evaluates evidence sufficiency.
_RETRIEVE_MORE_TOOL = {
    "name": "retrieve_more",
    "description": (
        "Request an additional retrieval pass when the existing evidence "
        "demonstrably cannot answer the query. Only call this tool when a "
        "specific, identifiable fact is missing — not when the answer could "
        "theoretically be improved."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Sub-question that targets the missing evidence.",
            },
            "company_filter": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict the sub-retrieval to these company names. "
                    "Pass an empty list for no restriction."
                ),
            },
        },
        "required": ["query", "company_filter"],
        "additionalProperties": False,
    },
}

_SUFFICIENCY_SYSTEM = (
    "You are a retrieval orchestrator for a financial document Q&A system. "
    "Your only job is to decide whether the current evidence is sufficient to "
    "answer the user's query, and if not, to call the retrieve_more tool with "
    "a focused sub-question that targets the specific missing fact.\n\n"
    "Rules:\n"
    "- Call retrieve_more ONLY when a specific, identifiable fact is absent "
    "from the evidence summary and that fact is necessary to answer the query.\n"
    "- Do NOT call retrieve_more when the evidence could theoretically be "
    "improved but already contains enough material to produce a reasonable answer.\n"
    "- Do NOT call retrieve_more if the query is unanswerable regardless of "
    "additional retrieval (e.g. the fact does not exist in financial documents).\n"
    "- If evidence is sufficient, respond with end_turn — do not explain yourself.\n"
    "- Sub-questions must be concise and specific (under 30 words)."
)


def _build_evidence_summary(contexts: list[RetrievedChunk]) -> str:
    """Produce a compact evidence summary for the Haiku sufficiency evaluator.

    Uses metadata and a short text hint rather than the full chunk text to
    keep the Haiku prompt small and cheap.
    """
    if not contexts:
        return "(no evidence retrieved yet)"
    lines: list[str] = []
    for i, rc in enumerate(contexts, start=1):
        c = rc.chunk
        hint = (c.chunk_text or "")[:120].replace("\n", " ")
        score_str = f"{rc.rerank_score:.2f}" if rc.rerank_score is not None else "—"
        lines.append(
            f"[{i}] {c.company} | {c.doc_type} | {c.report_date} "
            f"| p.{c.page_number or '?'} | score={score_str} "
            f"| section={c.section_title or '—'} | hint: {hint!r}"
        )
    return "\n".join(lines)


def _log_hop(record: dict) -> None:
    """Append a JSON record to the hop audit log; swallow all I/O errors."""
    try:
        _HOP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _HOP_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _get_haiku_client():  # type: ignore[return]
    """Return a lazy-initialised Anthropic client for adaptive hop calls."""
    from anthropic import Anthropic as _Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return _Anthropic(api_key=api_key)


def adaptive_retrieve(
    query: str,
    initial_result: "RetrievalResult",
    conn,
) -> "tuple[RetrievalResult, list[str]]":
    """Adaptive multi-hop loop: ask Haiku whether current evidence suffices.

    If Haiku judges the evidence insufficient and proposes a sub-question,
    runs another retrieval pass using the existing infrastructure with the
    sub-question and merges contexts by chunk ID dedupe. Caps at
    MAX_ADAPTIVE_HOPS additional hops. On any Haiku API error, returns
    the initial result unchanged so the user query is never blocked by
    a failed hop call.

    Args:
        query: The original user query.
        initial_result: Output of the first retrieval pass.
        conn: Open psycopg connection for additional retrieval passes.

    Returns:
        A tuple of (merged RetrievalResult, list of sub-queries fired).
    """
    # Lazy import: pipeline.py imports this module at its top.
    from src.retrieval.pipeline import RetrievalResult, _retrieve_core  # noqa: PLC0415

    seen_chunk_ids: set = {rc.chunk.id for rc in initial_result.contexts}
    merged_contexts: list[RetrievedChunk] = list(initial_result.contexts)
    sub_queries: list[str] = []

    try:
        client = _get_haiku_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Adaptive orchestrator unavailable: %s", exc)
        return initial_result, []

    for hop in range(MAX_ADAPTIVE_HOPS):
        evidence_summary = _build_evidence_summary(merged_contexts)
        user_message = (
            f"Original query: {query}\n\n"
            f"Current evidence ({len(merged_contexts)} chunks):\n{evidence_summary}"
        )

        try:
            response = client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=300,
                temperature=0.0,
                top_k=1,
                system=_SUFFICIENCY_SYSTEM,
                tools=[_RETRIEVE_MORE_TOOL],  # type: ignore[list-item]
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Adaptive orchestrator call failed on hop %d: %s", hop + 1, exc)
            _log_hop({"hop": hop + 1, "query": query, "error": str(exc)})
            break

        if response.stop_reason != "tool_use":
            logger.info("Adaptive orchestrator: evidence sufficient after %d hop(s)", hop)
            break

        # Extract the retrieve_more tool call.
        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            break

        try:
            sub_query: str = tool_block.input["query"]
            company_filter: list[str] = tool_block.input.get("company_filter") or []
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Adaptive orchestrator: malformed tool input: %s", exc)
            _log_hop({"hop": hop + 1, "query": query, "error": f"malformed_tool: {exc}"})
            break

        sub_queries.append(sub_query)
        logger.info(
            "Adaptive hop %d: sub_query=%r company_filter=%s",
            hop + 1, sub_query, company_filter,
        )
        _log_hop({
            "hop": hop + 1,
            "original_query": query,
            "sub_query": sub_query,
            "company_filter": company_filter,
        })

        hop_result = _retrieve_core(
            sub_query,
            company_filter=company_filter if company_filter else None,
            conn=conn,
            intent=initial_result.intent,
        )

        # Merge new contexts, deduping by chunk ID.
        new_count = 0
        for rc in hop_result.contexts:
            if rc.chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(rc.chunk.id)
                merged_contexts.append(rc)
                new_count += 1
        logger.info("Adaptive hop %d: +%d new chunks merged", hop + 1, new_count)

    # Apply deterministic sort over the merged context list.
    merged_contexts_sorted = sorted(
        merged_contexts,
        key=lambda rc: (rc.chunk.company, rc.chunk.report_date, rc.chunk.page_number or 0),
    )

    merged_result = RetrievalResult(
        query=initial_result.query,
        intent=initial_result.intent,
        contexts=merged_contexts_sorted,
        companies=initial_result.companies,
        abstain=initial_result.abstain,
        abstain_reason=initial_result.abstain_reason,
        retrieval_confidence=initial_result.retrieval_confidence,
        diagnostics=dict(initial_result.diagnostics),
    )
    return merged_result, sub_queries
