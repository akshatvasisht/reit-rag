"""Claude generation with citation enforcement and abstention.

Entry point: `answer(query)` runs the full retrieval+generation pipeline and
returns a structured `Answer`. `answer_stream(query)` is the streaming variant
used by the Streamlit UI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterator

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from src.generation.citation_check import CitationReport, check_citations
from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_message,
    render_abstention,
)
from src.models import RetrievedChunk
from src.retrieval.pipeline import RetrievalResult, retrieve

logger = logging.getLogger(__name__)

# Keep the model identifier centralized for straightforward updates.
MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 1500
TEMPERATURE = 0.0


_client: Anthropic | None = None
_GENERATION_ERRORS = (APIConnectionError, APITimeoutError, APIStatusError, RateLimitError)


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


# ---------------------------------------------------------------------------
# Non-streaming entry point
# ---------------------------------------------------------------------------


def answer(query: str) -> Answer:
    """Retrieve, optionally abstain, otherwise generate a cited answer."""
    retrieval = retrieve(query)

    if retrieval.abstain or not retrieval.contexts:
        logger.info("Abstaining: %s", retrieval.abstain_reason or "no contexts")
        return Answer(
            query=query,
            text=render_abstention(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=retrieval.diagnostics,
        )

    try:
        text = _generate(query, retrieval.contexts)
    except _GENERATION_ERRORS as e:
        logger.exception("Generation call failed: %s", type(e).__name__)
        diagnostics = dict(retrieval.diagnostics)
        diagnostics["generation_error"] = type(e).__name__
        return Answer(
            query=query,
            text=_render_generation_error(retrieval.contexts),
            abstained=True,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=diagnostics,
        )
    report = check_citations(text, retrieval.contexts)
    logger.info("Citation faithfulness: %d/%d supported (%.0f%%)",
                report.supported, report.total, report.faithfulness_ratio * 100)
    return Answer(
        query=query,
        text=text,
        abstained=False,
        contexts=retrieval.contexts,
        intent=retrieval.intent,
        diagnostics=retrieval.diagnostics,
        citation_report=report,
    )


def _generate(query: str, contexts: list[RetrievedChunk]) -> str:
    """Synchronously call Claude and return the full text."""
    client = _get_client()
    user_msg = build_user_message(query, contexts)

    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_k=1,  # greedy decoding — temperature=0 alone is not deterministic on Claude
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    # response.content is a heterogeneous block list; keep only text payloads.
    return "".join(getattr(block, "text", "") for block in resp.content)


# ---------------------------------------------------------------------------
# Streaming entry point — used by the Streamlit UI
# ---------------------------------------------------------------------------


def answer_stream(query: str) -> Iterator[dict]:
    """Stream the answer; first event is metadata, then text deltas, then final.

    Yields dicts with shapes:
        {"event": "retrieval", "result": RetrievalResult}
        {"event": "abstain", "text": str}                  # terminal if abstaining
        {"event": "delta",  "text": str}
        {"event": "final",  "answer": Answer}
    """
    retrieval = retrieve(query)
    yield {"event": "retrieval", "result": retrieval}

    if retrieval.abstain or not retrieval.contexts:
        text = render_abstention(retrieval.contexts)
        yield {"event": "abstain", "text": text}
        yield {
            "event": "final",
            "answer": Answer(
                query=query,
                text=text,
                abstained=True,
                contexts=retrieval.contexts,
                intent=retrieval.intent,
                diagnostics=retrieval.diagnostics,
            ),
        }
        return

    client = _get_client()
    user_msg = build_user_message(query, retrieval.contexts)
    chunks: list[str] = []

    try:
        with client.messages.stream(
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_k=1,  # greedy decoding — temperature=0 alone is not deterministic on Claude
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
                yield {"event": "delta", "text": text}
    except _GENERATION_ERRORS as e:
        logger.exception("Streaming generation failed: %s", type(e).__name__)
        text = _render_generation_error(retrieval.contexts)
        diagnostics = dict(retrieval.diagnostics)
        diagnostics["generation_error"] = type(e).__name__
        yield {"event": "abstain", "text": text}
        yield {
            "event": "final",
            "answer": Answer(
                query=query,
                text=text,
                abstained=True,
                contexts=retrieval.contexts,
                intent=retrieval.intent,
                diagnostics=diagnostics,
            ),
        }
        return

    final_text = "".join(chunks)
    report = check_citations(final_text, retrieval.contexts)
    logger.info("Citation faithfulness: %d/%d supported (%.0f%%)",
                report.supported, report.total, report.faithfulness_ratio * 100)
    yield {
        "event": "final",
        "answer": Answer(
            query=query,
            text=final_text,
            abstained=False,
            contexts=retrieval.contexts,
            intent=retrieval.intent,
            diagnostics=retrieval.diagnostics,
            citation_report=report,
        ),
    }


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
