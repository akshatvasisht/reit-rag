"""Corporate RAG — Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from src.generation.generator import answer_structured
from src.generation.prompts import format_citation_header
from src.models import RetrievedChunk
from src.retrieval.pipeline import confidence_band
from src.retrieval.reranker import RERANK_THRESHOLD


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Confidence indicator thresholds.
# ---------------------------------------------------------------------------

# Legacy threshold kept for any callers that compare raw logit scores.
CONFIDENCE_HIGH = 3.0
CONFIDENCE_LOW = RERANK_THRESHOLD


def _confidence_label(retrieval_confidence: float, abstained: bool) -> str:
    """Return plain-text confidence label for summary metrics.

    Translates a [0, 1] composite evidence-quality score into a display band.
    The binary abstain gate is unchanged — this function is only called after
    the abstain check passes.
    """
    if abstained:
        return "Abstained"
    band = confidence_band(retrieval_confidence)
    if band == "high":
        return "High confidence"
    if band == "medium":
        return "Medium confidence"
    return "Low confidence"


def _rerank_label(rerank_score: float | None) -> str:
    """Return a human-readable rerank score label for a retrieved chunk.

    Chunks injected via downstream pipeline stages (parent expansion, sibling
    expansion, table-pair expansion, conflict injection, per-issuer floor)
    do not pass through the cross-encoder and have no rerank score. The
    label below avoids enumerating an incomplete set of stage names — that
    enumeration drifts whenever a new stage is added.

    Args:
        rerank_score: The cross-encoder rerank score, or None for chunks
            that bypassed the reranker.

    Returns:
        A formatted score string (e.g. "1.23") for scored chunks, or a
        generic descriptive label for unscored expanded context chunks.
    """
    if rerank_score is None:
        return "unscored expanded context"
    return f"{rerank_score:.2f}"


def _render_sources(contexts: list[RetrievedChunk]) -> None:
    """Per-source expander panel showing every retrieved context."""
    if not contexts:
        st.info("No source excerpts to display.")
        return

    collision_counts: dict[tuple, int] = {}
    for rc in contexts:
        c = rc.chunk
        key = (
            getattr(c, "company", None),
            getattr(c, "doc_type", None),
            getattr(c, "report_date", None),
        )
        collision_counts[key] = collision_counts.get(key, 0) + 1
    colliding_keys = {k for k, n in collision_counts.items() if n >= 2}

    st.subheader(f"Sources ({len(contexts)})")
    for i, rc in enumerate(contexts, start=1):
        c = rc.chunk
        key = (
            getattr(c, "company", None),
            getattr(c, "doc_type", None),
            getattr(c, "report_date", None),
        )
        header = format_citation_header(c, disambiguate=key in colliding_keys)
        rerank = _rerank_label(rc.rerank_score)
        expander_label = f"{i}. {header} · rerank {rerank}"
        with st.expander(expander_label):
            left, right = st.columns(2)
            with left:
                st.caption(f"Company: {c.company} ({c.ticker})")
                st.caption(f"Document: {c.doc_type} · v{c.doc_version}")
                st.caption(f"Page: {c.page_number}")
            with right:
                st.caption(f"Section: {c.section_title or '—'}")
                st.caption(f"Content type: {c.content_type}")
            if c.content_type == "chart_caption":
                st.warning(
                    "This excerpt is a chart caption only — underlying chart "
                    "data was not text-extractable."
                )
            st.write(c.chunk_text)


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Corporate RAG for REIT Presentations",
    page_icon=":material/description:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading retrieval models (~30s on first run)…")
def _warm_models() -> None:
    """Pre-load the embedder and cross-encoder once per process so the first
    user query does not pay the ~30s cold-load latency mid-stream."""
    from src.ingestion.embedder import _get_model
    from src.retrieval.reranker import _get_cross_encoder
    _get_model()
    _get_cross_encoder()


@st.cache_resource(show_spinner=False)
def _startup_checks() -> str | None:
    """Run app-entrypoint checks that require the DB; invoked once per process.

    Returns None on success or a short user-facing error message on failure.
    Wrapped in a try/except so an unexpected import or DB error does not
    abort the Streamlit page render — the UI surfaces the message instead.
    """
    try:
        from src.retrieval.pipeline import check_contextual_activation
        check_contextual_activation()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


_warm_models()
_startup_error = _startup_checks()
if _startup_error:
    st.warning(f"Startup check reported a non-fatal issue: {_startup_error}")


st.title("Corporate RAG for REIT Investor Presentations")
st.caption(
    "Ask a question about the 10 REIT documents (investor presentations plus "
    "one thematic report). Every factual "
    "claim is cited with company, document, date, and page."
)

with st.sidebar:
    st.markdown("**Example queries**")
    example_queries = [
        "What is VICI's total revenue for Q4 2025?",
        "What is Realty Income's dividend yield guidance?",
        "How did DLR's leverage change between December 2025 and March 2026?",
        "What is EGP's occupancy trend over the past 4 quarters?",
        "Which company has the highest leverage?",
        "What is DLR's projected AI inference capacity in GW for 2030?",
        "What is Digital Realty's future development capacity in GW?",
        "What does Simon Property's report say about foot traffic trends?",
        "What is BXP CEO's favorite color?",
        "What is Equinix's portfolio size?",
    ]
    for i, q in enumerate(example_queries):
        if st.button(q, use_container_width=True, key=f"ex_{i}"):
            st.session_state["query"] = q

# Initialise session state
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = ""


def _submit_query() -> None:
    st.session_state["submitted_query"] = st.session_state["query"].strip()


query = st.text_input(
    "Question",
    key="query",
    label_visibility="collapsed",
    placeholder="e.g. What is Digital Realty's net debt to EBITDA?",
    on_change=_submit_query,
)

submitted_query = st.session_state.get("submitted_query", "")
if submitted_query:
    # ------------------------------------------------------------------
    # Run the query and persist the final answer in st.session_state so
    # subsequent reruns (for example, when opening a source expander)
    # retain the output.
    # ------------------------------------------------------------------
    answer_placeholder = st.empty()
    status_placeholder = st.empty()

    with status_placeholder.status("Retrieving and generating…", expanded=False) as status:
        st.write("Classifying query and searching the corpus…")

        # Streaming removed: structured output guarantees schema-compliant typed responses, which is the correct tradeoff for a financial citation system.
        final_answer = answer_structured(submitted_query)

        status.update(label="Done", state="complete", expanded=False)

    st.session_state["last_answer"] = final_answer

    # Clear the placeholder — the persistent block below renders the answer.
    answer_placeholder.empty()
    st.session_state["submitted_query"] = ""


# ---------------------------------------------------------------------------
# Persistent answer display — renders on every script run when an answer is
# available in session_state. This block preserves the answer across
# source-expander reruns (clicking an expander causes a rerun while submit
# state is false).
# ---------------------------------------------------------------------------
final_answer = st.session_state.get("last_answer")
if final_answer is not None:
    with st.container(border=True):
        # Answer text
        if final_answer.abstained:
            st.warning(final_answer.text)
        else:
            st.markdown(final_answer.text)

        # Citation badges from the answer's claim objects
        if not final_answer.abstained and hasattr(final_answer, "citation_report"):
            report = final_answer.citation_report
        else:
            report = None

        # Summary metrics (single place for diagnostics)
        retrieval_conf = final_answer.diagnostics.get("retrieval_confidence", 0.0)
        summary_left, summary_mid, summary_right, summary_last = st.columns(4)
        summary_left.metric("Intent", str(final_answer.intent))
        summary_mid.metric(
            "Contexts",
            str(final_answer.diagnostics.get("after_expansion", 0)),
        )
        summary_right.metric("Confidence", _confidence_label(retrieval_conf, final_answer.abstained))

        # Confidence band caveats — shown inline below the metrics row.
        if not final_answer.abstained:
            _band = confidence_band(retrieval_conf)
            if _band == "medium":
                st.caption(
                    "Retrieved evidence is moderate — verify against source documents "
                    "for critical decisions."
                )
            elif _band == "low":
                st.warning(
                    "Retrieved evidence is weak — treat this answer as a starting "
                    "point for manual review, not a verified fact."
                )

        if report is not None and report.total > 0:
            summary_last.metric(
                "Citations",
                f"{report.supported}/{report.total}",
                f"{len(report.unsupported)} unsupported" if report.unsupported else "all supported",
            )
        else:
            summary_last.metric("Citations", "n/a")

        # Retrieval hops badge — shown when adaptive multi-hop ran.
        retrieval_hops = final_answer.diagnostics.get("retrieval_hops", 0)
        if retrieval_hops and retrieval_hops > 0:
            sub_queries = final_answer.diagnostics.get("sub_queries", [])
            st.info(f"Retrieval hops: {retrieval_hops}")
            if sub_queries:
                with st.expander("Sub-queries fired during adaptive retrieval"):
                    for i, sq in enumerate(sub_queries, start=1):
                        st.caption(f"{i}. {sq}")

        # Value mismatch badge — shown only when numeric issues were detected.
        if report is not None and report.numeric_mismatches:
            st.error("⚠ Value mismatch detected")
            with st.expander("Value mismatch details"):
                st.caption(
                    "The following claimed values could not be verified in their cited source chunks."
                )
                for issue in report.numeric_mismatches:
                    if issue.get("type") == "numeric_mismatch":
                        st.markdown(
                            f"**Claim:** {issue['claim']}  \n"
                            f"**Claimed value:** `{issue['claimed_value']}`  \n"
                            f"**Values found in source:** "
                            f"{', '.join(f'`{v}`' for v in issue['chunk_values_found']) or '(none)'}"
                        )
                    elif issue.get("type") == "unsupported_citation":
                        st.markdown(
                            f"**Claim:** {issue['claim']}  \n"
                            f"**Claimed value:** `{issue['value']}`  \n"
                            "**Source:** cited chunk not found in retrieved context"
                        )
                    st.divider()

        st.caption("Source excerpts")
        _render_sources(final_answer.contexts)
