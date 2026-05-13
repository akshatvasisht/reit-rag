"""Corporate RAG — Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from src.generation.generator import answer_stream
from src.generation.prompts import format_citation_header
from src.models import RetrievedChunk
from src.retrieval.reranker import RERANK_THRESHOLD


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Confidence indicator thresholds.
# ---------------------------------------------------------------------------

# Heuristic thresholds for ms-marco cross-encoder logit scores:
#   ≥ HIGH      → green  ("Confident — direct evidence retrieved")
#   ≥ THRESHOLD → amber  ("Some evidence; verify before relying on figures")
#   < THRESHOLD → red    ("Abstained — no reliable evidence")
CONFIDENCE_HIGH = 3.0
CONFIDENCE_LOW = RERANK_THRESHOLD


def _confidence_label(top_score: float, abstained: bool) -> str:
    """Return plain-text confidence label for summary metrics."""
    if abstained:
        return "Abstained"
    if top_score >= CONFIDENCE_HIGH:
        return "Confident"
    if top_score >= CONFIDENCE_LOW:
        return "Partial evidence"
    return "Low confidence"


def _render_sources(contexts: list[RetrievedChunk]) -> None:
    """Per-source expander panel showing every retrieved context."""
    if not contexts:
        st.info("No source excerpts to display.")
        return

    st.subheader(f"Sources ({len(contexts)})")
    for i, rc in enumerate(contexts, start=1):
        c = rc.chunk
        header = format_citation_header(c)
        rerank = f"{rc.rerank_score:.2f}" if rc.rerank_score is not None else "—"
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


_warm_models()


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
    # Run the query. Stream tokens into a placeholder and, on completion,
    # persist the final answer in st.session_state so subsequent reruns
    # (for example, when opening a source expander) retain the output.
    # ------------------------------------------------------------------
    answer_placeholder = st.empty()
    status_placeholder = st.empty()

    with status_placeholder.status("Retrieving and generating…", expanded=False) as status:
        st.write("Classifying query and searching the corpus…")

        final_answer = None
        text_chunks: list[str] = []
        abstain_text: str | None = None

        for event in answer_stream(submitted_query):
            kind = event["event"]
            if kind == "retrieval":
                result = event["result"]
            elif kind == "abstain":
                abstain_text = event["text"]
                answer_placeholder.warning(abstain_text)
            elif kind == "delta":
                text_chunks.append(event["text"])
                answer_placeholder.markdown("".join(text_chunks))
            elif kind == "final":
                final_answer = event["answer"]
        status.update(label="Done", state="complete", expanded=False)

    if final_answer is not None:
        st.session_state["last_answer"] = final_answer

    # Clear streaming placeholders — render below from session_state in a
    # uniform layout that survives expander-click reruns.
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

        # Summary metrics (single place for diagnostics)
        top_score = final_answer.diagnostics.get("top_rerank_score", float("-inf"))
        report = final_answer.citation_report
        summary_left, summary_mid, summary_right, summary_last = st.columns(4)
        summary_left.metric("Intent", str(final_answer.intent))
        summary_mid.metric(
            "Contexts",
            str(final_answer.diagnostics.get("after_expansion", 0)),
        )
        summary_right.metric("Confidence", _confidence_label(top_score, final_answer.abstained))
        if report is not None and report.total > 0:
            summary_last.metric(
                "Citations",
                f"{report.supported}/{report.total}",
                f"{len(report.unsupported)} unsupported" if report.unsupported else "all supported",
            )
        else:
            summary_last.metric("Citations", "n/a")

        st.caption("Source excerpts")
        _render_sources(final_answer.contexts)
