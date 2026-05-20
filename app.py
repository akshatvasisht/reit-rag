"""Corporate RAG — Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import re

import streamlit as st

from src.generation.generator import answer_structured
from src.generation.prompts import format_citation_header
from src.models import RetrievedChunk
from src.retrieval.pipeline import confidence_band
from src.retrieval.reranker import RERANK_THRESHOLD


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Escape `$` before digits so Streamlit does not parse currency as LaTeX math.
_CURRENCY_DOLLAR_RE = re.compile(r"\$(?=[\d,])")


def _escape_currency_dollars(text: str | None) -> str:
    if not text:
        return text or ""
    return _CURRENCY_DOLLAR_RE.sub(r"\\$", text)


# ---------------------------------------------------------------------------
# Confidence indicator thresholds.
# ---------------------------------------------------------------------------

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


def _render_value_mismatch_block(issues: list[dict]) -> None:
    """Render the numeric-mismatch list as a single red-tinted collapsible
    block. The cover doubles as the warning indicator; expansion reveals the
    per-claim breakdown.
    """
    rows: list[str] = []
    for issue in issues:
        claim = html.escape(str(issue.get("claim", "")))
        if issue.get("type") == "numeric_mismatch":
            claimed = html.escape(str(issue.get("claimed_value", "")))
            chunk_values = issue.get("chunk_values_found") or []
            values_html = (
                ", ".join(f"<code>{html.escape(str(v))}</code>" for v in chunk_values)
                or "<em>(none)</em>"
            )
            rows.append(
                f"<div style='margin-bottom:10px'>"
                f"<div><strong>Claim:</strong> {claim}</div>"
                f"<div><strong>Claimed value:</strong> <code>{claimed}</code></div>"
                f"<div><strong>Values found in source:</strong> {values_html}</div>"
                f"</div>"
            )
        elif issue.get("type") == "unsupported_citation":
            value = html.escape(str(issue.get("value", "")))
            rows.append(
                f"<div style='margin-bottom:10px'>"
                f"<div><strong>Claim:</strong> {claim}</div>"
                f"<div><strong>Claimed value:</strong> <code>{value}</code></div>"
                f"<div><strong>Source:</strong> cited chunk not found in retrieved context</div>"
                f"</div>"
            )

    if not rows:
        return

    body = "<hr style='border:none;border-top:1px solid rgba(255,43,43,0.25);margin:8px 0'>".join(rows)
    st.markdown(
        f"""<details style="background-color:rgba(255,43,43,0.10);border:1px solid rgba(255,43,43,0.35);border-radius:6px;padding:10px 14px;margin:8px 0">
<summary style="cursor:pointer;font-weight:600;color:rgb(255,43,43);list-style:none">⚠ Value mismatch detected — click to inspect</summary>
<div style="margin-top:10px;font-size:0.92em">
<div style="margin-bottom:8px;color:rgba(0,0,0,0.7)">The following claimed values could not be verified in their cited source chunks.</div>
{body}
</div>
</details>""",
        unsafe_allow_html=True,
    )


def _reorder_by_citation_appearance(
    contexts: list[RetrievedChunk], answer_text: str
) -> list[RetrievedChunk]:
    """Reorder contexts so chunks appear in the order their citation header
    first occurs in *answer_text*. Chunks whose citation does not appear
    (parent-, sibling-, or table-pair-expanded context) keep their pipeline
    order and are pinned after the cited block.
    """
    if not answer_text or not contexts:
        return list(contexts)

    collision_counts: dict[tuple, int] = {}
    for rc in contexts:
        c = rc.chunk
        key = (c.company, c.doc_type, c.report_date)
        collision_counts[key] = collision_counts.get(key, 0) + 1
    colliding_keys = {k for k, n in collision_counts.items() if n >= 2}

    ranked: list[tuple[int, int, RetrievedChunk]] = []
    for idx, rc in enumerate(contexts):
        c = rc.chunk
        key = (c.company, c.doc_type, c.report_date)
        header = format_citation_header(c, disambiguate=key in colliding_keys)
        pos = answer_text.find(header)
        ranked.append((pos, idx, rc))

    cited = sorted([r for r in ranked if r[0] >= 0], key=lambda r: (r[0], r[1]))
    uncited = sorted([r for r in ranked if r[0] == -1], key=lambda r: r[1])
    return [rc for _, _, rc in cited + uncited]


def _render_sources(
    contexts: list[RetrievedChunk], *, answer_text: str | None = None
) -> None:
    """Per-source expander panel showing every retrieved context.

    When ``answer_text`` is provided, contexts are reordered so the chunk
    whose citation appears first in the answer comes first in the panel —
    the order a reader expects when scanning the answer top-to-bottom.
    """
    if not contexts:
        st.info("No source excerpts to display.")
        return

    if answer_text is not None:
        contexts = _reorder_by_citation_appearance(contexts, answer_text)

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
                st.caption(f"Ticker: {c.ticker}")
                st.caption(f"Version: {c.doc_version}")
            with right:
                st.caption(f"Section: {c.section_title or '—'}")
                st.caption(f"Content type: {c.content_type}")
            if c.content_type == "chart_caption":
                st.warning(
                    "This excerpt is a chart caption only — underlying chart "
                    "data was not text-extractable."
                )
            st.markdown(_escape_currency_dollars(c.chunk_text))


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
        "What is the status of BXP's 343 Madison Avenue development?",
        "What is BXP's leverage ratio and how does the company plan to deleverage to year-end 2027?",
        "What was Realty Income's full year 2025 AFFO per share?",
        "How did DLR's leverage change between December 2025 and March 2026?",
        "Are there conflicting data points across documents on Digital Realty leverage?",
        "What is BXP's latest occupancy outlook?",
        "What is Realty Income's dividend yield guidance?",
        "What is DLR's projected AI inference capacity in GW for 2030?",
        "Which company has the highest leverage?",
        "What does Simon Property's report say about foot traffic trends?",
        "What is Equinix's portfolio size?",
    ]
    for i, q in enumerate(example_queries):
        if st.button(q, use_container_width=True, key=f"ex_{i}"):
            st.session_state["query"] = q
            st.session_state["submitted_query"] = q

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
    with st.spinner("Classifying query and searching the corpus…"):
        final_answer = answer_structured(submitted_query)

    st.session_state["last_answer"] = final_answer
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
        if final_answer.abstained:
            st.warning(final_answer.text)
        else:
            st.markdown(_escape_currency_dollars(final_answer.text))

        if not final_answer.abstained and hasattr(final_answer, "citation_report"):
            report = final_answer.citation_report
        else:
            report = None

        retrieval_conf = final_answer.diagnostics.get("retrieval_confidence", 0.0)
        summary_left, summary_mid, summary_right = st.columns(3)
        summary_left.metric("Intent", str(final_answer.intent))
        summary_mid.metric("Confidence", _confidence_label(retrieval_conf, final_answer.abstained))

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
            summary_right.metric(
                "Citations",
                f"{report.supported}/{report.total}",
                f"{len(report.unsupported)} unsupported" if report.unsupported else "all supported",
            )
        else:
            summary_right.metric("Citations", "n/a")

        retrieval_hops = final_answer.diagnostics.get("retrieval_hops", 0)
        if retrieval_hops and retrieval_hops > 0:
            sub_queries = final_answer.diagnostics.get("sub_queries", [])
            st.info(f"Retrieval hops: {retrieval_hops}")
            if sub_queries:
                with st.expander("Sub-queries fired during adaptive retrieval"):
                    for i, sq in enumerate(sub_queries, start=1):
                        st.caption(f"{i}. {sq}")

        if report is not None and report.numeric_mismatches:
            _render_value_mismatch_block(report.numeric_mismatches)

        _render_sources(final_answer.contexts, answer_text=final_answer.text)
