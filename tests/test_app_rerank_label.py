"""Tests for the rerank score label helper in app.py.

Verifies that chunks with rerank_score=None show a descriptive label
instead of the bare "—" that was previously displayed.

Tests import _rerank_label directly — Streamlit is stubbed so no runtime
is needed.
"""
from __future__ import annotations

import sys
import types


def _install_streamlit_stub() -> None:
    """Install a minimal Streamlit stub so app.py can be imported in tests.

    Idempotent — safe to call multiple times; only installs once.
    """
    if "streamlit" in sys.modules:
        return

    st = types.ModuleType("streamlit")

    # --- Decorator stubs ---
    # Streamlit's @st.cache_resource is used both bare (decorates the
    # function directly) and parameterized (@st.cache_resource(...)).
    # Support both forms so a future shift between the two doesn't
    # break this test harness silently.
    def _cache_resource(*args, **kw):
        if len(args) == 1 and callable(args[0]) and not kw:
            return args[0]
        return lambda f: f

    st.cache_resource = _cache_resource  # type: ignore[attr-defined]

    # --- Context-manager stub (used by st.sidebar, st.container, etc.) ---
    class _CMStub:
        """A no-op context manager that also acts as a callable."""
        def __call__(self, *a, **kw):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        # Streamlit columns returns a list of column objects.
        def __iter__(self):
            return iter([self, self])

    _cm = _CMStub()
    st.sidebar = _cm  # type: ignore[attr-defined]
    st.container = _cm  # type: ignore[attr-defined]
    st.expander = _cm  # type: ignore[attr-defined]
    st.status = _cm  # type: ignore[attr-defined]

    def _columns(n, **kw):
        return [_CMStub() for _ in range(n if isinstance(n, int) else len(n))]

    st.columns = _columns  # type: ignore[attr-defined]

    # --- Plain callable no-ops ---
    _noop = lambda *a, **kw: None  # noqa: E731
    for attr in (
        "set_page_config", "title", "caption", "info", "subheader",
        "write", "warning", "error", "markdown", "divider",
        "button", "text_input", "empty", "metric",
    ):
        setattr(st, attr, _noop)

    # --- Session state ---
    st.session_state = {}  # type: ignore[attr-defined]

    sys.modules["streamlit"] = st


def _get_rerank_label():
    """Return the _rerank_label function from app.py with Streamlit stubbed."""
    _install_streamlit_stub()
    # Remove cached app so we always pick up the current version.
    sys.modules.pop("app", None)
    import app  # noqa: PLC0415
    return app._rerank_label


def _get_reorder_fn():
    """Return the _reorder_by_citation_appearance function from app.py."""
    _install_streamlit_stub()
    sys.modules.pop("app", None)
    import app  # noqa: PLC0415
    return app._reorder_by_citation_appearance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scored_chunk_shows_formatted_score():
    fn = _get_rerank_label()
    assert fn(1.23) == "1.23"


def test_none_score_is_not_bare_dash():
    fn = _get_rerank_label()
    assert fn(None) != "—"


def test_none_score_contains_expanded_keyword():
    fn = _get_rerank_label()
    label = fn(None)
    keywords = ["expanded", "unscored", "parent", "sibling", "conflict"]
    assert any(kw in label.lower() for kw in keywords), (
        f"Expected expansion context keyword in label, got: '{label}'"
    )


def test_zero_score_shows_zero_not_expanded_label():
    fn = _get_rerank_label()
    label = fn(0.0)
    assert "0.00" in label
    assert "expanded" not in label.lower()


def test_negative_score_shows_value():
    fn = _get_rerank_label()
    label = fn(-2.50)
    assert "-2.50" in label


# ---------------------------------------------------------------------------
# _reorder_by_citation_appearance — citation-order source panel
# ---------------------------------------------------------------------------


def _make_rc(
    company: str = "Digital Realty",
    doc_type: str = "investor_presentation",
    report_date: str = "2026-03",
    page_number: int = 5,
    rerank_score: float | None = 1.0,
):
    """Build a minimal RetrievedChunk for reorder tests."""
    from uuid import uuid4
    from src.models import Chunk, RetrievedChunk  # noqa: PLC0415
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="X",
        doc_type=doc_type,
        report_date=report_date,
        doc_version=report_date,
        chunk_text="body",
        content_type="text",
        page_number=page_number,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=rerank_score)


def test_reorder_moves_cited_chunks_to_their_appearance_order():
    fn = _get_reorder_fn()
    rc_p5 = _make_rc(page_number=5, report_date="2026-03")
    rc_p10 = _make_rc(page_number=10, report_date="2026-02")
    rc_p3 = _make_rc(page_number=3, report_date="2026-01")
    answer = (
        "Leverage was 4.9x [Digital Realty, Investor Presentation, February 2026, p.10]. "
        "Earlier coverage cited 5.1x [Digital Realty, Investor Presentation, January 2026, p.3]."
    )
    ordered = fn([rc_p5, rc_p10, rc_p3], answer)
    assert [rc.chunk.page_number for rc in ordered] == [10, 3, 5]


def test_reorder_pins_uncited_chunks_after_cited():
    fn = _get_reorder_fn()
    cited = _make_rc(page_number=7, report_date="2026-03")
    parent = _make_rc(page_number=99, report_date="2026-02", rerank_score=None)
    sibling = _make_rc(page_number=100, report_date="2026-01", rerank_score=None)
    answer = "Detail [Digital Realty, Investor Presentation, March 2026, p.7]."
    ordered = fn([parent, sibling, cited], answer)
    assert ordered[0].chunk.page_number == 7
    assert ordered[1].chunk.page_number == 99
    assert ordered[2].chunk.page_number == 100


def test_reorder_no_answer_text_returns_pipeline_order():
    fn = _get_reorder_fn()
    rc_a = _make_rc(page_number=1)
    rc_b = _make_rc(page_number=2)
    ordered = fn([rc_a, rc_b], "")
    assert ordered == [rc_a, rc_b]


def test_reorder_disambiguates_collision_with_subtype():
    """When two chunks share (company, doc_type, report_date), the citation
    header includes the subtype; the reorder must match that extended form."""
    fn = _get_reorder_fn()
    from uuid import uuid4
    from src.models import Chunk, RetrievedChunk  # noqa: PLC0415
    def _make(subtype: str, page: int):
        return RetrievedChunk(chunk=Chunk(
            id=uuid4(), document_id=uuid4(), company="BXP", ticker="BXP",
            doc_type="investor_presentation", report_date="2025-12",
            doc_version="2025-12", chunk_text="body", content_type="text",
            page_number=page, doc_subtype=subtype,
        ), rerank_score=1.0)
    quarterly = _make("quarterly_investor_deck", page=4)
    investor_day = _make("investor_day_session", page=12)
    answer = (
        "Occupancy was 88% [BXP, Investor Presentation (Investor Day Session), "
        "December 2025, p.12]. Q4 deck reported 87% [BXP, Investor Presentation "
        "(Quarterly Investor Deck), December 2025, p.4]."
    )
    ordered = fn([quarterly, investor_day], answer)
    assert ordered[0].chunk.doc_subtype == "investor_day_session"
    assert ordered[1].chunk.doc_subtype == "quarterly_investor_deck"
