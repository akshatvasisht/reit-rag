"""Tests for the rerank score label helper in app.py (fix 2h).

Verifies that chunks with rerank_score=None show a descriptive label
instead of the bare "—" that was previously displayed.

Tests import _rerank_label directly — Streamlit is stubbed so no runtime
is needed.
"""
from __future__ import annotations

import contextlib
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
    def _cache_resource(**kw):
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
