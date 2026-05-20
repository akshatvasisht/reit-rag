"""Tests that vector_search issues SET hnsw.ef_search before ANN queries.

All tests use mocked connections — no real database or embedder access.
Each test completes in well under 2 seconds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from src.retrieval.vector import HNSW_EF_SEARCH, vector_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conn(rows=None, cols=None):
    """Return a mock psycopg connection whose cursor records executed statements."""
    if rows is None:
        rows = []
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = rows
    if cols:
        col_desc = [MagicMock(name=c) for c in cols]
        for desc, col_name in zip(col_desc, cols):
            desc.name = col_name
        mock_cursor.description = col_desc
    else:
        mock_cursor.description = None

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHnswEfSearchConstant:
    def test_constant_value(self):
        assert HNSW_EF_SEARCH == 200


class TestVectorSearchSetsEfSearch:
    """vector_search must issue SET hnsw.ef_search = 200 before the SELECT."""

    def _run_vector_search(self, companies=None, content_types=None):
        mock_conn, mock_cursor = _make_mock_conn(rows=[], cols=None)
        with patch("src.retrieval.vector.embed_query", return_value=[0.0] * 1024):
            vector_search("test query", limit=5, conn=mock_conn,
                          companies=companies, content_types=content_types)
        return mock_cursor

    def _get_executed_sql(self, mock_cursor) -> list[str]:
        """Extract the SQL string from each cur.execute() call."""
        stmts = []
        for c in mock_cursor.execute.call_args_list:
            args = c[0]
            stmts.append(args[0] if args else "")
        return stmts

    def test_ef_search_set_before_select_no_filter(self):
        cur = self._run_vector_search()
        stmts = self._get_executed_sql(cur)
        assert len(stmts) >= 2, "Expected at least SET + SELECT"
        assert stmts[0] == f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"

    def test_ef_search_set_before_select_with_company_filter(self):
        cur = self._run_vector_search(companies=["Acme Corp"])
        stmts = self._get_executed_sql(cur)
        assert stmts[0] == f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"

    def test_ef_search_set_before_select_with_content_type_filter(self):
        cur = self._run_vector_search(content_types=["text"])
        stmts = self._get_executed_sql(cur)
        assert stmts[0] == f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"

    def test_ef_search_set_before_select_with_both_filters(self):
        cur = self._run_vector_search(companies=["Acme Corp"], content_types=["text"])
        stmts = self._get_executed_sql(cur)
        assert stmts[0] == f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"

    def test_set_statement_uses_module_constant(self):
        """The emitted SET value must match HNSW_EF_SEARCH, not a hardcoded literal."""
        cur = self._run_vector_search()
        stmts = self._get_executed_sql(cur)
        expected = f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"
        assert expected in stmts

    def test_set_appears_exactly_once_per_call(self):
        cur = self._run_vector_search()
        stmts = self._get_executed_sql(cur)
        set_stmts = [s for s in stmts if "hnsw.ef_search" in s]
        assert len(set_stmts) == 1

    def test_select_comes_after_set(self):
        cur = self._run_vector_search()
        stmts = self._get_executed_sql(cur)
        set_idx = next(i for i, s in enumerate(stmts) if "hnsw.ef_search" in s)
        select_idx = next(i for i, s in enumerate(stmts) if "SELECT" in s.upper())
        assert set_idx < select_idx, "SET must precede the SELECT"
