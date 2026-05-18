"""Tests for contextual retrieval activation checks.

All DB interactions are mocked — no real database or network access required.
Each test completes in well under 2 seconds.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cursor(row: tuple | None) -> MagicMock:
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchone.return_value = row
    return cur


def _make_conn(row: tuple | None) -> MagicMock:
    cur = _make_cursor(row)
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@contextmanager
def _mock_connect(row: tuple | None):
    conn = _make_conn(row)

    @contextmanager
    def _fake_connect():
        yield conn

    with patch("src.db.connect", side_effect=_fake_connect):
        yield conn


# ---------------------------------------------------------------------------
# verify_contextual_retrieval_activated — tests
# ---------------------------------------------------------------------------


class TestVerifyContextualRetrievalActivated:
    def _import_fn(self):
        # Import fresh each time in case module-level state differs
        import scripts.contextualize as ctx  # noqa: PLC0415
        return ctx.verify_contextual_retrieval_activated

    def test_high_population_no_exception(self, capsys):
        """pct=98.0 (>=95) → no exception; confirmation line printed."""
        fn = self._import_fn()
        row = (490, 500, 98.0)  # populated, total, pct_populated

        with patch("scripts.contextualize.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            fn()  # must not raise

    def test_low_population_exits(self):
        """pct=40.0 (<95) → sys.exit called with message containing key text."""
        fn = self._import_fn()
        row = (200, 500, 40.0)

        with patch("scripts.contextualize.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(SystemExit) as exc_info:
                fn()

        msg = str(exc_info.value)
        assert "CONTEXTUAL RETRIEVAL NOT FULLY ACTIVATED" in msg
        assert "40" in msg

    def test_zero_total_no_exception(self):
        """total=0 → warning printed, no exception raised."""
        fn = self._import_fn()
        row = (0, 0, None)  # pct is NULL when total=0

        with patch("scripts.contextualize.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            fn()  # must not raise

    def test_exactly_95_percent_no_exception(self):
        """pct=95.0 exactly (boundary inclusive) → no exception."""
        fn = self._import_fn()
        row = (475, 500, 95.0)

        with patch("scripts.contextualize.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            fn()  # must not raise

    def test_94_9_percent_exits(self):
        """pct=94.9 (just below threshold) → sys.exit."""
        fn = self._import_fn()
        row = (474, 500, 94.9)

        with patch("scripts.contextualize.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(SystemExit):
                fn()


# ---------------------------------------------------------------------------
# check_contextual_activation — tests
# ---------------------------------------------------------------------------


class TestCheckContextualActivationAtStartup:
    def _get_fn(self):
        from src.retrieval import pipeline  # noqa: PLC0415
        return pipeline.check_contextual_activation

    def test_high_population_no_warning(self):
        """pct=98.0 → logger.warning NOT called."""
        fn = self._get_fn()
        row = (490, 500, 98.0)

        with patch("src.retrieval.pipeline.logger") as mock_logger, \
             patch("src.db.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            fn()

        mock_logger.warning.assert_not_called()

    def test_low_population_emits_warning(self):
        """pct=50.0 → logger.warning called with text mentioning '50' and 'COALESCE'."""
        fn = self._get_fn()
        row = (250, 500, 50.0)

        with patch("src.retrieval.pipeline.logger") as mock_logger, \
             patch("src.db.connect") as mock_connect:
            mock_conn = _make_conn(row)
            mock_connect.return_value.__enter__ = lambda s: mock_conn
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            fn()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        # Reconstruct the formatted message from the format string + args
        fmt = call_args[0][0]
        pos_args = call_args[0][1:]
        formatted = fmt % pos_args
        assert "50" in formatted
        assert "COALESCE" in formatted

    def test_db_exception_logged_no_crash(self):
        """DB raises → caught, logged at WARNING, no crash. Both the exception
        class name AND the specific message must appear in the formatted log
        record so operators can diagnose the failure from logs alone."""
        fn = self._get_fn()

        exc_message = "no db: connection refused at 127.0.0.1:5433"
        with patch("src.retrieval.pipeline.logger") as mock_logger, \
             patch("src.db.connect", side_effect=RuntimeError(exc_message)):
            fn()  # must not raise

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        message = call_args.args[0]
        assert "startup check failed" in message.lower(), (
            f"warning template missing 'startup check failed': {message!r}"
        )
        assert "RuntimeError" in str(call_args), (
            f"warning args missing exception class: {call_args!r}"
        )
        assert exc_message in str(call_args), (
            f"warning args missing exception message: {call_args!r}"
        )
        mock_logger.error.assert_not_called()

    def test_module_import_survives_db_error(self):
        """Importing src.retrieval.pipeline must not raise even when connect() raises."""
        import src.retrieval as _retrieval_pkg  # noqa: PLC0415

        mod_name = "src.retrieval.pipeline"
        cached_mod = sys.modules.pop(mod_name, None)
        cached_attr = getattr(_retrieval_pkg, "pipeline", None)
        try:
            with patch("src.db.connect", side_effect=RuntimeError("connection refused")):
                imported = importlib.import_module(mod_name)
            assert imported is not None
        finally:
            # Restore both sys.modules entry and the package attribute so
            # other test modules that hold a reference to the original object
            # (e.g. `from src.retrieval import pipeline as p`) are not affected.
            if cached_mod is not None:
                sys.modules[mod_name] = cached_mod
                _retrieval_pkg.pipeline = cached_mod  # type: ignore[attr-defined]
            else:
                sys.modules.pop(mod_name, None)
                if cached_attr is not None:
                    _retrieval_pkg.pipeline = cached_attr  # type: ignore[attr-defined]
