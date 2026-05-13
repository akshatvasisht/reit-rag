from __future__ import annotations

import pytest

from src import db


def test_connect_fails_fast_when_database_url_missing(monkeypatch) -> None:
    monkeypatch.setattr(db, "DATABASE_URL", None)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        with db.connect():
            pass
