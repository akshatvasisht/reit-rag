"""Tests for fetch_embeddings_by_ids — guards the pgvector HalfVector→list path.

The live regression this covers: pgvector returns halfvec columns as HalfVector
objects, not plain iterables. A prior version did `[float(x) for x in emb]`
directly, which passed unit tests using list fixtures but raised
"'HalfVector' object is not iterable" against the real database.
"""
from __future__ import annotations

from src.retrieval.vector import fetch_embeddings_by_ids


class _FakeHalfVector:
    """Stands in for pgvector's HalfVector: not directly iterable; has to_list()."""

    def __init__(self, vals: list[float]) -> None:
        self._vals = vals

    def to_list(self) -> list[float]:
        return self._vals

    def __iter__(self):  # pragma: no cover - mirrors HalfVector being non-iterable
        raise TypeError("'HalfVector' object is not iterable")


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, *args, **kwargs) -> None:
        pass

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)


def test_fetch_converts_halfvector_via_to_list() -> None:
    rows = [("id1", _FakeHalfVector([0.1, 0.2, 0.3]))]
    out = fetch_embeddings_by_ids(["id1"], _FakeConn(rows))  # type: ignore[arg-type]
    assert out["id1"] == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in out["id1"])


def test_fetch_accepts_plain_list_fallback() -> None:
    rows = [("id2", [0.4, 0.5])]
    out = fetch_embeddings_by_ids(["id2"], _FakeConn(rows))  # type: ignore[arg-type]
    assert out["id2"] == [0.4, 0.5]


def test_fetch_skips_none_embeddings() -> None:
    rows = [("id1", None), ("id2", _FakeHalfVector([1.0]))]
    out = fetch_embeddings_by_ids(["id1", "id2"], _FakeConn(rows))  # type: ignore[arg-type]
    assert "id1" not in out
    assert out["id2"] == [1.0]


def test_fetch_empty_ids_returns_empty() -> None:
    assert fetch_embeddings_by_ids([], _FakeConn([])) == {}  # type: ignore[arg-type]
