"""Tests for the batch-state persistence in scripts/contextualize.py.

Without persistence, a Ctrl-C between Batch API submit and result-write loses
the batch handle and the next run re-submits (and re-bills) the same chunks
while the original batch continues processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import contextualize as ctx


@pytest.fixture
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the module-level _STATE_FILE to a temp path so tests do not
    touch the real data/ directory."""
    fake = tmp_path / "data" / ".contextualize_state.json"
    monkeypatch.setattr(ctx, "_STATE_FILE", fake)
    return fake


def test_save_and_load_round_trip(state_file: Path) -> None:
    """_save_state writes JSON with doc_id and batch_id; _load_state reads it back."""
    ctx._save_state("doc-uuid-1", "batch-abc")
    assert state_file.exists()

    payload = json.loads(state_file.read_text())
    assert payload == {"doc_id": "doc-uuid-1", "batch_id": "batch-abc"}

    assert ctx._load_state() == payload


def test_load_state_missing_returns_none(state_file: Path) -> None:
    """When no sidecar exists, _load_state returns None (not raises)."""
    assert not state_file.exists()
    assert ctx._load_state() is None


def test_load_state_corrupt_returns_none(state_file: Path) -> None:
    """Malformed JSON does not raise — _load_state returns None and logs."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("{not valid json")
    assert ctx._load_state() is None


def test_clear_state_removes_file(state_file: Path) -> None:
    """After clear, the sidecar is gone and load returns None."""
    ctx._save_state("doc-uuid-2", "batch-def")
    assert state_file.exists()
    ctx._clear_state()
    assert not state_file.exists()
    assert ctx._load_state() is None


def test_clear_state_no_file_is_noop(state_file: Path) -> None:
    """Clearing when no sidecar exists is safe."""
    assert not state_file.exists()
    ctx._clear_state()  # must not raise


def test_resume_no_state_returns_immediately(state_file: Path) -> None:
    """Without a sidecar, _resume_in_flight_batch is a no-op — does not call
    the Anthropic client or the embedder."""
    with patch.object(ctx, "_get_embedder") as embedder, \
         patch.object(ctx, "_poll_until_done") as poll, \
         patch.object(ctx, "_write_results") as writer:
        ctx._resume_in_flight_batch(client=object())
    embedder.assert_not_called()
    poll.assert_not_called()
    writer.assert_not_called()


def test_save_state_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_save_state creates the parent directory when it does not exist (the
    `data/` dir might be absent on a very fresh clone)."""
    fake = tmp_path / "nested" / "dir" / ".state.json"
    monkeypatch.setattr(ctx, "_STATE_FILE", fake)
    assert not fake.parent.exists()
    ctx._save_state("doc-uuid-3", "batch-ghi")
    assert fake.exists()
    assert json.loads(fake.read_text())["batch_id"] == "batch-ghi"
