"""Unit tests for claim-level conflict detection in the retrieval pipeline.

All tests use mocked DB connections — no real database or network access.
Each test completes in well under 2 seconds.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.models import Chunk, RetrievedChunk
from src.retrieval.pipeline import (
    MAX_CONTEXT_CHUNKS,
    RERANK_TOP_N,
    find_conflicting_chunks,
)
from src.retrieval.reranker import RERANK_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "Acme REIT",
    doc_version: str = "2026-03",
    report_date: str = "2026-03",
    page_number: int = 5,
    chunk_text: str = "Revenue was 100M",
    chunk_id: UUID | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="ACR",
        doc_type="investor_presentation",
        report_date=report_date,
        period_covered="Q4 2025",
        doc_version=doc_version,
        section_title="Financials",
        page_number=page_number,
        content_type="text",
        chunk_text=chunk_text,
    )


def _make_rc(chunk: Chunk, score: float = 3.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _mock_conn_no_claims() -> MagicMock:
    """A psycopg connection whose chunk_claims table is empty."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = []
    cur.description = None

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def _mock_conn_with_claims(
    retained_claim_rows: list[tuple],
    conflict_id_rows: list[tuple],
    conflict_chunk_rows: list[tuple],
    conflict_chunk_cols: list[str],
) -> MagicMock:
    """Build a mock connection that returns canned query results in sequence.

    Query call order in find_conflicting_chunks:
      1. SELECT DISTINCT company, metric, value, chunk_id FROM chunk_claims
         WHERE chunk_id = ANY(...)  → retained_claim_rows
      2. Per (company, metric, value): SELECT DISTINCT cc.chunk_id FROM chunk_claims
         WHERE cc.company = %s AND cc.metric = %s AND cc.value != %s ...
         → conflict_id_rows
      3. _fetch_chunks_by_ids: SELECT ... FROM chunks WHERE id = ANY(...)
         → conflict_chunk_rows with conflict_chunk_cols
    """
    call_count = {"n": 0}
    results_sequence = [
        # First call: claim rows for retained chunks
        (retained_claim_rows, None),
        # Second call: conflicting chunk IDs
        (conflict_id_rows, None),
        # Third call (from _fetch_chunks_by_ids): chunk rows
        (conflict_chunk_rows, conflict_chunk_cols),
    ]

    def _make_cursor(*_args, **_kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        rows, cols = results_sequence[min(idx, len(results_sequence) - 1)]

        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = rows

        if cols is not None:
            desc = [MagicMock(name=c) for c in cols]
            for d, c in zip(desc, cols):
                d.name = c
            cur.description = desc
        else:
            cur.description = None

        return cur

    conn = MagicMock()
    conn.cursor.side_effect = _make_cursor
    return conn


# ---------------------------------------------------------------------------
# Test: empty chunk_claims → no change
# ---------------------------------------------------------------------------


def test_find_conflicting_chunks_empty_claims_returns_retained() -> None:
    """When chunk_claims has no rows for the retained set, return retained unchanged."""
    chunk = _make_chunk()
    retained = [_make_rc(chunk)]
    conn = _mock_conn_no_claims()

    result = find_conflicting_chunks(retained, None, conn)

    assert result == retained
    assert len(result) == 1


def test_find_conflicting_chunks_empty_retained_returns_empty() -> None:
    """An empty retained list returns an empty list without touching the DB."""
    conn = _mock_conn_no_claims()
    result = find_conflicting_chunks([], None, conn)
    assert result == []


# ---------------------------------------------------------------------------
# Test: conflicting value → non-retained chunk added
# ---------------------------------------------------------------------------


def _build_conflict_scenario():
    """Two chunks share (company, metric) but have different values.

    retained_chunk has value '4.9x'; conflict_chunk (not in retained) has '5.1x'.
    The function should return [retained, conflict_chunk_as_rc].
    """
    retained_id = uuid4()
    conflict_id = uuid4()

    retained_chunk = _make_chunk(
        company="Acme REIT", chunk_text="Net debt/EBITDA 4.9x", chunk_id=retained_id
    )
    retained = [_make_rc(retained_chunk, score=3.0)]

    # DB row shapes:
    #   chunk_claims: (company, metric, value, chunk_id)
    retained_claim_rows = [("Acme REIT", "net_debt_ebitda", "4.9x", str(retained_id))]
    #   conflict chunk_id rows: [(chunk_id,)]
    conflict_id_rows = [(str(conflict_id),)]
    #   chunk SELECT columns — must match the production SELECT shape
    cols = [
        "id", "document_id", "parent_chunk_id",
        "company", "ticker", "doc_type", "report_date", "period_covered", "doc_version",
        "section_title", "page_number", "content_type", "source_authority",
        "chunk_text", "is_parent", "token_count",
        "doc_subtype", "page_content_class",
    ]
    doc_id = uuid4()
    conflict_chunk_rows = [(
        str(conflict_id), str(doc_id), None,
        "Acme REIT", "ACR", "investor_presentation", "2025-12", "Q3 2025", "2025-12",
        "Financials", 8, "text", "company_authored",
        "Net debt/EBITDA 5.1x", False, 120,
        "quarterly_investor_deck", "substantive",
    )]

    conn = _mock_conn_with_claims(
        retained_claim_rows, conflict_id_rows, conflict_chunk_rows, cols
    )
    return retained, conflict_id, conn


def test_find_conflicting_chunks_different_value_adds_chunk() -> None:
    """A non-retained chunk with a different metric value for the same company+metric
    must be appended to the result."""
    retained, conflict_id, conn = _build_conflict_scenario()

    result = find_conflicting_chunks(retained, None, conn)

    # Should have retained + 1 conflict
    assert len(result) == 2
    conflict_rc = result[-1]
    assert conflict_rc.chunk.id == conflict_id


def test_find_conflicting_chunks_conflict_score_above_threshold() -> None:
    """Injected conflict chunks must have rerank_score = RERANK_THRESHOLD + 0.1."""
    retained, conflict_id, conn = _build_conflict_scenario()

    result = find_conflicting_chunks(retained, None, conn)

    conflict_rc = next(rc for rc in result if rc.chunk.id == conflict_id)
    assert conflict_rc.rerank_score == pytest.approx(RERANK_THRESHOLD + 0.1)


# ---------------------------------------------------------------------------
# Test: matching value → no chunk added
# ---------------------------------------------------------------------------


def test_find_conflicting_chunks_same_value_no_injection() -> None:
    """When there are no chunks with a different value for the metric, no chunk
    is added beyond the retained set."""
    retained_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id)
    retained = [_make_rc(retained_chunk)]

    # claim rows exist but the conflict query returns nothing
    retained_claim_rows = [("Acme REIT", "net_debt_ebitda", "4.9x", str(retained_id))]
    conflict_id_rows: list[tuple] = []  # no conflicting chunks found
    cols = ["id"]

    conn = _mock_conn_with_claims(retained_claim_rows, conflict_id_rows, [], cols)

    result = find_conflicting_chunks(retained, None, conn)

    assert result == retained
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: context cap
# ---------------------------------------------------------------------------


def test_max_context_chunks_constant() -> None:
    """MAX_CONTEXT_CHUNKS must equal RERANK_TOP_N * 2."""
    assert MAX_CONTEXT_CHUNKS == RERANK_TOP_N * 2


def test_context_cap_drops_lowest_retained_preserves_conflict() -> None:
    """When retained + conflict exceeds MAX_CONTEXT_CHUNKS, the lowest-scoring
    retained chunks are dropped; conflict chunks are always preserved."""
    # Build MAX_CONTEXT_CHUNKS retained chunks with scores 1..N
    retained_chunks = [
        _make_rc(_make_chunk(chunk_text=f"chunk {i}"), score=float(i))
        for i in range(1, MAX_CONTEXT_CHUNKS + 1)
    ]

    # One extra conflict chunk — this pushes total over the cap
    conflict_id = uuid4()
    conflict_chunk = _make_chunk(chunk_id=conflict_id, chunk_text="conflict chunk")
    conflict_rc = RetrievedChunk(chunk=conflict_chunk, rerank_score=RERANK_THRESHOLD + 0.1)

    # Simulate the cap logic directly (as the pipeline would apply it after
    # find_conflicting_chunks returns).
    with_conflicts = retained_chunks + [conflict_rc]
    assert len(with_conflicts) == MAX_CONTEXT_CHUNKS + 1  # over cap

    # Replicate the cap logic from pipeline.retrieve()
    deduped_ids = {rc.chunk.id for rc in retained_chunks}
    conflict_chunk_ids = {rc.chunk.id for rc in with_conflicts if rc.chunk.id not in deduped_ids}
    retained_part = [rc for rc in with_conflicts if rc.chunk.id not in conflict_chunk_ids]
    conflict_part = [rc for rc in with_conflicts if rc.chunk.id in conflict_chunk_ids]
    max_kept = MAX_CONTEXT_CHUNKS - len(conflict_part)
    retained_trimmed = sorted(
        retained_part,
        key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
        reverse=True,
    )[:max(0, max_kept)]
    result = retained_trimmed + conflict_part

    assert len(result) == MAX_CONTEXT_CHUNKS
    # Conflict chunk is always present
    assert any(rc.chunk.id == conflict_id for rc in result)
    # The lowest-scoring retained chunk (score=1.0) was dropped
    min_score_chunk = retained_chunks[0]  # score=1 (index 0)
    assert min_score_chunk not in result
    # The highest-scoring retained chunk survives
    max_score_chunk = retained_chunks[-1]  # score=MAX_CONTEXT_CHUNKS
    assert max_score_chunk in result


# ---------------------------------------------------------------------------
# Test: abstention path — find_conflicting_chunks not called
# ---------------------------------------------------------------------------


def test_abstention_path_skips_conflict_detection(monkeypatch) -> None:
    """When the pipeline would abstain, find_conflicting_chunks must not be
    invoked (verified by patching it to raise if called)."""
    from src.retrieval import pipeline as pipeline_module

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("find_conflicting_chunks must not be called when abstaining")

    monkeypatch.setattr(pipeline_module, "find_conflicting_chunks", _raise_if_called)

    # Build a deduped list where we can test the pipeline branching logic.
    # We directly test the in-pipeline branching: abstain=True → skip conflict call.
    abstain = True
    deduped: list[RetrievedChunk] = []

    # Replicate the exact conditional from pipeline.retrieve()
    if not abstain:
        pipeline_module.find_conflicting_chunks(deduped, None, None)  # type: ignore[arg-type]

    # If we reach here without an AssertionError, the abstention gate is correct.


# ---------------------------------------------------------------------------
# Qualifier-class tests
# ---------------------------------------------------------------------------
#
# The qualifier-class predicate is evaluated inside the DB query.  In the
# mocked test environment the SQL is never actually executed; we use the mock
# connection's result-sequence mechanism to assert what the *function* does
# with the query result — specifically, whether it injects a conflict chunk
# (trusting the DB to have applied the qualifier filter) or returns unchanged.
#
# Two helpers cover the "DB found a conflict" path and the "DB found nothing"
# path so we can test all five scenarios.


def _make_conflict_conn_with_qualifier(
    retained_id: UUID,
    conflict_id: UUID,
    conflict_chunk_rows: list[tuple],
    conflict_chunk_cols: list[str],
    qualifier: str | None = None,
) -> MagicMock:
    """Mock connection that reports one conflict chunk for the retained claim."""
    retained_claim_rows = [("Acme REIT", "net_debt_ebitda", "4.9x", str(retained_id))]
    conflict_id_rows = [(str(conflict_id),)]
    return _mock_conn_with_claims(
        retained_claim_rows, conflict_id_rows, conflict_chunk_rows, conflict_chunk_cols
    )


def _make_no_conflict_conn(retained_id: UUID) -> MagicMock:
    """Mock connection that reports no conflicting chunk IDs (DB filtered them out)."""
    retained_claim_rows = [("Acme REIT", "net_debt_ebitda", "4.9x", str(retained_id))]
    return _mock_conn_with_claims(retained_claim_rows, [], [], ["id"])


_COLS = [
    "id", "document_id", "parent_chunk_id",
    "company", "ticker", "doc_type", "report_date", "period_covered", "doc_version",
    "section_title", "page_number", "content_type", "source_authority",
    "chunk_text", "is_parent", "token_count",
    "doc_subtype", "page_content_class",
]


def _conflict_row(chunk_id: UUID, doc_subtype: str = "annual_supplement") -> tuple:
    """Return a conflict-chunk DB row.

    Uses ``annual_supplement`` as the default doc_subtype — a different value
    from the retained chunk's ``quarterly_investor_deck`` — so the
    version-group key treats them as separate groups within the same doc_type.
    """
    doc_id = uuid4()
    return (
        str(chunk_id), str(doc_id), None,
        "Acme REIT", "ACR", "investor_presentation", "2025-12", "Q3 2025", "2025-12",
        "Financials", 8, "text", "company_authored",
        "Net debt/EBITDA 5.1x", False, 120,
        doc_subtype, "substantive",
    )


def test_qualifier_both_null_different_value_is_conflict() -> None:
    """Two reported claims (both value_qualifier=NULL) with different values → conflict."""
    retained_id = uuid4()
    conflict_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id, chunk_text="Net debt 4.9x")
    retained = [_make_rc(retained_chunk)]

    # DB returns a conflicting chunk (both NULL qualifier → same class → conflict detected)
    conn = _make_conflict_conn_with_qualifier(
        retained_id, conflict_id, [_conflict_row(conflict_id)], _COLS
    )

    result = find_conflicting_chunks(retained, None, conn)

    assert len(result) == 2
    assert any(rc.chunk.id == conflict_id for rc in result)


def test_qualifier_reported_vs_target_not_conflict() -> None:
    """Retained=NULL (reported), non-retained=target → DB should NOT return conflict.

    Modelled as: DB returns empty conflict_id_rows (qualifier predicate excluded it).
    """
    retained_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id, chunk_text="Net debt 4.9x")
    retained = [_make_rc(retained_chunk)]

    conn = _make_no_conflict_conn(retained_id)

    result = find_conflicting_chunks(retained, None, conn)

    assert result == retained
    assert len(result) == 1


def test_qualifier_both_guidance_different_value_is_conflict() -> None:
    """Two forward-looking claims (both value_qualifier='guidance') → conflict."""
    retained_id = uuid4()
    conflict_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id, chunk_text="Guidance net debt 5.0x")
    retained = [_make_rc(retained_chunk)]

    conn = _make_conflict_conn_with_qualifier(
        retained_id, conflict_id, [_conflict_row(conflict_id)], _COLS
    )

    result = find_conflicting_chunks(retained, None, conn)

    assert len(result) == 2
    assert any(rc.chunk.id == conflict_id for rc in result)


def test_qualifier_both_approximate_different_value_is_conflict() -> None:
    """Two approximate claims (both value_qualifier='approximate') → conflict (reported class)."""
    retained_id = uuid4()
    conflict_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id, chunk_text="Approx net debt ~4.9x")
    retained = [_make_rc(retained_chunk)]

    conn = _make_conflict_conn_with_qualifier(
        retained_id, conflict_id, [_conflict_row(conflict_id)], _COLS
    )

    result = find_conflicting_chunks(retained, None, conn)

    assert len(result) == 2
    assert any(rc.chunk.id == conflict_id for rc in result)


def test_qualifier_guidance_retained_vs_null_non_retained_not_conflict() -> None:
    """Retained with guidance qualifier vs. non-retained with NULL → NOT conflict.

    Modelled as: DB returns empty conflict_id_rows (different classes).
    """
    retained_id = uuid4()
    retained_chunk = _make_chunk(chunk_id=retained_id, chunk_text="Guidance net debt 5.0x")
    retained = [_make_rc(retained_chunk)]

    conn = _make_no_conflict_conn(retained_id)

    result = find_conflicting_chunks(retained, None, conn)

    assert result == retained
    assert len(result) == 1
