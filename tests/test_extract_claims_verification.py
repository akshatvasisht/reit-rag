"""Tests for the claim verification pass in scripts/extract_claims.py.

All tests stub the Anthropic Batches API — no real API calls are made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# Allow importing scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_claims import _run_verification_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_result(custom_id: str, verdict: dict) -> MagicMock:
    """Fake a successful batch result with a JSON text block."""
    text_block = MagicMock()
    text_block.text = json.dumps(verdict)

    message = MagicMock()
    message.content = [text_block]

    result_inner = MagicMock()
    result_inner.type = "succeeded"
    result_inner.message = message

    result = MagicMock()
    result.custom_id = custom_id
    result.result = result_inner
    return result


def _make_failed_batch_result(custom_id: str) -> MagicMock:
    """Fake a batch result that did not succeed."""
    result_inner = MagicMock()
    result_inner.type = "errored"

    result = MagicMock()
    result.custom_id = custom_id
    result.result = result_inner
    return result


def _make_client(batch_results: list) -> MagicMock:
    """Build a minimal stub for the Anthropic client's batches interface."""
    batch_obj = MagicMock()
    batch_obj.id = "vbatch_test_001"

    batches = MagicMock()
    batches.create.return_value = batch_obj
    batches.retrieve.return_value = MagicMock(processing_status="ended")
    batches.results.return_value = iter(batch_results)

    messages = MagicMock()
    messages.batches = batches

    client = MagicMock()
    client.messages = messages
    return client


def _make_extracted(chunk_id: str, chunk_meta: dict, claims: list[dict]) -> dict:
    return {"chunk_id": chunk_id, "meta": chunk_meta, "claims": claims}


def _basic_meta(chunk_id: str) -> dict:
    return {
        "id": chunk_id,
        "company": "Acme REIT",
        "doc_version": "2026-03",
        "page_number": 5,
        "chunk_text": "Net debt/EBITDA was 4.9x as of Q4 2025.",
        "content_type": "text",
    }


# ---------------------------------------------------------------------------
# Test: all-yes + qualifier=none → claim written with value_qualifier=NULL
# ---------------------------------------------------------------------------


def test_all_yes_qualifier_none_writes_with_null_qualifier() -> None:
    chunk_id = str(uuid4())
    claim = {"metric": "net_debt_ebitda", "value": "4.9x", "period": "Q4 2025"}
    extracted = [_make_extracted(chunk_id, _basic_meta(chunk_id), [claim])]
    chunk_by_id = {chunk_id: _basic_meta(chunk_id)}

    verdict = {
        "value_in_text": "yes",
        "metric_accurate": "yes",
        "period_correct": "yes",
        "qualifier": "none",
    }
    client = _make_client([_make_batch_result(f"verify:{chunk_id}:0", verdict)])

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    assert rejected == 0
    assert qualified == 0
    assert len(verified) == 1
    assert len(verified[0]["claims"]) == 1
    assert verified[0]["claims"][0]["value_qualifier"] is None


# ---------------------------------------------------------------------------
# Test: all-yes + qualifier=approximate → claim written with value_qualifier set
# ---------------------------------------------------------------------------


def test_all_yes_qualifier_approximate_writes_with_qualifier() -> None:
    chunk_id = str(uuid4())
    claim = {"metric": "net_debt_ebitda", "value": "~4.9x", "period": "Q4 2025"}
    meta = {**_basic_meta(chunk_id), "chunk_text": "Net debt ~4.9x as of Q4 2025."}
    extracted = [_make_extracted(chunk_id, meta, [claim])]
    chunk_by_id = {chunk_id: meta}

    verdict = {
        "value_in_text": "yes",
        "metric_accurate": "yes",
        "period_correct": "yes",
        "qualifier": "approximate",
    }
    client = _make_client([_make_batch_result(f"verify:{chunk_id}:0", verdict)])

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    assert rejected == 0
    assert qualified == 1
    assert len(verified) == 1
    assert verified[0]["claims"][0]["value_qualifier"] == "approximate"


# ---------------------------------------------------------------------------
# Test: any-no → claim NOT written, rejected count incremented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failing_field", ["value_in_text", "metric_accurate", "period_correct"])
def test_any_no_rejects_claim(failing_field: str) -> None:
    chunk_id = str(uuid4())
    claim = {"metric": "net_debt_ebitda", "value": "4.9x", "period": "Q4 2025"}
    extracted = [_make_extracted(chunk_id, _basic_meta(chunk_id), [claim])]
    chunk_by_id = {chunk_id: _basic_meta(chunk_id)}

    verdict: dict[str, str] = {
        "value_in_text": "yes",
        "metric_accurate": "yes",
        "period_correct": "yes",
        "qualifier": "none",
    }
    verdict[failing_field] = "no"
    client = _make_client([_make_batch_result(f"verify:{chunk_id}:0", verdict)])

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    assert rejected == 1
    assert verified == []


# ---------------------------------------------------------------------------
# Test: count accuracy — extract 5, reject 2, qualify 1
# ---------------------------------------------------------------------------


def test_counts_extract5_reject2_qualify1() -> None:
    """5 extracted claims: 3 pass verification (1 with qualifier), 2 rejected."""
    chunk_id = str(uuid4())
    claims = [
        {"metric": f"metric_{i}", "value": f"{i}.0x", "period": "Q4 2025"}
        for i in range(5)
    ]
    extracted = [_make_extracted(chunk_id, _basic_meta(chunk_id), claims)]
    chunk_by_id = {chunk_id: _basic_meta(chunk_id)}

    # Verdicts: 0=pass/none, 1=pass/approximate (qualified), 2=pass/none,
    #           3=fail(value_in_text=no), 4=fail(metric_accurate=no)
    verdicts = [
        {"value_in_text": "yes", "metric_accurate": "yes", "period_correct": "yes", "qualifier": "none"},
        {"value_in_text": "yes", "metric_accurate": "yes", "period_correct": "yes", "qualifier": "approximate"},
        {"value_in_text": "yes", "metric_accurate": "yes", "period_correct": "yes", "qualifier": "none"},
        {"value_in_text": "no",  "metric_accurate": "yes", "period_correct": "yes", "qualifier": "none"},
        {"value_in_text": "yes", "metric_accurate": "no",  "period_correct": "yes", "qualifier": "none"},
    ]
    batch_results = [
        _make_batch_result(f"verify:{chunk_id}:{i}", verdicts[i])
        for i in range(5)
    ]
    client = _make_client(batch_results)

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    claims_verified = sum(len(e["claims"]) for e in verified)

    assert rejected == 2
    assert qualified == 1
    assert claims_verified == 3


# ---------------------------------------------------------------------------
# Test: errored batch result → treated as rejection
# ---------------------------------------------------------------------------


def test_errored_batch_result_counts_as_rejection() -> None:
    chunk_id = str(uuid4())
    claim = {"metric": "net_debt_ebitda", "value": "4.9x", "period": "Q4 2025"}
    extracted = [_make_extracted(chunk_id, _basic_meta(chunk_id), [claim])]
    chunk_by_id = {chunk_id: _basic_meta(chunk_id)}

    client = _make_client([_make_failed_batch_result(f"verify:{chunk_id}:0")])

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    assert rejected == 1
    assert verified == []


# ---------------------------------------------------------------------------
# Test: chart_description chunk — is_chart flag propagated (smoke test)
# ---------------------------------------------------------------------------


def test_chart_description_chunk_passes_verification() -> None:
    chunk_id = str(uuid4())
    claim = {"metric": "occupancy_rate", "value": "94.1%", "period": "Q4 2025"}
    meta = {
        **_basic_meta(chunk_id),
        "content_type": "chart_description",
        "chunk_text": "Chart shows occupancy rate at approximately 94.1% in Q4 2025.",
    }
    extracted = [_make_extracted(chunk_id, meta, [claim])]
    chunk_by_id = {chunk_id: meta}

    verdict = {
        "value_in_text": "yes",
        "metric_accurate": "yes",
        "period_correct": "yes",
        "qualifier": "approximate",
    }
    client = _make_client([_make_batch_result(f"verify:{chunk_id}:0", verdict)])

    verified, rejected, qualified = _run_verification_batch(client, extracted, chunk_by_id)

    assert rejected == 0
    assert qualified == 1
    assert verified[0]["claims"][0]["value_qualifier"] == "approximate"
