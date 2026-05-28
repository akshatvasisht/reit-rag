"""Tests for the intra-document divergence instruction in SYSTEM_PROMPT.

A prompt change cannot be unit-tested via a live LLM call cheaply, so these
tests assert the standing instruction block is present and structurally
covers the two divergence cases it must address:

1. Same metric + same entity + same as-of date but different value
   (intra-document inconsistency, NOT temporal growth) — e.g. two differing
   customer counts on different pages, both footnoted as of the same date.
2. Same metric reported at different scope qualifiers (e.g. total footprint
   vs a narrower geographic scope) — e.g. a total country count vs a
   region-scoped count.

These guard against accidental removal/weakening of the instruction and
against the prompt-assembly paths dropping it.
"""

from __future__ import annotations

from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
)


# ---------------------------------------------------------------------------
# Presence: the standing instruction block exists
# ---------------------------------------------------------------------------


def test_system_prompt_has_divergence_block() -> None:
    """The intra-document divergence header must be present in SYSTEM_PROMPT."""
    assert "INTRA-DOCUMENT DIVERGENCE — MANDATORY:" in SYSTEM_PROMPT


def test_build_system_prompt_carries_divergence_block() -> None:
    """The accessor used by the generator must include the block."""
    assert "INTRA-DOCUMENT DIVERGENCE — MANDATORY:" in build_system_prompt()


# ---------------------------------------------------------------------------
# Coverage: both divergence cases are addressed
# ---------------------------------------------------------------------------


def test_block_covers_same_asof_inconsistency() -> None:
    """Case 1: same as-of date + different value -> flag as inconsistency,
    not temporal change."""
    block = _divergence_block()
    # Must steer away from assuming temporal change when as-of dates match.
    assert "as-of" in block.lower()
    assert "inconsist" in block.lower()
    # Must explicitly caution against assuming growth/decline absent a real
    # as-of date difference.
    lowered = block.lower()
    assert "temporal" in lowered or "period-over-period" in lowered
    assert "genuinely differ" in lowered


def test_block_covers_scope_qualifiers() -> None:
    """Case 2: same metric at different scopes -> surface ALL scopes."""
    block = _divergence_block().lower()
    assert "scope" in block
    # Must instruct surfacing every scope rather than collapsing to one number.
    assert "all scopes" in block


def test_block_has_worked_examples_for_both_cases() -> None:
    """Both worked examples should appear in the block so the instruction style
    matches the rest of the prompt."""
    block = _divergence_block()
    # Case 1 worked example: same as-of date, two differing counts on different pages.
    assert "customers" in block.lower()
    assert "internal inconsistency" in block.lower()
    # Case 2 worked example: same metric at total vs narrower scope.
    assert "countries" in block.lower()
    assert "region" in block.lower()


# ---------------------------------------------------------------------------
# Assembly: the block survives user-message construction wrappers
# ---------------------------------------------------------------------------


def test_user_message_does_not_strip_system_block() -> None:
    """build_user_message operates on the user role and must not duplicate or
    interfere with the system-prompt divergence block; the system prompt
    remains the carrier."""
    # The instruction lives in SYSTEM_PROMPT, not the user message — confirm
    # the user message builder runs cleanly and the system prompt is intact
    # regardless of intent routing.
    for intent in ("latest", "conflict", "all_company_synthesis"):
        msg = build_user_message("How many countries?", [], intent=intent)  # type: ignore[arg-type]
        assert isinstance(msg, str) and msg
    assert "INTRA-DOCUMENT DIVERGENCE — MANDATORY:" in build_system_prompt()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _divergence_block() -> str:
    """Return the divergence instruction block extracted from SYSTEM_PROMPT.

    Extraction stops at the next all-caps MANDATORY/CRITICAL header or the
    submit_answer footer so assertions target only the new block.
    """
    marker = "INTRA-DOCUMENT DIVERGENCE — MANDATORY:"
    start = SYSTEM_PROMPT.index(marker)
    tail = SYSTEM_PROMPT[start:]
    # Cut at the next top-level instruction section that follows the block.
    for terminator in (
        "\nCRITICAL — when the excerpt",
        "\nFOOTNOTE ANCHOR DISAMBIGUATION",
        "\nUse the submit_answer tool",
    ):
        idx = tail.find(terminator)
        if idx != -1:
            tail = tail[:idx]
            break
    return tail
