"""Tests for the peer-set-completeness and cross-document-complementarity
instruction blocks in SYSTEM_PROMPT.

A prompt change cannot be unit-tested via a live LLM call cheaply, so these
tests assert the standing instruction blocks are present and structurally cover
the cases they must address:

1. Peer-set completeness — when summarizing a chart/ranking/peer table, name
   every entity present and flag when the peer set varies across metrics
   (e.g. a peer table where the lowest-ranked operator must not be dropped).
2. Cross-document complementarity — when evidence for one company spans
   multiple deck subtypes, draw complementary figures from each rather than
   answering from a single source (e.g. positioning supported by figures from
   both an Investor Day Session and a Quarterly Investor Deck).

These guard against accidental removal/weakening of the instructions and
against the prompt-assembly paths dropping them.
"""

from __future__ import annotations

from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
)


# ---------------------------------------------------------------------------
# Presence: the standing instruction blocks exist
# ---------------------------------------------------------------------------


def test_system_prompt_has_peer_completeness_block() -> None:
    assert "PEER-SET COMPLETENESS — MANDATORY:" in SYSTEM_PROMPT


def test_system_prompt_has_cross_document_block() -> None:
    assert "CROSS-DOCUMENT COMPLEMENTARITY — MANDATORY:" in SYSTEM_PROMPT


def test_build_system_prompt_carries_both_blocks() -> None:
    """The accessor used by the generator must include both blocks."""
    prompt = build_system_prompt()
    assert "PEER-SET COMPLETENESS — MANDATORY:" in prompt
    assert "CROSS-DOCUMENT COMPLEMENTARITY — MANDATORY:" in prompt


# ---------------------------------------------------------------------------
# Coverage: each block addresses its case
# ---------------------------------------------------------------------------


def test_peer_block_demands_full_coverage_and_varying_sets() -> None:
    """Name every entity; flag when the peer set differs across metrics."""
    block = _block(
        "PEER-SET COMPLETENESS — MANDATORY:",
        "\nCROSS-DOCUMENT COMPLEMENTARITY",
    )
    lowered = block.lower()
    assert "every entity" in lowered
    # Must steer away from dropping low-ranked peers.
    assert "rank" in lowered
    # Must address the peer set varying across metrics.
    assert "differs across metrics" in lowered
    # Worked example demands naming the lowest-ranked peer, not just the top.
    assert "lowest-ranked" in block.lower()


def test_cross_document_block_demands_complementary_synthesis() -> None:
    """Draw complementary figures from each subtype; attribute each."""
    block = _block(
        "CROSS-DOCUMENT COMPLEMENTARITY — MANDATORY:",
        "\nUse the submit_answer tool",
    )
    lowered = block.lower()
    assert "subtype" in lowered
    assert "complementary" in lowered
    # Must instruct using each source rather than only the top-ranked one.
    assert "rather than answering from only one" in lowered
    # Must not treat distinct-metric subtypes as a conflict.
    assert "not in conflict" in lowered


# ---------------------------------------------------------------------------
# Assembly: the blocks survive user-message construction wrappers
# ---------------------------------------------------------------------------


def test_user_message_does_not_strip_system_blocks() -> None:
    """build_user_message operates on the user role and must not interfere with
    the system-prompt blocks; the system prompt remains the carrier."""
    for intent in ("latest", "conflict", "all_company_synthesis"):
        msg = build_user_message("Compare the peers.", [], intent=intent)  # type: ignore[arg-type]
        assert isinstance(msg, str) and msg
    prompt = build_system_prompt()
    assert "PEER-SET COMPLETENESS — MANDATORY:" in prompt
    assert "CROSS-DOCUMENT COMPLEMENTARITY — MANDATORY:" in prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(marker: str, terminator: str) -> str:
    """Return the instruction block beginning at ``marker`` up to ``terminator``."""
    start = SYSTEM_PROMPT.index(marker)
    tail = SYSTEM_PROMPT[start:]
    idx = tail.find(terminator)
    if idx != -1:
        tail = tail[:idx]
    return tail
