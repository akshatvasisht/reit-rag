"""Structural integrity checks for the evaluation set."""

from __future__ import annotations

from typing import get_args

from scripts import evaluate
from src.generation.generator import Answer
from src.models import StructuredAnswer
from tests.evaluation_set import (
    EVALUATION_SET,
    ExpectedIntent,
    EvaluationQuery,
    PassCategory,
)


# ---------------------------------------------------------------------------
# Evaluation set structural checks
# ---------------------------------------------------------------------------


def test_golden_set_count() -> None:
    assert len(EVALUATION_SET) == 28, (
        f"Expected 28 entries (g01, g02, g04-g25, g26-g29); got {len(EVALUATION_SET)}"
    )


def test_new_query_ids_present() -> None:
    ids = {gq.id for gq in EVALUATION_SET}
    for expected_id in ("g19", "g20", "g21"):
        assert expected_id in ids, f"{expected_id} not found in EVALUATION_SET"


def test_new_query_notes_contain_verification_statement() -> None:
    for gq in EVALUATION_SET:
        if gq.id in ("g19", "g20", "g21"):
            lower = gq.notes.lower()
            assert any(word in lower for word in ("verified", "found", "absent", "not present")), (
                f"{gq.id}: notes field must contain a verification statement; got: {gq.notes!r}"
            )


def test_unique_ids() -> None:
    ids = [gq.id for gq in EVALUATION_SET]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {sorted(ids)}"


def test_all_categories_are_valid() -> None:
    valid = set(get_args(PassCategory))
    for gq in EVALUATION_SET:
        assert gq.category in valid, (
            f"{gq.id}: category '{gq.category}' is not in PassCategory"
        )


def test_all_intents_are_valid() -> None:
    valid = set(get_args(ExpectedIntent))
    for gq in EVALUATION_SET:
        if gq.expected_intent is not None:
            assert gq.expected_intent in valid, (
                f"{gq.id}: expected_intent '{gq.expected_intent}' is not in ExpectedIntent"
            )


def test_expect_both_dlr_versions_only_on_g02_and_g16() -> None:
    dlr_ids = {gq.id for gq in EVALUATION_SET if gq.expect_both_dlr_versions}
    assert dlr_ids == {"g02", "g16"}, (
        f"expect_both_dlr_versions=True on unexpected entries: {dlr_ids}"
    )


def test_expect_min_companies_in_context_set_on_correct_queries() -> None:
    min_company_ids = {gq.id for gq in EVALUATION_SET if gq.expect_min_companies_in_context is not None}
    assert min_company_ids == {"g04", "g07", "g17", "g29"}, (
        f"expect_min_companies_in_context set on unexpected entries: {min_company_ids}"
    )


def test_expect_forward_looking_on_g24() -> None:
    g24 = next(gq for gq in EVALUATION_SET if gq.id == "g24")
    assert g24.expect_forward_looking is True


def test_expect_forward_looking_false_for_all_others() -> None:
    for gq in EVALUATION_SET:
        if gq.id != "g24":
            assert gq.expect_forward_looking is False, (
                f"{gq.id}: expect_forward_looking should be False"
            )


# ---------------------------------------------------------------------------
# forward_looking_check auto-check unit tests
# ---------------------------------------------------------------------------


def _make_answer(abstained: bool, forward_looking: bool) -> Answer:
    """Build a minimal Answer with a real StructuredAnswer attached."""
    structured = StructuredAnswer(
        answer_prose="Some answer text.",
        claims=[],
        abstain=abstained,
        abstain_reason="",
        forward_looking=forward_looking,
    )
    return Answer(
        query="What is management guidance?",
        text="Some answer text.",
        abstained=abstained,
        contexts=[],
        intent="latest",
        diagnostics={"top_rerank_score": 0.9, "companies_filter": []},
        citation_report=None,
        structured=structured,
    )


def test_forward_looking_check_passes_when_flag_matches() -> None:
    gq = EvaluationQuery(
        id="t_fl_01",
        query="What is management guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expect_forward_looking=True,
    )
    a = _make_answer(abstained=False, forward_looking=True)
    result = evaluate._grade(gq, a)
    assert result.forward_looking_check is True
    assert "forward_looking_check" in result.auto_pass_reasons


def test_forward_looking_check_fails_when_answer_not_forward_looking() -> None:
    gq = EvaluationQuery(
        id="t_fl_02",
        query="What is management guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expect_forward_looking=True,
    )
    a = _make_answer(abstained=False, forward_looking=False)
    result = evaluate._grade(gq, a)
    assert result.forward_looking_check is False
    assert "forward_looking_check" in result.auto_fail_reasons


def test_forward_looking_check_fails_when_answer_abstained() -> None:
    gq = EvaluationQuery(
        id="t_fl_03",
        query="What is management guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expect_forward_looking=True,
    )
    a = _make_answer(abstained=True, forward_looking=True)
    result = evaluate._grade(gq, a)
    assert result.forward_looking_check is False


def test_forward_looking_check_not_applied_when_flag_false() -> None:
    gq = EvaluationQuery(
        id="t_fl_04",
        query="What is leverage?",
        category="factual_citation",
        expected_intent="latest",
        expect_forward_looking=False,
    )
    a = _make_answer(abstained=False, forward_looking=False)
    result = evaluate._grade(gq, a)
    assert result.forward_looking_check is None


def test_forward_looking_check_none_when_no_structured_answer() -> None:
    gq = EvaluationQuery(
        id="t_fl_05",
        query="What is management guidance?",
        category="forward_looking_label",
        expected_intent="latest",
        expect_forward_looking=True,
    )
    # An Answer with structured=None (e.g. produced by a path that had no
    # StructuredAnswer available) should yield forward_looking_check=None.
    a = Answer(
        query="What is management guidance?",
        text="Some answer.",
        abstained=False,
        contexts=[],
        intent="latest",
        diagnostics={"top_rerank_score": 0.9, "companies_filter": []},
        citation_report=None,
        structured=None,
    )
    result = evaluate._grade(gq, a)
    assert result.forward_looking_check is None
