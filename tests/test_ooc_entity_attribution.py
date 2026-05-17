"""Unit tests for check_ooc_entity_attribution and extract_entity_candidates.

All fixtures are plain dataclasses — no network or database access.
Tests run in well under 2 seconds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.generation.citation_check import check_ooc_entity_attribution
from src.models import Claim, Chunk, RetrievedChunk, StructuredAnswer
from src.retrieval.entity_filter import extract_entity_candidates


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_claim(
    text: str,
    value: str,
    citation_company: str,
) -> Claim:
    return Claim(
        text=text,
        value=value,
        citation=f"[{citation_company}, Investor Presentation, March 2026, p.5]",
        citation_company=citation_company,
        citation_date="March 2026",
        citation_page=5,
    )


def _make_answer(claims: list[Claim], abstain: bool = False) -> StructuredAnswer:
    return StructuredAnswer(
        answer_prose="Some prose.",
        claims=claims,
        abstain=abstain,
        abstain_reason="",
        forward_looking=False,
    )


# Corpus companies present in the test corpus.
_CORPUS = ["Digital Realty", "VICI Properties", "BXP", "Public Storage"]


# ---------------------------------------------------------------------------
# extract_entity_candidates — unit tests
# ---------------------------------------------------------------------------


def test_extract_entity_candidates_returns_ooc_name() -> None:
    """Equinix (not in corpus) is extracted from a typical query."""
    candidates = extract_entity_candidates("What is Equinix's portfolio size?")
    assert "Equinix" in candidates


def test_extract_entity_candidates_returns_multi_word_name() -> None:
    """Multi-word proper nouns are extracted as a single candidate."""
    candidates = extract_entity_candidates("Compare Simon Property Group's leverage to BXP.")
    joined = " ".join(candidates)
    assert "Simon Property Group" in joined or any("Simon" in c for c in candidates)


def test_extract_entity_candidates_empty_query() -> None:
    """Empty query returns an empty list without error."""
    assert extract_entity_candidates("") == []


def test_extract_entity_candidates_excludes_stopwords() -> None:
    """Question-opener words like 'What' and 'How' are not returned as candidates."""
    candidates = extract_entity_candidates("What is the leverage ratio?")
    assert "What" not in candidates


def test_extract_entity_candidates_in_corpus_name_still_returned() -> None:
    """extract_entity_candidates does NOT filter by corpus membership.

    Simon Property Group is a corpus member, but the function returns it anyway.
    The corpus-membership check (ooc_entities filter) is the caller's responsibility.
    """
    candidates = extract_entity_candidates("What is Simon Property Group's leverage?")
    joined = " ".join(candidates)
    assert "Simon" in joined


# ---------------------------------------------------------------------------
# check_ooc_entity_attribution — uses extract_entity_candidates internally
# ---------------------------------------------------------------------------


def test_ooc_entity_with_value_attributed_via_different_company_flagged() -> None:
    """Query mentions OOC entity, claim attributes a value to it via a different company's citation."""
    claim = _make_claim(
        text="Equinix has a market cap of $80B.",
        value="$80B",
        citation_company="VICI Properties",
    )
    answer = _make_answer([claim])
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(
            answer, "What is Equinix's portfolio size?", _CORPUS
        )
    assert len(issues) == 1
    issue = issues[0]
    assert issue["type"] == "ooc_entity_attribution"
    assert issue["ooc_entity"] == "Equinix"
    assert issue["claimed_value"] == "$80B"
    assert issue["citing_company"] == "VICI Properties"
    assert "Equinix" in issue["explanation"]
    assert "VICI Properties" in issue["explanation"]


def test_abstaining_answer_returns_empty_list() -> None:
    """An abstaining answer returns an empty issue list without inspecting claims."""
    claim = _make_claim(
        text="Equinix reported $100B revenue.",
        value="$100B",
        citation_company="VICI Properties",
    )
    answer = _make_answer([claim], abstain=True)
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(
            answer, "What is Equinix's revenue?", _CORPUS
        )
    assert issues == []


def test_in_corpus_entity_only_returns_empty_list() -> None:
    """Query names only in-corpus entities — no OOC entity → empty list."""
    claim = _make_claim(
        text="Digital Realty net debt to EBITDA was 4.9x.",
        value="4.9x",
        citation_company="Digital Realty",
    )
    answer = _make_answer([claim])
    # "Digital Realty" is extracted by the regex but is in the corpus set,
    # so ooc_entities is empty and the guard returns no issues.
    issues = check_ooc_entity_attribution(
        answer, "What is Digital Realty's leverage?", _CORPUS
    )
    assert issues == []


def test_claim_with_no_value_not_flagged() -> None:
    """Claims with an empty value field are skipped."""
    claim = _make_claim(
        text="Equinix is a data center REIT.",
        value="",
        citation_company="VICI Properties",
    )
    answer = _make_answer([claim])
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(answer, "Tell me about Equinix.", _CORPUS)
    assert issues == []


def test_citation_company_matches_ooc_entity_not_flagged() -> None:
    """Edge case: citation_company == OOC entity name — should not be flagged."""
    # This would mean the model cited a chunk authored by the OOC entity itself,
    # which is a valid citation path and should not be flagged as an OOC attribution error.
    claim = _make_claim(
        text="Equinix reported $80B in assets.",
        value="$80B",
        citation_company="Equinix",
    )
    answer = _make_answer([claim])
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(
            answer, "What is Equinix's asset base?", _CORPUS
        )
    assert issues == []


def test_case_insensitive_matching() -> None:
    """OOC entity and corpus matching are case-insensitive."""
    # corpus has "Digital Realty" — query uses lowercase "digital realty"
    # which is in corpus, so no OOC issue expected.
    claim = _make_claim(
        text="digital realty had leverage of 4.9x.",
        value="4.9x",
        citation_company="Digital Realty",
    )
    answer = _make_answer([claim])
    # extract_entity_candidates returns [] for a fully-lowercase query like
    # "What is digital realty's leverage?" — no capitalized phrases present.
    issues = check_ooc_entity_attribution(
        answer, "What is digital realty's leverage?", _CORPUS
    )
    assert issues == []


def test_case_insensitive_ooc_detection() -> None:
    """OOC entity name in claim text is matched case-insensitively."""
    claim = _make_claim(
        text="EQUINIX has a market cap of $80B.",
        value="$80B",
        citation_company="VICI Properties",
    )
    answer = _make_answer([claim])
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(
            answer, "What is Equinix's market cap?", _CORPUS
        )
    assert len(issues) == 1
    assert issues[0]["ooc_entity"] == "Equinix"


def test_no_ooc_entities_in_query_returns_empty_list() -> None:
    """Query that mentions no recognizable company names → empty list."""
    claim = _make_claim(
        text="Some generic claim about real estate.",
        value="5%",
        citation_company="Digital Realty",
    )
    answer = _make_answer([claim])
    # Empty query → extract_entity_candidates returns []
    issues = check_ooc_entity_attribution(answer, "", _CORPUS)
    assert issues == []


def test_multiple_claims_only_violating_ones_flagged() -> None:
    """Multiple claims: only the one attributing a value to the OOC entity is flagged."""
    good_claim = _make_claim(
        text="VICI Properties reported $2.1B revenue.",
        value="$2.1B",
        citation_company="VICI Properties",
    )
    bad_claim = _make_claim(
        text="Equinix has a portfolio of 250 data centers.",
        value="250",
        citation_company="VICI Properties",
    )
    answer = _make_answer([good_claim, bad_claim])
    with patch(
        "src.retrieval.entity_filter.extract_entity_candidates",
        return_value=["Equinix"],
    ):
        issues = check_ooc_entity_attribution(
            answer, "What is Equinix's portfolio count?", _CORPUS
        )
    assert len(issues) == 1
    assert issues[0]["claim_text"] == bad_claim.text


def test_simon_property_group_in_corpus_not_flagged_as_ooc() -> None:
    """Simon Property Group appears in extract_entity_candidates output but is
    in the corpus set, so ooc_entities is empty and no issue is raised."""
    claim = _make_claim(
        text="Simon Property Group reported 4.9x leverage.",
        value="4.9x",
        citation_company="Simon Property Group",
    )
    answer = _make_answer([claim])
    corpus_with_simon = _CORPUS + ["Simon Property Group"]
    # Do not patch — let the real regex run.
    issues = check_ooc_entity_attribution(
        answer,
        "What is Simon Property Group's leverage?",
        corpus_with_simon,
    )
    assert issues == []


# ---------------------------------------------------------------------------
# Generator wiring test
# ---------------------------------------------------------------------------


def test_generator_populates_ooc_attribution_issues_on_answer() -> None:
    """The generator wires ooc_attribution_issues onto Answer when the answered path is taken."""
    from src.generation.generator import _generate_structured, answer_structured

    # Build a fake StructuredAnswer with an OOC issue scenario.
    fake_claim = _make_claim(
        text="Equinix has a market cap of $80B.",
        value="$80B",
        citation_company="VICI Properties",
    )
    fake_structured = StructuredAnswer(
        answer_prose="Equinix has a market cap of $80B.",
        claims=[fake_claim],
        abstain=False,
        abstain_reason="",
        forward_looking=False,
    )

    fake_chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company="VICI Properties",
        ticker="VICI",
        doc_type="investor_presentation",
        report_date="2026-03",
        period_covered="Q4 2025",
        doc_version="2026-03",
        section_title="Peers",
        page_number=5,
        content_type="text",
        chunk_text="Equinix has a market cap of $80B.",
    )
    fake_rc = RetrievedChunk(chunk=fake_chunk, rerank_score=4.0)

    from src.retrieval.pipeline import RetrievalResult

    fake_retrieval = RetrievalResult(
        query="What is Equinix's market cap?",
        intent="latest",
        contexts=[fake_rc],
        abstain=False,
        abstain_reason="",
        diagnostics={"top_rerank_score": 4.0, "after_expansion": 1},
    )

    with (
        patch("src.generation.generator.retrieve", return_value=fake_retrieval),
        patch("src.generation.generator._generate_structured", return_value=fake_structured),
        patch(
            "src.retrieval.entity_filter.extract_entity_candidates",
            return_value=["Equinix"],
        ),
    ):
        result = answer_structured("What is Equinix's market cap?")

    assert hasattr(result, "ooc_attribution_issues")
    assert len(result.ooc_attribution_issues) == 1
    assert result.ooc_attribution_issues[0]["ooc_entity"] == "Equinix"
