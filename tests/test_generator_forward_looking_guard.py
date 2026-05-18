"""Tests for 2d (forward-looking integrity guard) and 2k (corpus-membership framing).

probe_id: bxp-morn-adv-2 (2d anchor), xdoc-std-5/xdoc-adv-6/xdoc-std-4 (2k anchors).

Test design:
- All tests mock retrieve() and _generate_structured() so no DB or API access.
- 2d tests verify the system prompt injected into _generate_structured carries
  forward-looking-absent framing when the query is guidance-intent and chunks
  lack forward-looking language for the metric.
- 2k tests verify the system prompt informs the LLM about corpus companies so
  it can distinguish "not retrieved" from "not in corpus".
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import pytest

from src.generation import generator
from src.models import Chunk, RetrievedChunk, StructuredAnswer
from src.retrieval.pipeline import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    company: str = "BXP",
    text: str = "BXP reported FFO of $7.10 per share for full-year 2025.",
    page: int = 46,
    section: str = "FFO",
) -> RetrievedChunk:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        company=company,
        ticker="BXP",
        doc_type="investor_presentation",
        report_date="2025-12",
        period_covered="Q4 2025",
        doc_version="2025-12",
        section_title=section,
        page_number=page,
        content_type="text",
        chunk_text=text,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=7.42)


def _make_retrieval(
    contexts: list[RetrievedChunk],
    query: str = "test",
    intent: str = "latest",
) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        intent=intent,
        contexts=contexts,
        companies=["BXP"],
        abstain=False,
        diagnostics={"top_rerank_score": 7.42},
        retrieval_confidence=0.9,
    )


def _make_sa(
    prose: str = "BXP reported $7.10 FFO.",
    abstain: bool = False,
    abstain_reason: str = "",
    forward_looking: bool = False,
) -> StructuredAnswer:
    return StructuredAnswer(
        answer_prose=prose,
        claims=[],
        abstain=abstain,
        abstain_reason=abstain_reason,
        forward_looking=forward_looking,
    )


# ---------------------------------------------------------------------------
# 2d — Forward-looking intent detection (unit tests on the guard function)
# ---------------------------------------------------------------------------


def test_detect_forward_looking_intent_guidance_keyword() -> None:
    """Query containing 'guidance' must be detected as forward-looking."""
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What is BXP's 2025 FFO guidance?") is True


def test_detect_forward_looking_intent_target_keyword() -> None:
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What is BXP's FFO target for 2026?") is True


def test_detect_forward_looking_intent_outlook_keyword() -> None:
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What is BXP's earnings outlook?") is True


def test_detect_forward_looking_intent_projection_keyword() -> None:
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What are BXP's FFO projections for next year?") is True


def test_detect_forward_looking_intent_forecast_keyword() -> None:
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("FFO forecast for DLR in 2026?") is True


def test_detect_forward_looking_intent_expects_keyword() -> None:
    from src.generation.generator import _is_forward_looking_query
    # "expected" in the query (asking what is expected)
    assert _is_forward_looking_query("What FFO per share is expected for 2026?") is True


def test_detect_forward_looking_intent_negative_historical_query() -> None:
    """A historical query must NOT be detected as forward-looking — NEGATIVE test."""
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What was BXP's 2025 FFO?") is False


def test_detect_forward_looking_intent_negative_reported_query() -> None:
    """A 'what did X report' query is not forward-looking — NEGATIVE test."""
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What did BXP report for FFO in Q4 2025?") is False


def test_detect_forward_looking_intent_negative_leverage_query() -> None:
    """A leverage question with no guidance keywords — NEGATIVE test."""
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What is BXP's net debt to EBITDA?") is False


def test_detect_forward_looking_intent_negative_occupancy_query() -> None:
    """Occupancy is reported, not guided — NEGATIVE test."""
    from src.generation.generator import _is_forward_looking_query
    assert _is_forward_looking_query("What is BXP's portfolio occupancy?") is False


# ---------------------------------------------------------------------------
# 2d — chunks_have_forward_looking_support (unit tests)
# ---------------------------------------------------------------------------


def test_chunks_support_guidance_language_present() -> None:
    """Chunk with 'guidance' token must return True."""
    from src.generation.generator import _chunks_have_forward_looking_support
    chunks = [_make_chunk(text="Management guided 2026 FFO of $7.10-$7.30 per share.")]
    assert _chunks_have_forward_looking_support(chunks) is True


def test_chunks_support_expect_language_present() -> None:
    from src.generation.generator import _chunks_have_forward_looking_support
    chunks = [_make_chunk(text="We expect FFO of $7.10 to $7.30 for the full year 2026.")]
    assert _chunks_have_forward_looking_support(chunks) is True


def test_chunks_support_target_language_present() -> None:
    from src.generation.generator import _chunks_have_forward_looking_support
    chunks = [_make_chunk(text="Our target range for 2026 FFO is $7.00-$7.50.")]
    assert _chunks_have_forward_looking_support(chunks) is True


def test_chunks_support_projected_language_present() -> None:
    from src.generation.generator import _chunks_have_forward_looking_support
    chunks = [_make_chunk(text="FFO is projected at $6.90 for fiscal 2026.")]
    assert _chunks_have_forward_looking_support(chunks) is True


def test_chunks_support_forecast_language_present() -> None:
    from src.generation.generator import _chunks_have_forward_looking_support
    chunks = [_make_chunk(text="Consensus forecast for FFO in 2026 is $7.20.")]
    assert _chunks_have_forward_looking_support(chunks) is True


def test_chunks_no_forward_looking_language_returns_false() -> None:
    """Chunk with only reported actuals, no guidance language — must return False.

    This is the anchor failure from bxp-morn-adv-2: the deck's p46 chunk contains
    '$7.10' as an actual but carries NO guidance/target/expect/project/forecast
    language. The guard must fire.
    """
    from src.generation.generator import _chunks_have_forward_looking_support
    # bxp-morn-adv-2 anchor: reported actual, no forward-looking language
    chunks = [_make_chunk(text="BXP reported FFO of $7.10 per share for full-year 2025.")]
    assert _chunks_have_forward_looking_support(chunks) is False


def test_chunks_empty_returns_false() -> None:
    from src.generation.generator import _chunks_have_forward_looking_support
    assert _chunks_have_forward_looking_support([]) is False


# ---------------------------------------------------------------------------
# 2d — Integration: system prompt gets forward-looking-absent injection
#       when query is forward-looking and chunks lack support
# ---------------------------------------------------------------------------


def test_system_prompt_gets_fl_injection_when_query_is_guidance_and_chunks_lack_support(
    monkeypatch,
) -> None:
    """Anchor case (bxp-morn-adv-2): query='BXP 2025 FFO guidance', chunks contain
    only reported actuals. The system prompt passed to the LLM must contain the
    forward-looking-absent instruction.
    """
    # Chunk with only reported actual — no guidance language
    ctx = [_make_chunk(text="BXP reported FFO of $7.10 per share for full-year 2025.")]
    retrieval = _make_retrieval(ctx, query="What is BXP's 2025 FFO guidance?")
    sa = _make_sa(prose="BXP's guidance was not explicitly stated.")

    captured_system: list[str] = []

    def fake_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):  # noqa: ANN001
        if system_prompt_override is not None:
            captured_system.append(system_prompt_override)
        return sa

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval)
    monkeypatch.setattr(generator, "_generate_structured", fake_generate)
    monkeypatch.setattr(generator, "check_citations", lambda *a, **k: MagicMock(
        supported=0, total=0, faithfulness_ratio=1.0, numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda *a, **k: [])

    result = generator.answer("What is BXP's 2025 FFO guidance?")

    assert len(captured_system) == 1, "system_prompt_override must be passed when FL guard fires"
    injected = captured_system[0]
    assert "not disclose" in injected.lower() or "not guided" in injected.lower() or \
           "forward-looking" in injected.lower() or "guidance" in injected.lower(), \
        f"Injected system prompt should contain FL-absent instruction, got: {injected[:200]}"


def test_no_fl_injection_when_chunks_have_guidance_language(monkeypatch) -> None:
    """When chunks DO contain guidance language, the FL-absent instruction must NOT
    appear in the system prompt passed to the LLM.

    The corpus-aware prompt is always passed (2k fix), but the FL-absent block
    (2d fix) must be absent when chunks already contain forward-looking language.
    """
    from src.generation.generator import _FL_ABSENT_INSTRUCTION

    ctx = [_make_chunk(text="Management guided 2026 FFO of $7.10–$7.30 per share.")]
    retrieval = _make_retrieval(ctx, query="What is BXP's 2026 FFO guidance?")
    sa = _make_sa(prose="BXP guided 2026 FFO of $7.10–$7.30 per share.")

    captured_system: list[str] = []

    def fake_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):  # noqa: ANN001
        if system_prompt_override is not None:
            captured_system.append(system_prompt_override)
        return sa

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval)
    monkeypatch.setattr(generator, "_generate_structured", fake_generate)
    monkeypatch.setattr(generator, "check_citations", lambda *a, **k: MagicMock(
        supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda *a, **k: [])

    generator.answer("What is BXP's 2026 FFO guidance?")

    # The corpus-aware prompt is always passed (2k); but the FL-absent block
    # (2d) must NOT be injected since chunks have forward-looking language.
    assert len(captured_system) == 1, "Corpus-aware system prompt must always be passed"
    assert "FORWARD-LOOKING GUARD" not in captured_system[0], \
        "FL-absent instruction must NOT be injected when chunks have FL support"


def test_no_fl_injection_for_non_forward_looking_query(monkeypatch) -> None:
    """Non-forward-looking query ('What was BXP's 2025 FFO?') must NOT have the
    FL-absent instruction injected. This is the critical negative test: the guard
    must not over-refuse on historical queries even when chunks lack FL language.
    """
    from src.generation.generator import _FL_ABSENT_INSTRUCTION

    # Same chunks as the forward-looking test — but query is historical
    ctx = [_make_chunk(text="BXP reported FFO of $7.10 per share for full-year 2025.")]
    retrieval = _make_retrieval(ctx, query="What was BXP's 2025 FFO?")
    sa = _make_sa(prose="BXP reported FFO of $7.10 per share.")

    captured_system: list[str] = []

    def fake_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):  # noqa: ANN001
        if system_prompt_override is not None:
            captured_system.append(system_prompt_override)
        return sa

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval)
    monkeypatch.setattr(generator, "_generate_structured", fake_generate)
    monkeypatch.setattr(generator, "check_citations", lambda *a, **k: MagicMock(
        supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda *a, **k: [])

    generator.answer("What was BXP's 2025 FFO?")

    # Corpus-aware prompt is always passed (2k); FL-absent block (2d) must NOT fire.
    assert len(captured_system) == 1, "Corpus-aware system prompt must always be passed"
    assert "FORWARD-LOOKING GUARD" not in captured_system[0], \
        "Guard must NOT fire on non-forward-looking query — over-refusal risk"


# ---------------------------------------------------------------------------
# 2d — RI dividend yield anchor case (§3 anchor)
# ---------------------------------------------------------------------------


def test_ri_dividend_yield_guidance_fires_guard(monkeypatch) -> None:
    """RI dividend yield guidance anchor: deck shows reported ~5.7% yield (p34 fn3)
    but no yield guidance. Query='What is Realty Income's dividend yield guidance?'
    Chunks contain only reported yield — guard must fire.
    """
    ctx = [_make_chunk(
        company="Realty Income",
        text="The company's dividend yield was approximately 5.7% as of December 31, 2025.",
        section="Dividend",
    )]
    retrieval = RetrievalResult(
        query="What is Realty Income's dividend yield guidance?",
        intent="latest",
        contexts=ctx,
        companies=["Realty Income"],
        abstain=False,
        diagnostics={"top_rerank_score": 5.0},
        retrieval_confidence=0.8,
    )
    sa = _make_sa(prose="Realty Income does not disclose dividend yield guidance.")

    captured_system: list[str] = []
    captured_forward_looking: list[bool] = []

    def fake_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):  # noqa: ANN001
        captured_forward_looking.append(forward_looking_hint)
        if system_prompt_override is not None:
            captured_system.append(system_prompt_override)
        return sa

    # The guard fires only when retrieval.forward_looking is True (the
    # classifier signal). Patch retrieve() to return a forward-looking-
    # flagged result so the guard branch is exercised.
    retrieval.forward_looking = True

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval)
    monkeypatch.setattr(generator, "_generate_structured", fake_generate)
    monkeypatch.setattr(generator, "check_citations", lambda *a, **k: MagicMock(
        supported=0, total=0, faithfulness_ratio=1.0, numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda *a, **k: [])

    generator.answer("What is Realty Income's dividend yield guidance?")

    # Guard fired: a system prompt override was applied (i.e. the
    # FL-absent instruction was appended).
    assert len(captured_system) == 1, "Guard must fire for RI dividend yield guidance query"
    # And the override actually contains the forward-looking guard text,
    # not just any non-empty override.
    injected = captured_system[0].lower()
    assert (
        "not disclose" in injected
        or "not guided" in injected
        or "forward-looking" in injected
        or "guidance" in injected
    ), f"Injected system prompt should contain FL-absent instruction; got: {captured_system[0][:300]}"
    # The classifier signal was propagated through to the generation call.
    assert captured_forward_looking == [True]


# ---------------------------------------------------------------------------
# 2k — Corpus-company list in system prompt
# ---------------------------------------------------------------------------


def test_system_prompt_contains_corpus_companies() -> None:
    """build_system_prompt_with_corpus() must embed all canonical corpus companies
    so the LLM can distinguish retrieval-miss from corpus-absence.

    Anchor probes: xdoc-std-5, xdoc-adv-6, xdoc-std-4 — all three had the LLM
    assert 'not in corpus' for RI/PSA which ARE in corpus.
    """
    from src.generation.generator import _build_system_prompt_with_corpus
    from src.corpus_registry import CORPUS_REGISTRY_SEED

    corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY_SEED})
    prompt = _build_system_prompt_with_corpus(corpus_companies)

    # Every corpus company must appear in the prompt
    for company in corpus_companies:
        assert company in prompt, \
            f"Corpus company '{company}' must appear in system prompt for 2k fix"


def test_system_prompt_contains_retrieval_miss_framing() -> None:
    """The system prompt must instruct the LLM to use 'did not retrieve' framing
    for in-corpus companies it missed, not 'not in corpus' framing.
    """
    from src.generation.generator import _build_system_prompt_with_corpus

    prompt = _build_system_prompt_with_corpus(["Realty Income", "Public Storage"])

    # Must contain the "did not retrieve" instruction
    assert "did not retrieve" in prompt or "not retrieve" in prompt, \
        "Prompt must distinguish retrieval miss from corpus absence"


def test_answer_uses_corpus_aware_prompt_for_generation(monkeypatch) -> None:
    """When answer() calls _generate_structured, the system_prompt_override passed
    must include the corpus company list (2k fix).

    Anchor: xdoc-adv-6 — RI was in companies_filter but LLM said 'RI not in corpus'.
    We verify the system prompt injected by answer() into _generate_structured
    includes all corpus companies.
    """
    from src.corpus_registry import CORPUS_REGISTRY_SEED

    corpus_companies = sorted({e["company"] for e in CORPUS_REGISTRY_SEED})

    ctx = [_make_chunk(
        company="BXP",
        text="BXP net debt to EBITDA was 8.18x as of June 30, 2025.",
    )]
    retrieval = _make_retrieval(ctx, query="What is BXP leverage?")
    sa = _make_sa(prose="BXP leverage was 8.18x.")

    captured_overrides: list[str] = []

    def fake_generate(query, contexts, intent="latest", forward_looking_hint=False, system_prompt_override=None):  # noqa: ANN001
        if system_prompt_override is not None:
            captured_overrides.append(system_prompt_override)
        return sa

    monkeypatch.setattr(generator, "retrieve", lambda q: retrieval)
    monkeypatch.setattr(generator, "_generate_structured", fake_generate)
    monkeypatch.setattr(generator, "check_citations", lambda *a, **k: MagicMock(
        supported=1, total=1, faithfulness_ratio=1.0, numeric_mismatches=[]
    ))
    monkeypatch.setattr(generator, "check_numeric_consistency", lambda *a, **k: [])

    generator.answer("What is BXP leverage?")

    assert len(captured_overrides) == 1, "system_prompt_override must be passed by answer()"
    system_used = captured_overrides[0]
    # All corpus companies must appear in the system prompt override
    for company in corpus_companies:
        assert company in system_used, \
            f"Corpus company '{company}' must appear in system prompt used for generation"


# ---------------------------------------------------------------------------
# simon-std-1 — Footnote anchor disambiguation in system prompt
# ---------------------------------------------------------------------------


def test_system_prompt_contains_footnote_anchor_instruction() -> None:
    """build_system_prompt_with_corpus() must embed the footnote-anchor disambiguation
    instruction so the Simon p.4 $478/fn.3 vs ICSC/fn.2 trap is caught at generation
    time regardless of query intent.

    Anchor probe: simon-std-1 — user asks "what does ICSC say about $478 per sq ft"
    but the chunk attributes $478 to fn.3 (Simon analysis), not fn.2 (ICSC).
    The instruction must be present unconditionally (not gated on FL intent).
    """
    from src.generation.generator import _build_system_prompt_with_corpus

    prompt = _build_system_prompt_with_corpus(["BXP", "Simon Property Group"])

    assert "footnote" in prompt.lower() or "anchor" in prompt.lower(), (
        "System prompt must contain footnote-anchor disambiguation instruction "
        f"(probe: simon-std-1); got first 500 chars: {prompt[:500]}"
    )


def test_system_prompt_contains_citation_page_anchoring_instruction() -> None:
    """The system prompt must embed an explicit "every cited page must match an
    actually-retrieved excerpt" rule. The instruction has to be unconditional —
    the failure mode is fabricated page tuples on all-company synthesis queries
    where the model has the right value but invents a plausible-looking page
    number that does not match any retrieved chunk.

    Anchor probe: g07 — "Rank all companies by Net Debt to EBITDA" produced an
    answer whose citation page tuples did not exist in any retrieved chunk.
    """
    from src.generation.generator import _build_system_prompt_with_corpus

    prompt = _build_system_prompt_with_corpus(["BXP", "Digital Realty"])

    lowered = prompt.lower()
    assert "page" in lowered and "retrieved" in lowered, (
        "System prompt must reference 'page' and 'retrieved' in the page-anchoring "
        "rule; got first 800 chars: " + prompt[:800]
    )
    # The instruction must specifically forbid inferring a page from where the
    # data 'should' be — the prior failure mode was the model guessing a page
    # because the figure was correct.
    assert "should" in lowered or "infer" in lowered or "invent" in lowered, (
        "System prompt must forbid inferring page numbers; got: " + prompt[:800]
    )


def test_system_prompt_contains_disambiguated_citation_example() -> None:
    """The SYSTEM_PROMPT must include a worked example showing the disambiguated
    citation form [Company, Doc Type (Subtype), Month Year, p.N], so the LLM
    preserves the full doc-type+subtype string instead of collapsing to just
    the subtype on synthesis-table output.

    Anchor probe: xdoc-adv-7 — the model produced [VICI Properties, Company Update,
    March 2026, p.7] (collapsed) instead of [VICI Properties, Investor Presentation
    (Company Update), March 2026, p.7] (full disambiguated form).
    """
    from src.generation.generator import _build_system_prompt_with_corpus
    prompt = _build_system_prompt_with_corpus(["BXP", "VICI Properties"])

    # The example must use the parenthetical subtype form
    assert "Investor Presentation (Investor Day Session)" in prompt or \
           "Investor Presentation (Quarterly Investor Deck)" in prompt, (
        "SYSTEM_PROMPT must show a worked example using the disambiguated "
        "citation form `[Company, Doc Type (Subtype), Month Year, p.N]`; "
        f"got prompt head[:800]={prompt[:800]}"
    )
    # And must explicitly forbid the collapse-to-subtype behavior
    assert "abbreviate" in prompt.lower() or "verbatim" in prompt.lower() or "preserve" in prompt.lower(), (
        "SYSTEM_PROMPT must explicitly instruct not to abbreviate the citation "
        "to just the subtype; got: " + prompt[:1500]
    )


def test_system_prompt_in_corpus_company_not_found_framing(monkeypatch) -> None:
    """When an in-corpus company is NOT in retrieved contexts, the LLM should use
    'I did not retrieve a chunk from...' framing, NOT 'not in corpus'.

    We test this by asserting the generated answer text (from a mocked LLM that
    simply repeats the system-prompt instruction) follows the retrieval-miss framing.

    This is a prompt-level fix — we verify the system prompt contains the
    correct framing instruction.
    """
    from src.generation.generator import _build_system_prompt_with_corpus

    # Simulate the scenario from xdoc-adv-6: RI is in corpus but not retrieved
    corpus_companies = ["Realty Income", "BXP", "Digital Realty"]
    prompt = _build_system_prompt_with_corpus(corpus_companies)

    # The prompt must NOT instruct the model to say "not in corpus" for in-corpus companies
    # It must give a "did not retrieve" framing instruction
    assert "did not retrieve" in prompt or "retrieval" in prompt.lower(), \
        "Prompt must contain retrieval-miss framing to prevent false corpus-absence claims"

    # The corpus list must allow the LLM to check membership
    assert "Realty Income" in prompt
    assert "BXP" in prompt
    assert "Digital Realty" in prompt
