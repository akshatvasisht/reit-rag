"""Tests for extract_json_object — recovers JSON from LLM bodies that wrap it in a fence plus trailing reasoning prose."""

from __future__ import annotations

import pytest

from src.ingestion.json_response import extract_json_object


def test_bare_json_object() -> None:
    assert extract_json_object('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_fenced_json_object() -> None:
    text = '```json\n{"page_content_class": "substantive", "confidence": 0.95}\n```'
    assert extract_json_object(text) == {
        "page_content_class": "substantive",
        "confidence": 0.95,
    }


def test_fenced_json_followed_by_reasoning_prose() -> None:
    """The real Haiku failure mode: fence + trailing (possibly cut-off) prose."""
    text = (
        "```json\n"
        '{\n  "page_content_class": "substantive",\n  "confidence": 0.95\n}\n'
        "```\n\n"
        '**Reasoning:** This chunk presents an action plan with specific strategic '
        'priorities ("Continued CBD Trophy Asset F'
    )
    assert extract_json_object(text) == {
        "page_content_class": "substantive",
        "confidence": 0.95,
    }


def test_prose_then_json() -> None:
    text = 'Here is the classification you asked for:\n{"doc_subtype": "investor_day_session", "confidence": 0.9}'
    assert extract_json_object(text) == {
        "doc_subtype": "investor_day_session",
        "confidence": 0.9,
    }


def test_nested_object_recovered_whole() -> None:
    text = 'noise {"outer": {"inner": 2}, "k": 3} trailing'
    assert extract_json_object(text) == {"outer": {"inner": 2}, "k": 3}


def test_first_brace_malformed_then_valid_object() -> None:
    """A stray '{' that doesn't parse must not abort the search."""
    text = 'prefix { not json } then {"confidence": 0.8}'
    assert extract_json_object(text) == {"confidence": 0.8}


def test_no_json_raises() -> None:
    with pytest.raises(ValueError):
        extract_json_object("there is no json here at all")


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        extract_json_object("")
