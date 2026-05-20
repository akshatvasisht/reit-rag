"""Tests for the markdown JSON-fence stripper used before json.loads.

Haiku 4.5 wraps JSON-schema responses in ```json ... ``` fences; the helper
strips those fences so json.loads succeeds. Without it, the broad-except in
classify_page_content swallows the JSONDecodeError and the chunk falls back
to 'unknown'.
"""
from __future__ import annotations

import json

from src.ingestion.page_classifier import _strip_json_fences


def test_strip_json_language_fence_parses() -> None:
    """Input wrapped in ```json ... ``` parses correctly after stripping."""
    wrapped = '```json\n{"page_content_class": "substantive", "confidence": 0.9}\n```'
    parsed = json.loads(_strip_json_fences(wrapped))
    assert parsed == {"page_content_class": "substantive", "confidence": 0.9}


def test_no_fence_unchanged_and_parses() -> None:
    """Input with no fence is returned unchanged and still parses."""
    raw = '{"page_content_class": "boilerplate_legal", "confidence": 0.85}'
    assert _strip_json_fences(raw) == raw
    parsed = json.loads(_strip_json_fences(raw))
    assert parsed["page_content_class"] == "boilerplate_legal"


def test_strip_plain_fence_no_language_tag() -> None:
    """Plain ``` ... ``` (no 'json' language tag) is also stripped."""
    wrapped = '```\n{"page_content_class": "index_reference", "confidence": 0.95}\n```'
    parsed = json.loads(_strip_json_fences(wrapped))
    assert parsed == {"page_content_class": "index_reference", "confidence": 0.95}


def test_partial_or_malformed_fence_does_not_crash() -> None:
    """A lone opening fence with no matching close is left unchanged.

    The helper must never raise; the existing broad-except in the classifier
    will then catch the downstream JSONDecodeError and fall back to 'unknown'.
    """
    malformed = '```json\n{"page_content_class": "substantive", "confidence": 0.9}'
    out = _strip_json_fences(malformed)
    # Returned unchanged because no matching closing fence exists.
    assert out == malformed
    # json.loads should fail on this — that's the expected fallback path.
    try:
        json.loads(out)
        raised = False
    except json.JSONDecodeError:
        raised = True
    assert raised, "Expected JSONDecodeError on malformed fenced input"


def test_uppercase_language_tag_stripped() -> None:
    """Case-insensitive: ```JSON fences are also stripped."""
    wrapped = '```JSON\n{"page_content_class": "cover_page", "confidence": 0.8}\n```'
    parsed = json.loads(_strip_json_fences(wrapped))
    assert parsed["page_content_class"] == "cover_page"


def test_extra_whitespace_around_fences_tolerated() -> None:
    """Leading/trailing whitespace around the fence pair is tolerated."""
    wrapped = '   ```json\n{"page_content_class": "substantive", "confidence": 1.0}\n```   '
    parsed = json.loads(_strip_json_fences(wrapped))
    assert parsed["confidence"] == 1.0
