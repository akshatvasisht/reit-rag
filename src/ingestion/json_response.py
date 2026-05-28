"""Robust JSON-object extraction from LLM text responses.

Small classification models increasingly wrap their JSON in a ```json fence and
append reasoning prose after it (and may hit ``max_tokens`` mid-prose). A strict
"the whole body is one JSON value" parse fails on those responses, which would
otherwise collapse a classification straight to its ``"unknown"`` fallback.

``extract_json_object`` recovers the first complete JSON object whether it is
bare, fenced, or embedded in surrounding prose, by scanning for the first
balanced ``{...}`` span that parses.
"""

from __future__ import annotations

import json


def extract_json_object(text: str) -> dict:
    """Return the first complete JSON object found in ``text``.

    Tries a direct parse first, then scans for the first balanced ``{...}``
    object — covering fenced blocks and JSON followed by (or preceded by)
    commentary. Raises ``ValueError`` when no JSON object can be parsed.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break  # malformed span; resume search after this '{'
        start = text.find("{", start + 1)

    raise ValueError("no JSON object found in LLM response")
