"""Anthropic-style Contextual Retrieval: generate a 1-line context per chunk.

For each chunk, send Claude Haiku 4.5 the full document plus the chunk text and
ask for a short context that situates the chunk within the document — Anthropic's
published technique (Sept 2024). The document portion uses prompt caching so the
per-document cost amortizes across all of that document's chunks.

This module owns ONLY the Claude API call. The orchestrator
(`scripts/contextualize.py`) handles document reconstruction, batching, and DB
writes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

# Haiku 4.5 supports prompt caching and is ~5× cheaper than Sonnet for the
# short-summary task this prompt expects.
CONTEXT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 200
# Anthropic Tier 1 Haiku has ~50 RPM; bursting 400+ calls produces sustained
# 429s. Retry envelope goes up to ~5 min so a stuck rate window clears.
MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 5.0

# The instruction part of the prompt — verbatim from Anthropic's published example.
_INSTRUCTION = (
    "Here is the chunk to situate within the full document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. Answer "
    "only with the succinct context and nothing else."
)


_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=api_key)
    return _client


def generate_context(document_text: str, chunk_text: str) -> str:
    """Generate a 1-line context for `chunk_text` given its parent document.

    Args:
        document_text: Full text of the document the chunk came from. This
            portion is marked for prompt caching, so repeated calls with the
            same document amortize the cost.
        chunk_text: The chunk to situate.

    Returns:
        A short string (typically 50-100 tokens) describing where this chunk
        sits in the document.
    """
    client = _get_client()
    resp = _call_with_retry(client, document_text, chunk_text)
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()


def _call_with_retry(client: Anthropic, document_text: str, chunk_text: str):
    """Call Claude Haiku with exponential backoff on transient failures."""
    delay = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(
                model=CONTEXT_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<document>\n{document_text}\n</document>",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": _INSTRUCTION.format(chunk=chunk_text),
                            },
                        ],
                    }
                ],
            )
        except (APIConnectionError, RateLimitError) as e:
            last_err = e
            if attempt == MAX_RETRIES:
                break
            logger.warning(
                "Contextualizer retry %d/%d after %s: sleeping %.1fs",
                attempt, MAX_RETRIES, type(e).__name__, delay,
            )
            time.sleep(delay)
            delay *= 2
        except APIStatusError as e:
            if e.status_code is None or e.status_code < 500:
                raise
            last_err = e
            if attempt == MAX_RETRIES:
                break
            logger.warning(
                "Contextualizer retry %d/%d after HTTP %s: sleeping %.1fs",
                attempt, MAX_RETRIES, e.status_code, delay,
            )
            time.sleep(delay)
            delay *= 2

    assert last_err is not None
    raise last_err
