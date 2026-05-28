"""Chart-description extraction via Claude vision.

Used by `scripts/enrich_charts.py` to upgrade Docling's captionless PictureItems
into structured chart descriptions that retrieve and cite like any other chunk.

This module owns:
  - the structured extraction prompt
  - the Anthropic vision API call
  - PIL image → base64 conversion
  - the NOT_CHART sentinel parsing
  - extraction confidence filtering and fallback chunk text assembly
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from typing import Optional, Tuple

from PIL import Image
from anthropic import Anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

# Use the same model as the answer generator for consistency across
# ingestion-time and query-time prompts.
VISION_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 600  # extraction output is short; cap to control cost
MIN_IMAGE_DIM = 100  # below this on either axis → assume logo / icon, skip

# Confidence thresholds for chart extraction quality filtering.
# Descriptions below these bars are treated as low-confidence and replaced
# with a chart_context fallback that preserves page context without noise.
MIN_DESCRIPTION_LENGTH = 150
MIN_NUMERICAL_VALUES = 1

# Sentinel the prompt instructs the model to emit for non-chart images.
NOT_CHART_SENTINEL = "NOT_CHART"
_NOT_CHART_RE = re.compile(r"\bNOT_CHART\b", re.IGNORECASE)

# Retry parameters for transient Anthropic API failures (connection drops, 5xx,
# and rate limits). Retries use exponential backoff to reduce silent extraction
# loss during intermittent failures.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

# Footer appended to every chart_context fallback chunk so retrievers know to
# consult the source document for the actual chart data.
_CHART_CONTEXT_FOOTER = (
    "Chart detected on this page — visual extraction did not meet confidence threshold. "
    "Consult source document for chart data."
)


def _is_confident_extraction(description: str) -> bool:
    """Return True when the vision extraction meets minimum quality standards.

    A description is accepted only when it is long enough, contains at least
    one numerical value, and uses the expected structural keywords that the
    extraction prompt requests.  Short or keyword-free outputs indicate the
    model could not read the chart reliably.
    """
    if len(description) < MIN_DESCRIPTION_LENGTH:
        return False
    if description.strip().upper() == "NOT_CHART":
        return False
    has_number = bool(re.search(r'\d+(?:\.\d+)?', description))
    if not has_number:
        return False
    has_structure = any(kw in description.lower() for kw in
                        ["chart type", "title", "key values", "axes", "trend"])
    return has_structure


def build_chart_context_text(
    section_title: Optional[str],
    sibling_texts: list[str],
) -> str:
    """Assemble the chunk_text for a chart_context fallback chunk.

    The fallback preserves as much page context as possible (section heading +
    sibling text-chunk content) so the page is still retrievable even though
    the chart image could not be reliably described.
    """
    parts: list[str] = []
    if section_title:
        parts.append(section_title)
    if sibling_texts:
        parts.append("\n".join(t for t in sibling_texts if t))
    parts.append(_CHART_CONTEXT_FOOTER)
    return "\n\n".join(p for p in parts if p)


def _parse_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Parse a float env var with validation and fallback."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.2f", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%.2f is below minimum %.2f; using default %.2f", name, value, minimum, default)
        return default
    return value


# Proactive pacing to reduce 429 frequency on Tier-1 limits.
MIN_REQUEST_INTERVAL_SECONDS = _parse_float_env(
    "VISION_MIN_INTERVAL_SECONDS", 1.4, minimum=0.0
)

VISION_PROMPT = """Examine this image from a REIT investor presentation.

First decide whether the image carries extractable data.

Respond with exactly the single word NOT_CHART when it is decorative or has no readable data: a logo, headshot, photograph, icon, background texture, or a map/illustration with no legible labels and no legend.

Otherwise — a chart, table-as-image, data visualization, OR a map/diagram that carries legible text labels or a legend — extract its content in this format:

Chart type: <bar / line / pie / scatter / waterfall / table / map / diagram / other>
Title: <if visible, else "(none)">
Axes / categories: <X-axis and Y-axis labels with units; or, for a map/diagram, the legend categories and what each color or symbol denotes>
Data series: <each series name and what it represents; for a map, what the plotted locations or regions represent>
Key values: <list the readable data points, or the labeled locations/entities each with its label and — where a legend assigns one — its category (e.g. "Caesars Palace - VICI/Caesars"); transcribe every label exactly as printed>
Insight: <one sentence on what the visual communicates: trend, comparison, count, or geographic concentration>

Be precise:
- Transcribe labels, names, and numbers EXACTLY as printed. Never invent, infer, translate, or complete a label, location, category, or value.
- If a label, its color/category, or a value is not legible, omit it rather than guessing — never approximate text. You may mark a partially-readable number "approximately" ("approximately 5.2x", "between 30% and 40%").
- Report only what is visibly present in this image."""


_client: Anthropic | None = None
_last_request_ts: float = 0.0


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic(api_key=api_key)
    return _client


def _image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG."""
    buf = io.BytesIO()
    # Convert any palette / RGBA-with-transparency to a clean RGB so encoders agree
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_chart(image: Image.Image) -> Tuple[Optional[str], bool]:
    """Call Claude vision to extract a structured chart description.

    Args:
        image: PIL Image of a single PictureItem from Docling.

    Returns:
        A tuple ``(description, needs_fallback)`` where:
        - ``description`` is the structured text when the model found a chart
          that passes the confidence gate, or None otherwise.
        - ``needs_fallback`` is True when the model produced output that looks
          like a chart but did not meet the confidence gate — the caller should
          create a chart_context fallback chunk in that case.

        When ``description`` is None and ``needs_fallback`` is False, the image
        was filtered by size, identified as NOT_CHART, or returned empty output.
    """
    width, height = image.size
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        logger.debug("Skipping small image (%dx%d) — likely a logo/icon", width, height)
        return None, False

    b64 = _image_to_base64_png(image)
    client = _get_client()
    resp = _call_with_retry(client, b64)

    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()

    if not text:
        return None, False
    if _NOT_CHART_RE.search(text):
        return None, False

    # Apply the confidence gate.  A passing description is stored as
    # chart_description.  A failing one signals the caller to create a
    # chart_context fallback so the page remains findable without noise.
    if not _is_confident_extraction(text):
        logger.debug(
            "Extraction failed confidence gate (%d chars) — fallback chunk needed",
            len(text),
        )
        return None, True

    return text, False


def _call_with_retry(client: Anthropic, b64_image: str):
    """Send the vision request with exponential backoff on transient failures.

    Retries `APIConnectionError`, `RateLimitError`, and 5xx `APIStatusError`.
    Re-raises 4xx errors (auth, payload too large, etc.) immediately — those
    do not improve with retry.
    """
    delay = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _pace_requests()
            return client.messages.create(
                model=VISION_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_image,
                                },
                            },
                            {"type": "text", "text": VISION_PROMPT},
                        ],
                    }
                ],
            )
        except (APIConnectionError, RateLimitError) as e:
            last_err = e
            if attempt == MAX_RETRIES:
                break
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                delay = max(delay, retry_after)
            logger.warning(
                "Vision call retry %d/%d after %s: sleeping %.1fs",
                attempt, MAX_RETRIES, type(e).__name__, delay,
            )
            time.sleep(delay)
            delay *= 2
        except APIStatusError as e:
            # Retry 429 and 5xx. Other 4xx are caller faults; bail immediately.
            status = e.status_code
            if status not in (429,) and (status is None or status < 500):
                raise
            last_err = e
            if attempt == MAX_RETRIES:
                break
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                delay = max(delay, retry_after)
            logger.warning(
                "Vision call retry %d/%d after HTTP %s: sleeping %.1fs",
                attempt, MAX_RETRIES, e.status_code, delay,
            )
            time.sleep(delay)
            delay *= 2

    assert last_err is not None
    raise last_err


def _pace_requests() -> None:
    """Enforce a minimum gap between outbound vision requests."""
    global _last_request_ts
    now = time.monotonic()
    elapsed = now - _last_request_ts
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_ts = time.monotonic()


def _retry_after_seconds(err: Exception) -> float | None:
    """Extract Retry-After header when present."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        value = float(retry_after)
    except (TypeError, ValueError):
        return None
    return max(0.0, value)
