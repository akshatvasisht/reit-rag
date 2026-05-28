"""Tests for chart extraction confidence filtering and fallback chunk assembly.

All tests are pure unit tests — no real DB, no network, no Docling PDF parsing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from src.ingestion.chart_extractor import (
    MIN_DESCRIPTION_LENGTH,
    VISION_PROMPT,
    _is_confident_extraction,
    build_chart_context_text,
    extract_chart,
)


# ---------------------------------------------------------------------------
# _is_confident_extraction
# ---------------------------------------------------------------------------

_GOOD_DESCRIPTION = (
    "Chart type: bar\n"
    "Title: Same-Store NOI Growth by Region\n"
    "Axes: X-axis: Region; Y-axis: NOI Growth (%)\n"
    "Key values: North America 4.2%, Europe 3.8%, Asia Pacific 5.1%\n"
    "Insight: Asia Pacific led NOI growth trend across all regions in Q4 2025."
)  # 223 chars, has number, has structure keywords


def test_confident_extraction_passes_all_signals() -> None:
    """Description with length, number, and structure keyword → True."""
    assert len(_GOOD_DESCRIPTION) >= MIN_DESCRIPTION_LENGTH
    assert _is_confident_extraction(_GOOD_DESCRIPTION) is True


def test_confident_extraction_fails_on_missing_structure_keyword() -> None:
    """A description that is long and has numbers but lacks structure keywords → False."""
    # Replace all structure keywords with neutral words
    desc = (
        "This is a visualization showing 4.2 percent and 3.8 percent values "
        "across several categories in the dataset. The bars represent different "
        "segments of the portfolio broken down by geographic location. Overall "
        "the pattern indicates upward momentum in most areas reviewed by analysts."
    )
    # Must be long enough to pass that check
    assert len(desc) >= MIN_DESCRIPTION_LENGTH
    # Must have a digit
    assert any(c.isdigit() for c in desc)
    # Must NOT have structure keywords
    keywords = ["chart type", "title", "key values", "axes", "trend"]
    assert not any(kw in desc.lower() for kw in keywords)
    assert _is_confident_extraction(desc) is False


def test_confident_extraction_fails_on_not_chart_sentinel() -> None:
    """Sentinel value NOT_CHART → False regardless of length."""
    assert _is_confident_extraction("NOT_CHART") is False


def test_confident_extraction_fails_on_not_chart_sentinel_whitespace() -> None:
    """NOT_CHART with surrounding whitespace also returns False."""
    assert _is_confident_extraction("  NOT_CHART  ") is False


def test_confident_extraction_fails_on_no_digits() -> None:
    """A structurally sound description with no digits → False."""
    desc = (
        "Chart type: bar\n"
        "Title: Portfolio Overview\n"
        "Axes: Regions vs Categories\n"
        "Key values: North, South, East, West regions are shown\n"
        "Insight: The trend across all segments shows a consistent pattern."
    )
    assert len(desc) >= MIN_DESCRIPTION_LENGTH
    assert not any(c.isdigit() for c in desc)
    assert _is_confident_extraction(desc) is False


def test_confident_extraction_fails_on_short_description() -> None:
    """A short description with a digit and structure keyword → False."""
    short = "Chart type: bar. Key values: 42. Trend is up."
    assert len(short) < MIN_DESCRIPTION_LENGTH
    assert _is_confident_extraction(short) is False


# ---------------------------------------------------------------------------
# _is_likely_chart (import from enrich_charts)
# ---------------------------------------------------------------------------

# Import at function level to avoid triggering docling imports at collection time.

def _make_bbox(left: float, top: float, right: float, bottom: float):
    return SimpleNamespace(l=left, t=top, r=right, b=bottom)


def _make_prov(bbox):
    return SimpleNamespace(bbox=bbox, page_no=1)


def _make_picture_item(bbox=None, has_prov: bool = True):
    if not has_prov:
        return SimpleNamespace(prov=None)
    if bbox is None:
        return SimpleNamespace(prov=[SimpleNamespace(bbox=None, page_no=1)])
    return SimpleNamespace(prov=[_make_prov(bbox)])


def test_likely_chart_large_bbox() -> None:
    """Bbox area >= MIN_CHART_AREA_PX → True."""
    # 200 x 100 = 20_000 >= 10_000
    item = _make_picture_item(_make_bbox(0, 0, 200, 100))
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is True


def test_likely_chart_small_bbox() -> None:
    """Bbox area < MIN_CHART_AREA_PX → False."""
    # 50 x 100 = 5_000 < 10_000
    item = _make_picture_item(_make_bbox(0, 0, 50, 100))
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is False


def test_likely_chart_missing_prov_fails_open() -> None:
    """Missing prov → True (fail open to preserve prior behavior)."""
    item = _make_picture_item(has_prov=False)
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is True


def test_likely_chart_missing_bbox_fails_open() -> None:
    """Prov present but bbox is None → True (fail open)."""
    item = _make_picture_item(bbox=None)
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is True


def test_likely_chart_exactly_at_threshold() -> None:
    """Bbox area exactly equal to MIN_CHART_AREA_PX → True."""
    # 100 x 100 = 10_000
    item = _make_picture_item(_make_bbox(0, 0, 100, 100))
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is True


def test_likely_chart_bottomleft_origin_large() -> None:
    """BOTTOMLEFT-origin bboxes (top > bottom) yield negative b-t; a large chart must still register as likely, so unsigned area is used."""
    # width 343, |height| 340 -> 116_620 >= 10_000
    item = _make_picture_item(_make_bbox(0, 400, 343, 60))
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is True


def test_likely_chart_bottomleft_origin_small() -> None:
    """BOTTOMLEFT origin small icon (|area| < threshold) → False."""
    # width 50, |height| 80 -> 4_000 < 10_000
    item = _make_picture_item(_make_bbox(0, 100, 50, 20))
    from scripts.enrich_charts import _is_likely_chart
    assert _is_likely_chart(item) is False


# ---------------------------------------------------------------------------
# build_chart_context_text
# ---------------------------------------------------------------------------

_FOOTER = (
    "Chart detected on this page — visual extraction did not meet confidence threshold. "
    "Consult source document for chart data."
)


def test_fallback_chunk_contains_section_title() -> None:
    """Fallback text includes the section title when provided."""
    text = build_chart_context_text("Revenue Overview", [])
    assert "Revenue Overview" in text


def test_fallback_chunk_contains_sibling_text() -> None:
    """Fallback text includes sibling chunk content."""
    text = build_chart_context_text(None, ["NOI increased 4.2% year-over-year."])
    assert "NOI increased 4.2% year-over-year." in text


def test_fallback_chunk_contains_footer_sentence() -> None:
    """Fallback text always ends with the fixed footer sentence."""
    text = build_chart_context_text("Section", ["Some text."])
    assert _FOOTER in text


def test_fallback_chunk_all_three_parts() -> None:
    """Fallback text joins section title, sibling text, and footer."""
    title = "Quarterly Results"
    sibling = "Occupancy reached 95% in Q4 2025."
    text = build_chart_context_text(title, [sibling])
    assert title in text
    assert sibling in text
    assert _FOOTER in text


def test_fallback_chunk_no_section_title() -> None:
    """When section_title is None, fallback text still includes sibling and footer."""
    text = build_chart_context_text(None, ["Sibling text here."])
    assert "Sibling text here." in text
    assert _FOOTER in text


def test_fallback_chunk_no_siblings() -> None:
    """When sibling_texts is empty, fallback text has title and footer only."""
    text = build_chart_context_text("My Section", [])
    assert "My Section" in text
    assert _FOOTER in text
    # No extra blank section
    assert text.count("\n\n") <= 1


# ---------------------------------------------------------------------------
# extract_chart integration: confident path still produces chart_description signal
# ---------------------------------------------------------------------------


def _make_mock_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_image(width: int = 500, height: int = 400) -> MagicMock:
    img = MagicMock()
    img.size = (width, height)
    return img


def test_extract_chart_confident_returns_description_no_fallback() -> None:
    """A confident extraction returns (description, False) — no fallback needed."""
    with (
        patch("src.ingestion.chart_extractor._get_client") as mock_get_client,
        patch("src.ingestion.chart_extractor._image_to_base64_png", return_value="b64"),
        patch("src.ingestion.chart_extractor._pace_requests"),
    ):
        client = MagicMock()
        client.messages.create.return_value = _make_mock_response(_GOOD_DESCRIPTION)
        mock_get_client.return_value = client

        description, needs_fallback = extract_chart(_make_image())

    assert description == _GOOD_DESCRIPTION
    assert needs_fallback is False


def test_extract_chart_low_confidence_returns_none_with_fallback_flag() -> None:
    """A low-confidence extraction returns (None, True) — fallback chunk needed."""
    short_response = "Bar chart. Some data."  # short, no structure keywords
    with (
        patch("src.ingestion.chart_extractor._get_client") as mock_get_client,
        patch("src.ingestion.chart_extractor._image_to_base64_png", return_value="b64"),
        patch("src.ingestion.chart_extractor._pace_requests"),
    ):
        client = MagicMock()
        client.messages.create.return_value = _make_mock_response(short_response)
        mock_get_client.return_value = client

        description, needs_fallback = extract_chart(_make_image())

    assert description is None
    assert needs_fallback is True


def test_extract_chart_not_chart_returns_none_no_fallback() -> None:
    """NOT_CHART sentinel returns (None, False) — no chunk of any kind."""
    with (
        patch("src.ingestion.chart_extractor._get_client") as mock_get_client,
        patch("src.ingestion.chart_extractor._image_to_base64_png", return_value="b64"),
        patch("src.ingestion.chart_extractor._pace_requests"),
    ):
        client = MagicMock()
        client.messages.create.return_value = _make_mock_response("NOT_CHART")
        mock_get_client.return_value = client

        description, needs_fallback = extract_chart(_make_image())

    assert description is None
    assert needs_fallback is False


def test_extract_chart_small_image_returns_none_no_fallback() -> None:
    """Images below MIN_IMAGE_DIM return (None, False) without calling the API."""
    with patch("src.ingestion.chart_extractor._get_client") as mock_get_client:
        description, needs_fallback = extract_chart(_make_image(width=50, height=50))

    mock_get_client.assert_not_called()
    assert description is None
    assert needs_fallback is False


# ---------------------------------------------------------------------------
# VISION_PROMPT content — map/diagram extraction + anti-hallucination
# ---------------------------------------------------------------------------


def test_prompt_extracts_labeled_maps_and_diagrams() -> None:
    """The prompt must treat label/legend-bearing maps and diagrams as extractable."""
    p = VISION_PROMPT.lower()
    assert "map" in p and "diagram" in p
    assert "legend" in p


def test_prompt_still_skips_decorative_or_labelless_maps() -> None:
    """Decorative / label-less maps must still route to the NOT_CHART sentinel."""
    p = VISION_PROMPT
    assert "NOT_CHART" in p
    assert "no legible labels and no legend" in p.lower()


def test_prompt_forbids_inventing_labels_not_just_numbers() -> None:
    """Anti-hallucination must cover labels/locations, not only numeric values."""
    p = VISION_PROMPT.lower()
    assert "exactly as printed" in p
    assert "never invent" in p
    # the rule must name non-numeric targets (label/location/category), not just numbers
    assert any(tok in p for tok in ("label", "location", "category"))
