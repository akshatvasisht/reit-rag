"""Unit tests for the full-page fallback helper in enrich_charts.

Pure unit tests — no DB, no network, no Docling parse.
"""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from scripts.enrich_charts import _page_image


def _doc_with_page(page_no: int, pil: Image.Image) -> SimpleNamespace:
    img_ref = SimpleNamespace(pil_image=pil)
    page_item = SimpleNamespace(image=img_ref)
    return SimpleNamespace(pages={page_no: page_item})


def test_page_image_none_for_none_page() -> None:
    doc = _doc_with_page(1, Image.new("RGB", (10, 10)))
    assert _page_image(doc, None) is None


def test_page_image_none_when_page_absent() -> None:
    doc = _doc_with_page(1, Image.new("RGB", (10, 10)))
    assert _page_image(doc, 5) is None


def test_page_image_none_when_no_image_on_page() -> None:
    doc = SimpleNamespace(pages={2: SimpleNamespace(image=None)})
    assert _page_image(doc, 2) is None


def test_page_image_none_when_doc_has_no_pages() -> None:
    assert _page_image(SimpleNamespace(), 1) is None


def test_page_image_returns_pil_when_present() -> None:
    img = Image.new("RGB", (20, 20))
    doc = _doc_with_page(3, img)
    assert _page_image(doc, 3) is img
