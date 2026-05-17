"""Tests for embed_query instruction-prefix behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.embedder import FINANCIAL_QUERY_INSTRUCTION


def _fake_encode(text, **kwargs):
    """Return a small numpy array regardless of input."""
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture()
def mock_model():
    """Patch _get_model so no real SentenceTransformer is loaded."""
    m = MagicMock()
    m.encode.side_effect = _fake_encode
    with patch("src.ingestion.embedder._get_model", return_value=m):
        yield m


def test_embed_query_input_starts_with_instruct(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    call_args = mock_model.encode.call_args
    actual_input = call_args[0][0]
    assert actual_input.startswith("Instruct: "), (
        f"Expected input to start with 'Instruct: ', got: {actual_input!r}"
    )


def test_embed_query_input_contains_financial_instruction(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    call_args = mock_model.encode.call_args
    actual_input = call_args[0][0]
    assert FINANCIAL_QUERY_INSTRUCTION in actual_input


def test_embed_query_input_contains_query_text(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    call_args = mock_model.encode.call_args
    actual_input = call_args[0][0]
    assert "Query: test query" in actual_input


def test_embed_query_does_not_pass_prompt_name(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    call_kwargs = mock_model.encode.call_args[1]
    assert "prompt_name" not in call_kwargs, (
        f"prompt_name should not be passed to encode; got kwargs: {call_kwargs}"
    )


def test_embed_query_preserves_normalize_embeddings(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs.get("normalize_embeddings") is True


def test_embed_query_calls_encode_exactly_once(mock_model):
    from src.ingestion.embedder import embed_query

    embed_query("test query")

    assert mock_model.encode.call_count == 1


def test_embed_query_returns_list(mock_model):
    from src.ingestion.embedder import embed_query

    result = embed_query("test query")
    assert isinstance(result, list)


def test_embed_chunks_does_not_include_instruct_prefix(mock_model):
    """Document-side encoding must not receive an instruction prefix."""
    from src.ingestion.embedder import embed_chunks
    from src.models import Chunk
    from uuid import uuid4

    chunk = Chunk(
        document_id=uuid4(),
        company="ExampleREIT",
        ticker="EXRT",
        doc_type="investor_presentation",
        report_date="2026-03",
        doc_version="2026-03",
        chunk_text="FFO per share was $1.42 for the quarter.",
        content_type="text",
    )

    def fake_encode_docs(texts, **kwargs):
        return np.zeros((len(texts), 1024))

    mock_model.encode.side_effect = fake_encode_docs

    embed_chunks([chunk])

    call_args = mock_model.encode.call_args
    texts_arg = call_args[0][0]
    for t in texts_arg:
        assert not t.startswith("Instruct:"), (
            f"Document text must not start with 'Instruct:'; got: {t!r}"
        )
