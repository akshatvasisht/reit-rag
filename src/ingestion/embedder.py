"""Self-hosted embedding via Qwen3-Embedding-0.6B (1024 dims, sentence-transformers).

No API key. The model is downloaded from Hugging Face on first use and cached
in ~/.cache/huggingface/. Loads lazily so importing this module is cheap.

Qwen3-Embedding follows the asymmetric retrieval convention:
  - Queries are encoded with `prompt_name="query"` (adds the task-specific
    instruction prefix the model was trained with).
  - Documents are encoded plain.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from src.models import Chunk

logger = logging.getLogger(__name__)

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Pinned commit SHA — supply-chain mitigation against an upstream model swap.
EMBED_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBED_DIMS = 1024


def _parse_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Parse an integer env var with validation and fallback."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%d is below minimum %d; using default %d", name, value, minimum, default)
        return default
    return value


# All three knobs honor environment variables so the same code adapts to
# CPU-only, constrained GPU (RTX 3060 6 GB), or high-memory GPU runs without
# code changes.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
EMBED_BATCH = _parse_int_env("EMBED_BATCH", 16, minimum=1)
EMBED_DTYPE = os.environ.get(
    "EMBED_DTYPE",
    "float16" if EMBED_DEVICE.startswith("cuda") else "float32",
)


_model: Optional[SentenceTransformer] = None


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _torch_dtype(name: str) -> torch.dtype:
    """Map env-var dtype strings to torch dtypes; raise on invalid input."""
    dtype = _DTYPE_ALIASES.get(name.lower())
    if dtype is None:
        valid = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(
            f"Invalid EMBED_DTYPE={name!r}. Set EMBED_DTYPE to one of: {valid}."
        )
    return dtype


def _get_model() -> SentenceTransformer:
    """Return a lazily-loaded SentenceTransformer for Qwen3-Embedding-0.6B."""
    global _model
    if _model is None:
        logger.info(
            "Loading embedding model: %s on device=%s dtype=%s batch=%d "
            "(initial call may download ~1.2 GB)",
            EMBED_MODEL, EMBED_DEVICE, EMBED_DTYPE, EMBED_BATCH,
        )
        _model = SentenceTransformer(
            EMBED_MODEL,
            revision=EMBED_MODEL_REVISION,
            device=EMBED_DEVICE,
            model_kwargs={"torch_dtype": _torch_dtype(EMBED_DTYPE)},
        )
    return _model


def embed_chunks(chunks: list[Chunk], batch_size: int = EMBED_BATCH) -> list[Chunk]:
    """Populate `chunk.embedding` for every chunk via Qwen3-Embedding-0.6B.

    Encodes in micro-batches and runs `torch.cuda.empty_cache()` between them
    when on CUDA to prevent attention-memory fragmentation from accumulating
    across documents.

    Args:
        chunks: Chunks whose `chunk_text` is set.
        batch_size: Forwarded to sentence-transformers `model.encode`.

    Returns:
        The input list with `embedding` set on every chunk.
    """
    if not chunks:
        return chunks

    model = _get_model()
    on_cuda = EMBED_DEVICE.startswith("cuda") and torch.cuda.is_available()
    logger.info("Embedding %d chunks with %s (dim=%d, device=%s, dtype=%s, batch=%d)",
                len(chunks), EMBED_MODEL, EMBED_DIMS, EMBED_DEVICE, EMBED_DTYPE, batch_size)

    # Process in slices to clear the CUDA allocator between batches.
    for start in range(0, len(chunks), batch_size):
        slice_ = chunks[start : start + batch_size]
        texts = [c.chunk_text for c in slice_]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if vectors.shape[1] != EMBED_DIMS:
            raise RuntimeError(
                f"Expected {EMBED_DIMS} dims from {EMBED_MODEL}, got {vectors.shape[1]}"
            )
        for chunk, vec in zip(slice_, vectors):
            chunk.embedding = vec.tolist()
        if on_cuda:
            torch.cuda.empty_cache()
        logger.info("  Embedded %d / %d chunks", min(start + batch_size, len(chunks)), len(chunks))

    return chunks


def embed_query(text: str) -> list[float]:
    """Embed a single query for retrieval.

    Uses Qwen3's `prompt_name="query"` so the encoded vector is in the same
    space as documents but with the retrieval-task instruction prefix.

    Args:
        text: Natural-language user query.

    Returns:
        1024-element float list (normalized).
    """
    model = _get_model()
    vec = model.encode(
        text,
        prompt_name="query",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.tolist()
