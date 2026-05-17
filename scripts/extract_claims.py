"""Extract typed atomic numerical claims from chunks and store in chunk_claims.

Uses the Anthropic Message Batches API to submit one extraction request per
chunk.  For each chunk the model returns a JSON array of claims, each with
a metric name (snake_case), a verbatim value string, and an optional period.

The script is idempotent: chunks that already have rows in chunk_claims are
skipped unless --force is passed.  Use --limit N to cap the number of chunks
processed in a single run; use --batch-size to tune the Batches API payload
size.

Usage:
    python scripts/extract_claims.py [--force] [--limit N] [--batch-size N]

Do NOT execute during normal query serving — claim extraction is a background
ingestion step.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_claims")

POLL_INTERVAL_SECONDS = 30

# Greedy decoding for reproducibility.
CLAIMS_MODEL = "claude-haiku-4-5-20251001"
CLAIMS_MAX_TOKENS = 300
CLAIMS_TEMPERATURE = 0.0
CLAIMS_TOP_K = 1

VERIFICATION_MODEL = "claude-haiku-4-5-20251001"
VERIFICATION_MAX_TOKENS = 150
VERIFICATION_TEMPERATURE = 0.0
VERIFICATION_TOP_K = 1

# Schema constraining the model output to an array of typed claim objects.
# The model must return only explicitly stated numerical facts; it must
# not infer or extrapolate.
CLAIMS_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "value":  {"type": "string"},
                    "period": {"type": ["string", "null"]},
                },
                "required": ["metric", "value", "period"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "value_in_text":   {"type": "string", "enum": ["yes", "no"]},
        "metric_accurate": {"type": "string", "enum": ["yes", "no"]},
        "period_correct":  {"type": "string", "enum": ["yes", "no"]},
        "qualifier":       {"type": "string", "enum": ["none", "approximate", "guidance", "target", "estimated"]},
    },
    "required": ["value_in_text", "metric_accurate", "period_correct", "qualifier"],
    "additionalProperties": False,
}

VERIFICATION_SYSTEM_PROMPT = (
    "You are a financial claim verifier. "
    "Given a source text excerpt and an extracted claim (metric, value, period), "
    "judge whether the extraction is faithful to the source text. "
    "Set value_in_text='yes' only if the value string appears verbatim or "
    "equivalently in the source text. "
    "Set metric_accurate='yes' only if the metric name correctly describes what is "
    "measured in the source text. "
    "Set period_correct='yes' only if the period matches what is stated in the source "
    "text, or if period is null and no period is stated. "
    "Set qualifier to the qualifier class that applies to the value as stated in the "
    "source text: 'none' for a plain reported figure, 'approximate' for approximate "
    "or rounded figures, 'guidance' for management guidance, 'target' for stated "
    "targets or goals, 'estimated' for estimates or projections."
)

# System prompt for claim extraction.  Instructs the model to extract only
# explicitly stated numerical facts (no inference) and to normalise metric
# names to snake_case.
CLAIMS_SYSTEM_PROMPT = (
    "You are a financial data extractor. "
    "Given a document excerpt, extract every explicitly stated numerical claim. "
    "Do NOT infer, compute, or extrapolate any value — only report numbers that "
    "appear verbatim in the text. "
    "Normalise the metric name to snake_case (e.g. net_debt_ebitda, "
    "occupancy_rate, total_revenue, ffo_per_share). "
    "Preserve the raw value string exactly as written in the text "
    "(e.g. '4.9x', '$1.2B', '94.1%'). "
    "Set period to the reporting period stated for this value, or null if none. "
    "Return an empty claims array for excerpts with no numerical claims."
)


def _get_client():
    """Construct an Anthropic client from ``ANTHROPIC_API_KEY``."""
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=api_key)


def _load_unprocessed_chunks(conn, force: bool, limit: int | None) -> list[dict]:
    """Return chunks that do not yet have claim rows, unless --force."""
    sql = """
        SELECT c.id, c.company, c.doc_version, c.page_number,
               c.chunk_text
        FROM chunks c
        WHERE c.is_parent = FALSE
    """
    if not force:
        sql += """
          AND NOT EXISTS (
              SELECT 1 FROM chunk_claims cc WHERE cc.chunk_id = c.id
          )
        """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _build_batch_request(chunk: dict) -> dict[str, Any]:
    """Build a single Batches API request entry for one chunk."""
    return {
        "custom_id": str(chunk["id"]),
        "params": {
            "model": CLAIMS_MODEL,
            "max_tokens": CLAIMS_MAX_TOKENS,
            "temperature": CLAIMS_TEMPERATURE,
            "top_k": CLAIMS_TOP_K,
            "system": CLAIMS_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": chunk["chunk_text"]}
            ],
        },
    }


def _poll_until_complete(client, batch_id: str) -> Any:
    """Block until the batch reaches processing_status == 'ended'."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        logger.info("Batch %s status: %s", batch_id, status)
        if status == "ended":
            return batch
        if status in ("canceling", "canceled", "errored"):
            raise RuntimeError(f"Batch {batch_id} ended in status: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)


def _upsert_claims(conn, batch_chunk_ids: list[UUID], results: list[dict]) -> int:
    """Delete existing claim rows for this batch and insert fresh ones.

    Each entry in `results` must carry a ``value_qualifier`` key (None or a
    non-empty string) that was resolved by the verification pass.

    Returns the number of claim rows inserted.
    """
    str_ids = [str(cid) for cid in batch_chunk_ids]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunk_claims WHERE chunk_id = ANY(%s)",
            [str_ids],
        )

    rows_inserted = 0
    with conn.cursor() as cur:
        for entry in results:
            chunk_id = UUID(entry["chunk_id"])
            chunk_meta = entry["meta"]
            for claim in entry["claims"]:
                cur.execute(
                    """
                    INSERT INTO chunk_claims
                        (chunk_id, company, doc_version, metric, value, period,
                         page_number, value_qualifier)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        str(chunk_id),
                        chunk_meta["company"],
                        chunk_meta["doc_version"],
                        claim["metric"],
                        claim["value"],
                        claim.get("period"),
                        chunk_meta.get("page_number"),
                        claim.get("value_qualifier"),
                    ],
                )
                rows_inserted += 1
    conn.commit()
    return rows_inserted


def _build_verification_request(
    chunk_id: str,
    claim_idx: int,
    chunk_text: str,
    claim: dict,
    is_chart: bool,
) -> dict[str, Any]:
    """Build one verification request for a single (chunk, claim) pair."""
    chart_note = (
        " Note: this excerpt is LLM-generated from a chart image, so treat "
        "minor paraphrasing of the value as acceptable."
        if is_chart else ""
    )
    user_content = (
        f"Source text:{chart_note}\n{chunk_text}\n\n"
        f"Extracted claim:\n"
        f"  metric: {claim['metric']}\n"
        f"  value:  {claim['value']}\n"
        f"  period: {claim.get('period') or 'null'}"
    )
    return {
        "custom_id": f"verify:{chunk_id}:{claim_idx}",
        "params": {
            "model": VERIFICATION_MODEL,
            "max_tokens": VERIFICATION_MAX_TOKENS,
            "temperature": VERIFICATION_TEMPERATURE,
            "top_k": VERIFICATION_TOP_K,
            "system": VERIFICATION_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def _run_verification_batch(
    client,
    extracted: list[dict],
    chunk_by_id: dict[str, dict],
) -> tuple[list[dict], int, int]:
    """Submit a verification batch for all extracted claims.

    Args:
        client: Anthropic client.
        extracted: List of dicts with keys chunk_id, meta, claims.
        chunk_by_id: Map from chunk_id string to chunk metadata dict.

    Returns:
        (verified_results, claims_rejected_count, claims_qualified_count)
        where verified_results has the same shape as extracted but with only
        claims that passed verification, each annotated with value_qualifier.
    """
    # Build one verification request per (chunk, claim) pair.
    verification_requests: list[dict] = []
    # Index: custom_id → (chunk_id, claim_idx, original_claim)
    request_index: dict[str, tuple[str, int, dict]] = {}

    for entry in extracted:
        cid = entry["chunk_id"]
        chunk_meta = chunk_by_id.get(cid, {})
        is_chart = chunk_meta.get("content_type") == "chart_description"
        chunk_text = chunk_meta.get("chunk_text", "")
        for idx, claim in enumerate(entry["claims"]):
            req = _build_verification_request(cid, idx, chunk_text, claim, is_chart)
            custom_id = req["custom_id"]
            verification_requests.append(req)
            request_index[custom_id] = (cid, idx, claim)

    if not verification_requests:
        return [], 0, 0

    logger.info("Submitting verification batch of %d request(s)", len(verification_requests))
    v_batch = client.messages.batches.create(requests=verification_requests)
    v_batch_id = v_batch.id
    logger.info("Verification batch submitted: %s", v_batch_id)

    _poll_until_complete(client, v_batch_id)

    # Collect pass/fail decisions per (chunk_id, claim_idx).
    # passed_claims[chunk_id] = list of (claim_idx, claim_with_qualifier)
    passed_claims: dict[str, list[tuple[int, dict]]] = {}
    claims_rejected_count = 0
    claims_qualified_count = 0

    for result in client.messages.batches.results(v_batch_id):
        custom_id = result.custom_id
        if custom_id not in request_index:
            continue
        cid, idx, original_claim = request_index[custom_id]

        if result.result.type != "succeeded":
            logger.warning(
                "Verification failed for chunk %s claim %d: %s",
                cid, idx, result.result.type,
            )
            claims_rejected_count += 1
            continue

        content = result.result.message.content
        raw = getattr(content[0], "text", None) if content else None
        if raw is None:
            claims_rejected_count += 1
            continue

        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse verification JSON for %s claim %d", cid, idx)
            claims_rejected_count += 1
            continue

        # Apply decision rule: all three fields must be "yes".
        if (
            verdict.get("value_in_text") != "yes"
            or verdict.get("metric_accurate") != "yes"
            or verdict.get("period_correct") != "yes"
        ):
            claims_rejected_count += 1
            continue

        qualifier_str = verdict.get("qualifier", "none")
        value_qualifier: str | None = None if qualifier_str == "none" else qualifier_str
        if value_qualifier is not None:
            claims_qualified_count += 1

        enriched_claim = dict(original_claim)
        enriched_claim["value_qualifier"] = value_qualifier

        if cid not in passed_claims:
            passed_claims[cid] = []
        passed_claims[cid].append((idx, enriched_claim))

    # Rebuild results preserving original chunk order; only verified claims survive.
    verified_results: list[dict] = []
    for entry in extracted:
        cid = entry["chunk_id"]
        surviving = passed_claims.get(cid, [])
        if not surviving:
            continue
        # Restore original claim order.
        ordered = [claim for _, claim in sorted(surviving, key=lambda t: t[0])]
        verified_results.append({
            "chunk_id": cid,
            "meta": entry["meta"],
            "claims": ordered,
        })

    return verified_results, claims_rejected_count, claims_qualified_count


def _process_batch(
    client,
    conn,
    chunks: list[dict],
) -> tuple[int, int, int, int]:
    """Submit extraction + verification batches, persist only verified claims.

    Returns:
        (claims_extracted, claims_verified, claims_rejected, claims_qualified)
    """
    requests = [_build_batch_request(c) for c in chunks]
    logger.info("Submitting extraction batch of %d chunk(s)", len(requests))
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("Extraction batch submitted: %s", batch_id)

    _poll_until_complete(client, batch_id)

    chunk_by_id = {str(c["id"]): c for c in chunks}
    extracted: list[dict] = []

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type != "succeeded":
            logger.warning("Chunk %s failed extraction: %s", cid, result.result.type)
            continue
        content = result.result.message.content
        if not content:
            continue
        raw = getattr(content[0], "text", None)
        if raw is None:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON for chunk %s", cid)
            continue

        claims = parsed.get("claims", [])
        if not claims:
            continue

        meta = chunk_by_id.get(cid, {})
        extracted.append({"chunk_id": cid, "meta": meta, "claims": claims})

    claims_extracted = sum(len(e["claims"]) for e in extracted)
    logger.info("Extracted %d claim(s) from %d chunk(s)", claims_extracted, len(extracted))

    # Run the verification pass before writing to chunk_claims.
    verified_results, claims_rejected, claims_qualified = _run_verification_batch(
        client, extracted, chunk_by_id
    )
    claims_verified = claims_extracted - claims_rejected

    batch_ids = [UUID(c["id"]) for c in chunks]
    inserted = _upsert_claims(conn, batch_ids, verified_results)
    logger.info(
        "Batch %s: extracted=%d verified=%d rejected=%d qualified=%d inserted=%d",
        batch_id, claims_extracted, claims_verified, claims_rejected, claims_qualified, inserted,
    )
    return claims_extracted, claims_verified, claims_rejected, claims_qualified


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract numerical claims from chunks.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even for chunks that already have claim rows.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of chunks to process.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of chunks per Batches API submission.")
    args = parser.parse_args()

    client = _get_client()

    total_extracted = 0
    total_verified = 0
    total_rejected = 0
    total_qualified = 0

    with connect() as conn:
        chunks = _load_unprocessed_chunks(conn, force=args.force, limit=args.limit)
        logger.info("Found %d chunk(s) to process", len(chunks))

        for start in range(0, len(chunks), args.batch_size):
            batch_chunks = chunks[start: start + args.batch_size]
            extracted, verified, rejected, qualified = _process_batch(client, conn, batch_chunks)
            total_extracted += extracted
            total_verified += verified
            total_rejected += rejected
            total_qualified += qualified

    summary = {
        "claims_extracted_count": total_extracted,
        "claims_verified_count": total_verified,
        "claims_rejected_count": total_rejected,
        "claims_qualified_count": total_qualified,
    }
    print(json.dumps(summary))
    logger.info(
        "Done. claims_extracted_count=%d claims_verified_count=%d "
        "claims_rejected_count=%d claims_qualified_count=%d",
        total_extracted, total_verified, total_rejected, total_qualified,
    )


if __name__ == "__main__":
    main()
