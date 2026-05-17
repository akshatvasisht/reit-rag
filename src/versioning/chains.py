"""Version-group grouping and deduplication for retrieved chunks."""

from __future__ import annotations

from uuid import UUID

from src.models import Chunk, RetrievedChunk
from src.versioning.classifier import TemporalIntent


def version_group_key(chunk: Chunk) -> tuple[str, str, str]:
    """Return the version-group identifier for a chunk.

    A version group is the set of all documents for the same company,
    document type, and document subtype ordered by ``doc_version``.  Two
    chunks share a group if and only if they have the same
    ``(company, doc_type, doc_subtype)`` triple.

    Keying on ``doc_subtype`` prevents structurally distinct documents that
    happen to share a ``doc_type`` label (e.g. a quarterly deck and an
    investor-day session both labelled ``investor_presentation``) from being
    collapsed into the same version group, which would cause the deduplicator
    to suppress one document entirely.

    Args:
        chunk: The chunk whose version-group key is needed.

    Returns:
        ``(company, doc_type, doc_subtype)`` as a 3-tuple of strings.
    """
    return (chunk.company, chunk.doc_type, chunk.doc_subtype)


def dedupe_by_version_group(
    retrieved: list[RetrievedChunk],
    intent: TemporalIntent,
) -> list[RetrievedChunk]:
    """Suppress older-version chunks unless the query intent demands them.

    Behaviour by intent:

    * ``"latest"``               — for every version group, keep ONLY chunks
                                   whose ``doc_version`` equals the maximum
                                   ``doc_version`` present in that group among
                                   the retrieved set. All other (older-version)
                                   chunks are dropped.
    * ``"comparison"``           — keep every chunk unchanged; the caller wants
                                   to see multiple versions explicitly.
    * ``"historical"``           — keep every chunk unchanged; retrieval ranking
                                   is already biased to the period named in the
                                   query.
    * ``"conflict"``             — keep every chunk unchanged; all versions are
                                   needed to surface contradictions across
                                   documents.
    * ``"all_company_synthesis"``— keep every chunk unchanged; per-company
                                   sub-retrieval already handles versioning and
                                   the full multi-company set must reach the
                                   generation layer.

    Groups with only one ``doc_version`` represented (e.g. VICI, EGP, RLT)
    are a no-op under all four intents: there is nothing to suppress.

    Input order is preserved for chunks that survive the filter.

    Args:
        retrieved: Ordered list of retrieved chunks from the reranker.
        intent: Temporal intent from ``classify_intent``.

    Returns:
        Filtered list of ``RetrievedChunk`` objects, preserving original order.
    """
    if intent in ("conflict", "all_company_synthesis"):
        # All retrieved versions must reach the LLM so it can surface
        # contradictions (conflict) or cross-company data (all_company_synthesis).
        # Per-company sub-retrieval handles versioning for synthesis queries.
        return list(retrieved)

    if intent != "latest":
        return list(retrieved)

    # Find the maximum (most-recent) doc_version per version group among
    # the retrieved set.  doc_version is "YYYY-MM", so lexicographic
    # comparison is equivalent to chronological comparison.
    latest_version: dict[tuple[str, str, str], str] = {}
    for rc in retrieved:
        key = version_group_key(rc.chunk)
        current_best = latest_version.get(key)
        if current_best is None or rc.chunk.doc_version > current_best:
            latest_version[key] = rc.chunk.doc_version

    return [
        rc
        for rc in retrieved
        if rc.chunk.doc_version == latest_version[version_group_key(rc.chunk)]
    ]


def expand_to_parents(
    retrieved: list[RetrievedChunk],
    conn=None,
) -> list[RetrievedChunk]:
    """Replace child chunks with their parent chunks for generation context.

    For each retrieved chunk that has a ``parent_chunk_id``, fetches the
    parent row from the ``chunks`` table and substitutes it as the chunk
    carried forward.  Multiple children that share the same parent are
    merged into a single ``RetrievedChunk`` whose ``rerank_score`` is the
    maximum of the children's scores (best signal wins).

    Chunks with no ``parent_chunk_id`` (already parents, or root-level
    chunks) are passed through unchanged.

    Input order is preserved: when the first child referencing a parent is
    encountered, the parent's position in the output list is fixed there.
    Subsequent children pointing to the same parent only update the score.

    Args:
        retrieved: Ordered list of retrieved chunks after deduplication.
        conn: An open ``psycopg.Connection``.  If ``None``, a connection is
            opened via ``src.db.connect()`` and closed automatically.

    Returns:
        Deduplicated list of ``RetrievedChunk`` objects with parent bodies,
        preserving the order of first appearance.
    """
    # Separate chunks that need parent expansion from passthrough chunks.
    needs_expansion: dict[UUID, RetrievedChunk] = {}
    parent_order: list[UUID] = []
    passthrough: list[tuple[int, RetrievedChunk]] = []

    position = 0
    for rc in retrieved:
        pid = rc.chunk.parent_chunk_id
        if pid is None:
            passthrough.append((position, rc))
            position += 1
        else:
            if pid not in needs_expansion:
                needs_expansion[pid] = rc
                parent_order.append(pid)
                position += 1
            else:
                # Merge: keep the higher rerank_score on the existing slot.
                existing = needs_expansion[pid]
                if (
                    rc.rerank_score is not None
                    and (
                        existing.rerank_score is None
                        or rc.rerank_score > existing.rerank_score
                    )
                ):
                    needs_expansion[pid] = RetrievedChunk(
                        chunk=existing.chunk,  # body not yet fetched
                        bm25_rank=existing.bm25_rank,
                        vector_rank=existing.vector_rank,
                        fused_score=existing.fused_score,
                        rerank_score=rc.rerank_score,
                    )

    if not needs_expansion:
        return [rc for _, rc in sorted(passthrough, key=lambda x: x[0])]

    # Fetch all required parent rows in a single query.
    parent_ids = list(needs_expansion.keys())

    def _fetch(connection) -> dict[UUID, Chunk]:
        placeholders = ", ".join(["%s"] * len(parent_ids))
        sql = f"""
            SELECT id, document_id, parent_chunk_id, company, ticker,
                   doc_type, report_date, period_covered, doc_version,
                   section_title, page_number, content_type, source_authority,
                   chunk_text, is_parent, token_count,
                   doc_subtype, page_content_class
            FROM chunks
            WHERE id IN ({placeholders})
        """
        with connection.cursor() as cur:
            cur.execute(sql, [str(pid) for pid in parent_ids])
            cols = [d.name for d in cur.description]
            return {chunk.id: chunk for chunk in (Chunk.from_row(row, cols) for row in cur.fetchall())}

    if conn is not None:
        parent_map = _fetch(conn)
    else:
        from src.db import connect
        with connect() as _conn:
            parent_map = _fetch(_conn)

    # Interleave parent-expanded slots with passthrough chunks by assigning
    # each output item its original position and sorting on that key.
    slot_map: dict[int, RetrievedChunk] = {}

    # Re-derive slot positions (passthrough already has them).
    for orig_pos, rc in passthrough:
        slot_map[orig_pos] = rc

    # Parent slots: rebuild with fetched parent body.
    parent_slot_position: dict[UUID, int] = {}
    slot = 0
    for rc in retrieved:
        pid = rc.chunk.parent_chunk_id
        if pid is None:
            slot += 1
        else:
            if pid not in parent_slot_position:
                parent_slot_position[pid] = slot
                slot += 1
            # else: already assigned

    for pid, winner_rc in needs_expansion.items():
        parent_chunk = parent_map.get(pid)
        if parent_chunk is None:
            # Parent row missing (unexpected); fall back to child.
            expanded = winner_rc
        else:
            expanded = RetrievedChunk(
                chunk=parent_chunk,
                bm25_rank=winner_rc.bm25_rank,
                vector_rank=winner_rc.vector_rank,
                fused_score=winner_rc.fused_score,
                rerank_score=winner_rc.rerank_score,
            )
        slot_map[parent_slot_position[pid]] = expanded

    return [rc for _, rc in sorted(slot_map.items())]
