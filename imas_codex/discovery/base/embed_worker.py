"""Shared embedding workers for all discovery engines.

Two async workers that poll the graph for nodes with text but no
embeddings, embed them in batches, and persist the results.  Designed
to run alongside LLM scoring workers in any discovery CLI.

``embed_description_worker``
    Targets nodes with a ``description`` field (FacilityPath,
    FacilitySignal, WikiPage, etc.).  Round-robin across labels.

``embed_text_worker``
    Targets chunk nodes with a ``text`` field (CodeChunk, WikiChunk).
    Runs asynchronously so the ingestion pipeline can write chunks
    without blocking on GPU embedding.

Both workers are label-agnostic and accept an explicit label list.

Integration:
    from imas_codex.discovery.base.embed_worker import (
        embed_description_worker,
        embed_text_worker,
    )

    # Description embeddings (wiki/signals/paths pattern):
    WorkerSpec("embed", "enrich_phase", embed_description_worker,
              group="embed", kwargs={"labels": ["CodeExample"]})

    # Chunk text embeddings (code/wiki pattern):
    WorkerSpec("chunk_embed", "code_phase", embed_text_worker,
              group="embed", kwargs={"labels": ["CodeChunk"]})
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Default batch size for embedding operations
DEFAULT_EMBED_BATCH_SIZE = 100

# Default batch size for text (chunk) embeddings — smaller than
# description batches because code chunks are up to 10 KB each.
# The pipeline pre-splits at 4K chars (~1K tokens); the embed server
# adaptively sub-batches at 16 for texts >2K chars.  24 items per
# request balances GPU utilisation against graph query overhead on
# P100-16GB.
DEFAULT_TEXT_EMBED_BATCH_SIZE = 24

# How long to sleep when no work is found (seconds)
IDLE_SLEEP = 3.0

# Cypher fragments that join from node ``n`` to the ancestor carrying
# ``score_composite``.  When a ``min_score`` filter is active the
# fragment is appended to the MATCH/WHERE of ``_fetch_unembedded`` and
# ``_count_unembedded``.  Labels not listed here are assumed to carry
# ``score_composite`` directly on the node.
_SCORE_JOINS: dict[str, str] = {
    "CodeChunk": (
        "MATCH (n)<-[:HAS_CHUNK]-(:CodeExample)<-[:HAS_EXAMPLE]-(scored:CodeFile)\n"
        "WHERE scored.score_composite >= $min_score"
    ),
    "CodeExample": (
        "MATCH (n)-[:FROM_FILE]->(scored:CodeFile)\n"
        "WHERE scored.score_composite >= $min_score"
    ),
}

# Maximum backoff when embed server is down (seconds)
ERROR_BACKOFF_MAX = 60.0

# Target text length (chars) for embedding.  Qwen3-Embedding-0.6B supports
# 32K tokens (~100K chars) but self-attention is O(n²) in sequence length —
# long texts exhaust GPU VRAM.  4000 chars ≈ 1K tokens which is safe for
# P100-16GB batches of 6 chunks.  Oversized chunks are split at this boundary.
TARGET_EMBED_TEXT_CHARS = 4000


def _split_oversized_text(
    text: str, max_chars: int = TARGET_EMBED_TEXT_CHARS
) -> list[str]:
    """Split oversized text into embeddable segments.

    Splits on newline boundaries to preserve code structure. Each segment
    is embedded separately and the results are averaged.

    Args:
        text: Text that may exceed max_chars
        max_chars: Maximum chars per segment

    Returns:
        List of text segments, each <= max_chars
    """
    if len(text) <= max_chars:
        return [text]

    segments: list[str] = []
    lines = text.split("\n")
    current_segment: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > max_chars and current_segment:
            segments.append("\n".join(current_segment))
            current_segment = []
            current_len = 0
        current_segment.append(line)
        current_len += line_len

    if current_segment:
        segments.append("\n".join(current_segment))

    return segments


def _get_description_labels() -> list[str]:
    """Derive embeddable labels from schema (nodes with description + embedding)."""
    from imas_codex.graph.schema import get_schema

    return get_schema().description_embeddable_labels


def _get_text_labels() -> list[str]:
    """Derive text-embeddable labels from schema (nodes with text + embedding)."""
    from imas_codex.graph.schema import get_schema

    return get_schema().text_embeddable_labels


def _mark_embed_failed(label: str, ids: list[str]) -> None:
    """Mark items as permanently failed for embedding.

    Sets ``embed_failed_at`` so these nodes are excluded from future
    embedding attempts.  To retry, clear the property manually.
    """
    if not ids:
        return
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            f"""
            UNWIND $ids AS item_id
            MATCH (n:{label} {{id: item_id}})
            SET n.embed_failed_at = datetime()
            """,
            ids=ids,
        )
    logger.warning("Marked %d %s items as embed-failed", len(ids), label)


def _fetch_unembedded(
    label: str,
    facility: str | None,
    batch_size: int,
    text_field: str = "description",
    min_score: float | None = None,
    score_joins: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch nodes with text but no embedding.

    Args:
        label: Node label (e.g., "FacilityPath")
        facility: Optional facility filter
        batch_size: Max items to fetch
        text_field: Property containing text to embed
        min_score: Minimum score threshold.  When set, nodes below
            this score (or whose scored ancestor is below it) are
            excluded.
        score_joins: Per-label Cypher fragments that traverse from
            ``n`` to the ancestor carrying ``score_composite``.
            Falls back to ``_SCORE_JOINS`` then to a direct property
            check on ``n``.

    Returns:
        List of dicts with 'id' and text field
    """
    from imas_codex.graph import GraphClient

    facility_filter = ""
    params: dict[str, Any] = {"batch_size": batch_size}

    if facility:
        facility_filter = "AND n.facility_id = $facility"
        params["facility"] = facility

    score_filter = ""
    if min_score is not None:
        params["min_score"] = min_score
        joins = score_joins or _SCORE_JOINS
        if label in joins:
            score_filter = joins[label]
        else:
            score_filter = "WITH n WHERE n.score_composite >= $min_score"

    query = f"""
        MATCH (n:{label})
        WHERE n.{text_field} IS NOT NULL
          AND n.{text_field} <> ''
          AND trim(n.{text_field}) <> ''
          AND n.embedding IS NULL
          AND n.embed_failed_at IS NULL
          {facility_filter}
        {score_filter}
        RETURN n.id AS id, n.{text_field} AS text
        ORDER BY rand()
        LIMIT $batch_size
    """

    with GraphClient() as gc:
        result = gc.query(query, **params)
        return [{"id": r["id"], "text": r["text"]} for r in result]


def _count_unembedded(
    label: str,
    facility: str | None,
    text_field: str = "description",
    min_score: float | None = None,
    score_joins: dict[str, str] | None = None,
) -> int:
    """Count nodes with text but no embedding."""
    from imas_codex.graph import GraphClient

    facility_filter = ""
    params: dict[str, Any] = {}

    if facility:
        facility_filter = "AND n.facility_id = $facility"
        params["facility"] = facility

    score_filter = ""
    if min_score is not None:
        params["min_score"] = min_score
        joins = score_joins or _SCORE_JOINS
        if label in joins:
            score_filter = joins[label]
        else:
            score_filter = "WITH n WHERE n.score_composite >= $min_score"

    query = f"""
        MATCH (n:{label})
        WHERE n.{text_field} IS NOT NULL
          AND n.{text_field} <> ''
          AND trim(n.{text_field}) <> ''
          AND n.embedding IS NULL
          AND n.embed_failed_at IS NULL
          {facility_filter}
        {score_filter}
        RETURN count(n) AS total
    """

    with GraphClient() as gc:
        result = gc.query(query, **params)
        return result[0]["total"] if result else 0


def _persist_embeddings(
    label: str,
    items: list[dict[str, Any]],
) -> int:
    """Persist embeddings back to graph nodes.

    Sets both the ``embedding`` vector and an ``embedded_at`` timestamp so
    downstream consumers can verify when the embedding was computed.

    Args:
        label: Node label
        items: List of dicts with 'id' and 'embedding' keys

    Returns:
        Number of nodes updated
    """
    from imas_codex.graph import GraphClient

    # Filter to items that actually got an embedding
    items_with_emb = [i for i in items if i.get("embedding") is not None]
    if not items_with_emb:
        return 0

    batch = [{"id": i["id"], "embedding": i["embedding"]} for i in items_with_emb]

    with GraphClient() as gc:
        gc.query(
            f"""
            UNWIND $batch AS item
            MATCH (n:{label} {{id: item.id}})
            SET n.embedding = item.embedding,
                n.embedded_at = datetime()
            """,
            batch=batch,
        )

    return len(items_with_emb)


def embed_batch_sync(
    label: str,
    facility: str | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    text_field: str = "description",
    min_score: float | None = None,
    score_joins: dict[str, str] | None = None,
) -> tuple[int, int, list[str]]:
    """Synchronous single-batch embed for one label.

    Fetches unembedded nodes, embeds them, persists results.  Oversized
    texts are split into segments and their embeddings are averaged.

    Args:
        label: Node label to process
        facility: Optional facility filter
        batch_size: Max items per batch
        text_field: Property containing text to embed
        min_score: Minimum score threshold (propagated to fetch query)
        score_joins: Per-label Cypher score traversal fragments

    Returns:
        (fetched, embedded, ids) — counts and list of embedded node IDs
    """
    import numpy as np

    from imas_codex.embeddings.description import embed_descriptions_batch

    items = _fetch_unembedded(
        label,
        facility,
        batch_size,
        text_field=text_field,
        min_score=min_score,
        score_joins=score_joins,
    )
    if not items:
        return 0, 0, []

    node_ids = [item["id"] for item in items]

    # Split oversized texts into segments for embedding.
    # Track segment -> item mapping for averaging later.
    segments: list[dict[str, Any]] = []
    segment_to_item: list[int] = []  # segment index -> item index

    for i, item in enumerate(items):
        text = item.pop("text")
        text_segments = _split_oversized_text(text)
        if len(text_segments) > 1:
            logger.debug(
                "Split oversized %s %s (%d chars) into %d segments",
                label,
                item["id"],
                len(text),
                len(text_segments),
            )
        for seg in text_segments:
            segments.append(
                {"id": f"{item['id']}:seg{len(segments)}", "description": seg}
            )
            segment_to_item.append(i)

    # Embed all segments
    segments = embed_descriptions_batch(segments)

    # Average embeddings for items with multiple segments
    for i, item in enumerate(items):
        item_segments = [
            segments[j] for j in range(len(segments)) if segment_to_item[j] == i
        ]
        embeddings = [
            seg["embedding"]
            for seg in item_segments
            if seg.get("embedding") is not None
        ]
        if embeddings:
            if len(embeddings) == 1:
                item["embedding"] = embeddings[0]
            else:
                # Average and normalize
                avg = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                item["embedding"] = avg.tolist()
        else:
            item["embedding"] = None

    embedded = _persist_embeddings(label, items)

    # Batch failed (e.g. CUDA OOM) — fall back to one-at-a-time
    if embedded == 0 and len(items) > 1:
        logger.info(
            "Batch embed failed for %d %s items — retrying one-at-a-time",
            len(items),
            label,
        )
        failed_ids: list[str] = []
        for item in items:
            if item.get("embedding") is not None:
                continue
            # Re-fetch and retry single item
            single_items = _fetch_unembedded(
                label,
                facility,
                1,
                text_field=text_field,
                min_score=min_score,
                score_joins=score_joins,
            )
            if not single_items:
                break  # no more items to try
            single_item = single_items[0]
            text = single_item.pop("text")
            text_segments = _split_oversized_text(text)
            if len(text_segments) > 1:
                logger.debug(
                    "Split oversized %s %s (%d chars) into %d segments",
                    label,
                    single_item["id"],
                    len(text),
                    len(text_segments),
                )
            seg_items = [
                {"id": f"seg{j}", "description": seg}
                for j, seg in enumerate(text_segments)
            ]
            seg_items = embed_descriptions_batch(seg_items)
            embeddings = [s["embedding"] for s in seg_items if s.get("embedding")]
            if embeddings:
                if len(embeddings) == 1:
                    single_item["embedding"] = embeddings[0]
                else:
                    avg = np.mean(embeddings, axis=0)
                    norm = np.linalg.norm(avg)
                    if norm > 0:
                        avg = avg / norm
                    single_item["embedding"] = avg.tolist()
                _persist_embeddings(label, [single_item])
                embedded += 1
            else:
                # Item failed even individually — mark as permanently
                # failed so it stops poisoning future batches.
                failed_ids.append(single_item["id"])

        if failed_ids:
            _mark_embed_failed(label, failed_ids)

    return len(items), embedded, node_ids


async def embed_description_worker(
    state: Any,
    *,
    labels: list[str] | None = None,
    facility: str | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    on_progress: Callable | None = None,
    min_score: float | None = None,
    score_joins: dict[str, str] | None = None,
    done_check: Callable[[], bool] | None = None,
) -> None:
    """Async worker that continuously embeds descriptions for specified labels.

    Polls the graph for nodes with descriptions but no embeddings,
    embeds them in batches, and persists the results. Runs until
    the parent state signals stop or ``done_check`` returns True
    while the worker is idle.

    The worker processes labels round-robin to distribute embedding
    work evenly across node types.

    Args:
        state: Discovery state object (must have facility attribute and
               a way to signal stop — checked via should_stop() or
               stop_requested attribute)
        labels: Node labels to process. Defaults to all embeddable labels.
        facility: Override facility (defaults to state.facility)
        batch_size: Items per embedding batch
        on_progress: Optional progress callback(msg, stats)
        min_score: Minimum score threshold.  Defaults to
            ``state.min_score`` when available.
        score_joins: Per-label Cypher traversal fragments for score
            filtering (defaults to built-in ``_SCORE_JOINS``).
        done_check: Optional callable returning True when the worker
            should exit (e.g. ``lambda: state.enrich_phase.done``).
            Checked when idle — allows embed workers to terminate
            when their upstream phase has completed.
    """
    from imas_codex.discovery.base.progress import WorkerStats

    worker_id = id(asyncio.current_task())
    facility = facility or getattr(state, "facility", None)
    labels = labels or _get_description_labels()
    stats = WorkerStats()

    # Resolve min_score from explicit kwarg → state.min_score → None
    if min_score is None:
        min_score = getattr(state, "min_score", None)

    logger.info(
        "embed_worker started (task=%s, labels=%s, facility=%s, min_score=%s)",
        worker_id,
        labels,
        facility,
        min_score,
    )

    def _should_stop() -> bool:
        """Check stop condition across different state types."""
        if hasattr(state, "should_stop") and callable(state.should_stop):
            return state.should_stop()
        return getattr(state, "stop_requested", False)

    # Initialize encoder eagerly so first batch doesn't pay cold-start cost
    try:
        await asyncio.to_thread(_warmup_encoder)
        logger.debug("embed_worker %s: encoder warmed up", worker_id)
    except Exception as e:
        logger.warning(
            "embed_worker %s: encoder warmup failed: %s (will retry per batch)",
            worker_id,
            e,
        )

    idle_count = 0
    error_count = 0

    while not _should_stop():
        # Gate on service monitor embed health if available
        monitor = getattr(state, "service_monitor", None)
        if monitor is not None:
            embed_status = monitor.get_service_status("embed")
            if embed_status and not embed_status.is_healthy:
                logger.debug(
                    "embed_worker %s: embed service unhealthy, waiting",
                    worker_id,
                )
                if on_progress:
                    on_progress("waiting (embed server down)", stats)
                await asyncio.sleep(ERROR_BACKOFF_MAX)
                continue

        total_fetched = 0
        total_embedded = 0
        had_errors = False

        # Process each label round-robin
        for label in labels:
            if _should_stop():
                break

            batch_start = _time.monotonic()
            try:
                fetched, embedded, _ids = await asyncio.to_thread(
                    embed_batch_sync,
                    label,
                    facility,
                    batch_size,
                    min_score=min_score,
                    score_joins=score_joins,
                )
            except Exception as e:
                logger.warning(
                    "embed_worker %s: error embedding %s: %s",
                    worker_id,
                    label,
                    e,
                )
                stats.errors += 1
                error_count += 1
                had_errors = True
                continue

            total_fetched += fetched
            total_embedded += embedded

            if embedded > 0:
                batch_elapsed = _time.monotonic() - batch_start
                stats.processed += embedded
                stats.last_batch_time = batch_elapsed
                stats.record_batch(embedded)
                error_count = 0  # Reset error streak on success
                logger.debug(
                    "embed_worker %s: embedded %d/%d %s",
                    worker_id,
                    embedded,
                    fetched,
                    label,
                )
                if on_progress:
                    results = [{"id": nid, "label": label} for nid in _ids[:embedded]]
                    on_progress(f"embedded {embedded} {label}", stats, results)

        # Backoff logic: distinguish idle (no work) from errors (work but
        # embed server failing).  When items are fetched but none embedded,
        # the embed server is down — back off exponentially.
        # Important: exceptions during fetch/embed cause `continue`,
        # leaving total_fetched=0 even though work exists.  Treat this
        # as an error, not idle, so idle_count does not falsely advance.
        if total_fetched == 0 and not had_errors:
            idle_count += 1
            error_count = 0
            # Exit when idle and phase is done (no more upstream work coming)
            if done_check is not None and idle_count >= 3 and done_check():
                logger.info(
                    "embed_worker %s: phase done and idle — exiting",
                    worker_id,
                )
                break
            if on_progress and idle_count <= 3:
                on_progress("idle", stats)
            await asyncio.sleep(min(IDLE_SLEEP * idle_count, 15.0))
        elif total_fetched == 0 and had_errors:
            # Errors prevented fetching — back off but don't count as idle
            backoff = min(IDLE_SLEEP * (2**error_count), ERROR_BACKOFF_MAX)
            if on_progress:
                on_progress(f"embed errors (backing off {backoff:.0f}s)", stats)
            await asyncio.sleep(backoff)
        elif total_embedded == 0:
            # Fetched items but failed to embed any — server is down
            error_count += 1
            idle_count = 0
            backoff = min(IDLE_SLEEP * (2**error_count), ERROR_BACKOFF_MAX)
            logger.info(
                "embed_worker %s: fetched %d but embedded 0 "
                "(embed server likely down), backing off %.0fs",
                worker_id,
                total_fetched,
                backoff,
            )
            if on_progress:
                on_progress(f"embed failed (backing off {backoff:.0f}s)", stats)
            await asyncio.sleep(backoff)
        else:
            idle_count = 0
            error_count = 0

    logger.info(
        "embed_worker %s: stopped after embedding %d descriptions",
        worker_id,
        stats.processed,
    )


async def embed_text_worker(
    state: Any,
    *,
    labels: list[str] | None = None,
    facility: str | None = None,
    batch_size: int = DEFAULT_TEXT_EMBED_BATCH_SIZE,
    on_progress: Callable | None = None,
    min_score: float | None = None,
    score_joins: dict[str, str] | None = None,
    done_check: Callable[[], bool] | None = None,
) -> None:
    """Async worker that embeds the ``text`` field of chunk nodes.

    Mirrors ``embed_description_worker`` but targets nodes whose
    searchable content lives in a ``text`` property (CodeChunk,
    WikiChunk) rather than ``description``.  This decouples embedding
    from the ingestion pipeline so chunks can be written to the graph
    immediately and embedded asynchronously by the GPU.

    Args:
        state: Discovery state object (must have facility attribute)
        labels: Node labels to process.  Defaults to all
            text-embeddable labels from the schema.
        facility: Override facility (defaults to state.facility)
        batch_size: Items per embedding batch
        on_progress: Optional progress callback(msg, stats)
        min_score: Minimum score threshold.  Defaults to
            ``state.min_score`` when available.
        score_joins: Per-label Cypher traversal fragments for score
            filtering (defaults to built-in ``_SCORE_JOINS``).
        done_check: Optional callable returning True when the worker
            should exit (e.g. ``lambda: state.code_phase.done``).
    """
    from imas_codex.discovery.base.progress import WorkerStats

    worker_id = id(asyncio.current_task())
    facility = facility or getattr(state, "facility", None)
    labels = labels or _get_text_labels()
    stats = WorkerStats()

    # Resolve min_score from explicit kwarg → state.min_score → None
    if min_score is None:
        min_score = getattr(state, "min_score", None)

    logger.info(
        "embed_text_worker started (task=%s, labels=%s, facility=%s, min_score=%s)",
        worker_id,
        labels,
        facility,
        min_score,
    )

    def _should_stop() -> bool:
        if hasattr(state, "should_stop") and callable(state.should_stop):
            return state.should_stop()
        return getattr(state, "stop_requested", False)

    # Warm up encoder eagerly
    try:
        await asyncio.to_thread(_warmup_encoder)
        logger.debug("embed_text_worker %s: encoder warmed up", worker_id)
    except Exception as e:
        logger.warning("embed_text_worker %s: encoder warmup failed: %s", worker_id, e)

    idle_count = 0
    error_count = 0

    while not _should_stop():
        total_fetched = 0
        total_embedded = 0
        had_errors = False

        for label in labels:
            if _should_stop():
                break

            batch_start = _time.monotonic()
            try:
                fetched, embedded, node_ids = await asyncio.to_thread(
                    embed_batch_sync,
                    label,
                    facility,
                    batch_size,
                    text_field="text",
                    min_score=min_score,
                    score_joins=score_joins,
                )
            except Exception as e:
                logger.warning(
                    "embed_text_worker %s: error embedding %s: %s",
                    worker_id,
                    label,
                    e,
                )
                stats.errors += 1
                error_count += 1
                had_errors = True
                continue

            total_fetched += fetched
            total_embedded += embedded

            if embedded > 0:
                batch_elapsed = _time.monotonic() - batch_start
                stats.processed += embedded
                stats.last_batch_time = batch_elapsed
                stats.record_batch(embedded)
                error_count = 0
                logger.debug(
                    "embed_text_worker %s: embedded %d/%d %s",
                    worker_id,
                    embedded,
                    fetched,
                    label,
                )
                if on_progress:
                    results = [
                        {"id": nid, "label": label} for nid in node_ids[:embedded]
                    ]
                    on_progress(f"embedded {embedded} {label}", stats, results)

        if total_fetched == 0 and not had_errors:
            idle_count += 1
            error_count = 0
            # Exit when idle and phase is done (no more upstream work coming)
            if done_check is not None and idle_count >= 3 and done_check():
                logger.info(
                    "embed_text_worker %s: phase done and idle — exiting",
                    worker_id,
                )
                break
            if on_progress and idle_count <= 3:
                on_progress("idle", stats)
            await asyncio.sleep(min(IDLE_SLEEP * idle_count, 15.0))
        elif total_fetched == 0 and had_errors:
            # Errors prevented fetching — back off but don't count as idle
            backoff = min(IDLE_SLEEP * (2**error_count), ERROR_BACKOFF_MAX)
            if on_progress:
                on_progress(f"embed errors (backing off {backoff:.0f}s)", stats)
            await asyncio.sleep(backoff)
        elif total_embedded == 0:
            error_count += 1
            idle_count = 0
            backoff = min(IDLE_SLEEP * (2**error_count), ERROR_BACKOFF_MAX)
            logger.info(
                "embed_text_worker %s: fetched %d but embedded 0, backing off %.0fs",
                worker_id,
                total_fetched,
                backoff,
            )
            if on_progress:
                on_progress(f"embed failed (backing off {backoff:.0f}s)", stats)
            await asyncio.sleep(backoff)
        else:
            idle_count = 0
            error_count = 0

    logger.info(
        "embed_text_worker %s: stopped after embedding %d chunks",
        worker_id,
        stats.processed,
    )


def _warmup_encoder() -> None:
    """Pre-initialize the encoder to avoid cold-start latency."""
    from imas_codex.embeddings.description import embed_description

    # Embed a tiny test string to warm up the model
    embed_description("warmup")
