"""Shared embed-description worker for all discovery engines.

A single async worker that polls the graph for nodes with descriptions
but no embeddings, embeds them in batches, and persists the results.
Designed to run alongside LLM scoring workers in any discovery CLI.

The worker is label-agnostic: it accepts a list of node labels to process
(e.g., ["FacilityPath", "WikiPage", "WikiArtifact", "FacilitySignal", "Image"])
and embeds descriptions round-robin across all labels.

Integration:
    # In any discovery engine's run_parallel_* function:
    from imas_codex.discovery.base.embed_worker import embed_description_worker

    # Add as a supervised worker (wiki/data pattern):
    embed_status = worker_group.create_status("embed_worker", group="ingest")
    worker_group.add_task(
        asyncio.create_task(
            supervised_worker(
                embed_description_worker,
                "embed_worker",
                state,
                state.should_stop,
                labels=["WikiPage", "WikiArtifact", "FacilitySignal"],
                on_progress=on_embed_progress,
                status_tracker=embed_status,
            )
        )
    )

    # Or as a plain async task (paths pattern):
    embed_task = asyncio.create_task(
        embed_description_worker(
            state, labels=["FacilityPath"], on_progress=callback
        )
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Default batch size for embedding operations
DEFAULT_EMBED_BATCH_SIZE = 100

# How long to sleep when no work is found (seconds)
IDLE_SLEEP = 3.0

# Labels that use 'description' field for embedding
DESCRIPTION_LABELS = {
    "FacilityPath",
    "FacilitySignal",
    "TreeNode",
    "WikiArtifact",
    "WikiPage",
}

# Labels that use 'caption' as the text field for embedding
CAPTION_LABELS = {
    "Image",
}


def _text_field_for_label(label: str) -> str:
    """Return the text field name used for embedding a given label."""
    if label in CAPTION_LABELS:
        return "caption"
    return "description"


def _fetch_unembedded(
    label: str,
    facility: str | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Fetch nodes with text but no embedding.

    Args:
        label: Node label (e.g., "FacilityPath")
        facility: Optional facility filter
        batch_size: Max items to fetch

    Returns:
        List of dicts with 'id' and text field
    """
    from imas_codex.graph import GraphClient

    text_field = _text_field_for_label(label)
    facility_filter = ""
    params: dict[str, Any] = {"batch_size": batch_size}

    if facility:
        facility_filter = "AND n.facility_id = $facility"
        params["facility"] = facility

    query = f"""
        MATCH (n:{label})
        WHERE n.{text_field} IS NOT NULL
          AND n.{text_field} <> ''
          AND n.embedding IS NULL
          {facility_filter}
        RETURN n.id AS id, n.{text_field} AS text
        LIMIT $batch_size
    """

    with GraphClient() as gc:
        result = gc.query(query, **params)
        return [{"id": r["id"], "text": r["text"]} for r in result]


def _count_unembedded(
    label: str,
    facility: str | None,
) -> int:
    """Count nodes with text but no embedding."""
    from imas_codex.graph import GraphClient

    text_field = _text_field_for_label(label)
    facility_filter = ""
    params: dict[str, Any] = {}

    if facility:
        facility_filter = "AND n.facility_id = $facility"
        params["facility"] = facility

    query = f"""
        MATCH (n:{label})
        WHERE n.{text_field} IS NOT NULL
          AND n.{text_field} <> ''
          AND n.embedding IS NULL
          {facility_filter}
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
) -> tuple[int, int]:
    """Synchronous single-batch embed for one label.

    Fetches unembedded nodes, embeds them, persists results.

    Args:
        label: Node label to process
        facility: Optional facility filter
        batch_size: Max items per batch

    Returns:
        (fetched, embedded) counts
    """
    from imas_codex.embeddings.description import embed_descriptions_batch

    items = _fetch_unembedded(label, facility, batch_size)
    if not items:
        return 0, 0

    # embed_descriptions_batch expects 'description' key by default
    # Rename 'text' → 'description' for the batch embedder
    for item in items:
        item["description"] = item.pop("text")

    items = embed_descriptions_batch(items)
    embedded = _persist_embeddings(label, items)

    return len(items), embedded


async def embed_description_worker(
    state: Any,
    *,
    labels: list[str] | None = None,
    facility: str | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    on_progress: Callable | None = None,
) -> None:
    """Async worker that continuously embeds descriptions for specified labels.

    Polls the graph for nodes with descriptions but no embeddings,
    embeds them in batches, and persists the results. Runs until
    the parent state signals stop.

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
    """
    from imas_codex.discovery.base.progress import WorkerStats

    worker_id = id(asyncio.current_task())
    facility = facility or getattr(state, "facility", None)
    labels = labels or list(DESCRIPTION_LABELS | CAPTION_LABELS)
    stats = WorkerStats()

    logger.info(
        "embed_worker started (task=%s, labels=%s, facility=%s)",
        worker_id,
        labels,
        facility,
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

    while not _should_stop():
        total_fetched = 0
        total_embedded = 0

        # Process each label round-robin
        for label in labels:
            if _should_stop():
                break

            try:
                fetched, embedded = await asyncio.to_thread(
                    embed_batch_sync, label, facility, batch_size
                )
            except Exception as e:
                logger.warning(
                    "embed_worker %s: error embedding %s: %s",
                    worker_id,
                    label,
                    e,
                )
                stats.errors += 1
                continue

            total_fetched += fetched
            total_embedded += embedded

            if embedded > 0:
                stats.processed += embedded
                logger.debug(
                    "embed_worker %s: embedded %d/%d %s",
                    worker_id,
                    embedded,
                    fetched,
                    label,
                )
                if on_progress:
                    on_progress(f"embedded {embedded} {label}", stats)

        if total_fetched == 0:
            idle_count += 1
            if on_progress and idle_count <= 3:
                on_progress("idle", stats)
            # Exponential backoff when idle, capped at IDLE_SLEEP
            await asyncio.sleep(min(IDLE_SLEEP * idle_count, 15.0))
        else:
            idle_count = 0

    logger.info(
        "embed_worker %s: stopped after embedding %d descriptions",
        worker_id,
        stats.processed,
    )


def _warmup_encoder() -> None:
    """Pre-initialize the encoder to avoid cold-start latency."""
    from imas_codex.embeddings.description import embed_description

    # Embed a tiny test string to warm up the model
    embed_description("warmup")
