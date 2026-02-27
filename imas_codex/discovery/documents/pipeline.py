"""Document discovery pipeline.

Orchestrates document scanning, image fetching, and VLM captioning.
Operates on Document nodes created by the document scanner.

Separated from the code discovery pipeline for independent lifecycle,
cost budgets, and domain clears.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import (
    PipelinePhase,
    SupervisedWorkerGroup,
    run_supervised_loop,
    supervised_worker,
)

from .workers import image_fetch_worker, image_score_worker

logger = logging.getLogger(__name__)


@dataclass
class DocumentDiscoveryState:
    """Shared state for document discovery workers."""

    facility: str
    ssh_host: str
    cost_limit: float = 2.0
    min_score: float = 0.5
    deadline: float | None = None
    store_images: bool = False
    scan_only: bool = False
    focus: str | None = None

    # Worker stats
    image_stats: WorkerStats = field(default_factory=WorkerStats)
    image_score_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    image_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("image"))
    image_score_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("image_score")
    )

    # Control
    stop_requested: bool = False

    @property
    def score_only(self) -> bool:
        return False

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            return True
        return False

    @property
    def budget_exhausted(self) -> bool:
        return self.image_score_stats.cost >= self.cost_limit

    @property
    def total_cost(self) -> float:
        return self.image_score_stats.cost


def _has_pending_image_documents(facility: str) -> bool:
    """Check if there are Document nodes with document_type='image' pending processing."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (d:Document {facility_id: $facility, document_type: 'image'})
            WHERE d.status = 'discovered'
            RETURN count(d) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def _has_pending_image_scores(facility: str) -> bool:
    """Check if there are Image nodes pending VLM scoring."""

    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            WHERE img.status = 'ingested' AND img.claimed_at IS NULL
            RETURN count(img) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


async def run_document_discovery(
    facility: str,
    ssh_host: str,
    *,
    cost_limit: float = 2.0,
    min_score: float = 0.5,
    num_image_workers: int = 2,
    num_vlm_workers: int = 1,
    store_images: bool = False,
    scan_only: bool = False,
    focus: str | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run document discovery pipeline.

    Processes Document nodes through:
    1. Image workers: Fetch image Documents via SCP, create Image nodes
    2. VLM workers: Caption images and score relevance

    Args:
        facility: Facility ID
        ssh_host: SSH host alias
        cost_limit: Maximum VLM spend in USD
        min_score: Minimum document interest score
        num_image_workers: Number of fetch/process workers
        num_vlm_workers: Number of VLM captioning workers
        store_images: Keep image bytes in graph
        scan_only: Only fetch, skip VLM captioning
        focus: Natural language focus for VLM scoring
        deadline: Absolute time when discovery should stop

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    state = DocumentDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        cost_limit=cost_limit,
        min_score=min_score,
        deadline=deadline,
        store_images=store_images,
        scan_only=scan_only,
        focus=focus,
    )

    state.image_phase.set_has_work_fn(lambda: _has_pending_image_documents(facility))
    state.image_score_phase.set_has_work_fn(
        lambda: _has_pending_image_scores(facility) or not state.image_phase.done
    )

    worker_group = SupervisedWorkerGroup()

    for i in range(num_image_workers):
        worker_name = f"image_worker_{i}"
        status = worker_group.create_status(worker_name, group="image")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    image_fetch_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    status_tracker=status,
                )
            )
        )

    if not scan_only:
        for i in range(num_vlm_workers):
            worker_name = f"vlm_worker_{i}"
            status = worker_group.create_status(worker_name, group="vlm")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        image_score_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.image_score_phase.mark_done()

    logger.info(
        "Started %d document workers: fetch=%d vlm=%d",
        worker_group.get_active_count(),
        num_image_workers,
        num_vlm_workers if not scan_only else 0,
    )

    await run_supervised_loop(worker_group, state.should_stop)
    state.stop_requested = True

    elapsed = time.time() - start_time

    return {
        "images_fetched": state.image_stats.processed,
        "images_captioned": state.image_score_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "image_errors": state.image_stats.errors,
        "vlm_errors": state.image_score_stats.errors,
    }
