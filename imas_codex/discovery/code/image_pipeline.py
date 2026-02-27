"""Image discovery pipeline.

Standalone orchestration for image file discovery, fetch, and VLM captioning.
Operates on SourceFile nodes with file_category='image' that were created
by the code discovery scan phase.

Separated from the code discovery pipeline to allow independent lifecycle,
cost budgets, and worker scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    run_supervised_loop,
    supervised_worker,
)

from .workers import image_score_worker, image_worker

logger = logging.getLogger(__name__)


@dataclass
class ImageDiscoveryState:
    """Shared state for image discovery workers."""

    facility: str
    ssh_host: str
    cost_limit: float = 2.0
    min_score: float = 0.7
    max_paths: int = 50
    deadline: float | None = None
    store_images: bool = False
    scan_only: bool = False

    # Worker stats
    image_stats: WorkerStats = field(default_factory=WorkerStats)
    image_score_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False

    # Aliases for worker compatibility (workers expect these attribute names)
    @property
    def score_only(self) -> bool:
        return False

    @property
    def focus(self) -> str | None:
        return None

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


async def run_image_discovery(
    facility: str,
    ssh_host: str,
    *,
    cost_limit: float = 2.0,
    min_score: float = 0.7,
    max_paths: int = 50,
    num_image_workers: int = 2,
    num_vlm_workers: int = 1,
    store_images: bool = False,
    scan_only: bool = False,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run image discovery pipeline.

    Processes SourceFile nodes with file_category='image' through:
    1. Image workers: Fetch via SCP, downsample, create Image nodes
    2. VLM workers: Caption images and score relevance

    Args:
        facility: Facility ID
        ssh_host: SSH host alias
        cost_limit: Maximum VLM spend in USD
        min_score: Minimum source file score
        num_image_workers: Number of fetch/process workers
        num_vlm_workers: Number of VLM captioning workers
        store_images: Keep image bytes in graph
        scan_only: Only fetch, skip VLM captioning
        deadline: Absolute time when discovery should stop

    Returns:
        Dict with discovery statistics
    """
    from imas_codex.discovery.base.supervision import PipelinePhase

    start_time = time.time()

    state = ImageDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        cost_limit=cost_limit,
        min_score=min_score,
        max_paths=max_paths,
        deadline=deadline,
        store_images=store_images,
        scan_only=scan_only,
    )

    # Create pipeline phases for worker coordination
    image_phase = PipelinePhase("image")
    image_score_phase = PipelinePhase("image_score")

    # Attach phases to state for worker access
    state.image_phase = image_phase  # type: ignore[attr-defined]
    state.image_score_phase = image_score_phase  # type: ignore[attr-defined]

    from .graph_ops import has_pending_image_score_work, has_pending_image_work

    image_phase.set_has_work_fn(lambda: has_pending_image_work(facility))
    image_score_phase.set_has_work_fn(
        lambda: has_pending_image_score_work(facility) or not image_phase.done
    )

    worker_group = SupervisedWorkerGroup()

    for i in range(num_image_workers):
        worker_name = f"image_worker_{i}"
        status = worker_group.create_status(worker_name, group="image")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    image_worker,
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
        image_score_phase.mark_done()

    logger.info(
        "Started %d image workers: fetch=%d vlm=%d",
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
