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

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase

from .workers import image_fetch_worker, image_score_worker

logger = logging.getLogger(__name__)


@dataclass
class DocumentDiscoveryState(DiscoveryStateBase):
    """Shared state for document discovery workers."""

    ssh_host: str = ""
    cost_limit: float = 2.0
    min_score: float = 0.5
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

    @property
    def score_only(self) -> bool:
        return False

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
    state: DocumentDiscoveryState,
    *,
    num_image_workers: int = 2,
    num_vlm_workers: int = 1,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any = None,
) -> dict[str, Any]:
    """Run document discovery pipeline.

    Processes Document nodes through:
    1. Image workers: Fetch image Documents via SCP, create Image nodes
    2. VLM workers: Caption images and score relevance

    Args:
        state: Pre-built discovery state.
        num_image_workers: Number of fetch/process workers.
        num_vlm_workers: Number of VLM captioning workers.
        stop_event: Shutdown signal.
        on_worker_status: Callback for worker group status updates.

    Returns:
        Dict with discovery statistics.
    """
    start_time = time.time()
    facility = state.facility

    state.image_phase.set_has_work_fn(lambda: _has_pending_image_documents(facility))
    state.image_score_phase.set_has_work_fn(
        lambda: _has_pending_image_scores(facility) or not state.image_phase.done
    )

    workers = [
        WorkerSpec(
            "image",
            "image_phase",
            image_fetch_worker,
            count=num_image_workers,
        ),
        WorkerSpec(
            "vlm",
            "image_score_phase",
            image_score_worker,
            count=num_vlm_workers,
            enabled=not state.scan_only,
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
    )

    elapsed = time.time() - start_time

    return {
        "images_fetched": state.image_stats.processed,
        "images_captioned": state.image_score_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "image_errors": state.image_stats.errors,
        "vlm_errors": state.image_score_stats.errors,
    }
