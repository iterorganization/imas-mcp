"""SN generate pipeline orchestrator.

Wires the EXTRACT → COMPOSE → [REVIEW] → VALIDATE → CONSOLIDATE → PERSIST
workers into the generic discovery engine and runs them with supervision
and progress tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.standard_names.state import SNBuildState
from imas_codex.standard_names.workers import (
    compose_worker,
    consolidate_worker,
    extract_worker,
    persist_worker,
    review_worker,
    validate_worker,
)

logger = logging.getLogger(__name__)


async def run_sn_generate_engine(
    state: SNBuildState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN generate pipeline.

    Pipeline::

        EXTRACT → COMPOSE → [REVIEW] → VALIDATE → CONSOLIDATE → PERSIST

    Extract queries the graph for DD paths, builds cluster-based batches.
    Compose uses LLM to generate standard names from the batches.
    Review uses a different model family to cross-check composed names.
    Validate checks grammar compliance via round-trip + fields consistency.
    Consolidate performs cross-batch dedup, conflict detection, and coverage.
    Persist writes consolidated names to graph with provenance.

    When ``state.skip_review`` is True the REVIEW phase is disabled and
    its ``PipelinePhase`` is marked done immediately, so VALIDATE does
    not block.

    Args:
        state: Populated ``SNBuildState`` with source and filter config.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """
    # When review is skipped, validate depends directly on compose
    validate_deps = ["review_phase"] if not state.skip_review else ["compose_phase"]

    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            extract_worker,
        ),
        WorkerSpec(
            "compose",
            "compose_phase",
            compose_worker,
            depends_on=["extract_phase"],
        ),
        WorkerSpec(
            "review",
            "review_phase",
            review_worker,
            depends_on=["compose_phase"],
            enabled=not state.skip_review,
        ),
        WorkerSpec(
            "validate",
            "validate_phase",
            validate_worker,
            depends_on=validate_deps,
        ),
        WorkerSpec(
            "consolidate",
            "consolidate_phase",
            consolidate_worker,
            depends_on=["validate_phase"],
        ),
        WorkerSpec(
            "persist",
            "persist_phase",
            persist_worker,
            depends_on=["consolidate_phase"],
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
    )
