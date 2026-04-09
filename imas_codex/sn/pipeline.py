"""SN build pipeline orchestrator.

Wires the EXTRACT → COMPOSE → [REVIEW] → VALIDATE → PERSIST workers into
the generic discovery engine and runs them with supervision and progress
tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.sn.state import SNBuildState
from imas_codex.sn.workers import (
    compose_worker,
    extract_worker,
    persist_worker,
    review_worker,
    validate_worker,
)

logger = logging.getLogger(__name__)


async def run_sn_build_engine(
    state: SNBuildState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN build pipeline.

    Pipeline::

        EXTRACT → COMPOSE → [REVIEW] → VALIDATE → PERSIST

    Extract queries the graph for DD paths, builds cluster-based batches.
    Compose uses LLM to generate standard names from the batches.
    Review uses a different model family to cross-check composed names.
    Validate checks grammar compliance via round-trip + fields consistency.
    Persist writes validated names to graph with provenance.

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
            "persist",
            "persist_phase",
            persist_worker,
            depends_on=["validate_phase"],
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
    )
