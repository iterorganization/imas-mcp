"""SN generate pipeline orchestrator.

Wires the EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST
workers into the generic discovery engine and runs them with supervision
and progress tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.standard_names.state import StandardNameBuildState
from imas_codex.standard_names.workers import (
    compose_worker,
    consolidate_worker,
    extract_worker,
    persist_worker,
    validate_worker,
)

logger = logging.getLogger(__name__)


async def run_sn_pipeline(
    state: StandardNameBuildState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN generate pipeline.

    Pipeline::

        EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST

    Extract queries the graph for DD paths, builds cluster-based batches.
    Compose uses a reasoning model to generate standard names from the batches.
    Validate checks grammar compliance via round-trip + fields consistency.
    Consolidate performs cross-batch dedup, conflict detection, and coverage.
    Persist writes consolidated names to graph with provenance.

    Review/scoring is a separate process (``sn review``) that operates on the
    full catalog with cross-name visibility.

    Args:
        state: Populated ``StandardNameBuildState`` with source and filter config.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """

    # Downstream workers should run to completion even when cost limit is hit.
    # They only stop on CLI shutdown (stop_requested), not on budget.
    def _downstream_should_stop():
        return state.stop_requested

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
            "validate",
            "validate_phase",
            validate_worker,
            depends_on=["compose_phase"],
            group="finalize",
            should_stop_fn=_downstream_should_stop,
        ),
        WorkerSpec(
            "consolidate",
            "consolidate_phase",
            consolidate_worker,
            depends_on=["validate_phase"],
            group="finalize",
            should_stop_fn=_downstream_should_stop,
        ),
        WorkerSpec(
            "persist",
            "persist_phase",
            persist_worker,
            depends_on=["consolidate_phase"],
            group="finalize",
            should_stop_fn=_downstream_should_stop,
        ),
    ]

    # The supervised loop's stop function must NOT check budget_exhausted.
    # Budget only stops compose; downstream workers must run to completion.
    # Individual workers use their own should_stop_fn for fine-grained control.
    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
        stop_fn=lambda: state.stop_requested,
    )
