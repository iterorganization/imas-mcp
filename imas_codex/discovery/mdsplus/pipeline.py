"""Parallel tree discovery engine.

Main entry point for MDSplus tree discovery with async workers. Orchestrates:
- Seed: Create StructuralEpoch nodes from config or epoch detection
- Extract: SSH extraction + immediate ingestion per version/shot
- Units: Batched unit extraction for NUMERIC/SIGNAL nodes
- Promote: Create FacilitySignal nodes from leaf DataNodes

Use ``run_tree_discovery()`` as the main entry point.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.supervision import OrphanRecoverySpec
from imas_codex.graph import GraphClient

from .graph_ops import (
    has_pending_extract_work,
    reset_orphaned_static_claims,
    seed_versions,
)
from .state import TreeDiscoveryState
from .workers import (
    extract_worker,
    promote_worker,
    units_worker,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

logger = logging.getLogger(__name__)


async def run_tree_discovery(
    facility: str,
    ssh_host: str,
    data_source_name: str,
    tree_config: dict[str, Any],
    ver_list: list[int],
    *,
    cost_limit: float = 2.0,
    timeout: int = 600,
    batch_size: int = 40,
    force: bool = False,
    num_extract_workers: int = 1,
    deadline: float | None = None,
    on_extract_progress: Callable | None = None,
    on_units_progress: Callable | None = None,
    on_promote_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
    dry_run: bool = False,
    stop_event: asyncio.Event | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run parallel tree discovery with async workers.

    Handles versioned (machine-description), shot-scoped (dynamic), and
    epoched MDSplus trees through a unified pipeline.

    Phases:
    1. Seed StructuralEpoch nodes from config or epoch detection
    2. Extract workers claim versions, SSH extract, ingest to graph
    3. Units worker extracts units for latest version
    4. Promote worker creates FacilitySignal nodes from leaf DataNodes

    Returns:
        Dict with extraction/promotion statistics.
    """
    start_time = time.time()

    # Release orphaned claims from previous runs
    reset_orphaned_static_claims(facility, data_source_name, silent=True)

    # Ensure Facility node exists
    with GraphClient() as gc:
        gc.ensure_facility(facility)

    # Seed versions into graph
    version_configs = tree_config.get("versions", [])
    if force:
        _force_reset_versions(facility, data_source_name, ver_list)

    seeded = seed_versions(facility, data_source_name, ver_list, version_configs)
    logger.info(
        "Seeded %d new StructuralEpoch nodes for %s:%s (total %d versions)",
        seeded,
        facility,
        data_source_name,
        len(ver_list),
    )

    state = TreeDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        data_source_name=data_source_name,
        tree_config=tree_config,
        cost_limit=cost_limit,
        timeout=timeout,
        batch_size=batch_size,
        deadline=deadline,
        force=force,
        service_monitor=service_monitor,
    )

    # Wire up graph-backed has_work_fn on each phase
    state.extract_phase.set_has_work_fn(
        lambda: has_pending_extract_work(facility, data_source_name)
    )
    state.units_phase.set_has_work_fn(lambda: False)  # units runs once then exits
    state.promote_phase.set_has_work_fn(lambda: False)  # promote runs once then exits

    # Pre-warm SSH ControlMaster
    logger.info("Pre-warming SSH ControlMaster to %s...", ssh_host)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-O", "check", ssh_host],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["ssh", ssh_host, "true"],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.warning("Failed to pre-warm SSH to %s: %s", ssh_host, e)

    # Declare workers
    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            extract_worker,
            count=num_extract_workers,
            enabled=not dry_run,
            on_progress=on_extract_progress,
        ),
        WorkerSpec(
            "units",
            "units_phase",
            units_worker,
            enabled=not dry_run,
            on_progress=on_units_progress,
        ),
        WorkerSpec(
            "promote",
            "promote_phase",
            promote_worker,
            enabled=not dry_run,
            on_progress=on_promote_progress,
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=[
            OrphanRecoverySpec("StructuralEpoch"),
            OrphanRecoverySpec("DataNode", timeout_seconds=300),
        ],
        on_worker_status=on_worker_status,
    )

    elapsed = time.time() - start_time

    return {
        "versions_extracted": state.extract_stats.processed,
        "units_found": state.units_stats.processed,
        "signals_promoted": state.promote_stats.processed,
        "elapsed_seconds": elapsed,
        "extract_errors": state.extract_stats.errors,
    }


# Backward-compatible alias
run_parallel_static_discovery = run_tree_discovery


def _force_reset_versions(
    facility: str, data_source_name: str, ver_list: list[int]
) -> int:
    """Reset ingested versions back to discovered for re-extraction."""
    ids = [f"{facility}:{data_source_name}:v{v}" for v in ver_list]
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS vid
            MATCH (v:StructuralEpoch {id: vid})
            WHERE v.status = 'ingested'
            SET v.status = 'discovered', v.claimed_at = null
            RETURN count(v) AS reset
            """,
            ids=ids,
        )
        count = result[0]["reset"] if result else 0
    if count:
        logger.info("Force-reset %d versions back to discovered", count)
    return count
