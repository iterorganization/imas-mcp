"""Parallel static tree discovery engine.

Main entry point for static tree discovery with async workers. Orchestrates:
- Seed: Create TreeModelVersion nodes from facility config
- Extract: SSH extraction + immediate ingestion per version
- Units: Batched unit extraction for NUMERIC/SIGNAL nodes
- Enrich: LLM batch descriptions of tree nodes

Use ``run_parallel_static_discovery()`` as the main entry point.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
    supervised_worker,
)
from imas_codex.graph import GraphClient

from .graph_ops import (
    has_pending_enrich_work,
    has_pending_extract_work,
    has_pending_ingest_work,
    reset_orphaned_static_claims,
    seed_versions,
)
from .state import StaticDiscoveryState
from .workers import (
    enrich_worker,
    extract_worker,
    units_worker,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


async def run_parallel_static_discovery(
    facility: str,
    ssh_host: str,
    tree_name: str,
    tree_config: dict[str, Any],
    ver_list: list[int],
    *,
    cost_limit: float = 2.0,
    timeout: int = 600,
    batch_size: int = 40,
    enrich: bool = True,
    force: bool = False,
    num_extract_workers: int = 1,
    num_enrich_workers: int = 1,
    deadline: float | None = None,
    on_extract_progress: Callable | None = None,
    on_units_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
    dry_run: bool = False,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run parallel static tree discovery with async workers.

    Phases:
    1. Seed TreeModelVersion nodes from config (status=discovered)
    2. Extract workers claim versions, SSH extract, ingest to graph
    3. Units worker extracts units for latest version
    4. Enrich workers claim nodes, LLM describe, persist

    Returns:
        Dict with extraction/enrichment statistics.
    """
    start_time = time.time()

    # Release orphaned claims from previous runs
    reset_orphaned_static_claims(facility, tree_name, silent=True)

    # Ensure Facility node exists
    with GraphClient() as gc:
        gc.ensure_facility(facility)

    # Seed versions into graph
    version_configs = tree_config.get("versions", [])
    if force:
        # Force mode: reset ingested versions back to discovered
        _force_reset_versions(facility, tree_name, ver_list)

    seeded = seed_versions(facility, tree_name, ver_list, version_configs)
    logger.info(
        "Seeded %d new TreeModelVersion nodes for %s:%s (total %d versions)",
        seeded,
        facility,
        tree_name,
        len(ver_list),
    )

    state = StaticDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        tree_name=tree_name,
        tree_config=tree_config,
        cost_limit=cost_limit,
        timeout=timeout,
        batch_size=batch_size,
        enrich=enrich,
        deadline=deadline,
        force=force,
        service_monitor=service_monitor,
    )

    # Wire up graph-backed has_work_fn on each phase
    state.extract_phase.set_has_work_fn(
        lambda: has_pending_extract_work(facility, tree_name)
    )
    state.units_phase.set_has_work_fn(lambda: False)  # units runs once then exits
    state.enrich_phase.set_has_work_fn(
        lambda: (
            (enrich and has_pending_enrich_work(facility, tree_name))
            or not state.extract_phase.done
        )
    )
    state.ingest_phase.set_has_work_fn(
        lambda: has_pending_ingest_work(facility, tree_name)
    )

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

    worker_group = SupervisedWorkerGroup()

    # --- Extract workers ---
    if not dry_run:
        for i in range(num_extract_workers):
            worker_name = f"extract_worker_{i}"
            status = worker_group.create_status(worker_name, group="extract")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        extract_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_extract_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.extract_phase.mark_done()

    # --- Units worker (single instance) ---
    if not dry_run:
        worker_name = "units_worker_0"
        status = worker_group.create_status(worker_name, group="units")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    units_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_units_progress,
                    status_tracker=status,
                )
            )
        )
    else:
        state.units_phase.mark_done()

    # --- Enrich workers ---
    if enrich and not dry_run:
        for i in range(num_enrich_workers):
            worker_name = f"enrich_worker_{i}"
            status = worker_group.create_status(worker_name, group="enrich")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        enrich_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_enrich_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.enrich_phase.mark_done()

    # Ingest is done inline in extract_worker, mark done immediately
    state.ingest_phase.mark_done()

    # Periodic orphan recovery
    orphan_tick = make_orphan_recovery_tick(
        facility,
        [
            OrphanRecoverySpec("TreeModelVersion"),
            OrphanRecoverySpec("TreeNode", timeout_seconds=300),
        ],
    )

    await run_supervised_loop(
        worker_group,
        state.should_stop,
        on_worker_status=on_worker_status,
        on_tick=orphan_tick,
    )
    state.stop_requested = True

    elapsed = time.time() - start_time

    return {
        "versions_extracted": state.extract_stats.processed,
        "units_found": state.units_stats.processed,
        "nodes_enriched": state.enrich_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "extract_errors": state.extract_stats.errors,
        "enrich_errors": state.enrich_stats.errors,
    }


def _force_reset_versions(facility: str, tree_name: str, ver_list: list[int]) -> int:
    """Reset ingested versions back to discovered for re-extraction."""
    ids = [f"{facility}:{tree_name}:v{v}" for v in ver_list]
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS vid
            MATCH (v:TreeModelVersion {id: vid})
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
