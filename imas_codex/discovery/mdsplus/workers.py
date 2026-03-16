"""Async workers for parallel tree discovery.

Workers that process MDSplus trees through the pipeline:
- extract_worker: Claim SignalEpoch, SSH extract, ingest to graph
- units_worker: Batched unit extraction for NUMERIC/SIGNAL nodes
- promote_worker: Create FacilitySignal nodes from leaf DataNodes

Workers coordinate through graph_ops claim/mark functions using
claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from imas_codex.discovery.base.progress import format_count

from .state import TreeDiscoveryState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Extract Worker
# ============================================================================


async def extract_worker(
    state: TreeDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Extract worker: claim a SignalEpoch, SSH extract, ingest to graph.

    Claims SignalEpoch nodes with status=discovered, runs SSH extraction
    for that version, then immediately ingests the results into the graph.
    Each version is claimed-extracted-ingested as a unit.
    """
    from imas_codex.mdsplus.extraction import (
        async_extract_tree_version,
        ingest_static_tree,
        merge_version_results,
    )

    from .graph_ops import (
        claim_version_for_extraction,
        mark_version_extracted,
        release_version_claim,
    )

    ssh_retry_count = 0
    max_ssh_retries = 5

    while not state.should_stop():
        # Claim a pending version atomically
        claimed = await asyncio.to_thread(
            claim_version_for_extraction,
            state.facility,
            state.data_source_name,
        )

        if not claimed:
            state.extract_phase.record_idle()
            if state.extract_phase.done:
                break
            if on_progress:
                on_progress("idle", state.extract_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.extract_phase.record_activity(1)
        version = claimed["version"]
        extraction_shot = claimed.get("first_shot") or version
        version_id = claimed["id"]

        if on_progress:
            on_progress(
                f"extracting v{version} {state.facility}:{state.data_source_name}",
                state.extract_stats,
                [{"version": version, "phase": "extract"}],
            )

        try:
            # SSH extraction
            node_usages = state.tree_config.get("node_usages")
            data = await async_extract_tree_version(
                facility=state.facility,
                data_source_name=state.data_source_name,
                shot=extraction_shot,
                timeout=state.timeout,
                node_usages=node_usages,
            )
            ssh_retry_count = 0  # Reset on success

            ver_data = data.get("versions", {}).get(str(extraction_shot), {})
            if "error" in ver_data:
                logger.warning(
                    "Extraction returned error for v%d: %s",
                    version,
                    ver_data["error"][:100],
                )
                await asyncio.to_thread(release_version_claim, version_id)
                state.extract_stats.errors += 1
                if on_progress:
                    on_progress(
                        f"v{version} error: {ver_data['error'][:60]}",
                        state.extract_stats,
                        None,
                    )
                continue

            node_count = ver_data.get("node_count", 0)
            tags = len(ver_data.get("tags", {}))

            # Ingest immediately
            merged = merge_version_results([data])
            from imas_codex.graph import GraphClient

            with GraphClient() as client:
                stats = ingest_static_tree(client, state.facility, merged)

            # Mark version as extracted in graph
            await asyncio.to_thread(mark_version_extracted, version_id, node_count)

            state.extract_stats.processed += 1
            state.extract_stats.record_batch(1)

            if on_progress:
                on_progress(
                    f"v{version} done — {format_count(node_count)} nodes, {tags} tags",
                    state.extract_stats,
                    [
                        {
                            "version": version,
                            "node_count": node_count,
                            "tags": tags,
                            "nodes_created": stats.get("nodes_created", 0),
                        }
                    ],
                )

        except Exception as e:
            ssh_retry_count += 1
            logger.warning(
                "Extract v%d failed (%d/%d): %s",
                version,
                ssh_retry_count,
                max_ssh_retries,
                e,
            )
            state.extract_stats.errors += 1
            await asyncio.to_thread(release_version_claim, version_id)

            if ssh_retry_count >= max_ssh_retries:
                logger.error(
                    "SSH failed after %d attempts — extract worker stopping",
                    max_ssh_retries,
                )
                state.extract_phase.mark_done()
                if on_progress:
                    on_progress(
                        f"SSH failed: {str(e)[:100]}",
                        state.extract_stats,
                        None,
                    )
                break

            backoff = min(2**ssh_retry_count, 32)
            if on_progress:
                on_progress(
                    f"SSH retry {ssh_retry_count} in {backoff}s",
                    state.extract_stats,
                    None,
                )
            await asyncio.sleep(backoff)
            continue

        await asyncio.sleep(0.1)


# ============================================================================
# Units Worker
# ============================================================================


async def units_worker(
    state: TreeDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Units worker: extract units for the latest tree version.

    Runs batched unit extraction via SSH for NUMERIC/SIGNAL nodes,
    then creates Unit nodes and HAS_UNIT relationships in the graph.
    Tracks completion via SignalEpoch.units_extracted flag so
    re-runs are no-ops for already-processed versions.
    """
    from imas_codex.mdsplus.extraction import async_extract_units_for_version

    from .graph_ops import (
        get_latest_ingested_epoch_target,
        has_pending_units_work,
        mark_all_versions_units_extracted,
        merge_units_to_graph,
    )

    # Wait for at least one version to be extracted
    while not state.should_stop():
        if state.extract_phase.done:
            break
        if state.extract_stats.processed > 0:
            break
        if on_progress:
            on_progress("awaiting extract", state.units_stats, None)
        await asyncio.sleep(1.0)

    if state.should_stop():
        return

    # Check if units already extracted (graph is ledger)
    pending = await asyncio.to_thread(
        has_pending_units_work, state.facility, state.data_source_name
    )
    if not pending:
        if on_progress:
            on_progress("already extracted", state.units_stats, None)
        state.units_phase.mark_done()
        return

    latest_epoch = await asyncio.to_thread(
        get_latest_ingested_epoch_target, state.facility, state.data_source_name
    )
    latest_version = latest_epoch["latest_version"] if latest_epoch else 1
    latest_shot = latest_epoch["latest_shot"] if latest_epoch else latest_version

    if on_progress:
        on_progress(
            f"v{latest_version} — batched extraction",
            state.units_stats,
            None,
        )

    units_batch_size = 500

    def _on_ssh_progress(checked: int, total: int, found: int) -> None:
        state.units_stats.processed = checked
        state.units_stats.record_batch()
        if on_progress:
            batch_num = (checked // units_batch_size) + 1
            total_batches = (total + units_batch_size - 1) // units_batch_size
            on_progress(
                f"{format_count(checked)}/{format_count(total)} checked, {found} with units",
                state.units_stats,
                [
                    {
                        "checked": checked,
                        "total": total,
                        "found": found,
                        "batch": f"{batch_num}/{total_batches}",
                    }
                ],
            )

    try:
        units = await async_extract_units_for_version(
            state.facility,
            state.data_source_name,
            latest_shot,
            timeout=state.timeout,
            batch_size=units_batch_size,
            on_progress=_on_ssh_progress,
        )

        if units:
            # Create Unit nodes and HAS_UNIT relationships
            created = await asyncio.to_thread(
                merge_units_to_graph, state.facility, state.data_source_name, units
            )
            logger.info(
                "Created %d HAS_UNIT relationships for %d unique unit symbols",
                created,
                len(set(units.values())),
            )

            if on_progress:
                on_progress(
                    f"{len(units)} paths with units — complete",
                    state.units_stats,
                    [{"found": len(units), "status": "complete"}],
                )
        else:
            if on_progress:
                on_progress("no units found", state.units_stats, None)

        # Mark ALL ingested versions as units-extracted since units are
        # per-tree (shared nodes), not per-version.
        await asyncio.to_thread(
            mark_all_versions_units_extracted,
            state.facility,
            state.data_source_name,
            len(units) if units else 0,
        )

    except Exception as e:
        logger.exception("Units extraction failed")
        state.units_stats.errors += 1
        if on_progress:
            on_progress(
                f"failed: {str(e)[:80]}",
                state.units_stats,
                None,
            )

    # Units worker runs once then exits
    state.units_phase.mark_done()


# ============================================================================
# Promote Worker
# ============================================================================


async def promote_worker(
    state: TreeDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Promote worker: create FacilitySignal nodes from leaf DataNodes.

    After extraction and units complete, queries for leaf DataNodes
    (NUMERIC/SIGNAL usage) and creates FacilitySignal nodes with
    status=discovered. Descriptions come later from the signals
    enrichment pipeline.
    """
    from .graph_ops import promote_leaf_nodes_to_signals

    # Wait for extraction and units to complete
    while not state.should_stop():
        if state.extract_phase.done and state.units_phase.done:
            break
        if on_progress:
            on_progress("awaiting extract+units", state.promote_stats, None)
        await asyncio.sleep(1.0)

    if state.should_stop():
        return

    if on_progress:
        on_progress("promoting leaf nodes", state.promote_stats, None)

    try:
        promoted = await asyncio.to_thread(
            promote_leaf_nodes_to_signals,
            state.facility,
            state.data_source_name,
        )

        state.promote_stats.processed = promoted
        state.promote_stats.record_batch(promoted)

        if on_progress:
            on_progress(
                f"{promoted} signals promoted",
                state.promote_stats,
                [{"promoted": promoted}],
            )

    except Exception as e:
        logger.exception("Promote worker failed")
        state.promote_stats.errors += 1
        if on_progress:
            on_progress(
                f"failed: {str(e)[:80]}",
                state.promote_stats,
                None,
            )

    state.promote_phase.mark_done()
