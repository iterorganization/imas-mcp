"""
Parallel data signal discovery engine with async workers.

Architecture:
- Three async workers: Discover, Enrich, Check
- Graph + claimed_at timestamp for coordination (same pattern as wiki/paths)
- Status transitions:
  - discovered → enriched (LLM classification)
  - enriched → checked (data access test)
- Workers claim signals by setting claimed_at, release by clearing it
- Orphan recovery: signals with claimed_at > 5 min old are reclaimed

Resilience:
- Supervised workers with automatic restart on crash (via base.supervision)
- Exponential backoff on infrastructure errors (Neo4j, network, SSH)
- Graceful degradation when services are temporarily unavailable

Workflow:
1. SCAN: Enumerate signals from data sources (MDSplus trees, TDI functions)
2. ENRICH: LLM classification of physics_domain, description generation
3. CHECK: Test data access with example_shot, verify units/sign conventions
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    supervised_worker,
)
from imas_codex.graph import GraphClient
from imas_codex.graph.models import FacilitySignalStatus
from imas_codex.remote.executor import run_python_script

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Claim timeout - signals claimed longer than this are reclaimed
CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


def get_checkpoint_dir() -> Path:
    """Get checkpoint directory for data discovery, creating if needed.

    Returns ~/.local/share/imas-codex/checkpoints/data/
    """
    checkpoint_dir = Path.home() / ".local/share/imas-codex/checkpoints/data"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


# =============================================================================
# Data Discovery State
# =============================================================================


@dataclass
class DataDiscoveryState:
    """Shared state for parallel data discovery."""

    facility: str
    ssh_host: str | None = None

    # Data source configuration
    reference_shot: int | None = None
    tdi_path: str | None = None

    # Limits
    cost_limit: float = 10.0
    signal_limit: int | None = None
    focus: str | None = None

    # Worker stats
    discover_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    check_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    discover_idle_count: int = 0
    enrich_idle_count: int = 0
    check_idle_count: int = 0

    @property
    def total_cost(self) -> float:
        return self.enrich_stats.cost

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def signal_limit_reached(self) -> bool:
        if self.signal_limit is None:
            return False
        return self.enrich_stats.processed >= self.signal_limit

    def should_stop(self) -> bool:
        """Check if ALL workers should terminate."""
        if self.stop_requested:
            return True
        all_idle = (
            self.discover_idle_count >= 3
            and self.enrich_idle_count >= 3
            and self.check_idle_count >= 3
        )
        if all_idle:
            if has_pending_work(self.facility):
                self.discover_idle_count = 0
                self.enrich_idle_count = 0
                self.check_idle_count = 0
                return False
            return True
        return False

    def should_stop_discovering(self) -> bool:
        """Check if discover workers should stop."""
        if self.stop_requested:
            return True
        if self.discover_idle_count >= 3:
            return True
        return False

    def should_stop_enriching(self) -> bool:
        """Check if enrich workers should stop."""
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.signal_limit_reached:
            return True
        return False

    def should_stop_checking(self) -> bool:
        """Check if check workers should stop."""
        if self.stop_requested:
            return True
        if self.check_idle_count >= 3:
            # Only stop if enriching is done AND no pending validation work
            enriching_done = self.enrich_idle_count >= 3 or self.budget_exhausted
            if enriching_done and not has_pending_check_work(self.facility):
                return True
        return False


# =============================================================================
# Graph Queries
# =============================================================================


def has_pending_work(facility: str) -> bool:
    """Check if there's any pending work for this facility."""
    return has_pending_enrich_work(facility) or has_pending_check_work(facility)


def has_pending_enrich_work(facility: str) -> bool:
    """Check if there are signals awaiting enrichment."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $discovered
                  AND s.claimed_at IS NULL
                RETURN count(s) > 0 AS has_work
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
            )
            return result[0]["has_work"] if result else False
    except Exception:
        return False


def has_pending_check_work(facility: str) -> bool:
    """Check if there are enriched signals awaiting check."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $enriched
                  AND s.claimed_at IS NULL
                RETURN count(s) > 0 AS has_work
                """,
                facility=facility,
                enriched=FacilitySignalStatus.enriched.value,
            )
            return result[0]["has_work"] if result else False
    except Exception:
        return False


def get_data_discovery_stats(facility: str) -> dict[str, Any]:
    """Get current discovery statistics from graph.

    Returns counts of signals by status, pending work, and accumulated
    enrichment cost for historical tracking across runs.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                RETURN
                    count(s) AS total,
                    sum(CASE WHEN s.status = $discovered THEN 1 ELSE 0 END) AS discovered,
                    sum(CASE WHEN s.status = $enriched THEN 1 ELSE 0 END) AS enriched,
                    sum(CASE WHEN s.status = $checked THEN 1 ELSE 0 END) AS checked,
                    sum(CASE WHEN s.status = $skipped THEN 1 ELSE 0 END) AS skipped,
                    sum(CASE WHEN s.status = $failed THEN 1 ELSE 0 END) AS failed,
                    sum(CASE WHEN s.status = $discovered AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_enrich,
                    sum(CASE WHEN s.status = $enriched AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_check,
                    sum(coalesce(s.enrichment_cost, 0)) AS accumulated_cost
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                enriched=FacilitySignalStatus.enriched.value,
                checked=FacilitySignalStatus.checked.value,
                skipped=FacilitySignalStatus.skipped.value,
                failed=FacilitySignalStatus.failed.value,
            )
            if result:
                return dict(result[0])
            return {}
    except Exception as e:
        logger.warning("Could not get discovery stats: %s", e)
        return {}


def reset_transient_signals(facility: str, silent: bool = False) -> dict[str, int]:
    """Reset orphaned signals from previous runs.

    Clears claimed_at for any signal that's been claimed too long.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                f"""
                MATCH (s:FacilitySignal {{facility_id: $facility}})
                WHERE s.claimed_at IS NOT NULL
                  AND s.claimed_at < datetime() - duration('PT{CLAIM_TIMEOUT_SECONDS}S')
                SET s.claimed_at = null
                RETURN count(s) AS released
                """,
                facility=facility,
            )
            released = result[0]["released"] if result else 0
            if released > 0 and not silent:
                logger.info(
                    "Released %d orphaned signal claims for %s", released, facility
                )
            return {"released": released}
    except Exception as e:
        logger.warning("Could not reset transient signals: %s", e)
        return {"released": 0}


def clear_facility_signals(
    facility: str,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Clear all signal discovery data for a facility.

    Deletes FacilitySignal nodes, DataAccess nodes, TreeModelVersion
    (epoch) nodes, and clears epoch checkpoint files. Always cascades.

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch

    Returns:
        Dict with counts: signals_deleted, data_access_deleted,
        epochs_deleted, checkpoints_deleted
    """
    results = {
        "signals_deleted": 0,
        "data_access_deleted": 0,
        "epochs_deleted": 0,
        "checkpoints_deleted": 0,
    }

    try:
        with GraphClient() as gc:
            # Delete FacilitySignal nodes in batches
            while True:
                result = gc.query(
                    """
                    MATCH (s:FacilitySignal {facility_id: $facility})
                    WITH s LIMIT $batch_size
                    DETACH DELETE s
                    RETURN count(s) AS deleted
                    """,
                    facility=facility,
                    batch_size=batch_size,
                )
                deleted = result[0]["deleted"] if result else 0
                results["signals_deleted"] += deleted
                if deleted < batch_size:
                    break

            # Delete orphaned DataAccess nodes for this facility
            result = gc.query(
                """
                MATCH (da:DataAccess {facility_id: $facility})
                WHERE NOT EXISTS { MATCH (da)<-[:DATA_ACCESS]-() }
                DETACH DELETE da
                RETURN count(da) AS deleted
                """,
                facility=facility,
            )
            results["data_access_deleted"] = result[0]["deleted"] if result else 0

            # Delete TreeModelVersion (epoch) nodes in batches
            while True:
                result = gc.query(
                    """
                    MATCH (v:TreeModelVersion {facility_id: $facility})
                    WITH v LIMIT $batch_size
                    DETACH DELETE v
                    RETURN count(v) AS deleted
                    """,
                    facility=facility,
                    batch_size=batch_size,
                )
                deleted = result[0]["deleted"] if result else 0
                results["epochs_deleted"] += deleted
                if deleted < batch_size:
                    break

        # Clear epoch checkpoint files
        checkpoint_dir = get_checkpoint_dir()
        for checkpoint_file in checkpoint_dir.glob(f"{facility}_*_epochs.json"):
            try:
                checkpoint_file.unlink()
                results["checkpoints_deleted"] += 1
            except Exception as e:
                logger.warning("Could not delete checkpoint %s: %s", checkpoint_file, e)

        logger.info(
            "Cleared signals for %s: %d signals, %d data_access, %d epochs, %d checkpoints",
            facility,
            results["signals_deleted"],
            results["data_access_deleted"],
            results["epochs_deleted"],
            results["checkpoints_deleted"],
        )
        return results

    except Exception as e:
        logger.error("Failed to clear facility signals: %s", e)
        raise


# =============================================================================
# Signal Claim/Release
# =============================================================================


def claim_signals_for_enrichment(
    facility: str,
    batch_size: int = 10,
) -> list[dict]:
    """Claim a batch of discovered signals for enrichment.

    Returns signals sorted by tdi_function to enable batching by function.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $discovered
                  AND s.claimed_at IS NULL
                WITH s ORDER BY s.tdi_function, s.id LIMIT $batch_size
                SET s.claimed_at = datetime()
                RETURN s.id AS id, s.accessor AS accessor, s.tree_name AS tree_name,
                       s.node_path AS node_path, s.units AS units, s.name AS name,
                       s.tdi_function AS tdi_function
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                batch_size=batch_size,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning("Could not claim signals for enrichment: %s", e)
        return []


def claim_signals_for_check(
    facility: str,
    batch_size: int = 5,
    reference_shot: int | None = None,
) -> list[dict]:
    """Claim a batch of enriched signals for check.

    Uses reference_shot from config for TDI-based checking.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $enriched
                  AND s.claimed_at IS NULL
                WITH s LIMIT $batch_size
                SET s.claimed_at = datetime()
                // Derive data_access ID based on signal type
                WITH s,
                     CASE WHEN s.tdi_function IS NOT NULL
                          THEN $facility + ':tdi:functions'
                          WHEN s.tree_name IS NOT NULL
                          THEN $facility + ':mdsplus:tree_tdi'
                          ELSE null END AS derived_data_access
                RETURN s.id AS id, s.accessor AS accessor, s.tree_name AS tree_name,
                       s.physics_domain AS physics_domain, s.tdi_function AS tdi_function,
                       COALESCE(s.data_access, derived_data_access) AS data_access
                """,
                facility=facility,
                enriched=FacilitySignalStatus.enriched.value,
                batch_size=batch_size,
            )
            # Add reference_shot to each result
            signals = list(result) if result else []
            if reference_shot:
                for sig in signals:
                    sig["check_shot"] = reference_shot
            return signals
    except Exception as e:
        logger.warning("Could not claim signals for check: %s", e)
        return []


def mark_signals_enriched(
    signals: list[dict],
    batch_cost: float = 0.0,
) -> int:
    """Mark signals as enriched with LLM-generated metadata.

    Expected signal dict keys:
    - id: signal ID
    - physics_domain: physics domain value
    - description: physics description
    - name: human-readable name
    - diagnostic: diagnostic system name (optional)
    - analysis_code: analysis code name (optional)
    - keywords: searchable keywords (optional)

    Args:
        signals: List of signal enrichment results
        batch_cost: Total LLM cost for this batch (distributed across signals)
    """
    if not signals:
        return 0

    # Calculate per-signal cost
    per_signal_cost = batch_cost / len(signals) if batch_cost > 0 else 0.0

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $signals AS sig
                MATCH (s:FacilitySignal {id: sig.id})
                SET s.status = $enriched,
                    s.physics_domain = sig.physics_domain,
                    s.description = sig.description,
                    s.name = sig.name,
                    s.diagnostic = CASE WHEN sig.diagnostic IS NOT NULL AND sig.diagnostic <> ''
                                        THEN sig.diagnostic ELSE s.diagnostic END,
                    s.analysis_code = CASE WHEN sig.analysis_code IS NOT NULL AND sig.analysis_code <> ''
                                           THEN sig.analysis_code ELSE s.analysis_code END,
                    s.keywords = CASE WHEN sig.keywords IS NOT NULL
                                      THEN sig.keywords ELSE s.keywords END,
                    s.enrichment_cost = $per_signal_cost,
                    s.enriched_at = datetime(),
                    s.claimed_at = null
                """,
                signals=signals,
                enriched=FacilitySignalStatus.enriched.value,
                per_signal_cost=per_signal_cost,
            )
        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals enriched: %s", e)
        return 0


def mark_signals_checked(
    signals: list[dict],
) -> int:
    """Mark signals as checked and create CHECKED_VIA relationship to DataAccess.

    Creates a CHECKED_VIA relationship from each FacilitySignal to its DataAccess,
    indicating that the accessor was tested and works with this data access node.
    This leaves checked signals linked to verified working data access nodes.

    Expected signal dict keys:
    - id: signal ID
    - data_access: DataAccess ID (optional, creates CHECKED_VIA if present)
    """
    if not signals:
        return 0

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $signals AS sig
                MATCH (s:FacilitySignal {id: sig.id})
                SET s.status = $checked,
                    s.checked = true,
                    s.checked_at = datetime(),
                    s.claimed_at = null
                WITH s, sig
                WHERE sig.data_access IS NOT NULL
                MATCH (da:DataAccess {id: sig.data_access})
                MERGE (s)-[:CHECKED_VIA]->(da)
                """,
                signals=signals,
                checked=FacilitySignalStatus.checked.value,
            )
        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals checked: %s", e)
        return 0


def mark_signal_skipped(signal_id: str, reason: str) -> None:
    """Mark a signal as skipped."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (s:FacilitySignal {id: $id})
                SET s.status = $skipped,
                    s.skip_reason = $reason,
                    s.claimed_at = null
                """,
                id=signal_id,
                skipped=FacilitySignalStatus.skipped.value,
                reason=reason,
            )
    except Exception as e:
        logger.warning("Could not mark signal %s as skipped: %s", signal_id, e)


def mark_signal_failed(signal_id: str, error: str, revert_status: str) -> None:
    """Mark a signal as failed with error message."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (s:FacilitySignal {id: $id})
                SET s.status = $failed,
                    s.validation_error = $error,
                    s.claimed_at = null
                """,
                id=signal_id,
                failed=FacilitySignalStatus.failed.value,
                error=error,
            )
    except Exception as e:
        logger.warning("Could not mark signal %s as failed: %s", signal_id, e)


def release_signal_claim(signal_id: str) -> None:
    """Release claim on a signal without changing status."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (s:FacilitySignal {id: $id})
                SET s.claimed_at = null
                """,
                id=signal_id,
            )
    except Exception as e:
        logger.warning("Could not release signal claim: %s", e)


# =============================================================================
# Epoch Detection Queries
# =============================================================================


def get_tree_epochs(facility: str, tree_name: str) -> list[dict]:
    """Get existing epochs for a tree from the graph.

    Returns list of epoch dicts with id, version, first_shot, last_shot.
    Empty list if no epochs exist.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (v:TreeModelVersion {facility_id: $facility, tree_name: $tree})
                RETURN v.id AS id, v.version AS version,
                       v.first_shot AS first_shot, v.last_shot AS last_shot,
                       v.node_count AS node_count
                ORDER BY v.version
                """,
                facility=facility,
                tree=tree_name,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning("Could not get epochs for %s:%s: %s", facility, tree_name, e)
        return []


def get_latest_epoch_shot(facility: str, tree_name: str) -> int | None:
    """Get the first_shot of the latest epoch for incremental scanning."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (v:TreeModelVersion {facility_id: $facility, tree_name: $tree})
                RETURN max(v.first_shot) AS latest_shot
                """,
                facility=facility,
                tree=tree_name,
            )
            return (
                result[0]["latest_shot"]
                if result and result[0]["latest_shot"]
                else None
            )
    except Exception as e:
        logger.warning("Could not get latest epoch shot: %s", e)
        return None


def ingest_epochs(
    epochs: list[dict],
    data_access_id: str | None = None,
    reference_shot: int | None = None,
) -> dict[str, int]:
    """Ingest epochs with symmetric signal lifecycle tracking.

    Creates TreeModelVersion nodes and FacilitySignal nodes with proper
    INTRODUCED_IN/REMOVED_IN relationships. Processes epochs in version order
    to maintain temporal consistency.

    Symmetric pattern:
    - added_paths → Create signal + INTRODUCED_IN edge to this epoch
    - removed_paths → Create REMOVED_IN edge to this epoch

    Args:
        epochs: List of epoch dicts from discover_epochs_optimized()
        data_access_id: Optional DataAccess ID for signal creation
        reference_shot: Optional reference shot for signal metadata

    Returns:
        Dict with counts: epochs, signals_created, introduced_edges, removed_edges
    """
    if not epochs:
        return {
            "epochs": 0,
            "signals_created": 0,
            "introduced_edges": 0,
            "removed_edges": 0,
        }

    # Sort epochs by version to process in temporal order
    sorted_epochs = sorted(epochs, key=lambda e: e["version"])

    results = {
        "epochs": 0,
        "signals_created": 0,
        "introduced_edges": 0,
        "removed_edges": 0,
    }

    try:
        with GraphClient() as gc:
            # Phase 1: Create all TreeModelVersion nodes
            clean_epochs = []
            for e in sorted_epochs:
                clean = {
                    "id": e["id"],
                    "tree_name": e["tree_name"],
                    "facility_id": e["facility_id"],
                    "version": e["version"],
                    "first_shot": e["first_shot"],
                    "last_shot": e.get("last_shot"),
                    "node_count": e.get("node_count", 0),
                    "nodes_added": e.get("nodes_added", 0),
                    "nodes_removed": e.get("nodes_removed", 0),
                    "added_subtrees": e.get("added_subtrees", []),
                    "removed_subtrees": e.get("removed_subtrees", []),
                }
                if e.get("predecessor"):
                    clean["predecessor"] = e["predecessor"]
                clean_epochs.append(clean)

            gc.query(
                """
                UNWIND $epochs AS ep
                MERGE (v:TreeModelVersion {id: ep.id})
                ON CREATE SET v += ep,
                              v.discovery_date = datetime()
                ON MATCH SET v.node_count = ep.node_count,
                             v.last_shot = ep.last_shot,
                             v.nodes_added = ep.nodes_added,
                             v.nodes_removed = ep.nodes_removed,
                             v.added_subtrees = ep.added_subtrees,
                             v.removed_subtrees = ep.removed_subtrees
                """,
                epochs=clean_epochs,
            )
            results["epochs"] = len(clean_epochs)

            # Phase 2: Create predecessor relationships
            gc.query(
                """
                UNWIND $epochs AS ep
                WITH ep WHERE ep.predecessor IS NOT NULL
                MATCH (v:TreeModelVersion {id: ep.id})
                MATCH (pred:TreeModelVersion {id: ep.predecessor})
                MERGE (v)-[:PRECEDED_BY]->(pred)
                """,
                epochs=clean_epochs,
            )

            # Phase 3: Process signal lifecycle (in version order for proper sequencing)
            for epoch in sorted_epochs:
                epoch_id = epoch["id"]
                facility_id = epoch["facility_id"]
                tree_name = epoch["tree_name"]
                added_paths = epoch.get("added_paths", [])
                removed_paths = epoch.get("removed_paths", [])

                # Create signals from added_paths with INTRODUCED_IN edge
                if added_paths:
                    signals = []
                    for path in added_paths:
                        name = path.split(":")[-1].split(".")[-1]
                        signal_id = f"{facility_id}:general/{tree_name}/{name.lower()}"
                        signals.append(
                            {
                                "id": signal_id,
                                "facility_id": facility_id,
                                "physics_domain": "general",
                                "name": name,
                                "accessor": f"data({path})",
                                "data_access": data_access_id
                                or f"{facility_id}:mdsplus:tree_tdi",
                                "tree_name": tree_name,
                                "node_path": path,
                                "units": "",
                                "status": FacilitySignalStatus.discovered.value,
                                "discovery_source": "epoch_detection",
                                "example_shot": reference_shot or epoch["first_shot"],
                                "epoch_id": epoch_id,
                            }
                        )

                    # Create signals and INTRODUCED_IN edges atomically
                    result = gc.query(
                        """
                        UNWIND $signals AS sig
                        MERGE (s:FacilitySignal {id: sig.id})
                        ON CREATE SET s += sig,
                                      s.discovered_at = datetime()
                        ON MATCH SET s.claimed_at = null
                        WITH s, sig
                        MATCH (v:TreeModelVersion {id: sig.epoch_id})
                        MERGE (s)-[:INTRODUCED_IN]->(v)
                        RETURN count(s) AS created
                        """,
                        signals=signals,
                    )
                    results["signals_created"] += len(signals)
                    results["introduced_edges"] += len(signals)

                # Create REMOVED_IN edges for removed_paths
                if removed_paths:
                    result = gc.query(
                        """
                        UNWIND $paths AS path
                        MATCH (s:FacilitySignal {facility_id: $facility_id})
                        WHERE s.node_path = path
                        MATCH (v:TreeModelVersion {id: $epoch_id})
                        WHERE NOT (s)-[:REMOVED_IN]->(:TreeModelVersion)
                        MERGE (s)-[:REMOVED_IN]->(v)
                        RETURN count(*) AS removed
                        """,
                        paths=removed_paths,
                        facility_id=facility_id,
                        epoch_id=epoch_id,
                    )
                    results["removed_edges"] += result[0]["removed"] if result else 0

        logger.info(
            "Ingested %d epochs: %d signals created, %d INTRODUCED_IN, %d REMOVED_IN",
            results["epochs"],
            results["signals_created"],
            results["introduced_edges"],
            results["removed_edges"],
        )
        return results

    except Exception as e:
        logger.error("Failed to ingest epochs: %s", e)
        return results


# =============================================================================
# MDSplus Discovery
# =============================================================================


def discover_mdsplus_signals(
    facility: str,
    ssh_host: str,
    tree_name: str,
    shot: int,
    data_access_id: str,
) -> list[dict]:
    """Discover signals from an MDSplus tree via SSH.

    Args:
        facility: Facility ID
        ssh_host: SSH host for remote access
        tree_name: MDSplus tree name
        shot: Reference shot number
        data_access_id: ID of DataAccess for this tree

    Returns:
        List of signal dicts ready for graph insertion
    """
    # Python script to run on remote
    remote_script = f'''
import json
import MDSplus

tree = MDSplus.Tree("{tree_name}", {shot}, "readonly")
nodes = list(tree.getNodeWild("***"))

signals = []
for node in nodes:
    try:
        usage = str(node.usage)
        # Only include data-bearing nodes
        if usage not in ("SIGNAL", "NUMERIC", "AXIS"):
            continue

        path = str(node.path)

        # Extract units
        try:
            units = str(node.units).strip() if hasattr(node, "units") else ""
        except:
            units = ""

        # Extract node name for human-readable display
        name = path.split(":")[-1].split(".")[-1]

        signals.append({{
            "path": path,
            "name": name,
            "units": units or "",
            "usage": usage,
        }})
    except Exception:
        pass

print(json.dumps(signals))
'''

    # Escape for SSH
    escaped_script = remote_script.replace("'", "'\"'\"'")
    cmd = ["ssh", ssh_host, f"python3 -c '{escaped_script}'"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        raw_signals = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("SSH timeout discovering %s on %s", tree_name, facility)
        return []
    except subprocess.CalledProcessError as e:
        logger.error("SSH failed: %s", e.stderr[:200] if e.stderr else str(e))
        return []
    except json.JSONDecodeError as e:
        logger.error("Failed to parse MDSplus response: %s", e)
        return []

    # Convert to FacilitySignal format
    signals = []
    for raw in raw_signals:
        path = raw["path"]
        name = raw["name"]

        # Generate signal ID: facility:general/signal_name
        # Physics domain will be classified during enrichment
        signal_id = f"{facility}:general/{tree_name}/{name.lower()}"

        signals.append(
            {
                "id": signal_id,
                "facility_id": facility,
                "physics_domain": "general",  # Will be enriched
                "name": name,
                "accessor": f"data({path})",
                "data_access": data_access_id,
                "tree_name": tree_name,
                "node_path": path,
                "units": raw.get("units", ""),
                "status": FacilitySignalStatus.discovered.value,
                "discovery_source": "tree_traversal",
                "example_shot": shot,
            }
        )

    logger.info("Discovered %d signals from %s:%s", len(signals), facility, tree_name)
    return signals


def discover_tdi_signals(
    facility: str,
    ssh_host: str,
    tdi_path: str,
    data_access_id: str,
) -> list[dict]:
    """Discover signals from TDI function files via SSH.

    Uses the extract_tdi_functions.py script for proper parsing of .fun files.
    Extracts function metadata, supported quantities, and classifies physics domains.

    TDI functions like tcv_get() and tcv_eq() provide:
    - Physics-level abstraction over raw MDSplus paths
    - Built-in versioning and source selection
    - Sign convention handling

    Args:
        facility: Facility ID
        ssh_host: SSH host for remote access
        tdi_path: Path to TDI function directory
        data_access_id: ID of DataAccess for TDI access

    Returns:
        List of signal dicts ready for graph insertion
    """
    # Run the extraction script synchronously (will be called in to_thread)
    import asyncio

    from imas_codex.discovery.data.tdi import (
        build_signal_id,
        build_tdi_accessor,
        classify_tdi_quantity,
        extract_tdi_functions,
    )

    try:
        functions = asyncio.get_event_loop().run_until_complete(
            extract_tdi_functions(ssh_host, tdi_path)
        )
    except RuntimeError:
        # No event loop running, create a new one
        functions = asyncio.run(extract_tdi_functions(ssh_host, tdi_path))

    # Source selector values to skip (not actual quantities)
    SOURCE_SELECTORS = {
        "FBTE",
        "FBTE.M",
        "LIUQE",
        "LIUQE.M",
        "LIUQE2",
        "LIUQE.M2",
        "LIUQE.M3",
        "LIUQE3",
        "FLAT",
        "FLAT.M",
        "RAMP",
        "RAMP.M",
        "RUNS",
        "RUNS.M",
        "MAGNETICS",
        "PCS",
    }

    signals = []
    seen_accessors: set[str] = set()

    for func in functions:
        # Skip internal functions
        if func.name.startswith("_"):
            continue

        for q in func.quantities:
            qty_name = q["name"]

            # Skip source selectors
            if qty_name in SOURCE_SELECTORS:
                continue

            # Build accessor and deduplicate
            accessor = build_tdi_accessor(func.name, qty_name)
            if accessor in seen_accessors:
                continue
            seen_accessors.add(accessor)

            # Classify physics domain
            physics_domain = classify_tdi_quantity(qty_name, func.name)

            # Build signal ID
            signal_id = build_signal_id(facility, physics_domain, qty_name)

            signals.append(
                {
                    "id": signal_id,
                    "facility_id": facility,
                    "physics_domain": physics_domain.value,
                    "name": qty_name,
                    "accessor": accessor,
                    "data_access": data_access_id,
                    "tdi_function": func.name,
                    "tdi_quantity": qty_name,
                    "status": FacilitySignalStatus.discovered.value,
                    "discovery_source": "tdi_introspection",
                }
            )

    logger.info("Discovered %d TDI signals from %s", len(signals), facility)
    return signals


def ingest_discovered_signals(signals: list[dict]) -> int:
    """Ingest discovered signals to graph with epoch relationships.

    Creates FacilitySignal nodes and INTRODUCED_IN relationships to
    their TreeModelVersion epoch (if epoch_id is present).
    """
    if not signals:
        return 0

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $signals AS sig
                MERGE (s:FacilitySignal {id: sig.id})
                ON CREATE SET s += sig,
                              s.discovered_at = datetime()
                ON MATCH SET s.claimed_at = null
                WITH s, sig
                WHERE sig.epoch_id IS NOT NULL
                MATCH (v:TreeModelVersion {id: sig.epoch_id})
                MERGE (s)-[:INTRODUCED_IN]->(v)
                """,
                signals=signals,
            )
        return len(signals)
    except Exception as e:
        logger.error("Failed to ingest signals: %s", e)
        return 0


# =============================================================================
# Async Workers
# =============================================================================


async def scan_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that scans TDI function files for signals.

    Discovers signals from TDI .fun files, creates TDIFunction nodes
    for LLM enrichment context.
    """
    from imas_codex.discovery.data.tdi import (
        create_tdi_data_access,
        discover_tdi_signals,
        ingest_tdi_functions,
        ingest_tdi_signals,
    )

    # Scan TDI functions (primary data source)
    if state.tdi_path and not state.should_stop_discovering():
        if on_progress:
            on_progress(
                f"scanning TDI {state.tdi_path}",
                state.discover_stats,
                [{"tree_name": "TDI", "node_path": state.tdi_path}],
            )

        try:
            with GraphClient() as gc:
                # Create/verify TDI access method
                am = await create_tdi_data_access(gc, state.facility)

                # Discover signals and extract TDI function metadata
                signals, functions = await discover_tdi_signals(
                    facility=state.facility,
                    ssh_host=state.ssh_host,
                    tdi_path=state.tdi_path,
                    data_access_id=am.id,
                )

                if signals:
                    # Ingest TDI functions (stores source_code for LLM enrichment)
                    if functions:
                        func_count = await ingest_tdi_functions(
                            gc, state.facility, functions
                        )
                        logger.info("Ingested %d TDI functions", func_count)

                    # Ingest signals
                    count = await ingest_tdi_signals(gc, signals)
                    state.discover_stats.processed += count

                    if on_progress:
                        results = [
                            {
                                "id": s.id,
                                "tree_name": "TDI",
                                "node_path": s.accessor,
                                "signals_in_tree": count,
                            }
                            for s in signals[:20]
                        ]
                        on_progress(
                            f"discovered {count} TDI signals ({len(functions)} functions)",
                            state.discover_stats,
                            results,
                        )
                else:
                    logger.info("No TDI signals discovered from %s", state.tdi_path)
                    if on_progress:
                        on_progress(
                            "no TDI signals found",
                            state.discover_stats,
                        )

        except Exception as e:
            logger.error("TDI discovery failed: %s", e)
            if on_progress:
                on_progress(
                    f"TDI discovery failed: {e}",
                    state.discover_stats,
                )

    # Mark scan as complete
    state.discover_idle_count = 100  # Signal scan is fully done

    if on_progress:
        on_progress("scan complete", state.discover_stats)


async def enrich_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that enriches signals with LLM classification.

    Groups signals by TDI function and includes function source code as context.
    Uses Jinja2 prompt template with schema-injected physics domains.
    Uses centralized LLM access via acall_llm_structured() from base.llm.
    """
    from collections import defaultdict

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.discovery.data.models import SignalEnrichmentBatch
    from imas_codex.settings import get_model

    # Get model configured for enrichment task
    model = get_model("language")

    # Render system prompt once (contains physics domains from schema)
    system_prompt = render_prompt("discovery/signal-enrichment")

    # Cache for TDI function source code
    tdi_source_cache: dict[str, str] = {}

    async def get_tdi_source(func_name: str) -> str:
        """Fetch TDI function source code from graph (cached)."""
        if func_name in tdi_source_cache:
            return tdi_source_cache[func_name]

        try:
            with GraphClient() as gc:
                result = gc.query(
                    """
                    MATCH (f:TDIFunction {facility_id: $facility, name: $func_name})
                    RETURN f.source_code AS source_code
                    """,
                    facility=state.facility,
                    func_name=func_name,
                )
                if result and result[0]["source_code"]:
                    source = result[0]["source_code"]
                    tdi_source_cache[func_name] = source
                    return source
        except Exception as e:
            logger.debug("Could not fetch TDI source for %s: %s", func_name, e)

        tdi_source_cache[func_name] = ""
        return ""

    while not state.should_stop_enriching():
        # Claim batch of signals (sorted by tdi_function)
        # Batch size 50 - signals grouped by function so one source code per group
        signals = await asyncio.to_thread(
            claim_signals_for_enrichment,
            state.facility,
            batch_size=50,
        )

        if not signals:
            state.enrich_idle_count += 1
            if on_progress:
                on_progress("idle", state.enrich_stats)
            await asyncio.sleep(1.0)
            continue

        state.enrich_idle_count = 0

        if on_progress:
            on_progress("enriching batch", state.enrich_stats)

        # Group signals by TDI function for efficient source code inclusion
        func_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, signal in enumerate(signals):
            func_name = signal.get("tdi_function") or "_none_"
            func_groups[func_name].append((i, signal))

        # Build user prompt with function source code context
        user_lines = [
            f"Classify these {len(signals)} signals.\n",
            "Return results in the same order using signal_index (1-based).\n",
        ]

        # For each function group, include source code once then list signals
        signal_index = 0
        for func_name, indexed_signals in func_groups.items():
            if func_name != "_none_":
                # Fetch function source code
                source_code = await get_tdi_source(func_name)
                if source_code:
                    # Truncate very long source files (>8000 chars)
                    if len(source_code) > 8000:
                        source_code = source_code[:8000] + "\n... (truncated)"
                    user_lines.append(f"\n## TDI Function: {func_name}")
                    user_lines.append("```tdi")
                    user_lines.append(source_code)
                    user_lines.append("```")
                    user_lines.append("\nSignals from this function:")

            for _, signal in indexed_signals:
                signal_index += 1
                user_lines.append(f"\n### Signal {signal_index}")
                user_lines.append(f"accessor: {signal['accessor']}")
                user_lines.append(f"name: {signal.get('name', 'unknown')}")
                if signal.get("units"):
                    user_lines.append(f"units: {signal['units']}")
                if signal.get("tree_name"):
                    user_lines.append(f"tree_name: {signal['tree_name']}")
                if signal.get("node_path"):
                    user_lines.append(f"node_path: {signal['node_path']}")

        user_prompt = "\n".join(user_lines)

        # Call LLM with structured output using shared base infrastructure
        # acall_llm_structured handles: retry, backoff, noise suppression,
        # API key, model prefix, JSON sanitization, cost tracking
        try:
            batch_result, batch_cost, total_tokens = await acall_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=SignalEnrichmentBatch,
                temperature=0.3,
            )
        except ValueError as e:
            logger.warning(
                "LLM enrichment failed for batch of %d signals: %s",
                len(signals),
                e,
            )
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            continue
        except Exception as e:
            logger.warning("LLM error (non-retryable): %s", e)
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            continue

        # Track cost
        state.enrich_stats.cost += batch_cost

        logger.debug(
            "LLM response: %d tokens for %d signals (cost=$%.4f)",
            total_tokens,
            len(signals),
            batch_cost,
        )

        # Match results back to signals by index (1-based signal_index)
        enriched = []
        matched_indices = set()

        for result in batch_result.results:
            # signal_index is 1-based, list is 0-based
            idx = result.signal_index - 1
            if 0 <= idx < len(signals):
                signal = signals[idx]
                matched_indices.add(idx)
                enriched.append(
                    {
                        "id": signal["id"],
                        "physics_domain": result.physics_domain.value,
                        "description": result.description,
                        "name": result.name,
                        "diagnostic": result.diagnostic,
                        "analysis_code": result.analysis_code,
                        "keywords": result.keywords,
                    }
                )

        # Release claims for unmatched signals
        for idx, signal in enumerate(signals):
            if idx not in matched_indices:
                await asyncio.to_thread(release_signal_claim, signal["id"])

        # Update graph with enrichment cost for historical tracking
        if enriched:
            await asyncio.to_thread(mark_signals_enriched, enriched, batch_cost)
            state.enrich_stats.processed += len(enriched)

            if on_progress:
                on_progress("enriched batch", state.enrich_stats, enriched)


async def check_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that checks signals by testing data access.

    Uses highly batched SSH execution with grouping by (tree, shot) to minimize
    MDSplus tree open overhead. Uses reference_shot from config for TDI signals.
    """
    # Large batch size - signals are grouped by tree/shot on the remote side
    # so we can efficiently check many signals with minimal tree opens
    BATCH_SIZE = 100

    while not state.should_stop_checking():
        # Claim batch of signals with reference_shot
        signals = await asyncio.to_thread(
            claim_signals_for_check,
            state.facility,
            batch_size=BATCH_SIZE,
            reference_shot=state.reference_shot,
        )

        if not signals:
            state.check_idle_count += 1
            if on_progress:
                on_progress("idle", state.check_stats)
            await asyncio.sleep(1.0)
            continue

        state.check_idle_count = 0

        if on_progress:
            on_progress(f"checking {len(signals)} signals", state.check_stats)

        # Prepare batch for remote check script
        # Build map of signal ID -> data_access for CHECKED_VIA relationship
        batch_input = []
        signal_data_access: dict[str, str | None] = {}
        for signal in signals:
            # Use reference_shot from config
            shot = signal.get("check_shot") or state.reference_shot
            if not shot:
                logger.warning(
                    "Signal %s has no check_shot and no reference_shot", signal["id"]
                )
                await asyncio.to_thread(release_signal_claim, signal["id"])
                continue

            signal_id = signal["id"]
            signal_data_access[signal_id] = signal.get("data_access")

            # TDI signals use tcv_shot tree
            tree_name = signal.get("tree_name")
            if signal.get("tdi_function") and not tree_name:
                tree_name = "tcv_shot"

            batch_input.append(
                {
                    "id": signal_id,
                    "accessor": signal["accessor"],
                    "tree_name": tree_name or "tcv_shot",
                    "shot": shot,
                }
            )

        if not batch_input:
            continue

        # Execute batched check via optimized remote script (single SSH call)
        # The script groups signals by tree/shot for efficient processing
        checked = []
        failed = []
        try:
            # Use batched script with 30s timeout per tree/shot group
            output = await asyncio.to_thread(
                run_python_script,
                "check_signals_batch.py",
                {"signals": batch_input, "timeout_per_group": 30},
                ssh_host=state.ssh_host,
                timeout=60 + len(batch_input),  # Base + 1s per signal
            )

            # Parse results - handle empty output gracefully
            if not output or not output.strip():
                logger.warning(
                    "Check script returned empty output for %d signals",
                    len(batch_input),
                )
                for sig in batch_input:
                    await asyncio.to_thread(release_signal_claim, sig["id"])
                continue

            try:
                # Find JSON in output (may have stderr mixed in)
                json_line = None
                for line in output.split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        json_line = line
                        break

                if not json_line:
                    logger.warning(
                        "No JSON found in check output: %s",
                        output[:200] if output else "(empty)",
                    )
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])
                    continue

                response = json.loads(json_line)

                # Check for script-level error
                if "error" in response and not response.get("results"):
                    logger.warning("Check script error: %s", response["error"])
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])
                    continue

                results = response.get("results", [])
                stats = response.get("stats", {})

                if stats:
                    logger.debug(
                        "Check batch: %d signals in %d groups, %d success, %d failed",
                        stats.get("total", 0),
                        stats.get("groups", 0),
                        stats.get("success", 0),
                        stats.get("failed", 0),
                    )

                for result in results:
                    signal_id = result.get("id")
                    if result.get("success"):
                        checked.append(
                            {
                                "id": signal_id,
                                "shape": result.get("shape"),
                                "dtype": result.get("dtype"),
                                "data_access": signal_data_access.get(signal_id),
                            }
                        )
                    else:
                        failed.append(
                            {
                                "id": signal_id,
                                "error": result.get("error", "check failed"),
                            }
                        )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to parse check response (%s): %s",
                    e,
                    output[:200] if output else "(empty)",
                )
                for sig in batch_input:
                    await asyncio.to_thread(release_signal_claim, sig["id"])
                continue

        except subprocess.CalledProcessError as e:
            stderr = e.stderr[:200] if e.stderr else str(e)
            logger.warning("Check script failed (exit %d): %s", e.returncode, stderr)
            for sig in batch_input:
                await asyncio.to_thread(release_signal_claim, sig["id"])
            continue
        except subprocess.TimeoutExpired:
            logger.warning(
                "Check script timed out for batch of %d signals", len(batch_input)
            )
            for sig in batch_input:
                await asyncio.to_thread(release_signal_claim, sig["id"])
            continue
        except Exception as e:
            logger.warning("Failed to run check: %s", e)
            for sig in batch_input:
                await asyncio.to_thread(release_signal_claim, sig["id"])
            continue

        # Mark failed signals
        for fail in failed:
            await asyncio.to_thread(
                mark_signal_failed,
                fail["id"],
                fail["error"],
                FacilitySignalStatus.enriched.value,
            )

        # Update graph with checked signals
        if checked:
            await asyncio.to_thread(mark_signals_checked, checked)
            state.check_stats.processed += len(checked)

            if on_progress:
                results = [
                    {"id": v["id"], "success": True, "shape": v.get("shape")}
                    for v in checked
                ]
                on_progress("checked batch", state.check_stats, results)


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_parallel_data_discovery(
    facility: str,
    ssh_host: str | None = None,
    tdi_path: str | None = None,
    reference_shot: int | None = None,
    cost_limit: float = 10.0,
    signal_limit: int | None = None,
    focus: str | None = None,
    num_enrich_workers: int = 2,
    num_check_workers: int = 1,
    discover_only: bool = False,
    enrich_only: bool = False,
    on_discover_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_check_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
) -> dict[str, Any]:
    """Run parallel data discovery with async workers.

    Discovers signals from TDI function files, enriches with LLM classification,
    and validates data access.

    Args:
        facility: Facility ID (e.g., "tcv")
        ssh_host: SSH host for remote discovery
        tdi_path: Path to TDI function directory
        reference_shot: Reference shot for validation
        cost_limit: Maximum LLM cost in USD
        signal_limit: Maximum signals to process
        focus: Focus area for discovery
        num_enrich_workers: Number of enrich workers
        num_check_workers: Number of check workers
        discover_only: Only discover, don't enrich
        enrich_only: Only enrich discovered signals
        on_*_progress: Progress callbacks

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    # Get facility config for defaults
    if not ssh_host:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)

    # Reset orphaned claims
    reset_transient_signals(facility)

    # Initialize state
    state = DataDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        reference_shot=reference_shot,
        tdi_path=tdi_path,
        cost_limit=cost_limit,
        signal_limit=signal_limit,
        focus=focus,
    )

    # Create worker group
    worker_group = SupervisedWorkerGroup()

    # Start scan worker (unless enrich_only)
    if not enrich_only:
        worker_name = "scan_worker_0"
        status = worker_group.create_status(worker_name)
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    scan_worker,
                    worker_name,
                    state,
                    state.should_stop_discovering,
                    on_progress=on_discover_progress,
                    status_tracker=status,
                )
            )
        )

    if discover_only:
        # Report worker status
        if on_worker_status:
            on_worker_status(worker_group)

        # Wait for scan worker to complete
        try:
            while state.discover_idle_count < 100:
                if on_worker_status:
                    on_worker_status(worker_group)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Discovery cancelled")
        finally:
            state.stop_requested = True
            await worker_group.cancel_all()

        return {
            "discovered": state.discover_stats.processed,
            "enriched": 0,
            "checked": 0,
            "cost": 0.0,
            "elapsed_seconds": time.time() - start_time,
        }

    # Start enrich workers
    for i in range(num_enrich_workers):
        worker_name = f"enrich_worker_{i}"
        status = worker_group.create_status(worker_name)
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    enrich_worker,
                    worker_name,
                    state,
                    state.should_stop_enriching,
                    on_progress=on_enrich_progress,
                    status_tracker=status,
                )
            )
        )

    # Start check workers (unless enrich_only)
    if not enrich_only:
        for i in range(num_check_workers):
            worker_name = f"check_worker_{i}"
            status = worker_group.create_status(worker_name)
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        check_worker,
                        worker_name,
                        state,
                        state.should_stop_checking,
                        on_progress=on_check_progress,
                        status_tracker=status,
                    )
                )
            )

    # Report worker status
    if on_worker_status:
        on_worker_status(worker_group)

    # Wait for completion
    try:
        while not state.should_stop():
            if on_worker_status:
                on_worker_status(worker_group)
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        logger.info("Discovery cancelled")
    finally:
        state.stop_requested = True
        await worker_group.cancel_all()

    elapsed = time.time() - start_time
    return {
        "discovered": state.discover_stats.processed,
        "enriched": state.enrich_stats.processed,
        "checked": state.check_stats.processed,
        "cost": state.enrich_stats.cost,
        "elapsed_seconds": elapsed,
        "discover_rate": state.discover_stats.processed / elapsed if elapsed > 0 else 0,
        "enrich_rate": state.enrich_stats.processed / elapsed if elapsed > 0 else 0,
        "check_rate": state.check_stats.processed / elapsed if elapsed > 0 else 0,
    }
