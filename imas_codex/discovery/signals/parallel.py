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
    OrphanRecoverySpec,
    PipelinePhase,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
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
    scanner_types: list[str] = field(default_factory=list)

    # Legacy — kept for backwards compat with progress display
    tdi_path: str | None = None

    # Wiki context for enrichment (populated by wiki scanner)
    # Keyed by MDSplus path (uppercase), values have description/units/page
    wiki_context: dict[str, dict[str, str]] = field(default_factory=dict)

    # Limits
    cost_limit: float = 10.0
    signal_limit: int | None = None
    focus: str | None = None
    deadline: float | None = None  # Unix timestamp when discovery should stop

    # Worker stats
    discover_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    check_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    enrich_only: bool = False  # When True, discover/check workers not started

    # Pipeline phases (initialized in __post_init__)
    scan_phase: PipelinePhase = field(init=False)
    enrich_phase: PipelinePhase = field(init=False)
    check_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.scan_phase = PipelinePhase(
            "scan"
        )  # No graph check — scan is deterministic
        self.enrich_phase = PipelinePhase(
            "enrich",
            has_work_fn=lambda: has_pending_enrich_work(self.facility),
        )
        self.check_phase = PipelinePhase(
            "check",
            has_work_fn=lambda: has_pending_check_work(self.facility),
        )

    # Backwards-compat idle count properties for progress display
    @property
    def discover_idle_count(self) -> int:
        return self.scan_phase.idle_count

    @discover_idle_count.setter
    def discover_idle_count(self, value: int) -> None:
        # Support legacy assignment (e.g., enrich_only mode)
        if value >= 100:
            self.scan_phase.mark_done()
        else:
            self.scan_phase._idle_count = value

    @property
    def enrich_idle_count(self) -> int:
        return self.enrich_phase.idle_count

    @enrich_idle_count.setter
    def enrich_idle_count(self, value: int) -> None:
        self.enrich_phase._idle_count = value

    @property
    def check_idle_count(self) -> int:
        return self.check_phase.idle_count

    @check_idle_count.setter
    def check_idle_count(self, value: int) -> None:
        self.check_phase._idle_count = value

    @property
    def total_cost(self) -> float:
        return self.enrich_stats.cost

    @property
    def deadline_expired(self) -> bool:
        """Check if the deadline has been reached."""
        if self.deadline is None:
            return False
        return time.time() >= self.deadline

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def signal_limit_reached(self) -> bool:
        if self.signal_limit is None:
            return False
        return self.enrich_stats.processed >= self.signal_limit

    def should_stop(self) -> bool:
        """Check if ALL workers should terminate.

        Uses PipelinePhase.done which combines idle detection with
        graph-level pending work checks.  This replaces the old
        idle-counter approach that was prone to race conditions.
        """
        if self.stop_requested:
            return True
        if self.deadline_expired:
            return True

        limit_done = self.budget_exhausted or self.signal_limit_reached

        # LLM workers: phase done OR limit-stopped counts as "done"
        enrich_done = self.enrich_phase.done or limit_done

        # Check phase considers upstream phases
        check_done = self.check_phase.done

        # In enrich_only mode, scan and check phases are pre-marked done
        all_done = self.scan_phase.done and enrich_done and check_done
        if all_done:
            if self.enrich_only:
                return limit_done or self.enrich_phase.done
            return True
        return False

    def should_stop_discovering(self) -> bool:
        """Check if discover workers should stop."""
        if self.stop_requested:
            return True
        return self.scan_phase.done

    def should_stop_enriching(self) -> bool:
        """Check if enrich workers should stop."""
        if self.stop_requested:
            return True
        if self.deadline_expired:
            return True
        if self.budget_exhausted:
            return True
        if self.signal_limit_reached:
            return True
        return False

    def should_stop_checking(self) -> bool:
        """Check if check workers should stop.

        Check workers must wait for both upstream phases (scan + enrich)
        to finish producing work before exiting.  Uses PipelinePhase.done
        which checks the graph for remaining unclaimed/claimed items.
        """
        if self.stop_requested:
            return True
        if self.deadline_expired:
            return True
        if not self.check_phase.idle:
            return False
        # Idle — but don't exit if upstream is still producing
        if not self.scan_phase.done:
            return False
        # Scan done — wait for enrichment to drain too
        enriching_done = self.enrich_phase.done or self.budget_exhausted
        if not enriching_done:
            return False
        # Both upstream phases done — check graph for remaining work
        return self.check_phase.done


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

    Returns counts of signals by status, pending work, accumulated
    enrichment cost, and access check outcomes for historical tracking.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                OPTIONAL MATCH (s)-[c:CHECKED_WITH]->()
                RETURN
                    count(DISTINCT s) AS total,
                    sum(CASE WHEN s.status = $discovered THEN 1 ELSE 0 END) AS discovered,
                    sum(CASE WHEN s.status = $enriched THEN 1 ELSE 0 END) AS enriched,
                    sum(CASE WHEN s.status = $checked THEN 1 ELSE 0 END) AS checked,
                    sum(CASE WHEN s.status = $skipped THEN 1 ELSE 0 END) AS skipped,
                    sum(CASE WHEN s.status = $failed THEN 1 ELSE 0 END) AS failed,
                    sum(CASE WHEN s.status = $discovered AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_enrich,
                    sum(CASE WHEN s.status = $enriched AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_check,
                    sum(coalesce(s.llm_cost, 0)) AS accumulated_cost,
                    count(CASE WHEN c.success = true THEN 1 END) AS checks_passed,
                    count(CASE WHEN c.success = false THEN 1 END) AS checks_failed
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
    Delegates to the common claims module for the actual reset.
    """
    from imas_codex.discovery.base.claims import reset_stale_claims

    try:
        released = reset_stale_claims(
            "FacilitySignal",
            facility,
            timeout_seconds=CLAIM_TIMEOUT_SECONDS,
            silent=silent,
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
    Uses claimed_at timeout for orphan recovery (parallel-safe).
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $discovered
                  AND (s.claimed_at IS NULL
                       OR s.claimed_at < datetime() - duration($cutoff))
                WITH s ORDER BY s.tdi_function, s.id LIMIT $batch_size
                SET s.claimed_at = datetime()
                RETURN s.id AS id, s.accessor AS accessor, s.tree_name AS tree_name,
                       s.node_path AS node_path, s.units AS units, s.name AS name,
                       s.tdi_function AS tdi_function,
                       s.discovery_source AS discovery_source,
                       s.description AS description
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                batch_size=batch_size,
                cutoff=cutoff,
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
    Uses claimed_at timeout for orphan recovery (parallel-safe).
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $enriched
                  AND (s.claimed_at IS NULL
                       OR s.claimed_at < datetime() - duration($cutoff))
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
                       s.discovery_source AS discovery_source, s.name AS name,
                       COALESCE(s.data_access, derived_data_access) AS data_access
                """,
                facility=facility,
                enriched=FacilitySignalStatus.enriched.value,
                batch_size=batch_size,
                cutoff=cutoff,
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

    Also creates/links Diagnostic nodes: the original diagnostic casing is
    preserved on the FacilitySignal for data access, while a canonical
    lowercase Diagnostic node is MERGE'd and linked via BELONGS_TO_DIAGNOSTIC.

    Expected signal dict keys:
    - id: signal ID
    - physics_domain: physics domain value
    - description: physics description
    - name: human-readable name
    - diagnostic: diagnostic system name (optional)
    - analysis_code: analysis code name (optional)
    - keywords: searchable keywords (optional)
    - sign_convention: sign convention description (optional)

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
                    s.sign_convention = CASE WHEN sig.sign_convention IS NOT NULL AND sig.sign_convention <> ''
                                             THEN sig.sign_convention ELSE s.sign_convention END,
                    s.llm_cost = $per_signal_cost,
                    s.enriched_at = datetime(),
                    s.claimed_at = null
                """,
                signals=signals,
                enriched=FacilitySignalStatus.enriched.value,
                per_signal_cost=per_signal_cost,
            )

            # Create Diagnostic nodes and BELONGS_TO_DIAGNOSTIC edges.
            # The diagnostic name on FacilitySignal preserves original casing
            # (needed for data access); the Diagnostic node uses canonical
            # lowercase for grouping.
            diag_signals = [
                s for s in signals if s.get("diagnostic") and s["diagnostic"].strip()
            ]
            if diag_signals:
                gc.query(
                    """
                    UNWIND $signals AS sig
                    MATCH (s:FacilitySignal {id: sig.id})
                    WITH s, toLower(trim(sig.diagnostic)) AS diag_name
                    WHERE diag_name <> ''
                    MERGE (d:Diagnostic {name: diag_name})
                    ON CREATE SET d.facility_id = s.facility_id
                    MERGE (s)-[:BELONGS_TO_DIAGNOSTIC]->(d)
                    WITH d, s
                    MATCH (f:Facility {id: s.facility_id})
                    MERGE (d)-[:AT_FACILITY]->(f)
                    """,
                    signals=diag_signals,
                )

        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals enriched: %s", e)
        return 0


def mark_signals_checked(
    signals: list[dict],
) -> int:
    """Mark signals as checked and create CHECKED_WITH relationship to DataAccess.

    Creates a CHECKED_WITH relationship from each FacilitySignal to its DataAccess
    node, carrying outcome metadata (success, shot, shape, dtype, error).
    Created for BOTH successful and failed data access checks.

    Expected signal dict keys:
    - id: signal ID
    - data_access: DataAccess ID (creates CHECKED_WITH relationship)
    - success: bool — whether the check succeeded
    - shot: int — shot number used for the check
    - shape: str — data shape (success only)
    - dtype: str — data type (success only)
    - error: str — error message (failure only)
    - error_type: str — error classification (failure only)
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
                MERGE (s)-[c:CHECKED_WITH]->(da)
                SET c.success = sig.success,
                    c.shot = sig.shot,
                    c.checked_at = datetime(),
                    c.shape = sig.shape,
                    c.dtype = sig.dtype,
                    c.error = sig.error,
                    c.error_type = sig.error_type
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
    """Mark a signal as failed due to infrastructure error.

    Use this for infrastructure-level failures (SSH timeout, script crash),
    NOT for data access failures (use mark_signals_checked with success=false).
    """
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (s:FacilitySignal {id: $id})
                SET s.status = $failed,
                    s.claimed_at = null
                """,
                id=signal_id,
                failed=FacilitySignalStatus.failed.value,
            )
    except Exception as e:
        logger.warning("Could not mark signal %s as failed: %s", signal_id, e)


def release_signal_claim(signal_id: str) -> None:
    """Release claim on a signal without changing status."""
    from imas_codex.discovery.base.claims import release_claim

    release_claim("FacilitySignal", signal_id)


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
                MERGE (v)-[:HAS_PREDECESSOR]->(pred)
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

                    # Create signals with AT_FACILITY edge (always)
                    gc.query(
                        """
                        UNWIND $signals AS sig
                        MERGE (s:FacilitySignal {id: sig.id})
                        ON CREATE SET s += sig,
                                      s.discovered_at = datetime()
                        ON MATCH SET s.claimed_at = null
                        WITH s, sig
                        MATCH (f:Facility {id: sig.facility_id})
                        MERGE (s)-[:AT_FACILITY]->(f)
                        """,
                        signals=signals,
                    )

                    # Create INTRODUCED_IN edges to epoch (separate query)
                    result = gc.query(
                        """
                        UNWIND $signals AS sig
                        MATCH (s:FacilitySignal {id: sig.id})
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

    from imas_codex.discovery.signals.tdi import (
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

    Creates FacilitySignal nodes with AT_FACILITY and DATA_ACCESS edges.
    Optionally creates INTRODUCED_IN relationships to TreeModelVersion
    epoch (if epoch_id is present).
    """
    if not signals:
        return 0

    try:
        with GraphClient() as gc:
            # Phase 1: Create/update signal nodes + AT_FACILITY edge (always)
            gc.query(
                """
                UNWIND $signals AS sig
                MERGE (s:FacilitySignal {id: sig.id})
                ON CREATE SET s += sig,
                              s.discovered_at = datetime()
                ON MATCH SET s.claimed_at = null
                WITH s, sig
                MATCH (f:Facility {id: sig.facility_id})
                MERGE (s)-[:AT_FACILITY]->(f)
                """,
                signals=signals,
            )

            # Phase 2: Create DATA_ACCESS edges for signals with data_access
            gc.query(
                """
                UNWIND $signals AS sig
                WITH sig WHERE sig.data_access IS NOT NULL
                MATCH (s:FacilitySignal {id: sig.id})
                MATCH (da:DataAccess {id: sig.data_access})
                MERGE (s)-[:DATA_ACCESS]->(da)
                """,
                signals=signals,
            )

            # Phase 3: Create INTRODUCED_IN edges for epoch-tracked signals
            gc.query(
                """
                UNWIND $signals AS sig
                WITH sig WHERE sig.epoch_id IS NOT NULL
                MATCH (s:FacilitySignal {id: sig.id})
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
    """Worker that dispatches to registered scanner plugins.

    Iterates through configured scanner_types, calling each scanner's scan()
    method and ingesting the discovered signals into the graph. Falls back
    to TDI-specific discovery if scanner_types includes 'tdi' and the
    legacy tdi_path is set.
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.signals.scanners.base import get_scanner

    facility_config = get_facility(state.facility)
    data_sources = facility_config.get("data_sources", {})
    ssh_host = state.ssh_host or facility_config.get("ssh_host", state.facility)

    total_discovered = 0

    for scanner_type in state.scanner_types:
        if state.should_stop_discovering():
            break

        if on_progress:
            on_progress(
                f"scanning {scanner_type}",
                state.discover_stats,
            )

        try:
            scanner = get_scanner(scanner_type)
        except KeyError:
            logger.warning("Scanner '%s' not registered, skipping", scanner_type)
            continue

        # Get scanner-specific config from facility data_sources
        scanner_config = data_sources.get(scanner_type, {})
        if not isinstance(scanner_config, dict):
            scanner_config = {}

        try:
            result = await scanner.scan(
                facility=state.facility,
                ssh_host=ssh_host,
                config=scanner_config,
                reference_shot=state.reference_shot,
            )

            # Store wiki context for use by enrich_worker
            if result.wiki_context:
                state.wiki_context.update(result.wiki_context)
                logger.info(
                    "Wiki context: %d entries for %s",
                    len(result.wiki_context),
                    state.facility,
                )

            if result.signals:
                # Ingest signals to graph
                count = ingest_discovered_signals(
                    [s.model_dump(exclude_none=True) for s in result.signals]
                )
                total_discovered += count
                state.discover_stats.processed += count

                # Ingest DataAccess node if provided
                if result.data_access:
                    try:
                        with GraphClient() as gc:
                            gc.query(
                                """
                                MERGE (da:DataAccess {id: $id})
                                SET da += $props
                                WITH da
                                MATCH (f:Facility {id: $facility})
                                MERGE (da)-[:AT_FACILITY]->(f)
                                """,
                                id=result.data_access.id,
                                props=result.data_access.model_dump(exclude_none=True),
                                facility=state.facility,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to ingest DataAccess for %s: %s",
                            scanner_type,
                            e,
                        )

                if on_progress:
                    on_progress(
                        f"{scanner_type}: discovered {count} signals",
                        state.discover_stats,
                        [
                            {
                                "id": s.id,
                                "tree_name": scanner_type,
                                "node_path": s.accessor,
                                "signals_in_tree": count,
                            }
                            for s in result.signals[:20]
                        ],
                    )
            else:
                logger.info(
                    "No signals from %s scanner for %s", scanner_type, state.facility
                )
                if on_progress:
                    on_progress(
                        f"{scanner_type}: no signals found",
                        state.discover_stats,
                    )

            # Handle scanner-specific metadata (e.g., TDI function ingestion)
            if scanner_type == "tdi" and result.metadata.get("functions"):
                try:
                    from imas_codex.discovery.signals.tdi import ingest_tdi_functions

                    with GraphClient() as gc:
                        func_count = await ingest_tdi_functions(
                            gc, state.facility, result.metadata["functions"]
                        )
                        logger.info("Ingested %d TDI functions", func_count)
                except Exception as e:
                    logger.warning("Failed to ingest TDI functions: %s", e)

        except Exception as e:
            logger.error("%s scan failed for %s: %s", scanner_type, state.facility, e)
            if on_progress:
                on_progress(
                    f"{scanner_type}: failed - {e}",
                    state.discover_stats,
                )

    # Mark scan as complete
    state.scan_phase.mark_done()

    if on_progress:
        on_progress(
            f"scan complete: {total_discovered} signals from {len(state.scanner_types)} scanners",
            state.discover_stats,
        )


async def enrich_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that enriches signals with LLM classification.

    Handles signals from all scanner types (TDI, PPF, EDAS, MDSplus, wiki).
    Groups signals by context source for efficient prompt construction:
    - TDI signals: grouped by function, source code included
    - PPF signals: grouped by DDA, access pattern noted
    - EDAS signals: grouped by category, existing description included
    - Other signals: grouped generically

    Wiki context is injected for any signal whose MDSplus path or accessor
    has a matching entry in state.wiki_context, reducing LLM hallucination.

    Uses Jinja2 prompt template with schema-injected physics domains.
    Uses centralized LLM access via acall_llm_structured() from base.llm.
    """
    from collections import defaultdict

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.discovery.signals.models import SignalEnrichmentBatch
    from imas_codex.settings import get_model

    # Get model configured for enrichment task
    model = get_model("language")

    # Render system prompt once (contains physics domains from schema)
    system_prompt = render_prompt("discovery/signal-enrichment")

    # Cache for TDI function source code
    tdi_source_cache: dict[str, str] = {}

    # Cache for semantic wiki context queries (avoids redundant embedding calls)
    wiki_context_cache: dict[str, list[dict[str, str]]] = {}

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

    def _find_wiki_context(signal: dict) -> dict[str, str] | None:
        """Find wiki context matching a signal's path or accessor.

        Tries multiple matching strategies to maximize cross-reference hits:
        1. Exact node_path match (uppercase-normalized)
        2. Accessor path extraction (data(...) pattern)
        3. Normalized path (strip leading backslashes, collapse separators)
        4. PPF-style DDA/DTYPE matching for JET signals
        """
        if not state.wiki_context:
            return None

        # Try node_path (uppercase for matching)
        node_path = signal.get("node_path")
        if node_path:
            normalized = node_path.upper().lstrip("\\")
            # Try with single backslash prefix (wiki format)
            ctx = state.wiki_context.get(f"\\{normalized}")
            if ctx:
                return ctx
            # Try with double backslash prefix
            ctx = state.wiki_context.get(f"\\\\{normalized}")
            if ctx:
                return ctx
            # Try raw
            ctx = state.wiki_context.get(node_path.upper())
            if ctx:
                return ctx

        # Try accessor pattern - extract path from data() or similar
        accessor = signal.get("accessor", "")
        if accessor.startswith("data(") and accessor.endswith(")"):
            path = accessor[5:-1].strip("'\"")
            normalized = path.upper().lstrip("\\")
            ctx = state.wiki_context.get(f"\\{normalized}")
            if ctx:
                return ctx
            ctx = state.wiki_context.get(path.upper())
            if ctx:
                return ctx

        # Try PPF-style matching (JET: DDA/DTYPE)
        name = signal.get("name", "")
        source = signal.get("discovery_source", "")
        if source == "ppf_enumeration" and "/" in name:
            ctx = state.wiki_context.get(name.upper())
            if ctx:
                return ctx

        return None

    def _signal_context_key(signal: dict) -> str:
        """Determine grouping key for a signal based on its discovery source."""
        # TDI signals: group by function name
        tdi_func = signal.get("tdi_function")
        if tdi_func:
            return f"tdi:{tdi_func}"

        # PPF signals: group by DDA (first part of name like "EFIT/IP")
        source = signal.get("discovery_source", "")
        name = signal.get("name", "")

        if source == "ppf_enumeration" and "/" in name:
            dda = name.split("/")[0]
            return f"ppf:{dda}"

        if source == "edas_enumeration" and "/" in name:
            cat = name.split("/")[0]
            return f"edas:{cat}"

        # MDSplus tree traversal: group by tree
        tree = signal.get("tree_name")
        if tree:
            return f"tree:{tree}"

        return "_ungrouped_"

    async def _fetch_facility_wiki_context(
        facility: str,
        cache: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, str]]:
        """Fetch facility-level wiki context (sign conventions, coordinates).

        Queries the wiki_chunk_embedding vector index for content about
        facility sign conventions and coordinate systems. Results are cached
        for the lifetime of the enrichment worker.
        """
        cache_key = f"_facility_{facility}"
        if cache_key in cache:
            return cache[cache_key]

        from imas_codex.discovery.signals.scanners.wiki import (
            fetch_semantic_wiki_context,
        )

        query = (
            f"{facility} sign conventions coordinate systems COCOS toroidal poloidal"
        )
        chunks = await asyncio.to_thread(
            fetch_semantic_wiki_context, facility, query, k=5, min_score=0.35
        )
        cache[cache_key] = chunks
        if chunks:
            logger.info(
                "Facility wiki context: %d chunks for %s (conventions, coordinates)",
                len(chunks),
                facility,
            )
        return chunks

    async def _fetch_group_wiki_context(
        facility: str,
        group_key: str,
        indexed_signals: list[tuple[int, dict]],
        cache: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, str]]:
        """Fetch group-level wiki context for a batch of related signals.

        Builds a semantic query from the group key and signal names, then
        searches the wiki_chunk_embedding index for relevant documentation
        about the diagnostic, tree, or analysis code.
        """
        if group_key in cache:
            return cache[group_key]

        from imas_codex.discovery.signals.scanners.wiki import (
            fetch_semantic_wiki_context,
        )

        # Build a targeted query from the group context
        if group_key.startswith("tdi:"):
            func_name = group_key[4:]
            signal_names = " ".join(
                s.get("name") or "" for _, s in indexed_signals[:10]
            )
            query = f"{func_name} TDI function {signal_names}"
        elif group_key.startswith("ppf:"):
            dda = group_key[4:]
            query = f"{dda} JET diagnostic processed pulse file"
        elif group_key.startswith("edas:"):
            cat = group_key[5:]
            query = f"{cat} JT-60SA diagnostic data"
        elif group_key.startswith("tree:"):
            tree = group_key[5:]
            signal_names = " ".join(
                s.get("name") or "" for _, s in indexed_signals[:10]
            )
            query = f"{tree} MDSplus tree {signal_names}"
        else:
            # Ungrouped — use first few signal names
            signal_names = " ".join(
                s.get("name") or "" for _, s in indexed_signals[:10]
            )
            query = signal_names

        if not query.strip():
            cache[group_key] = []
            return []

        chunks = await asyncio.to_thread(
            fetch_semantic_wiki_context, facility, query, k=3, min_score=0.4
        )
        cache[group_key] = chunks
        return chunks

    # Cache for code chunk context queries
    code_context_cache: dict[str, list[dict[str, str]]] = {}

    async def _fetch_code_context(
        group_key: str,
        indexed_signals: list[tuple[int, dict]],
    ) -> list[dict[str, str]]:
        """Fetch relevant source code chunks for a signal group.

        Uses the code_chunk_embedding vector index to find code that
        references signals in this group, providing usage patterns and
        computational context for the LLM.
        """
        if group_key in code_context_cache:
            return code_context_cache[group_key]

        from imas_codex.embeddings.config import EncoderConfig
        from imas_codex.embeddings.encoder import Encoder

        # Build query from group key and signal names
        signal_names = " ".join(s.get("name") or "" for _, s in indexed_signals[:10])
        if group_key.startswith("tdi:"):
            query_text = f"{group_key[4:]} {signal_names}"
        elif group_key.startswith("tree:"):
            query_text = f"{group_key[5:]} MDSplus {signal_names}"
        else:
            query_text = signal_names

        if not query_text.strip():
            code_context_cache[group_key] = []
            return []

        try:
            config = EncoderConfig()
            encoder = Encoder(config)
            embedding = encoder.embed_texts([query_text])[0].tolist()
        except Exception as e:
            logger.debug("Could not embed for code context: %s", e)
            code_context_cache[group_key] = []
            return []

        try:
            with GraphClient() as gc:
                results = gc.query(
                    """
                    CALL db.index.vector.queryNodes(
                        'code_chunk_embedding', $k, $embedding
                    )
                    YIELD node, score
                    WHERE score >= $min_score
                    OPTIONAL MATCH (src)-[:HAS_CHUNK]->(node)
                    WHERE src.facility_id = $facility
                    RETURN node.content AS content,
                           src.path AS source_path,
                           node.chunk_type AS chunk_type,
                           score
                    ORDER BY score DESC
                    """,
                    k=3,
                    embedding=embedding,
                    min_score=0.45,
                    facility=state.facility,
                )
                chunks = []
                for row in results:
                    content = row.get("content", "")
                    if len(content) > 600:
                        content = content[:600] + "..."
                    chunks.append(
                        {
                            "content": content,
                            "source_path": row.get("source_path", ""),
                            "chunk_type": row.get("chunk_type", ""),
                        }
                    )
                code_context_cache[group_key] = chunks
                return chunks
        except Exception as e:
            logger.debug("Code context search failed: %s", e)
            code_context_cache[group_key] = []
            return []

    while not state.should_stop_enriching():
        # Claim batch of signals (sorted by tdi_function)
        # Batch size 50 - signals grouped by function so one source code per group
        signals = await asyncio.to_thread(
            claim_signals_for_enrichment,
            state.facility,
            batch_size=50,
        )

        if not signals:
            state.enrich_phase.record_idle()
            if on_progress:
                on_progress("idle", state.enrich_stats)
            await asyncio.sleep(1.0)
            continue

        state.enrich_phase.record_activity(len(signals))

        if on_progress:
            on_progress("enriching batch", state.enrich_stats)

        # Group signals by context source for efficient prompting
        context_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, signal in enumerate(signals):
            key = _signal_context_key(signal)
            context_groups[key].append((i, signal))

        # Fetch facility-level wiki context (sign conventions, coordinate systems)
        # This is a single semantic search per batch, cached across batches
        facility_wiki_context = await _fetch_facility_wiki_context(
            state.facility, wiki_context_cache
        )

        # Build user prompt with context from each signal group
        user_lines = [
            f"Classify these {len(signals)} signals.\n",
            "Return results in the same order using signal_index (1-based).\n",
        ]

        # Inject facility-level wiki context (sign conventions, coordinates)
        if facility_wiki_context:
            user_lines.append("\n## Facility Wiki Reference")
            user_lines.append(
                "The following is authoritative documentation from the facility wiki. "
                "Use it to determine sign conventions, coordinate systems, and units.\n"
            )
            for chunk in facility_wiki_context:
                user_lines.append(f"### From: {chunk['page_title']}")
                if chunk.get("conventions"):
                    user_lines.append(f"Conventions: {', '.join(chunk['conventions'])}")
                user_lines.append(chunk["content"])
                user_lines.append("")

        signal_index = 0
        for group_key, indexed_signals in context_groups.items():
            # Add group-level context header
            if group_key.startswith("tdi:"):
                func_name = group_key[4:]
                source_code = await get_tdi_source(func_name)
                if source_code:
                    if len(source_code) > 8000:
                        source_code = source_code[:8000] + "\n... (truncated)"
                    user_lines.append(f"\n## TDI Function: {func_name}")
                    user_lines.append("```tdi")
                    user_lines.append(source_code)
                    user_lines.append("```")
                    user_lines.append("\nSignals from this function:")

            elif group_key.startswith("ppf:"):
                dda = group_key[4:]
                user_lines.append(f"\n## PPF DDA: {dda}")
                user_lines.append(
                    f"JET Processed Pulse File signals from diagnostic data area {dda}."
                )
                user_lines.append("Access: ppfdata(pulse, dda, dtype, uid='jetppf')")
                user_lines.append("\nSignals from this DDA:")

            elif group_key.startswith("edas:"):
                cat = group_key[5:]
                user_lines.append(f"\n## EDAS Category: {cat}")
                user_lines.append(
                    f"JT-60SA Experiment Data Access System signals from category {cat}."
                )
                user_lines.append(
                    "Access: eddbreadTime(shot, category, data_name, t1, t2)"
                )
                user_lines.append("\nSignals from this category:")

            elif group_key.startswith("tree:"):
                tree = group_key[5:]
                user_lines.append(f"\n## MDSplus Tree: {tree}")
                user_lines.append(f"Direct MDSplus tree traversal signals from {tree}.")
                user_lines.append("\nSignals from this tree:")

            # Fetch group-level semantic wiki context (e.g., wiki content
            # about a specific diagnostic, tree, or analysis code)
            group_wiki = await _fetch_group_wiki_context(
                state.facility, group_key, indexed_signals, wiki_context_cache
            )
            if group_wiki:
                user_lines.append("\n**Relevant wiki documentation:**")
                for chunk in group_wiki:
                    user_lines.append(f"- [{chunk['page_title']}] {chunk['content']}")

            # Fetch relevant source code chunks via code_chunk_embedding
            code_chunks = await _fetch_code_context(group_key, indexed_signals)
            if code_chunks:
                user_lines.append("\n**Relevant source code:**")
                for chunk in code_chunks:
                    path = chunk.get("source_path", "unknown")
                    ctype = chunk.get("chunk_type", "")
                    label = f"{path} ({ctype})" if ctype else path
                    user_lines.append(f"\n```python  # {label}")
                    user_lines.append(chunk["content"])
                    user_lines.append("```")

            # Add individual signal entries
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

                # Inject existing description from scanner (e.g., EDAS Japanese desc)
                if signal.get("description"):
                    user_lines.append(f"existing_description: {signal['description']}")

                # Inject wiki context if available
                wiki_ctx = _find_wiki_context(signal)
                if wiki_ctx:
                    if wiki_ctx.get("description"):
                        user_lines.append(
                            f"wiki_description: {wiki_ctx['description']}"
                        )
                    if wiki_ctx.get("units"):
                        user_lines.append(f"wiki_units: {wiki_ctx['units']}")
                    if wiki_ctx.get("page"):
                        user_lines.append(f"wiki_source: {wiki_ctx['page']}")

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
                        "sign_convention": result.sign_convention,
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


def _classify_check_error(error: str) -> str:
    """Classify a check error message into a category for analytics.

    Returns a short error_type string that can be aggregated across signals.
    """
    if not error:
        return "unknown"
    err_lower = error.lower()
    if "treeннф" in err_lower or "not found" in err_lower or "TreeNNF" in error:
        return "node_not_found"
    if "TreeNODATA" in error or "no data" in err_lower:
        return "no_data"
    if "TdiINVCLADSC" in error or "invalid" in err_lower:
        return "invalid_descriptor"
    if "timeout" in err_lower or "timed out" in err_lower:
        return "timeout"
    if "connection" in err_lower or "refused" in err_lower:
        return "connection_error"
    if "empty" in err_lower:
        return "empty_data"
    if "permission" in err_lower or "denied" in err_lower:
        return "permission_denied"
    if "segmentation" in err_lower or "segfault" in err_lower:
        return "segfault"
    return "other"


async def check_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that checks signals by testing data access.

    Routes signals to the appropriate check method based on discovery source:
    - TDI/MDSplus signals: batched SSH via check_signals_batch.py
    - PPF signals: scanner check() via check_ppf.py
    - EDAS signals: scanner check() via check_edas.py
    - Wiki-only signals: skipped (validated by primary scanner)

    Uses reference_shot from config for validation.
    """
    from collections import defaultdict

    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.signals.scanners.base import get_scanner
    from imas_codex.graph.models import FacilitySignal as FacilitySignalModel

    facility_config = get_facility(state.facility)
    data_sources = facility_config.get("data_sources", {})

    # Build fallback shots from static tree epochs (operational phases)
    # These provide representative shots from different machine configurations
    fallback_shots: list[int] = []
    mdsplus_config = data_sources.get("mdsplus", {})
    if isinstance(mdsplus_config, dict):
        static_trees = mdsplus_config.get("static_trees", [])
        for st in static_trees:
            if isinstance(st, dict):
                for v in st.get("versions", []):
                    first_shot = v.get("first_shot")
                    if first_shot and first_shot != state.reference_shot:
                        fallback_shots.append(first_shot)
        # Sort descending so we try newest shots first
        fallback_shots.sort(reverse=True)

    # Large batch size - signals are grouped by tree/shot on the remote side
    BATCH_SIZE = 100

    def _get_signal_scanner_type(signal: dict) -> str:
        """Determine which scanner should check this signal."""
        source = signal.get("discovery_source", "")
        if source == "ppf_enumeration":
            return "ppf"
        if source == "edas_enumeration":
            return "edas"
        if source == "wiki_extraction":
            return "wiki"
        # Default to MDSplus/TDI batch check
        return "mdsplus"

    while not state.should_stop_checking():
        # Claim batch of signals with reference_shot
        signals = await asyncio.to_thread(
            claim_signals_for_check,
            state.facility,
            batch_size=BATCH_SIZE,
            reference_shot=state.reference_shot,
        )

        if not signals:
            state.check_phase.record_idle()
            if on_progress:
                on_progress("idle", state.check_stats)
            await asyncio.sleep(1.0)
            continue

        state.check_phase.record_activity(len(signals))

        if on_progress:
            on_progress(f"checking {len(signals)} signals", state.check_stats)

        # Route signals by scanner type
        scanner_groups: dict[str, list[dict]] = defaultdict(list)
        signal_data_access: dict[str, str | None] = {}

        for signal in signals:
            scanner_type = _get_signal_scanner_type(signal)
            scanner_groups[scanner_type].append(signal)
            signal_data_access[signal["id"]] = signal.get("data_access")

        checked: list[dict] = []

        # --- Scanner-based checks (PPF, EDAS) ---
        for scanner_type in ("ppf", "edas"):
            group = scanner_groups.get(scanner_type, [])
            if not group:
                continue

            try:
                scanner = get_scanner(scanner_type)
                scanner_config = data_sources.get(scanner_type, {})
                if not isinstance(scanner_config, dict):
                    scanner_config = {}

                # Build FacilitySignal model instances for scanner.check()
                signal_models = [
                    FacilitySignalModel(
                        id=s["id"],
                        facility_id=state.facility,
                        name=s.get("name"),
                        accessor=s.get("accessor", ""),
                        physics_domain=s.get("physics_domain", "general"),
                        data_access=s.get("data_access", ""),
                        tree_name=s.get("tree_name"),
                    )
                    for s in group
                ]

                results = await scanner.check(
                    facility=state.facility,
                    ssh_host=state.ssh_host or state.facility,
                    signals=signal_models,
                    config=scanner_config,
                    reference_shot=state.reference_shot,
                )

                for r in results:
                    signal_id = r.get("signal_id")
                    shot = state.reference_shot
                    success = bool(r.get("valid", False))
                    entry = {
                        "id": signal_id,
                        "success": success,
                        "shot": shot,
                        "data_access": signal_data_access.get(signal_id),
                    }
                    if success:
                        entry["shape"] = r.get("shape")
                        entry["dtype"] = r.get("dtype")
                    else:
                        error_msg = r.get("error", "check failed")
                        entry["error"] = error_msg
                        entry["error_type"] = _classify_check_error(error_msg)
                    checked.append(entry)

            except Exception as e:
                logger.warning(
                    "%s check failed for %d signals: %s",
                    scanner_type,
                    len(group),
                    e,
                )
                for sig in group:
                    await asyncio.to_thread(release_signal_claim, sig["id"])

        # --- Wiki signals: auto-pass (validated by primary scanner) ---
        wiki_group = scanner_groups.get("wiki", [])
        for sig in wiki_group:
            checked.append(
                {
                    "id": sig["id"],
                    "success": True,
                    "shot": state.reference_shot,
                    "data_access": signal_data_access.get(sig["id"]),
                    "note": "wiki-sourced; validate via primary scanner",
                }
            )

        # --- MDSplus/TDI batch check (existing optimized path) ---
        mdsplus_group = scanner_groups.get("mdsplus", [])
        if mdsplus_group:
            batch_input = []
            for signal in mdsplus_group:
                shot = signal.get("check_shot") or state.reference_shot
                if not shot:
                    logger.warning(
                        "Signal %s has no check_shot and no reference_shot",
                        signal["id"],
                    )
                    await asyncio.to_thread(release_signal_claim, signal["id"])
                    continue

                # TDI signals use tcv_shot tree
                tree_name = signal.get("tree_name")
                if signal.get("tdi_function") and not tree_name:
                    tree_name = "tcv_shot"

                batch_input.append(
                    {
                        "id": signal["id"],
                        "accessor": signal["accessor"],
                        "tree_name": tree_name or "tcv_shot",
                        "shot": shot,
                        **(
                            {"fallback_shots": fallback_shots} if fallback_shots else {}
                        ),
                    }
                )

            if batch_input:
                try:
                    output = await asyncio.to_thread(
                        run_python_script,
                        "check_signals_batch.py",
                        {"signals": batch_input, "timeout_per_group": 30},
                        ssh_host=state.ssh_host,
                        timeout=60 + len(batch_input),
                    )

                    if not output or not output.strip():
                        logger.warning(
                            "Check script returned empty output for %d signals",
                            len(batch_input),
                        )
                        for sig in batch_input:
                            await asyncio.to_thread(release_signal_claim, sig["id"])
                    else:
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
                        else:
                            response = json.loads(json_line)

                            if "error" in response and not response.get("results"):
                                logger.warning(
                                    "Check script error: %s", response["error"]
                                )
                                for sig in batch_input:
                                    await asyncio.to_thread(
                                        release_signal_claim, sig["id"]
                                    )
                            else:
                                results = response.get("results", [])
                                stats = response.get("stats", {})

                                if stats:
                                    retry_count = stats.get("retry_success", 0)
                                    logger.debug(
                                        "Check batch: %d signals in %d groups, "
                                        "%d success, %d failed, %d via fallback",
                                        stats.get("total", 0),
                                        stats.get("groups", 0),
                                        stats.get("success", 0),
                                        stats.get("failed", 0),
                                        retry_count,
                                    )

                                for result in results:
                                    signal_id = result.get("id")
                                    # Use checked_shot from remote (may differ
                                    # from primary if fallback succeeded)
                                    shot = result.get("checked_shot")
                                    if not shot:
                                        for bi in batch_input:
                                            if bi["id"] == signal_id:
                                                shot = bi.get("shot")
                                                break
                                    success = bool(result.get("success"))
                                    entry = {
                                        "id": signal_id,
                                        "success": success,
                                        "shot": shot,
                                        "data_access": signal_data_access.get(
                                            signal_id
                                        ),
                                    }
                                    if success:
                                        entry["shape"] = result.get("shape")
                                        entry["dtype"] = result.get("dtype")
                                    else:
                                        error_msg = result.get("error", "check failed")
                                        entry["error"] = error_msg
                                        entry["error_type"] = _classify_check_error(
                                            error_msg
                                        )
                                    checked.append(entry)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "Failed to parse check response (%s): %s",
                        e,
                        str(e)[:200],
                    )
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr[:200] if e.stderr else str(e)
                    logger.warning(
                        "Check script failed (exit %d): %s", e.returncode, stderr
                    )
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Check script timed out for batch of %d signals",
                        len(batch_input),
                    )
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])
                except Exception as e:
                    logger.warning("Failed to run check: %s", e)
                    for sig in batch_input:
                        await asyncio.to_thread(release_signal_claim, sig["id"])

        # All results go through mark_signals_checked
        if checked:
            await asyncio.to_thread(mark_signals_checked, checked)
            state.check_stats.processed += len(checked)

            if on_progress:
                results = [
                    {
                        "id": v["id"],
                        "success": v.get("success", True),
                        "shape": v.get("shape"),
                    }
                    for v in checked
                ]
                on_progress("checked batch", state.check_stats, results)


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_parallel_data_discovery(
    facility: str,
    ssh_host: str | None = None,
    scanner_types: list[str] | None = None,
    tdi_path: str | None = None,
    reference_shot: int | None = None,
    cost_limit: float = 10.0,
    signal_limit: int | None = None,
    focus: str | None = None,
    num_enrich_workers: int = 2,
    num_check_workers: int = 1,
    discover_only: bool = False,
    enrich_only: bool = False,
    deadline: float | None = None,
    on_discover_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_check_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
) -> dict[str, Any]:
    """Run parallel data discovery with async workers.

    Dispatches to registered scanner plugins for signal discovery, enriches
    with LLM classification, and validates data access.

    Args:
        facility: Facility ID (e.g., "tcv")
        ssh_host: SSH host for remote discovery
        scanner_types: Scanner types to run (e.g., ["tdi", "ppf"]).
            Auto-detected from facility config if not specified.
        tdi_path: Legacy path to TDI directory (deprecated, use scanner config)
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

    # Ensure Facility node exists so AT_FACILITY relationships don't fail
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.ensure_facility(facility)

    # Auto-detect scanner types if not specified
    if not scanner_types:
        from imas_codex.discovery.signals.scanners.base import (
            get_scanners_for_facility,
        )

        scanner_instances = get_scanners_for_facility(facility)
        scanner_types = [s.scanner_type for s in scanner_instances]

    # Initialize state
    state = DataDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        reference_shot=reference_shot,
        scanner_types=scanner_types,
        tdi_path=tdi_path,
        cost_limit=cost_limit,
        signal_limit=signal_limit,
        focus=focus,
        enrich_only=enrich_only,
        deadline=deadline,
    )

    # Create worker group
    worker_group = SupervisedWorkerGroup()

    # Start scan worker (unless enrich_only)
    if not enrich_only:
        worker_name = "scan_worker_0"
        status = worker_group.create_status(worker_name, group="scan")
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

    # Periodic orphan recovery during discovery (every 60s)
    orphan_tick = make_orphan_recovery_tick(
        facility,
        [OrphanRecoverySpec("FacilitySignal", timeout_seconds=CLAIM_TIMEOUT_SECONDS)],
    )

    if discover_only:
        # Run supervision loop — scan worker only
        await run_supervised_loop(
            worker_group,
            lambda: state.scan_phase.done,
            on_worker_status=on_worker_status,
            on_tick=orphan_tick,
        )
        state.stop_requested = True

        return {
            "scanned": state.discover_stats.processed,
            "discovered": state.discover_stats.processed,
            "enriched": 0,
            "checked": 0,
            "cost": 0.0,
            "elapsed_seconds": time.time() - start_time,
        }

    # Start enrich workers
    for i in range(num_enrich_workers):
        worker_name = f"enrich_worker_{i}"
        status = worker_group.create_status(worker_name, group="enrich")
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
            status = worker_group.create_status(worker_name, group="check")
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
    else:
        # In enrich_only mode, discover and check workers are not started.
        # Mark their phases as done so should_stop() doesn't block on them.
        state.scan_phase.mark_done()
        state.check_phase.mark_done()

    # In enrich_only mode, scan worker was already skipped above.
    # Mark scan phase done for the general case too.
    if enrich_only:
        state.scan_phase.mark_done()

    # Embed description worker: embeds FacilitySignal descriptions as they are enriched
    from imas_codex.discovery.base.embed_worker import embed_description_worker

    embed_status = worker_group.create_status("embed_worker", group="scan")
    worker_group.add_task(
        asyncio.create_task(
            supervised_worker(
                embed_description_worker,
                "embed_worker",
                state,
                state.should_stop,
                labels=["FacilitySignal"],
                status_tracker=embed_status,
            )
        )
    )

    # Run supervision loop — handles status updates and clean shutdown
    await run_supervised_loop(
        worker_group,
        state.should_stop,
        on_worker_status=on_worker_status,
        on_tick=orphan_tick,
    )
    state.stop_requested = True

    elapsed = time.time() - start_time
    return {
        "scanned": state.discover_stats.processed,
        "discovered": state.discover_stats.processed,
        "enriched": state.enrich_stats.processed,
        "checked": state.check_stats.processed,
        "cost": state.enrich_stats.cost,
        "elapsed_seconds": elapsed,
        "discover_rate": state.discover_stats.processed / elapsed if elapsed > 0 else 0,
        "enrich_rate": state.enrich_stats.processed / elapsed if elapsed > 0 else 0,
        "check_rate": state.check_stats.processed / elapsed if elapsed > 0 else 0,
    }
