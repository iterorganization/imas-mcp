"""
Parallel data signal discovery engine with async workers.

Architecture:
- Independent async workers, each claiming batches from the graph:
  - seed: Seeds SignalEpoch nodes from config (fast, exits immediately)
  - epoch: Detects structural epochs for dynamic trees (runs independently)
  - extract: Claims SignalEpoch, SSH extract + ingest DataNodes
  - units: Extracts units for trees with ingested versions
  - promote: Creates FacilitySignal from leaf DataNodes
  - enrich: LLM classification of physics_domain, description generation
  - check: Test data access with example_shot, verify units/sign
  - embed: Embeds FacilitySignal descriptions for vector search
- Graph + claimed_at timestamp for coordination (same pattern as wiki/paths)
- Status transitions:
  - SignalEpoch: discovered → ingested
  - FacilitySignal: discovered → enriched → checked
- Workers claim items by setting claimed_at, release by clearing it
- Orphan recovery: items claimed longer than 5 min are reclaimed

Resilience:
- Supervised workers with automatic restart on crash (via base.supervision)
- Exponential backoff on infrastructure errors (Neo4j, network, SSH)
- Graceful degradation when services are temporarily unavailable

Key design principle:
- No stage blocking. Downstream workers start as soon as the first batch
  of upstream work is available. The epoch worker can take 15 minutes to
  detect all epochs, but extract/units/promote/enrich start immediately
  with the first seeded version (e.g., version 1 / shot 1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
)
from imas_codex.graph import GraphClient
from imas_codex.graph.models import FacilitySignalStatus
from imas_codex.remote.executor import run_python_script

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

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
class DataDiscoveryState(DiscoveryStateBase):
    """Shared state for parallel data discovery."""

    ssh_host: str | None = None

    # Pre-fetched facility config (avoids sync get_facility() in workers)
    facility_config: dict = field(default_factory=dict)

    # Pre-fetched graph counts (avoids sync graph queries in workers)
    initial_version_counts: dict = field(default_factory=dict)
    initial_signal_counts: dict = field(default_factory=dict)

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

    # Worker stats — one per worker group for accurate display
    discover_stats: WorkerStats = field(default_factory=WorkerStats)
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    units_stats: WorkerStats = field(default_factory=WorkerStats)
    promote_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    check_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    enrich_only: bool = False  # When True, discover/check workers not started

    # Pipeline phases (initialized in __post_init__)
    # Scan sub-phases: seed, epoch, extract, units, promote
    seed_phase: PipelinePhase = field(init=False)
    epoch_phase: PipelinePhase = field(init=False)
    extract_phase: PipelinePhase = field(init=False)
    units_phase: PipelinePhase = field(init=False)
    promote_phase: PipelinePhase = field(init=False)
    # Top-level phases
    enrich_phase: PipelinePhase = field(init=False)
    check_phase: PipelinePhase = field(init=False)

    # Backward-compat alias
    @property
    def scan_phase(self) -> PipelinePhase:
        """Scan is done when all scan sub-phases are done."""
        return self._scan_phase

    def __post_init__(self) -> None:
        from imas_codex.discovery.mdsplus.graph_ops import (
            has_pending_extract_work_facility,
            has_pending_promote_work_facility,
            has_pending_units_work_facility,
        )

        self.seed_phase = PipelinePhase("seed")  # deterministic — no graph check
        self.epoch_phase = PipelinePhase("epoch")  # deterministic — no graph check
        self.extract_phase = PipelinePhase(
            "extract",
            has_work_fn=lambda: has_pending_extract_work_facility(self.facility),
        )
        self.units_phase = PipelinePhase(
            "units",
            has_work_fn=lambda: has_pending_units_work_facility(self.facility),
        )
        self.promote_phase = PipelinePhase(
            "promote",
            has_work_fn=lambda: has_pending_promote_work_facility(self.facility),
        )
        self.enrich_phase = PipelinePhase(
            "enrich",
            has_work_fn=lambda: has_pending_enrich_work(self.facility),
        )
        self.check_phase = PipelinePhase(
            "check",
            has_work_fn=lambda: has_pending_check_work(self.facility),
        )
        # Composite scan phase — done when all sub-phases are done
        self._scan_phase = PipelinePhase("scan")

    def _update_scan_phase(self) -> None:
        """Update composite scan phase from sub-phases."""
        if (
            self.seed_phase.done
            and self.epoch_phase.done
            and self.extract_phase.done
            and self.units_phase.done
            and self.promote_phase.done
        ):
            self._scan_phase.mark_done()

    @property
    def total_cost(self) -> float:
        return self.enrich_stats.cost

    @property
    def signal_limit_reached(self) -> bool:
        if self.signal_limit is None:
            return False
        return self.enrich_stats.processed >= self.signal_limit

    def should_stop(self) -> bool:
        """Check if ALL workers should terminate.

        Uses PipelinePhase.done which combines idle detection with
        graph-level pending work checks.
        """
        if super().should_stop():
            return True

        limit_done = self.budget_exhausted or self.signal_limit_reached

        # Update composite scan phase
        self._update_scan_phase()

        # LLM workers: phase done OR limit-stopped counts as "done"
        enrich_done = self.enrich_phase.done or limit_done

        # Check phase considers upstream phases
        check_done = self.check_phase.done

        # In enrich_only mode, scan and check phases are pre-marked done
        all_done = self._scan_phase.done and enrich_done and check_done
        if all_done:
            if self.enrich_only:
                return limit_done or self.enrich_phase.done
            return True
        return False

    def should_stop_discovering(self) -> bool:
        """Check if discover workers should stop."""
        if self.stop_requested:
            return True
        self._update_scan_phase()
        return self._scan_phase.done

    def should_stop_enriching(self) -> bool:
        """Check if enrich workers should stop."""
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        if self.signal_limit_reached:
            return True
        return False

    def should_stop_checking(self) -> bool:
        """Check if check workers should stop.

        Check workers must wait for both upstream phases (scan + enrich)
        to finish producing work before exiting.
        """
        if super().should_stop():
            return True
        if not self.check_phase.idle:
            return False
        # Idle — but don't exit if upstream is still producing
        self._update_scan_phase()
        if not self._scan_phase.done:
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
                WHERE 'FacilitySignal' IN labels(s)
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
            stats = dict(result[0]) if result else {}

            # Signal group counts (separate query to avoid cross-product)
            grp = gc.query(
                """
                MATCH (sg:SignalGroup {facility_id: $facility})
                OPTIONAL MATCH (s:FacilitySignal)-[:MEMBER_OF]->(sg)
                RETURN count(DISTINCT sg) AS signal_groups,
                       count(s) AS grouped_signals
                """,
                facility=facility,
            )
            if grp:
                stats["signal_groups"] = grp[0]["signal_groups"]
                stats["grouped_signals"] = grp[0]["grouped_signals"]

            return stats
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

    Deletes FacilitySignal nodes, DataAccess nodes, SignalEpoch
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

            # Delete SignalEpoch (epoch) nodes in batches
            while True:
                result = gc.query(
                    """
                    MATCH (v:SignalEpoch {facility_id: $facility})
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


@retry_on_deadlock()
def claim_signals_for_enrichment(
    facility: str,
    batch_size: int = 10,
) -> list[dict]:
    """Claim a batch of discovered signals for enrichment.

    Returns signals sorted by tdi_function to enable batching by function.
    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Filters out numbered channel signals (e.g. CHANNEL_006, :003) that are
    array elements of a parent signal — these don't need individual LLM
    enrichment. They are marked as skipped with reason 'channel_element'.
    """
    import uuid as _uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    try:
        with GraphClient() as gc:
            # First skip channel signals in bulk so they don't clog the queue.
            gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $discovered
                  AND (
                    s.accessor =~ '.*[_:]CHANNEL_?\\d{2,3}\\)?$'
                    OR s.accessor =~ '.*[_:]\\d{2,3}\\)?$'
                    OR s.accessor =~ '.*CHANNEL_?\\d{2,3}:.*'
                    OR s.name =~ '^\\d{2,3}$'
                  )
                SET s.status = $skipped,
                    s.skip_reason = 'channel_element',
                    s.claimed_at = null
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                skipped=FacilitySignalStatus.skipped.value,
            )

            # Step 1: Claim with random ordering and unique token
            gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status IN [$discovered, $underspecified]
                  AND NOT EXISTS {
                    MATCH (s)-[:MEMBER_OF]->(sg:SignalGroup)
                    WHERE sg.representative_id <> s.id
                  }
                  AND (s.claimed_at IS NULL
                       OR s.claimed_at < datetime() - duration($cutoff))
                WITH s ORDER BY rand() LIMIT $batch_size
                SET s.claimed_at = datetime(), s.claim_token = $token
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                underspecified=FacilitySignalStatus.underspecified.value,
                batch_size=batch_size,
                cutoff=cutoff,
                token=claim_token,
            )

            # Step 2: Read back only signals WE successfully claimed
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility, claim_token: $token})
                RETURN s.id AS id, s.accessor AS accessor, s.data_source_name AS data_source_name,
                       s.data_source_path AS data_source_path, s.unit AS unit, s.name AS name,
                       s.tdi_function AS tdi_function,
                       s.tdi_quantity AS tdi_quantity,
                       s.discovery_source AS discovery_source,
                       s.description AS description,
                       s.data_source_node AS data_source_node,
                       s.facility_id AS facility_id
                """,
                facility=facility,
                token=claim_token,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning("Could not claim signals for enrichment: %s", e)
        return []


def fetch_tree_context(
    signal_ids: list[str],
) -> dict[str, dict]:
    """Fetch SignalNode context for signals with HAS_DATA_SOURCE_NODE edges.

    For each signal that has a HAS_DATA_SOURCE_NODE→SignalNode edge, returns:
    - tree_path: Full MDSplus path of the SignalNode
    - parent_path: Parent node's path (one level up)
    - sibling_paths: List of sibling node paths (same parent, same type)
    - tdi_source: TDI function source_code if linked via RESOLVES_TO_NODE
    - tdi_name: TDI function name
    - epoch_range: Dict with first_shot/last_shot if versioned

    Args:
        signal_ids: List of FacilitySignal IDs to fetch context for

    Returns:
        Dict mapping signal_id → context dict (only for signals with tree context)
    """
    if not signal_ids:
        return {}

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS sid
                MATCH (s:FacilitySignal {id: sid})-[:HAS_DATA_SOURCE_NODE]->(n:SignalNode)
                OPTIONAL MATCH (n)-[:CHILD_OF]->(parent:SignalNode)
                OPTIONAL MATCH (parent)<-[:CHILD_OF]-(sibling:SignalNode)
                WHERE sibling.id <> n.id
                  AND sibling.node_type IN ['NUMERIC', 'SIGNAL']
                OPTIONAL MATCH (tdi:TDIFunction)-[:RESOLVES_TO_NODE]->(n)
                OPTIONAL MATCH (v:SignalEpoch {
                    facility_id: n.facility_id, data_source_name: n.data_source_name
                })
                WITH s.id AS signal_id,
                     n.path AS tree_path,
                     parent.path AS parent_path,
                     collect(DISTINCT sibling.path)[..10] AS sibling_paths,
                     head(collect(DISTINCT tdi.source_code)) AS tdi_source,
                     head(collect(DISTINCT tdi.name)) AS tdi_name,
                     min(v.version) AS first_version,
                     max(v.version) AS last_version
                RETURN signal_id, tree_path, parent_path, sibling_paths,
                       tdi_source, tdi_name, first_version, last_version
                """,
                ids=signal_ids,
            )
            ctx = {}
            for row in result:
                entry: dict = {
                    "tree_path": row["tree_path"],
                    "parent_path": row.get("parent_path"),
                    "sibling_paths": row.get("sibling_paths", []),
                }
                if row.get("tdi_source"):
                    entry["tdi_source"] = row["tdi_source"]
                    entry["tdi_name"] = row.get("tdi_name")
                if row.get("first_version") and row.get("last_version"):
                    entry["epoch_range"] = {
                        "first_version": row["first_version"],
                        "last_version": row["last_version"],
                    }
                ctx[row["signal_id"]] = entry
            return ctx
    except Exception as e:
        logger.warning("Could not fetch tree context: %s", e)
        return {}


def fetch_signal_code_refs(
    signal_ids: list[str],
) -> dict[str, list[dict]]:
    """Fetch deterministic code references for signals via graph traversal.

    Traverses: FacilitySignal →[HAS_DATA_SOURCE_NODE]→ SignalNode
               ←[RESOLVES_TO_NODE]← DataReference ←[CONTAINS_REF]← CodeChunk

    Returns code snippets that directly reference the signal's backing
    SignalNode, providing real usage context for enrichment.

    Args:
        signal_ids: List of FacilitySignal IDs

    Returns:
        Dict mapping signal_id → list of code reference dicts
    """
    if not signal_ids:
        return {}

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS sid
                MATCH (s:FacilitySignal {id: sid})
                      -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
                      <-[:RESOLVES_TO_NODE]-(dr:DataReference)
                      <-[:CONTAINS_REF]-(cc:CodeChunk)
                WITH s.id AS signal_id, cc
                ORDER BY signal_id, cc.id
                WITH signal_id, collect(cc)[..3] AS chunks
                UNWIND chunks AS cc
                RETURN signal_id,
                       cc.text AS code,
                       cc.language AS language,
                       cc.source_file AS file
                """,
                ids=signal_ids,
            )
            refs: dict[str, list[dict]] = {}
            for row in result:
                sid = row["signal_id"]
                if sid not in refs:
                    refs[sid] = []
                text = row.get("code", "")
                if len(text) > 600:
                    text = text[:600] + "..."
                refs[sid].append(
                    {
                        "code": text,
                        "language": row.get("language", ""),
                        "file": row.get("file", ""),
                    }
                )
            return refs
    except Exception as e:
        logger.warning("Could not fetch signal code refs: %s", e)
        return {}


@retry_on_deadlock()
def claim_signals_for_check(
    facility: str,
    batch_size: int = 5,
    reference_shot: int | None = None,
) -> list[dict]:
    """Claim a batch of enriched signals for check.

    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.
    Uses reference_shot from config for TDI-based checking.
    """
    import uuid as _uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    try:
        with GraphClient() as gc:
            # Step 1: Claim with random ordering and unique token
            gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $enriched
                  AND (s.claimed_at IS NULL
                       OR s.claimed_at < datetime() - duration($cutoff))
                WITH s ORDER BY rand() LIMIT $batch_size
                SET s.claimed_at = datetime(), s.claim_token = $token
                """,
                facility=facility,
                enriched=FacilitySignalStatus.enriched.value,
                batch_size=batch_size,
                cutoff=cutoff,
                token=claim_token,
            )

            # Step 2: Read back only signals WE successfully claimed
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility, claim_token: $token})
                // Derive data_access ID based on signal type
                WITH s,
                     CASE WHEN s.tdi_function IS NOT NULL
                          THEN $facility + ':tdi:functions'
                          WHEN s.data_source_name IS NOT NULL
                          THEN $facility + ':mdsplus:tree_tdi'
                          ELSE null END AS derived_data_access
                RETURN s.id AS id, s.accessor AS accessor, s.data_source_name AS data_source_name,
                       s.data_source_path AS data_source_path,
                       s.physics_domain AS physics_domain, s.tdi_function AS tdi_function,
                       s.discovery_source AS discovery_source, s.name AS name,
                       s.data_source_node AS data_source_node,
                       COALESCE(s.data_access, derived_data_access) AS data_access
                """,
                facility=facility,
                token=claim_token,
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
                    s.enrichment_source = 'direct',
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
                    MERGE (d:Diagnostic {id: s.facility_id + ':' + diag_name})
                    ON CREATE SET d.name = diag_name, d.facility_id = s.facility_id
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


def mark_signals_underspecified(
    signals: list[dict],
    batch_cost: float = 0.0,
) -> int:
    """Mark signals as underspecified — LLM enrichment completed but with insufficient context.

    These signals received LLM processing but the model determined that the
    available context was too limited to produce a reliable enrichment. They
    should be re-enriched when additional context (e.g. TDI source code,
    wiki documentation) becomes available.

    Uses the same enrichment fields as mark_signals_enriched but sets status
    to 'underspecified' instead of 'enriched'.

    Args:
        signals: List of signal enrichment results (same schema as enriched)
        batch_cost: Total LLM cost for this batch (distributed across signals)
    """
    if not signals:
        return 0

    per_signal_cost = batch_cost / len(signals) if batch_cost > 0 else 0.0

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $signals AS sig
                MATCH (s:FacilitySignal {id: sig.id})
                SET s.status = $status,
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
                    s.enrichment_source = 'direct_underspecified',
                    s.llm_cost = $per_signal_cost,
                    s.enriched_at = datetime(),
                    s.claimed_at = null
                """,
                signals=signals,
                status=FacilitySignalStatus.underspecified.value,
                per_signal_cost=per_signal_cost,
            )
        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals underspecified: %s", e)
        return 0


@retry_on_deadlock()
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


# Maximum number of crash retries before marking a signal as checked-failed.
# Covers core dumps, timeouts, and other transient script failures.
MAX_CRASH_RETRIES = 3


def release_or_fail_after_crash(
    signal_id: str,
    error_type: str = "segfault",
    error_msg: str = "MDSplus process crash",
) -> bool:
    """Increment crash retry counter and either release or fail the signal.

    Tracks how many times a signal has caused a process crash (core dump,
    timeout, etc.) via the ``check_retries`` property.  After
    ``MAX_CRASH_RETRIES`` crashes, marks the signal as ``checked`` with
    ``success=false`` so it stops being retried.

    Returns True if the signal was marked as checked-failed (max retries
    exceeded), False if it was released for another attempt.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {id: $id})
                SET s.check_retries = coalesce(s.check_retries, 0) + 1,
                    s.claimed_at = null
                RETURN s.check_retries AS retries
                """,
                id=signal_id,
            )
            retries = result[0]["retries"] if result else 0
            if retries >= MAX_CRASH_RETRIES:
                gc.query(
                    """
                    MATCH (s:FacilitySignal {id: $id})
                    SET s.status = $checked,
                        s.checked = true,
                        s.checked_at = datetime()
                    """,
                    id=signal_id,
                    checked=FacilitySignalStatus.checked.value,
                )
                logger.info(
                    "Signal %s marked checked-failed after %d crash retries (%s)",
                    signal_id,
                    retries,
                    error_type,
                )
                return True
            return False
    except Exception as e:
        logger.warning(
            "Could not update crash retry for %s: %s — releasing claim",
            signal_id,
            e,
        )
        release_signal_claim(signal_id)
        return False


# =============================================================================
# Signal Pattern Detection & Propagation
# =============================================================================

# Regex to replace numeric segments (2+ digits) with NNN for pattern grouping.
# Matches: _007, _010, :003, pure digits at segment boundaries.
_PATTERN_RE = re.compile(r"\d{2,}")


def _accessor_to_pattern(accessor: str) -> str:
    """Convert a signal accessor to its pattern template.

    Replaces numeric segments (2+ digits) with NNN.
    E.g., 'CALIB_GAS_010:PROPERTIES:PARAM_048:LIM' -> 'CALIB_GAS_NNN:PROPERTIES:PARAM_NNN:LIM'
    """
    return _PATTERN_RE.sub("NNN", accessor)


def detect_signal_groups(
    facility: str,
    min_instances: int = 3,
) -> tuple[int, int]:
    """Detect indexed signal groups and create SignalGroup nodes with MEMBER_OF.

    Scans discovered FacilitySignals for accessor patterns where numeric
    segments vary (e.g., CALIB_GAS_010:...:PARAM_048:LIM). For each
    pattern with >= min_instances, creates a SignalGroup node and links
    all member signals via MEMBER_OF relationships. One signal per group
    is designated as the representative (stored on SignalGroup.representative_id).

    Status stays ``discovered`` — no transient status is introduced.
    ``claim_signals_for_enrichment`` skips signals that are MEMBER_OF
    a SignalGroup (unless they are the representative) so only
    representatives are sent to the LLM.

    After the representative is enriched, ``propagate_signal_group_enrichment()``
    copies the LLM-generated metadata to all members and moves them
    straight to ``enriched``.

    Idempotent: signals already linked to a SignalGroup via MEMBER_OF
    are excluded from the initial query.

    Args:
        facility: Facility ID
        min_instances: Minimum signals in a group to form a pattern

    Returns:
        Tuple of (groups_detected, signals_linked_as_members)
    """
    # Fetch all discovered signal accessors not yet in a signal group
    with GraphClient() as gc:
        results = gc.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WHERE s.status = $discovered
              AND NOT EXISTS { (s)-[:MEMBER_OF]->(:SignalGroup) }
            RETURN s.id AS id, s.accessor AS accessor
            """,
            facility=facility,
            discovered=FacilitySignalStatus.discovered.value,
        )

    if not results:
        return 0, 0

    # Group signals by pattern template
    pattern_groups: dict[str, list[dict]] = {}
    for r in results:
        if r["accessor"] is None:
            continue
        pattern = _accessor_to_pattern(r["accessor"])
        pattern_groups.setdefault(pattern, []).append(
            {"id": r["id"], "accessor": r["accessor"]}
        )

    # Filter to groups with enough instances
    multi_groups = {
        p: sigs for p, sigs in pattern_groups.items() if len(sigs) >= min_instances
    }

    if not multi_groups:
        return 0, 0

    # For each pattern group, create a SignalGroup and link all members via MEMBER_OF.
    total_members = 0
    with GraphClient() as gc:
        gc.ensure_facility(facility)
        for pattern, sigs in multi_groups.items():
            # Representative = first signal alphabetically (deterministic)
            sigs.sort(key=lambda s: s["accessor"])
            representative = sigs[0]

            group_id = f"{facility}:{pattern}"
            member_ids = [s["id"] for s in sigs]

            # Create SignalGroup and link all members
            gc.query(
                """
                MERGE (sg:SignalGroup {id: $group_id})
                ON CREATE SET
                    sg.facility_id = $facility,
                    sg.group_key = $pattern,
                    sg.member_count = $member_count,
                    sg.representative_id = $rep_id,
                    sg.status = 'discovered'
                WITH sg
                MATCH (f:Facility {id: $facility})
                MERGE (sg)-[:AT_FACILITY]->(f)
                WITH sg
                UNWIND $member_ids AS mid
                MATCH (s:FacilitySignal {id: mid})
                MERGE (s)-[:MEMBER_OF]->(sg)
                """,
                group_id=group_id,
                facility=facility,
                pattern=pattern,
                member_count=len(sigs),
                rep_id=representative["id"],
                member_ids=member_ids,
            )
            total_members += len(sigs)

    groups_detected = len(multi_groups)
    logger.info(
        "Detected %d signal groups for %s: %d members",
        groups_detected,
        facility,
        total_members,
    )
    return groups_detected, total_members


def propagate_signal_group_enrichment(
    representative_id: str,
    enrichment: dict,
    batch_cost: float = 0.0,
) -> int:
    """Propagate enrichment from a representative signal to all group members.

    After the representative signal is enriched by the LLM, this copies
    the physics_domain, description, name, diagnostic, keywords, etc.
    to all signals that are MEMBER_OF the same SignalGroup.

    Members are still ``discovered`` (no transient status).  This
    function transitions them directly to ``enriched``.

    Args:
        representative_id: ID of the enriched representative signal
        enrichment: Dict with enrichment fields (physics_domain, description, etc.)
        batch_cost: LLM cost to distribute across all propagated signals

    Returns:
        Number of member signals updated
    """
    with GraphClient() as gc:
        # Find members via MEMBER_OF → SignalGroup (excluding the representative)
        count_result = gc.query(
            """
            MATCH (rep:FacilitySignal {id: $rep_id})
                  -[:MEMBER_OF]->(sg:SignalGroup)
            MATCH (s:FacilitySignal)-[:MEMBER_OF]->(sg)
            WHERE s.id <> $rep_id
              AND s.status = $discovered
            RETURN count(s) AS cnt
            """,
            rep_id=representative_id,
            discovered=FacilitySignalStatus.discovered.value,
        )
        follower_count = count_result[0]["cnt"] if count_result else 0

        # Update the SignalGroup itself: status → enriched, copy description/keywords
        gc.query(
            """
            MATCH (rep:FacilitySignal {id: $rep_id})
                  -[:MEMBER_OF]->(sg:SignalGroup)
            SET sg.status = 'enriched',
                sg.description = $description,
                sg.keywords = $keywords,
                sg.physics_domain = $physics_domain,
                sg.claimed_at = null
            """,
            rep_id=representative_id,
            description=enrichment.get("description", ""),
            keywords=enrichment.get("keywords", []),
            physics_domain=enrichment.get("physics_domain", ""),
        )

        if follower_count == 0:
            return 0

        # Cost per signal (representative + followers)
        per_signal_cost = batch_cost / (follower_count + 1) if batch_cost > 0 else 0.0

        # Propagate enrichment: discovered → enriched
        result = gc.query(
            """
            MATCH (rep:FacilitySignal {id: $rep_id})
                  -[:MEMBER_OF]->(sg:SignalGroup)
            MATCH (s:FacilitySignal)-[:MEMBER_OF]->(sg)
            WHERE s.id <> $rep_id
              AND s.status = $discovered
            SET s.status = $enriched,
                s.physics_domain = $physics_domain,
                s.description = $description,
                s.name = $name,
                s.diagnostic = CASE WHEN $diagnostic <> '' THEN $diagnostic ELSE s.diagnostic END,
                s.analysis_code = CASE WHEN $analysis_code <> '' THEN $analysis_code ELSE s.analysis_code END,
                s.keywords = $keywords,
                s.sign_convention = CASE WHEN $sign_convention <> '' THEN $sign_convention ELSE s.sign_convention END,
                s.llm_cost = $per_signal_cost,
                s.enriched_at = datetime(),
                s.claimed_at = null,
                s.enrichment_source = 'signal_group_propagation'
            RETURN count(s) AS updated
            """,
            rep_id=representative_id,
            discovered=FacilitySignalStatus.discovered.value,
            enriched=FacilitySignalStatus.enriched.value,
            physics_domain=enrichment.get("physics_domain", ""),
            description=enrichment.get("description", ""),
            name=enrichment.get("name", ""),
            diagnostic=enrichment.get("diagnostic", ""),
            analysis_code=enrichment.get("analysis_code", ""),
            keywords=enrichment.get("keywords", []),
            sign_convention=enrichment.get("sign_convention", ""),
            per_signal_cost=per_signal_cost,
        )

        updated = result[0]["updated"] if result else 0

        # Also create Diagnostic nodes for members
        if enrichment.get("diagnostic"):
            gc.query(
                """
                MATCH (rep:FacilitySignal {id: $rep_id})
                      -[:MEMBER_OF]->(sg:SignalGroup)
                MATCH (s:FacilitySignal)-[:MEMBER_OF]->(sg)
                WHERE s.id <> $rep_id
                  AND s.status = $enriched
                WITH s, toLower(trim($diagnostic)) AS diag_name
                WHERE diag_name <> ''
                MERGE (d:Diagnostic {id: s.facility_id + ':' + diag_name})
                ON CREATE SET d.name = diag_name, d.facility_id = s.facility_id
                MERGE (s)-[:BELONGS_TO_DIAGNOSTIC]->(d)
                WITH d, s
                MATCH (f:Facility {id: s.facility_id})
                MERGE (d)-[:AT_FACILITY]->(f)
                """,
                rep_id=representative_id,
                enriched=FacilitySignalStatus.enriched.value,
                diagnostic=enrichment.get("diagnostic", ""),
            )

        if updated > 0:
            logger.info(
                "Propagated enrichment from %s to %d followers",
                representative_id,
                updated,
            )

        return updated


# =============================================================================
# Epoch Detection Queries
# =============================================================================


def get_tree_epochs(facility: str, data_source_name: str) -> list[dict]:
    """Get existing epochs for a tree from the graph.

    Returns list of epoch dicts with id, version, first_shot, last_shot.
    Empty list if no epochs exist.
    """
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (v:SignalEpoch {facility_id: $facility, data_source_name: $tree})
                RETURN v.id AS id, v.version AS version,
                       v.first_shot AS first_shot, v.last_shot AS last_shot,
                       v.node_count AS node_count
                ORDER BY v.version
                """,
                facility=facility,
                tree=data_source_name,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning(
            "Could not get epochs for %s:%s: %s", facility, data_source_name, e
        )
        return []


def get_latest_epoch_shot(facility: str, data_source_name: str) -> int | None:
    """Get the first_shot of the latest epoch for incremental scanning."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (v:SignalEpoch {facility_id: $facility, data_source_name: $tree})
                RETURN max(v.first_shot) AS latest_shot
                """,
                facility=facility,
                tree=data_source_name,
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

    Creates SignalEpoch nodes and FacilitySignal nodes with proper
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
            # Phase 1: Create all SignalEpoch nodes
            clean_epochs = []
            for e in sorted_epochs:
                clean = {
                    "id": e["id"],
                    "data_source_name": e["data_source_name"],
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
                MERGE (v:SignalEpoch {id: ep.id})
                ON CREATE SET v += ep,
                              v.discovery_date = datetime()
                ON MATCH SET v.node_count = ep.node_count,
                             v.last_shot = ep.last_shot,
                             v.nodes_added = ep.nodes_added,
                             v.nodes_removed = ep.nodes_removed,
                             v.added_subtrees = ep.added_subtrees,
                             v.removed_subtrees = ep.removed_subtrees
                WITH v, ep
                MATCH (f:Facility {id: ep.facility_id})
                MERGE (v)-[:AT_FACILITY]->(f)
                WITH v, ep
                MERGE (t:DataSource {id: ep.facility_id + ':' + ep.data_source_name})
                ON CREATE SET t.name = ep.data_source_name, t.facility_id = ep.facility_id
                MERGE (v)-[:IN_DATA_SOURCE]->(t)
                """,
                epochs=clean_epochs,
            )
            results["epochs"] = len(clean_epochs)

            # Phase 2: Create predecessor relationships
            gc.query(
                """
                UNWIND $epochs AS ep
                WITH ep WHERE ep.predecessor IS NOT NULL
                MATCH (v:SignalEpoch {id: ep.id})
                MATCH (pred:SignalEpoch {id: ep.predecessor})
                MERGE (v)-[:HAS_PREDECESSOR]->(pred)
                """,
                epochs=clean_epochs,
            )

            # Phase 3: Process signal lifecycle (in version order for proper sequencing)
            for epoch in sorted_epochs:
                epoch_id = epoch["id"]
                facility_id = epoch["facility_id"]
                data_source_name = epoch["data_source_name"]
                added_paths = epoch.get("added_paths", [])
                removed_paths = epoch.get("removed_paths", [])

                # Create signals from added_paths with INTRODUCED_IN edge
                if added_paths:
                    signals = []
                    for path in added_paths:
                        name = path.split(":")[-1].split(".")[-1]
                        signal_id = (
                            f"{facility_id}:general/{data_source_name}/{name.lower()}"
                        )
                        signals.append(
                            {
                                "id": signal_id,
                                "facility_id": facility_id,
                                "physics_domain": "general",
                                "name": name,
                                "accessor": f"data({path})",
                                "data_access": data_access_id
                                or f"{facility_id}:mdsplus:tree_tdi",
                                "data_source_name": data_source_name,
                                "data_source_path": path,
                                "unit": "",
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
                        MATCH (v:SignalEpoch {id: sig.epoch_id})
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
                        WHERE s.data_source_path = path
                        MATCH (v:SignalEpoch {id: $epoch_id})
                        WHERE NOT (s)-[:REMOVED_IN]->(:SignalEpoch)
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
    data_source_name: str,
    shot: int,
    data_access_id: str,
) -> list[dict]:
    """Discover signals from an MDSplus tree via SSH.

    Args:
        facility: Facility ID
        ssh_host: SSH host for remote access
        data_source_name: MDSplus tree name
        shot: Reference shot number
        data_access_id: ID of DataAccess for this tree

    Returns:
        List of signal dicts ready for graph insertion
    """
    # Python script to run on remote
    remote_script = f'''
import json
import MDSplus

tree = MDSplus.Tree("{data_source_name}", {shot}, "readonly")
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
            "unit": units or "",
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
        logger.error("SSH timeout discovering %s on %s", data_source_name, facility)
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
        signal_id = f"{facility}:general/{data_source_name}/{name.lower()}"

        signals.append(
            {
                "id": signal_id,
                "facility_id": facility,
                "physics_domain": "general",  # Will be enriched
                "name": name,
                "accessor": f"data({path})",
                "data_access": data_access_id,
                "data_source_name": data_source_name,
                "data_source_path": path,
                "unit": raw.get("units", ""),
                "status": FacilitySignalStatus.discovered.value,
                "discovery_source": "tree_traversal",
                "example_shot": shot,
            }
        )

    logger.info(
        "Discovered %d signals from %s:%s", len(signals), facility, data_source_name
    )
    return signals


def discover_tdi_signals(
    facility: str,
    ssh_host: str,
    tdi_path: str,
    data_access_id: str,
) -> list[dict]:
    """Discover signals from TDI function files via SSH.

    Uses the extract_tdi_signals.py script for proper parsing of .fun files.
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


def ingest_discovered_signals(signals: list[dict], *, batch_size: int = 500) -> int:
    """Ingest discovered signals to graph with epoch relationships.

    Creates FacilitySignal nodes with AT_FACILITY and DATA_ACCESS edges.
    Optionally creates INTRODUCED_IN relationships to SignalEpoch
    epoch (if epoch_id is present).

    Large signal lists (e.g., 5000+ from PPF) are batched to avoid
    transaction timeouts from a single massive UNWIND MERGE.
    """
    if not signals:
        return 0

    ingested = 0
    try:
        with GraphClient() as gc:
            for i in range(0, len(signals), batch_size):
                batch = signals[i : i + batch_size]

                # Phase 1: Create/update signal nodes + AT_FACILITY edge
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
                    signals=batch,
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
                    signals=batch,
                )

                # Phase 3: Create INTRODUCED_IN edges for epoch-tracked signals
                gc.query(
                    """
                    UNWIND $signals AS sig
                    WITH sig WHERE sig.epoch_id IS NOT NULL
                    MATCH (s:FacilitySignal {id: sig.id})
                    MATCH (v:SignalEpoch {id: sig.epoch_id})
                    MERGE (s)-[:INTRODUCED_IN]->(v)
                    """,
                    signals=batch,
                )

                ingested += len(batch)

            if len(signals) > batch_size:
                logger.info(
                    "Ingested %d signals in %d batches",
                    ingested,
                    (len(signals) + batch_size - 1) // batch_size,
                )
        return ingested
    except Exception as e:
        logger.error(
            "Failed to ingest signals (ingested %d/%d): %s", ingested, len(signals), e
        )
        return ingested


# =============================================================================
# Async Workers
# =============================================================================


async def seed_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that seeds discovery work into the graph.

    For MDSplus: creates SignalEpoch nodes from config (fast).
    For other scanners (TDI, PPF, EDAS, wiki): runs scanner.scan() and
    ingests signals directly to graph.

    Idempotent: uses MERGE so re-runs detect existing work. Reports
    both new and pre-existing items so the display reflects total state.
    """
    from imas_codex.discovery.mdsplus.graph_ops import (
        backfill_tree_relationships,
        seed_versions,
    )
    from imas_codex.discovery.signals.scanners.base import get_scanner

    facility_config = state.facility_config
    data_systems = facility_config.get("data_systems", {})
    ssh_host = state.ssh_host or facility_config.get("ssh_host", state.facility)

    # Use pre-fetched graph state to report resumed progress
    existing_versions = state.initial_version_counts
    existing_signals = state.initial_signal_counts
    existing_total = existing_signals.get("total", 0)

    if existing_total > 0:
        # Show signal count (not version count) for meaningful display
        state.discover_stats.processed = existing_total
        if on_progress:
            on_progress(
                f"resumed: {existing_versions.get('total', 0)} versions, "
                f"{existing_total:,} signals in graph",
                state.discover_stats,
            )

    total_discovered = existing_total

    for scanner_type in state.scanner_types:
        if state.stop_requested:
            break

        if scanner_type == "mdsplus":
            # MDSplus: seed SignalEpoch nodes for each tree
            mdsplus_config = data_systems.get("mdsplus", {})
            if not isinstance(mdsplus_config, dict):
                continue

            for tree_config in mdsplus_config.get("trees", []):
                data_source_name = tree_config.get("source_name")
                if not data_source_name:
                    continue

                # Expand subtrees
                subtrees = tree_config.get("subtrees", [])
                trees_to_process = (
                    [
                        (st["source_name"], {**tree_config, **st})
                        for st in subtrees
                        if st.get("source_name")
                    ]
                    if subtrees
                    else [(data_source_name, tree_config)]
                )

                for sub_name, sub_config in trees_to_process:
                    versions = sub_config.get("versions", [])
                    ver_list = [v["version"] for v in versions if "version" in v]
                    if not ver_list and state.reference_shot:
                        ver_list = [state.reference_shot]

                    if ver_list:
                        seeded = await asyncio.to_thread(
                            seed_versions,
                            state.facility,
                            sub_name,
                            ver_list,
                            versions,
                        )
                        total_discovered += seeded
                        state.discover_stats.processed += seeded
                        if on_progress and seeded:
                            on_progress(
                                f"seeded {sub_name}: {seeded} versions",
                                state.discover_stats,
                                [
                                    {
                                        "id": f"{state.facility}:{sub_name}",
                                        "data_source_name": sub_name,
                                        "signals_in_source": seeded,
                                    }
                                ],
                            )

                        # Merge setup_commands
                        if "setup_commands" not in sub_config:
                            sub_config["setup_commands"] = mdsplus_config.get(
                                "setup_commands", []
                            )

            # Create DataAccess node for MDSplus
            connection_tree = mdsplus_config.get("connection_tree")
            first_tree = next(
                (
                    t["source_name"]
                    for t in mdsplus_config.get("trees", [])
                    if t.get("source_name")
                ),
                None,
            )
            primary_tree = connection_tree or first_tree
            if primary_tree:
                try:

                    def _create_mdsplus_da(_facility: str, _primary_tree: str) -> None:
                        with GraphClient() as gc:
                            gc.query(
                                """
                                MATCH (f:Facility {id: $facility})
                                MERGE (da:DataAccess {id: $id})
                                SET da.facility_id = $facility,
                                    da.method_type = 'mdsplus',
                                    da.library = 'MDSplus',
                                    da.access_type = 'local',
                                    da.data_source = 'mdsplus',
                                    da.connection_template = $conn_tpl,
                                    da.data_template = $data_tpl
                                MERGE (da)-[:AT_FACILITY]->(f)
                                """,
                                id=f"{_facility}:mdsplus:tree_tdi",
                                facility=_facility,
                                conn_tpl=(
                                    f"import MDSplus\n"
                                    f"tree = MDSplus.Tree('{_primary_tree}', "
                                    f"{{shot}}, 'readonly')"
                                ),
                                data_tpl="data = tree.getNode('{data_source_path}').data()",
                            )

                    await asyncio.to_thread(
                        _create_mdsplus_da, state.facility, primary_tree
                    )
                except Exception as e:
                    logger.warning("Failed to create MDSplus DataAccess: %s", e)

            # Backfill IN_DATA_SOURCE edges for DataNodes ingested before
            # DataSource relationships were added to the schema
            await asyncio.to_thread(backfill_tree_relationships, state.facility)

            continue

        # Non-MDSplus scanners: run scan() and ingest signals directly
        if on_progress:
            on_progress(f"scanning {scanner_type}", state.discover_stats)

        try:
            scanner = get_scanner(scanner_type)
        except KeyError:
            logger.warning("Scanner '%s' not registered, skipping", scanner_type)
            continue

        scanner_config = data_systems.get(scanner_type, {})
        if not isinstance(scanner_config, dict):
            scanner_config = {}

        if state.signal_limit:
            scanner_config = {**scanner_config, "_scan_limit": state.signal_limit}

        try:
            result = await scanner.scan(
                facility=state.facility,
                ssh_host=ssh_host,
                config=scanner_config,
                reference_shot=state.reference_shot,
            )

            if result.wiki_context:
                state.wiki_context.update(result.wiki_context)

            if result.signals:
                count = await asyncio.to_thread(
                    ingest_discovered_signals,
                    [s.model_dump(exclude_none=True) for s in result.signals],
                )
                total_discovered += count
                state.discover_stats.processed += count

                if on_progress:
                    on_progress(
                        f"{scanner_type}: discovered {count} signals",
                        state.discover_stats,
                        [
                            {
                                "id": s.id,
                                "data_source_name": scanner_type,
                                "data_source_path": s.accessor,
                                "signals_in_source": count,
                            }
                            for s in result.signals[:20]
                        ],
                    )

            # Persist DataAccess node independently of signals —
            # thin-client scanners create access metadata without signals
            if result.data_access:
                try:
                    da = result.data_access

                    def _ingest_da(_da_id: str, _props: dict, _facility: str) -> None:
                        with GraphClient() as gc:
                            gc.query(
                                """
                                MERGE (da:DataAccess {id: $id})
                                SET da += $props
                                WITH da
                                MATCH (f:Facility {id: $facility})
                                MERGE (da)-[:AT_FACILITY]->(f)
                                """,
                                id=_da_id,
                                props=_props,
                                facility=_facility,
                            )

                    await asyncio.to_thread(
                        _ingest_da,
                        da.id,
                        da.model_dump(exclude_none=True),
                        state.facility,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to ingest DataAccess for %s: %s",
                        scanner_type,
                        e,
                    )

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

    state.seed_phase.mark_done()
    if on_progress:
        on_progress(
            f"seed complete: {total_discovered} items from {len(state.scanner_types)} sources",
            state.discover_stats,
        )


async def epoch_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that detects structural epochs for dynamic MDSplus trees.

    Runs independently from seed — can take 15+ minutes for large trees.
    Seeds additional SignalEpoch nodes as epochs are detected.
    """
    from imas_codex.discovery.mdsplus.epochs import detect_epochs_for_tree
    from imas_codex.discovery.mdsplus.graph_ops import seed_versions

    facility_config = state.facility_config
    data_systems = facility_config.get("data_systems", {})
    mdsplus_config = data_systems.get("mdsplus", {})
    if not isinstance(mdsplus_config, dict):
        state.epoch_phase.mark_done()
        return

    has_mdsplus = "mdsplus" in state.scanner_types

    if not has_mdsplus:
        state.epoch_phase.mark_done()
        return

    trees_with_epochs = []
    for tree_config in mdsplus_config.get("trees", []):
        data_source_name = tree_config.get("source_name")
        if not data_source_name:
            continue
        # Only detect epochs for trees without static versions
        versions = tree_config.get("versions", [])
        if not versions and state.reference_shot:
            trees_with_epochs.append((data_source_name, tree_config))

    if not trees_with_epochs:
        state.epoch_phase.mark_done()
        if on_progress:
            on_progress("no dynamic trees", state.discover_stats)
        return

    for data_source_name, tree_config in trees_with_epochs:
        if state.stop_requested or state.deadline_expired:
            break

        if on_progress:
            on_progress(
                f"detecting epochs for {data_source_name}",
                state.discover_stats,
                [
                    {
                        "data_source_name": data_source_name,
                        "epoch_progress": {"phase": "coarse"},
                    }
                ],
            )

        try:
            # Bound epoch detection to remaining deadline so it doesn't
            # run for 15+ minutes when the user set --time 1.
            timeout = None
            if state.deadline is not None:
                remaining = state.deadline - time.time()
                if remaining <= 0:
                    break
                timeout = remaining

            coro = asyncio.to_thread(
                detect_epochs_for_tree,
                facility=state.ssh_host or state.facility,
                data_source_name=data_source_name,
                tree_config=tree_config,
            )
            if timeout is not None:
                epochs = await asyncio.wait_for(coro, timeout=timeout)
            else:
                epochs = await coro

            if epochs:
                # Ingest epoch SignalEpoch nodes
                await asyncio.to_thread(
                    ingest_epochs,
                    epochs,
                    data_access_id=f"{state.facility}:mdsplus:tree_tdi",
                    reference_shot=state.reference_shot,
                )
                # Also seed versions from detected epochs for extraction
                epoch_versions = [e["version"] for e in epochs if "version" in e]
                if epoch_versions:
                    await asyncio.to_thread(
                        seed_versions,
                        state.facility,
                        data_source_name,
                        epoch_versions,
                        [],
                    )

                if on_progress:
                    on_progress(
                        f"{data_source_name}: {len(epochs)} epochs detected",
                        state.discover_stats,
                        [
                            {
                                "data_source_name": data_source_name,
                                "epoch_progress": {
                                    "phase": "complete",
                                    "boundaries_found": len(epochs),
                                },
                            }
                        ],
                    )
            else:
                if on_progress:
                    on_progress(f"{data_source_name}: no epochs", state.discover_stats)

        except TimeoutError:
            logger.info(
                "Epoch detection for %s timed out (deadline reached)", data_source_name
            )
            if on_progress:
                on_progress(
                    f"{data_source_name}: epoch detection skipped (deadline)",
                    state.discover_stats,
                )
            break

        except Exception as e:
            logger.error("Epoch detection failed for %s: %s", data_source_name, e)
            if on_progress:
                on_progress(
                    f"{data_source_name}: epoch detection failed - {e}",
                    state.discover_stats,
                )

    state.epoch_phase.mark_done()
    if on_progress:
        on_progress("epoch detection complete", state.discover_stats)


async def mdsplus_extract_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that extracts MDSplus tree versions via SSH (facility-wide).

    Claims SignalEpoch nodes with status=discovered across ALL trees,
    runs SSH extraction, and ingests DataNodes into the graph.
    """
    from imas_codex.discovery.mdsplus.graph_ops import (
        claim_version_for_extraction_facility,
        mark_version_extracted,
        release_version_claim,
    )
    from imas_codex.mdsplus.extraction import (
        async_extract_tree_version,
        ingest_static_tree,
        merge_version_results,
    )

    facility_config = state.facility_config
    data_systems = facility_config.get("data_systems", {})
    mdsplus_config = data_systems.get("mdsplus", {})
    node_usages = None
    if isinstance(mdsplus_config, dict):
        # Find node_usages from first tree config
        for tc in mdsplus_config.get("trees", []):
            if tc.get("node_usages"):
                node_usages = tc["node_usages"]
                break

    has_mdsplus = "mdsplus" in state.scanner_types
    if not has_mdsplus:
        state.extract_phase.mark_done()
        return

    # Report existing extracted signals for idempotent restart
    existing_versions = state.initial_version_counts
    existing_sigs = state.initial_signal_counts
    if existing_versions.get("ingested", 0) > 0:
        state.extract_stats.processed = existing_sigs.get("total", 0)
        if on_progress:
            on_progress(
                f"resumed: {existing_sigs.get('total', 0)} signals already extracted",
                state.extract_stats,
            )

    ssh_retry_count = 0
    max_ssh_retries = 5

    while not state.stop_requested:
        claimed = await asyncio.to_thread(
            claim_version_for_extraction_facility,
            state.facility,
        )

        if not claimed:
            state.extract_phase.record_idle()
            if state.extract_phase.done:
                break
            if on_progress:
                on_progress("idle", state.extract_stats)
            await asyncio.sleep(2.0)
            continue

        state.extract_phase.record_activity(1)
        version = claimed["version"]
        version_id = claimed["id"]
        data_source_name = claimed["data_source_name"]

        if on_progress:
            on_progress(
                f"extracting v{version} {state.facility}:{data_source_name}",
                state.extract_stats,
                [
                    {
                        "id": version_id,
                        "data_source_name": data_source_name,
                        "data_source_path": f"v{version}",
                        "signals_in_source": 0,
                    }
                ],
            )

        try:
            data = await async_extract_tree_version(
                facility=state.facility,
                data_source_name=data_source_name,
                shot=version,
                timeout=600,
                node_usages=node_usages,
            )
            ssh_retry_count = 0

            ver_data = data.get("versions", {}).get(str(version), {})
            if "error" in ver_data:
                logger.warning(
                    "Extraction error for v%d %s: %s",
                    version,
                    data_source_name,
                    ver_data["error"][:100],
                )
                await asyncio.to_thread(release_version_claim, version_id)
                state.extract_stats.errors += 1
                continue

            node_count = ver_data.get("node_count", 0)

            merged = merge_version_results([data])

            def _ingest_tree(_facility: str, _merged: dict) -> None:
                from imas_codex.graph import GraphClient as GC2

                with GC2() as client:
                    ingest_static_tree(client, _facility, _merged)

            await asyncio.to_thread(_ingest_tree, state.facility, merged)

            await asyncio.to_thread(mark_version_extracted, version_id, node_count)
            state.extract_stats.processed += 1
            state.extract_stats.record_batch(1)

            if on_progress:
                on_progress(
                    f"v{version} {data_source_name} — {node_count:,} nodes",
                    state.extract_stats,
                    [
                        {
                            "id": version_id,
                            "data_source_name": data_source_name,
                            "data_source_path": f"v{version}",
                            "signals_in_source": node_count,
                        }
                    ],
                )

        except Exception as e:
            ssh_retry_count += 1
            logger.warning(
                "Extract v%d %s failed (%d/%d): %s",
                version,
                data_source_name,
                ssh_retry_count,
                max_ssh_retries,
                e,
            )
            state.extract_stats.errors += 1
            await asyncio.to_thread(release_version_claim, version_id)

            if ssh_retry_count >= max_ssh_retries:
                logger.error("SSH failed after %d attempts — stopping", max_ssh_retries)
                state.extract_phase.mark_done()
                break

            backoff = min(2**ssh_retry_count, 32)
            await asyncio.sleep(backoff)
            continue

        await asyncio.sleep(0.1)


async def mdsplus_units_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that extracts units for trees with ingested versions (facility-wide).

    Claims trees needing unit extraction, runs batched SSH extraction,
    and creates Unit nodes + HAS_UNIT relationships.
    """
    from imas_codex.discovery.mdsplus.graph_ops import (
        claim_tree_for_units,
        mark_all_versions_units_extracted,
        merge_units_to_graph,
    )
    from imas_codex.mdsplus.extraction import async_extract_units_for_version

    has_mdsplus = "mdsplus" in state.scanner_types
    if not has_mdsplus:
        state.units_phase.mark_done()
        return

    # Wait for at least one extraction before trying units
    while not state.stop_requested:
        if state.extract_stats.processed > 0:
            break
        if state.extract_phase.done:
            break
        if on_progress:
            on_progress("awaiting extract", state.units_stats)
        await asyncio.sleep(2.0)

    if state.stop_requested:
        return

    while not state.stop_requested:
        claimed = await asyncio.to_thread(
            claim_tree_for_units,
            state.facility,
        )

        if not claimed:
            state.units_phase.record_idle()
            if state.units_phase.done:
                break
            await asyncio.sleep(2.0)
            continue

        state.units_phase.record_activity(1)
        data_source_name = claimed["data_source_name"]
        latest_version = claimed["latest_version"]

        if on_progress:
            on_progress(
                f"extracting units for {data_source_name} v{latest_version}",
                state.units_stats,
            )

        try:
            units = await async_extract_units_for_version(
                state.facility,
                data_source_name,
                latest_version,
                timeout=600,
                batch_size=500,
            )

            if units:
                await asyncio.to_thread(
                    merge_units_to_graph, state.facility, data_source_name, units
                )
                state.units_stats.processed += 1
                if on_progress:
                    on_progress(
                        f"{data_source_name}: {len(units)} paths with units",
                        state.units_stats,
                    )

            await asyncio.to_thread(
                mark_all_versions_units_extracted,
                state.facility,
                data_source_name,
                len(units) if units else 0,
            )

        except Exception as e:
            logger.error("Units extraction failed for %s: %s", data_source_name, e)
            state.units_stats.errors += 1

    state.units_phase.mark_done()


async def mdsplus_promote_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that creates FacilitySignals from leaf DataNodes (facility-wide).

    Claims trees with un-promoted leaf nodes and runs promote_leaf_nodes_to_signals.
    """
    from imas_codex.discovery.mdsplus.graph_ops import (
        claim_tree_for_promote,
        promote_leaf_nodes_to_signals,
    )

    has_mdsplus = "mdsplus" in state.scanner_types
    if not has_mdsplus:
        state.promote_phase.mark_done()
        return

    # Report existing promoted signals for idempotent restart (pre-fetched)
    existing = state.initial_signal_counts
    if existing.get("total", 0) > 0:
        state.promote_stats.processed = existing["total"]
        if on_progress:
            on_progress(
                f"resumed: {existing['total']} signals already promoted",
                state.promote_stats,
            )

    # Wait for at least one extraction
    while not state.stop_requested:
        if state.extract_stats.processed > 0:
            break
        if state.extract_phase.done:
            break
        if on_progress:
            on_progress("awaiting extract", state.promote_stats)
        await asyncio.sleep(2.0)

    if state.stop_requested:
        return

    while not state.stop_requested:
        data_source_name = await asyncio.to_thread(
            claim_tree_for_promote,
            state.facility,
        )

        if not data_source_name:
            state.promote_phase.record_idle()
            # Wait for extract and units to finish before declaring done
            if state.extract_phase.done and state.units_phase.done:
                if state.promote_phase.done:
                    break
            await asyncio.sleep(2.0)
            continue

        state.promote_phase.record_activity(1)

        if on_progress:
            on_progress(
                f"promoting {data_source_name}",
                state.promote_stats,
            )

        try:
            promoted = await asyncio.to_thread(
                promote_leaf_nodes_to_signals,
                state.facility,
                data_source_name,
            )
            state.promote_stats.processed += promoted
            state.promote_stats.record_batch(promoted)

            if on_progress:
                on_progress(
                    f"{data_source_name}: {promoted} signals promoted",
                    state.promote_stats,
                    [
                        {
                            "id": f"{state.facility}:{data_source_name}",
                            "data_source_name": data_source_name,
                            "signals_in_source": promoted,
                        }
                    ],
                )

            # Run TDI linkage after promotion
            try:
                from imas_codex.discovery.mdsplus.tdi_linkage import (
                    link_tdi_to_data_nodes,
                )

                tdi_links = link_tdi_to_data_nodes(state.facility)
                if tdi_links:
                    logger.info(
                        "TDI linkage: %d edges for %s", tdi_links, data_source_name
                    )
            except Exception as e:
                logger.warning("TDI linkage failed: %s", e)

        except Exception as e:
            logger.error("Promote failed for %s: %s", data_source_name, e)
            state.promote_stats.errors += 1

    state.promote_phase.mark_done()


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
    from imas_codex.discovery.base.llm import (
        ProviderBudgetExhausted,
        acall_llm_structured,
    )
    from imas_codex.discovery.signals.models import (
        ContextQuality,
        SignalEnrichmentBatch,
    )
    from imas_codex.settings import get_model

    # Get model configured for enrichment task
    model = get_model("language")

    # Render system prompt once (contains physics domains from schema)
    system_prompt = render_prompt("signals/enrichment")

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
        1. Exact data_source_path match (uppercase-normalized)
        2. Accessor path extraction (data(...) pattern)
        3. Normalized path (strip leading backslashes, collapse separators)
        4. PPF-style DDA/DTYPE matching for JET signals
        """
        if not state.wiki_context:
            return None

        # Try data_source_path first, then fall back to accessor
        data_source_path = signal.get("data_source_path") or signal.get("accessor")
        if data_source_path:
            normalized = data_source_path.upper().lstrip("\\")
            # Try with single backslash prefix (wiki format)
            ctx = state.wiki_context.get(f"\\{normalized}")
            if ctx:
                return ctx
            # Try with double backslash prefix
            ctx = state.wiki_context.get(f"\\\\{normalized}")
            if ctx:
                return ctx
            # Try raw
            ctx = state.wiki_context.get(data_source_path.upper())
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
        tree = signal.get("data_source_name")
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
                    RETURN node.text AS text,
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
                    text = row.get("text", "")
                    if len(text) > 600:
                        text = text[:600] + "..."
                    chunks.append(
                        {
                            "text": text,
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

    # Circuit breaker: stop after consecutive batch failures to avoid
    # infinite claim/fail/release loops when LLM service is degraded.
    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures = 0

    while not state.should_stop_enriching():
        # --- Pattern detection phase ---
        # On each idle cycle, detect indexed signal patterns and mark
        # followers so they skip individual LLM enrichment. Once a
        # representative is enriched, its metadata is propagated.
        patterns_detected, followers_marked = await asyncio.to_thread(
            detect_signal_groups,
            state.facility,
        )
        if patterns_detected > 0 and on_progress:
            on_progress(
                f"detected {patterns_detected} patterns ({followers_marked} followers)",
                state.enrich_stats,
            )

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

        # Fetch tree context for signals with HAS_DATA_SOURCE_NODE edges.
        # Send all signal IDs — the Cypher MATCH naturally filters to only
        # signals with HAS_DATA_SOURCE_NODE edges (resilient to property gaps).
        all_signal_ids = [s["id"] for s in signals]
        tree_context = await asyncio.to_thread(fetch_tree_context, all_signal_ids)

        # Fetch deterministic code references via graph traversal
        # (CodeChunk → DataReference → SignalNode ← FacilitySignal)
        code_refs = await asyncio.to_thread(fetch_signal_code_refs, all_signal_ids)

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
                user_lines.append(chunk["text"])
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
                    user_lines.append(f"- [{chunk['page_title']}] {chunk['text']}")

            # Fetch relevant source code chunks via code_chunk_embedding
            code_chunks = await _fetch_code_context(group_key, indexed_signals)
            if code_chunks:
                user_lines.append("\n**Relevant source code:**")
                for chunk in code_chunks:
                    path = chunk.get("source_path", "unknown")
                    ctype = chunk.get("chunk_type", "")
                    label = f"{path} ({ctype})" if ctype else path
                    user_lines.append(f"\n```python  # {label}")
                    user_lines.append(chunk["text"])
                    user_lines.append("```")

            # Add individual signal entries
            for _, signal in indexed_signals:
                signal_index += 1
                user_lines.append(f"\n### Signal {signal_index}")
                user_lines.append(f"accessor: {signal['accessor']}")
                user_lines.append(f"name: {signal.get('name', 'unknown')}")
                if signal.get("tdi_quantity"):
                    user_lines.append(f"tdi_quantity: {signal['tdi_quantity']}")
                if signal.get("discovery_source"):
                    user_lines.append(f"discovery_source: {signal['discovery_source']}")
                if signal.get("unit"):
                    user_lines.append(f"units: {signal['unit']}")
                if signal.get("data_source_name"):
                    user_lines.append(f"data_source_name: {signal['data_source_name']}")
                if signal.get("data_source_path"):
                    user_lines.append(f"data_source_path: {signal['data_source_path']}")

                # Inject existing description from scanner (e.g., EDAS Japanese desc)
                if signal.get("description"):
                    user_lines.append(f"existing_description: {signal['description']}")

                # Inject tree context if available (HAS_DATA_SOURCE_NODE traversal)
                sig_tree_ctx = tree_context.get(signal["id"])
                if sig_tree_ctx:
                    if sig_tree_ctx.get("parent_path"):
                        user_lines.append(f"parent_node: {sig_tree_ctx['parent_path']}")
                    siblings = sig_tree_ctx.get("sibling_paths", [])
                    if siblings:
                        user_lines.append(f"siblings: {', '.join(siblings)}")
                    if sig_tree_ctx.get("tdi_name"):
                        user_lines.append(f"tdi_function: {sig_tree_ctx['tdi_name']}")
                        if sig_tree_ctx.get("tdi_source"):
                            src = sig_tree_ctx["tdi_source"]
                            if len(src) > 2000:
                                src = src[:2000] + "\n... (truncated)"
                            user_lines.append(f"tdi_source:\n```tdi\n{src}\n```")
                    epoch = sig_tree_ctx.get("epoch_range")
                    if epoch:
                        user_lines.append(
                            f"applicability: versions "
                            f"{epoch['first_version']}-{epoch['last_version']}"
                        )

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

                # Inject deterministic code references (graph traversal, not vector search)
                sig_code_refs = code_refs.get(signal["id"], [])
                if sig_code_refs:
                    user_lines.append("\n**Direct code references:**")
                    for ref in sig_code_refs:
                        lang = ref.get("language", "python")
                        fpath = ref.get("file", "unknown")
                        user_lines.append(f"\n```{lang}  # {fpath}")
                        user_lines.append(ref["code"])
                        user_lines.append("```")

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
        except ProviderBudgetExhausted as e:
            logger.error("LLM provider budget exhausted — stopping enrichment: %s", e)
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            state.enrich_phase.mark_done()
            if on_progress:
                on_progress("provider budget exhausted", state.enrich_stats)
            break
        except (ValueError, Exception) as e:
            consecutive_failures += 1
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "Enrichment stopped: %d consecutive batch failures. Last error: %s",
                    consecutive_failures,
                    e,
                )
                state.enrich_phase.mark_done()
                if on_progress:
                    on_progress(
                        f"stopped: {consecutive_failures} consecutive failures",
                        state.enrich_stats,
                    )
                break
            backoff = min(60.0, 5.0 * (2 ** (consecutive_failures - 1)))
            logger.warning(
                "Batch failure %d/%d: %s — backing off %.0fs",
                consecutive_failures,
                MAX_CONSECUTIVE_FAILURES,
                e,
                backoff,
            )
            if on_progress:
                on_progress(
                    f"LLM error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})",
                    state.enrich_stats,
                )
            await asyncio.sleep(backoff)
            continue

        # Track cost
        state.enrich_stats.cost += batch_cost
        consecutive_failures = 0  # Reset circuit breaker on success

        logger.debug(
            "LLM response: %d tokens for %d signals (cost=$%.4f)",
            total_tokens,
            len(signals),
            batch_cost,
        )

        # Match results back to signals by index (1-based signal_index)
        enriched = []
        underspecified = []
        matched_indices = set()

        for result in batch_result.results:
            # signal_index is 1-based, list is 0-based
            idx = result.signal_index - 1
            if 0 <= idx < len(signals):
                signal = signals[idx]
                matched_indices.add(idx)
                entry = {
                    "id": signal["id"],
                    "physics_domain": result.physics_domain.value,
                    "description": result.description,
                    "name": result.name,
                    "diagnostic": result.diagnostic,
                    "analysis_code": result.analysis_code,
                    "keywords": result.keywords,
                    "sign_convention": result.sign_convention,
                }
                if result.context_quality == ContextQuality.low:
                    underspecified.append(entry)
                else:
                    enriched.append(entry)

        # Release claims for unmatched signals
        for idx, signal in enumerate(signals):
            if idx not in matched_indices:
                await asyncio.to_thread(release_signal_claim, signal["id"])

        # Mark underspecified signals (low context quality)
        if underspecified:
            await asyncio.to_thread(
                mark_signals_underspecified, underspecified, batch_cost
            )
            state.enrich_stats.processed += len(underspecified)
            logger.info(
                "Marked %d signals as underspecified (low context)",
                len(underspecified),
            )

        # Update graph with enrichment cost for historical tracking
        if enriched:
            await asyncio.to_thread(mark_signals_enriched, enriched, batch_cost)
            state.enrich_stats.processed += len(enriched)

            # Propagate enrichment from representative signals to pattern followers
            total_propagated = 0
            for e in enriched:
                propagated = await asyncio.to_thread(
                    propagate_signal_group_enrichment,
                    e["id"],
                    e,
                    batch_cost / len(enriched) if batch_cost > 0 else 0.0,
                )
                total_propagated += propagated

            if total_propagated > 0:
                state.enrich_stats.processed += total_propagated
                if on_progress:
                    on_progress(
                        f"propagated to {total_propagated} pattern followers",
                        state.enrich_stats,
                        enriched,
                    )

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
    # Missing shared library — e.g. libjmmshr_gsl.so, libblas.so
    if "error loading" in err_lower or "cannot open shared object" in err_lower:
        return "missing_library"
    # Expression evaluation errors — TDI expression failures
    if "extra_arg" in err_lower or "roprand" in err_lower:
        return "expression_error"
    if "empty" in err_lower:
        return "empty_data"
    if "permission" in err_lower or "denied" in err_lower:
        return "permission_denied"
    if "segmentation" in err_lower or "segfault" in err_lower:
        return "segfault"
    return "other"


def _is_excluded_tdi_function(func_name: str, exclude_list: list[str]) -> bool:
    """Check if a TDI function should be excluded from checking.

    Args:
        func_name: TDI function name (e.g., "tcv_eq", "tile_store")
        exclude_list: List of excluded function names from facility config

    Returns:
        True if the function should be excluded from checking.
    """
    return func_name in exclude_list


def _resolve_check_tree(
    signal: dict,
    connection_tree: str,
    independent_trees: set[str],
    tree_shots: dict[str, list[int]],
    reference_shot: int,
) -> tuple[str, str, list[int]]:
    """Determine the correct data_source_name, accessor, and check_shots for a signal.

    Routes signals based on whether their tree is independent (opened directly)
    or a subtree of the connection tree (opened via the connection tree).

    For independent trees with configured versions (e.g., static), returns all
    version shots as check_shots. For subtrees, returns the reference_shot
    plus any connection tree version shots as check_shots.

    Args:
        signal: Signal dict with data_source_name, data_source_path, accessor, etc.
        connection_tree: Parent tree name (e.g., "tcv_shot")
        independent_trees: Set of tree names that are NOT subtrees
        tree_shots: Map of data_source_name -> list of version/epoch shots
        reference_shot: Default shot for subtree checking

    Returns:
        Tuple of (data_source_name, accessor, check_shots) where check_shots
        is a list of shots to try in order.
    """
    data_source_name = signal.get("data_source_name")
    data_source_path = signal.get("data_source_path")
    accessor = signal.get("accessor", "")

    # TDI functions use the connection tree
    if signal.get("tdi_function") and not data_source_name:
        return connection_tree, accessor, [reference_shot]

    # For tree_traversal signals, use full data_source_path as accessor
    if data_source_path and signal.get("discovery_source") == "tree_traversal":
        accessor = data_source_path

    # Independent trees are opened directly with their own shots
    if data_source_name and data_source_name in independent_trees:
        versions = tree_shots.get(data_source_name, [])
        if versions:
            return data_source_name, accessor, versions
        return data_source_name, accessor, [reference_shot]

    # Subtree signals go through the connection tree
    # Include connection tree version shots as additional check_shots
    conn_shots = tree_shots.get(connection_tree, [])
    check_shots = [reference_shot]
    for s in conn_shots:
        if s != reference_shot:
            check_shots.append(s)
    return connection_tree, accessor, check_shots


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

    from imas_codex.discovery.signals.scanners.base import get_scanner
    from imas_codex.graph.models import FacilitySignal as FacilitySignalModel

    facility_config = state.facility_config
    data_systems = facility_config.get("data_systems", {})

    # Build tree configuration from facility config
    mdsplus_config = data_systems.get("mdsplus", {})
    connection_tree = "tcv_shot"
    independent_trees: set[str] = set()
    tree_shots: dict[str, list[int]] = {}

    if isinstance(mdsplus_config, dict):
        connection_tree = mdsplus_config.get("connection_tree", "tcv_shot")

        # Read independent trees from data_access_patterns
        dap = facility_config.get("data_access_patterns", {})
        for t in dap.get("independent_trees", []):
            independent_trees.add(t)

        # Build version/epoch shot lists per tree
        all_trees = mdsplus_config.get("trees", [])
        for st in all_trees:
            if isinstance(st, dict):
                data_source_name = st.get("source_name", "")
                versions = st.get("versions", [])
                if versions:
                    # Collect all version shots for this tree
                    shots = [
                        v.get("first_shot") or v.get("version")
                        for v in versions
                        if v.get("first_shot") or v.get("version")
                    ]
                    if shots:
                        tree_shots[data_source_name] = sorted(shots)

    # Keep batch size moderate — MDSplus segfaults (core dumps) when
    # too many signals from the same tree are checked in a single process.
    # The remote script groups signals by (data_source_name, shot) so 100 signals
    # from the same tree all land in one group, overloading MDSplus.
    BATCH_SIZE = 20

    def _get_signal_scanner_type(signal: dict) -> str:
        """Determine which scanner should check this signal."""
        source = signal.get("discovery_source", "")
        if source == "ppf_enumeration":
            return "ppf"
        if source == "edas_enumeration":
            return "edas"
        if source == "wiki_extraction":
            return "wiki"
        if source in ("xml_extraction", "jec2020_xml", "magnetics_config"):
            return "device_xml"
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

        # --- Scanner-based checks (PPF, EDAS, device_xml) ---
        for scanner_type in ("ppf", "edas", "device_xml"):
            group = scanner_groups.get(scanner_type, [])
            if not group:
                continue

            try:
                scanner = get_scanner(scanner_type)
                scanner_config = data_systems.get(scanner_type, {})
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
                        data_source_name=s.get("data_source_name"),
                        data_source_node=s.get("data_source_node"),
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
                if not state.reference_shot:
                    logger.warning(
                        "Signal %s has no reference_shot",
                        signal["id"],
                    )
                    await asyncio.to_thread(release_signal_claim, signal["id"])
                    continue

                # Resolve the correct tree, accessor, and check_shots
                resolved_tree, resolved_accessor, check_shots = _resolve_check_tree(
                    signal,
                    connection_tree=connection_tree,
                    independent_trees=independent_trees,
                    tree_shots=tree_shots,
                    reference_shot=state.reference_shot,
                )

                batch_input.append(
                    {
                        "id": signal["id"],
                        "accessor": resolved_accessor,
                        "data_source_name": resolved_tree,
                        "check_shots": check_shots,
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
                        await asyncio.to_thread(
                            release_or_fail_after_crash,
                            sig["id"],
                            "parse_error",
                            f"Failed to parse check response: {e}",
                        )
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr[:200] if e.stderr else str(e)
                    # Core dumps indicate MDSplus segfault — log clearly
                    if "dumped core" in stderr or e.returncode < 0:
                        logger.warning(
                            "MDSplus core dump checking %d signals (exit %d) — "
                            "tracking retries (max %d)",
                            len(batch_input),
                            e.returncode,
                            MAX_CRASH_RETRIES,
                        )
                        error_type = "segfault"
                        error_msg = "MDSplus process crash (core dump)"
                    else:
                        logger.warning(
                            "Check script failed (exit %d): %s",
                            e.returncode,
                            stderr,
                        )
                        error_type = "script_crash"
                        error_msg = f"Check script failed (exit {e.returncode})"
                    for sig in batch_input:
                        await asyncio.to_thread(
                            release_or_fail_after_crash,
                            sig["id"],
                            error_type,
                            error_msg,
                        )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Check script timed out for batch of %d signals",
                        len(batch_input),
                    )
                    for sig in batch_input:
                        await asyncio.to_thread(
                            release_or_fail_after_crash,
                            sig["id"],
                            "timeout",
                            "Check script timed out",
                        )
                except Exception as e:
                    logger.warning("Failed to run check: %s", e)
                    for sig in batch_input:
                        await asyncio.to_thread(
                            release_or_fail_after_crash,
                            sig["id"],
                            "infrastructure",
                            str(e)[:200],
                        )

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
    on_extract_progress: Callable | None = None,
    on_promote_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_check_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    stop_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Run parallel data discovery with independent async workers.

    All workers start simultaneously and claim work from the graph:
    - seed: Seeds SignalEpoch nodes from config (exits fast)
    - epoch: Detects structural epochs independently
    - extract: Claims SignalEpoch, SSH extract DataNodes
    - units: Extracts units for ingested tree versions
    - promote: Creates FacilitySignal from leaf DataNodes
    - enrich: LLM classification (claims discovered FacilitySignals)
    - check: Test data access (claims enriched FacilitySignals)
    - embed: Embed descriptions for vector search

    No stage blocking — downstream workers start as soon as the
    first batch of upstream work is available.
    """
    start_time = time.time()

    # Pre-fetch all shared data in a background thread so the event loop
    # stays free for display ticking.  Every worker used to call
    # get_facility() + graph count queries synchronously at startup,
    # blocking the event loop for 30-60 s on remote Neo4j.
    def _preflight(
        _facility: str, _ssh_host: str | None, _scanner_types: list[str] | None
    ) -> tuple[dict, str | None, list[str], dict, dict]:
        from imas_codex.discovery.base.facility import get_facility
        from imas_codex.discovery.mdsplus.graph_ops import (
            get_signal_counts,
            get_version_counts,
        )
        from imas_codex.discovery.signals.scanners.base import (
            get_scanners_for_facility,
        )
        from imas_codex.graph import GraphClient as _GC

        config = get_facility(_facility)
        if not _ssh_host:
            _ssh_host = config.get("ssh_host", _facility)

        reset_transient_signals(_facility)

        with _GC() as gc:
            gc.ensure_facility(_facility)

        # Detect signal groups BEFORE workers start so enrich workers
        # won't waste LLM calls on non-representative group members.
        groups_detected, followers_marked = detect_signal_groups(_facility)
        if groups_detected > 0:
            logger.info(
                "Preflight: detected %d signal groups (%d followers) for %s",
                groups_detected,
                followers_marked,
                _facility,
            )

        if not _scanner_types:
            scanner_instances = get_scanners_for_facility(_facility)
            _scanner_types = [s.scanner_type for s in scanner_instances]

        version_counts = get_version_counts(_facility)
        signal_counts = get_signal_counts(_facility)

        return config, _ssh_host, _scanner_types, version_counts, signal_counts

    (
        facility_config,
        ssh_host,
        scanner_types,
        initial_version_counts,
        initial_signal_counts,
    ) = await asyncio.to_thread(_preflight, facility, ssh_host, scanner_types)

    # Initialize state
    state = DataDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        facility_config=facility_config,
        initial_version_counts=initial_version_counts,
        initial_signal_counts=initial_signal_counts,
        reference_shot=reference_shot,
        scanner_types=scanner_types,
        tdi_path=tdi_path,
        cost_limit=cost_limit,
        signal_limit=signal_limit,
        focus=focus,
        enrich_only=enrich_only,
        deadline=deadline,
    )

    orphan_specs = [
        OrphanRecoverySpec("FacilitySignal", timeout_seconds=CLAIM_TIMEOUT_SECONDS),
        OrphanRecoverySpec("SignalEpoch"),
    ]

    # Build worker specs
    workers: list[WorkerSpec] = []

    # --- Scan workers (unless enrich_only) ---
    if not enrich_only:
        workers.extend(
            [
                WorkerSpec(
                    "seed",
                    "seed_phase",
                    seed_worker,
                    group="seed",
                    should_stop_fn=lambda: state.stop_requested,
                    on_progress=on_discover_progress,
                ),
                WorkerSpec(
                    "epoch",
                    "epoch_phase",
                    epoch_worker,
                    group="seed",
                    should_stop_fn=lambda: state.stop_requested,
                    on_progress=on_discover_progress,
                ),
                WorkerSpec(
                    "extract",
                    "extract_phase",
                    mdsplus_extract_worker,
                    group="extract",
                    should_stop_fn=lambda: state.stop_requested,
                    on_progress=on_extract_progress or on_discover_progress,
                ),
                WorkerSpec(
                    "units",
                    "units_phase",
                    mdsplus_units_worker,
                    group="promote",
                    should_stop_fn=lambda: state.stop_requested,
                    on_progress=on_promote_progress or on_discover_progress,
                ),
                WorkerSpec(
                    "promote",
                    "promote_phase",
                    mdsplus_promote_worker,
                    group="promote",
                    should_stop_fn=lambda: state.stop_requested,
                    on_progress=on_promote_progress or on_discover_progress,
                ),
            ]
        )
    else:
        # In enrich_only mode, mark all scan sub-phases as done
        state.seed_phase.mark_done()
        state.epoch_phase.mark_done()
        state.extract_phase.mark_done()
        state.units_phase.mark_done()
        state.promote_phase.mark_done()
        state._scan_phase.mark_done()

    if discover_only:
        # Run engine with scan workers only — stop when scan phase completes
        await run_discovery_engine(
            state,
            workers,
            stop_event=stop_event,
            orphan_specs=orphan_specs,
            on_worker_status=on_worker_status,
            stop_fn=lambda: state._scan_phase.done,
        )

        return {
            "scanned": state.discover_stats.processed,
            "discovered": state.discover_stats.processed
            + state.promote_stats.processed,
            "enriched": 0,
            "checked": 0,
            "cost": 0.0,
            "elapsed_seconds": time.time() - start_time,
            "extract_count": state.extract_stats.processed,
            "units_count": state.units_stats.processed,
            "promote_count": state.promote_stats.processed,
        }

    # --- Enrich workers ---
    workers.append(
        WorkerSpec(
            "enrich",
            "enrich_phase",
            enrich_worker,
            count=num_enrich_workers,
            should_stop_fn=state.should_stop_enriching,
            on_progress=on_enrich_progress,
        )
    )

    # --- Check workers ---
    if num_check_workers > 0:
        workers.append(
            WorkerSpec(
                "check",
                "check_phase",
                check_worker,
                count=num_check_workers,
                should_stop_fn=state.should_stop_checking,
                on_progress=on_check_progress,
            )
        )

    # --- Embed worker ---
    from imas_codex.discovery.base.embed_worker import embed_description_worker

    workers.append(
        WorkerSpec(
            "embed",
            "enrich_phase",  # embed lifecycle follows enrich
            embed_description_worker,
            group="embed",
            kwargs={"labels": ["FacilitySignal"]},
        )
    )

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=orphan_specs,
        on_worker_status=on_worker_status,
    )

    elapsed = time.time() - start_time
    return {
        "scanned": state.discover_stats.processed,
        "discovered": state.discover_stats.processed + state.promote_stats.processed,
        "enriched": state.enrich_stats.processed,
        "checked": state.check_stats.processed,
        "cost": state.enrich_stats.cost,
        "elapsed_seconds": elapsed,
        "extract_count": state.extract_stats.processed,
        "units_count": state.units_stats.processed,
        "promote_count": state.promote_stats.processed,
    }
