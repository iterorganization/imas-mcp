"""
Parallel data signal discovery engine with async workers.

Architecture:
- Three async workers: Discover, Enrich, Validate
- Graph + claimed_at timestamp for coordination (same pattern as wiki/paths)
- Status transitions:
  - discovered → enriched (LLM classification)
  - enriched → validated (data access test)
- Workers claim signals by setting claimed_at, release by clearing it
- Orphan recovery: signals with claimed_at > 5 min old are reclaimed

Resilience:
- Supervised workers with automatic restart on crash (via base.supervision)
- Exponential backoff on infrastructure errors (Neo4j, network, SSH)
- Graceful degradation when services are temporarily unavailable

Workflow:
1. SCAN: Enumerate signals from data sources (MDSplus trees, TDI functions)
2. ENRICH: LLM classification of physics_domain, description generation
3. VALIDATE: Test data access with example_shot, verify units/sign conventions
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
    tree_names: list[str] = field(default_factory=list)
    reference_shot: int | None = None
    tdi_path: str | None = None

    # Flags
    force: bool = False  # Re-scan trees even if epochs exist

    # Limits
    cost_limit: float = 10.0
    signal_limit: int | None = None
    focus: str | None = None

    # Worker stats
    discover_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    discover_idle_count: int = 0
    enrich_idle_count: int = 0
    validate_idle_count: int = 0

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
            and self.validate_idle_count >= 3
        )
        if all_idle:
            if has_pending_work(self.facility):
                self.discover_idle_count = 0
                self.enrich_idle_count = 0
                self.validate_idle_count = 0
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

    def should_stop_validating(self) -> bool:
        """Check if validate workers should stop."""
        if self.stop_requested:
            return True
        if self.validate_idle_count >= 3:
            # Only stop if enriching is done AND no pending validation work
            enriching_done = self.enrich_idle_count >= 3 or self.budget_exhausted
            if enriching_done and not has_pending_validate_work(self.facility):
                return True
        return False


# =============================================================================
# Graph Queries
# =============================================================================


def has_pending_work(facility: str) -> bool:
    """Check if there's any pending work for this facility."""
    return has_pending_enrich_work(facility) or has_pending_validate_work(facility)


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


def has_pending_validate_work(facility: str) -> bool:
    """Check if there are signals awaiting validation."""
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
    """Get current discovery statistics from graph."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                RETURN
                    count(s) AS total,
                    sum(CASE WHEN s.status = $discovered THEN 1 ELSE 0 END) AS discovered,
                    sum(CASE WHEN s.status = $enriched THEN 1 ELSE 0 END) AS enriched,
                    sum(CASE WHEN s.status = $validated THEN 1 ELSE 0 END) AS validated,
                    sum(CASE WHEN s.status = $skipped THEN 1 ELSE 0 END) AS skipped,
                    sum(CASE WHEN s.status = $failed THEN 1 ELSE 0 END) AS failed,
                    sum(CASE WHEN s.status = $discovered AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_enrich,
                    sum(CASE WHEN s.status = $enriched AND s.claimed_at IS NULL THEN 1 ELSE 0 END) AS pending_validate
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                enriched=FacilitySignalStatus.enriched.value,
                validated=FacilitySignalStatus.validated.value,
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


# =============================================================================
# Signal Claim/Release
# =============================================================================


def claim_signals_for_enrichment(
    facility: str,
    batch_size: int = 10,
) -> list[dict]:
    """Claim a batch of discovered signals for enrichment."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $discovered
                  AND s.claimed_at IS NULL
                WITH s LIMIT $batch_size
                SET s.claimed_at = datetime()
                RETURN s.id AS id, s.accessor AS accessor, s.tree_name AS tree_name,
                       s.node_path AS node_path, s.units AS units, s.name AS name
                """,
                facility=facility,
                discovered=FacilitySignalStatus.discovered.value,
                batch_size=batch_size,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning("Could not claim signals for enrichment: %s", e)
        return []


def claim_signals_for_validation(
    facility: str,
    batch_size: int = 5,
) -> list[dict]:
    """Claim a batch of enriched signals for validation."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})
                WHERE s.status = $enriched
                  AND s.claimed_at IS NULL
                WITH s LIMIT $batch_size
                SET s.claimed_at = datetime()
                RETURN s.id AS id, s.accessor AS accessor, s.tree_name AS tree_name,
                       s.example_shot AS example_shot, s.physics_domain AS physics_domain
                """,
                facility=facility,
                enriched=FacilitySignalStatus.enriched.value,
                batch_size=batch_size,
            )
            return list(result) if result else []
    except Exception as e:
        logger.warning("Could not claim signals for validation: %s", e)
        return []


def mark_signals_enriched(
    signals: list[dict],
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
    """
    if not signals:
        return 0

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
                    s.enriched_at = datetime(),
                    s.claimed_at = null
                """,
                signals=signals,
                enriched=FacilitySignalStatus.enriched.value,
            )
        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals enriched: %s", e)
        return 0


def mark_signals_validated(
    signals: list[dict],
) -> int:
    """Mark signals as validated."""
    if not signals:
        return 0

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $signals AS sig
                MATCH (s:FacilitySignal {id: sig.id})
                SET s.status = $validated,
                    s.validated = true,
                    s.validated_at = datetime(),
                    s.claimed_at = null
                """,
                signals=signals,
                validated=FacilitySignalStatus.validated.value,
            )
        return len(signals)
    except Exception as e:
        logger.warning("Could not mark signals validated: %s", e)
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


def ingest_epochs(epochs: list[dict]) -> int:
    """Ingest TreeModelVersion nodes to the graph.

    Uses MERGE to ensure idempotency - existing epochs are updated, not duplicated.
    Does not reset status of any linked nodes.
    """
    if not epochs:
        return 0

    try:
        with GraphClient() as gc:
            # Clean epochs for ingestion (remove temporary fields)
            clean_epochs = []
            for e in epochs:
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

            # Create predecessor relationships
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

        return len(epochs)
    except Exception as e:
        logger.error("Failed to ingest epochs: %s", e)
        return 0


def create_signals_from_epoch(
    facility: str,
    tree_name: str,
    epoch_id: str,
    paths: list[str],
    access_method_id: str,
    reference_shot: int,
) -> list[dict]:
    """Create signal dicts from epoch paths for ingestion.

    Creates signals with epoch linkage for applicability tracking.
    """
    signals = []
    for path in paths:
        # Extract node name from path
        name = path.split(":")[-1].split(".")[-1]

        # Generate signal ID
        signal_id = f"{facility}:general/{tree_name}/{name.lower()}"

        signals.append(
            {
                "id": signal_id,
                "facility_id": facility,
                "physics_domain": "general",  # Will be enriched
                "name": name,
                "accessor": f"data({path})",
                "access_method": access_method_id,
                "tree_name": tree_name,
                "node_path": path,
                "units": "",  # Will be discovered during validation
                "status": FacilitySignalStatus.discovered.value,
                "discovery_source": "epoch_detection",
                "example_shot": reference_shot,
                "epoch_id": epoch_id,
            }
        )

    return signals


# =============================================================================
# MDSplus Discovery
# =============================================================================


def discover_mdsplus_signals(
    facility: str,
    ssh_host: str,
    tree_name: str,
    shot: int,
    access_method_id: str,
) -> list[dict]:
    """Discover signals from an MDSplus tree via SSH.

    Args:
        facility: Facility ID
        ssh_host: SSH host for remote access
        tree_name: MDSplus tree name
        shot: Reference shot number
        access_method_id: ID of AccessMethod for this tree

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
                "access_method": access_method_id,
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
    access_method_id: str,
) -> list[dict]:
    """Discover signals from TDI function files via SSH.

    Parses .fun files to extract case() statements which define
    available quantities.

    Args:
        facility: Facility ID
        ssh_host: SSH host for remote access
        tdi_path: Path to TDI function directory
        access_method_id: ID of AccessMethod for TDI access

    Returns:
        List of signal dicts ready for graph insertion
    """
    import re

    # List .fun files
    try:
        result = subprocess.run(
            ["ssh", ssh_host, f"find {tdi_path} -name '*.fun' -type f 2>/dev/null"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        files = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    except Exception as e:
        logger.error("Failed to list TDI files: %s", e)
        return []

    signals = []
    for filepath in files:
        func_name = filepath.rsplit("/", 1)[-1].replace(".fun", "")

        try:
            result = subprocess.run(
                ["ssh", ssh_host, f"cat '{filepath}'"],
                capture_output=True,
                timeout=30,
                check=True,
            )
            content = result.stdout.decode("latin-1", errors="replace")
        except Exception:
            continue

        # Extract case() quantities
        pattern = r'case\s*\(\s*["\']([A-Z_0-9]+)["\']\s*\)'
        quantities = sorted(set(re.findall(pattern, content, re.IGNORECASE)))

        for qty in quantities:
            qty_upper = qty.upper()
            signal_id = f"{facility}:general/{func_name}/{qty_upper.lower()}"

            signals.append(
                {
                    "id": signal_id,
                    "facility_id": facility,
                    "physics_domain": "general",  # Will be enriched
                    "name": qty_upper,
                    "accessor": f"{func_name}('{qty_upper}')",
                    "access_method": access_method_id,
                    "tdi_function": func_name,
                    "tdi_quantity": qty_upper,
                    "status": FacilitySignalStatus.discovered.value,
                    "discovery_source": "tdi_introspection",
                }
            )

    logger.info("Discovered %d TDI signals from %s", len(signals), facility)
    return signals


def ingest_discovered_signals(signals: list[dict]) -> int:
    """Ingest discovered signals to graph."""
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
    """Worker that scans data sources for signals with epoch detection.

    Uses batch_discovery.discover_epochs_optimized() for efficient epoch detection.
    Idempotent: skips trees that already have epochs unless force=True.
    """
    from imas_codex.mdsplus.batch_discovery import (
        EpochProgress,
        discover_epochs_optimized,
    )

    access_method_id = f"{state.facility}:mdsplus:tree"

    # Scan MDSplus trees with epoch detection
    if state.tree_names and state.reference_shot:
        for tree_name in state.tree_names:
            if state.should_stop_discovering():
                break

            # Check idempotency - skip if epochs already exist
            existing_epochs = await asyncio.to_thread(
                get_tree_epochs, state.facility, tree_name
            )

            if existing_epochs and not state.force:
                # Tree already scanned - skip
                epoch_count = len(existing_epochs)
                total_nodes = sum(e.get("node_count", 0) for e in existing_epochs)
                logger.info(
                    "Skipping %s:%s - already has %d epochs (%d nodes). "
                    "Use --force to re-scan.",
                    state.facility,
                    tree_name,
                    epoch_count,
                    total_nodes,
                )
                if on_progress:
                    on_progress(
                        f"skipped {tree_name} (already scanned)",
                        state.discover_stats,
                        [
                            {
                                "tree_name": tree_name,
                                "node_path": f"{epoch_count} epochs exist",
                            }
                        ],
                    )
                continue

            if on_progress:
                mode = "re-scanning" if existing_epochs else "scanning"
                on_progress(
                    f"{mode} {tree_name}",
                    state.discover_stats,
                    [{"tree_name": tree_name, "node_path": "detecting epochs..."}],
                )

            # Get incremental start shot if updating existing epochs
            start_shot = None
            if existing_epochs and state.force:
                latest = await asyncio.to_thread(
                    get_latest_epoch_shot, state.facility, tree_name
                )
                if latest:
                    start_shot = latest
                    logger.info(
                        "Incremental scan from shot %d for %s:%s",
                        start_shot,
                        state.facility,
                        tree_name,
                    )

            # Create epoch progress callback that bridges to main progress
            # This callback is called from the thread running discover_epochs_optimized
            # Use factory function to capture tree_name at creation time
            def make_epoch_callback(tree: str):
                def epoch_progress_callback(progress: EpochProgress) -> None:
                    """Report epoch detection progress to display."""
                    if not on_progress:
                        return

                    if progress.phase == "coarse":
                        pct = (
                            int(100 * progress.shots_scanned / progress.total_shots)
                            if progress.total_shots > 0
                            else 0
                        )
                        detail = f"coarse {pct}% shot {progress.current_shot}"
                    elif progress.phase == "refine":
                        detail = (
                            f"refine {progress.boundaries_refined}/"
                            f"{progress.boundaries_found}"
                        )
                    else:
                        detail = f"building {progress.boundaries_found} epochs"

                    on_progress(
                        f"epoch detection: {detail}",
                        state.discover_stats,
                        [
                            {
                                "tree_name": tree,
                                "node_path": detail,
                                "epoch_progress": {
                                    "phase": progress.phase,
                                    "current_shot": progress.current_shot,
                                    "shots_scanned": progress.shots_scanned,
                                    "total_shots": progress.total_shots,
                                    "boundaries_found": progress.boundaries_found,
                                    "boundaries_refined": progress.boundaries_refined,
                                },
                            }
                        ],
                    )

                return epoch_progress_callback

            epoch_callback = make_epoch_callback(tree_name)

            # Create checkpoint path for resumable epoch discovery
            checkpoint_path = (
                get_checkpoint_dir() / f"{state.facility}_{tree_name}_epochs.json"
            )

            # Discover epochs using optimized batch discovery
            try:
                with GraphClient() as gc:
                    # Run discover_epochs_optimized with progress callback
                    epochs, structures = await asyncio.to_thread(
                        discover_epochs_optimized,
                        state.facility,
                        tree_name,
                        start_shot=start_shot,
                        end_shot=state.reference_shot,
                        checkpoint_path=checkpoint_path,
                        client=gc if existing_epochs else None,
                        on_progress=epoch_callback,
                    )
            except Exception as e:
                logger.error(
                    "Epoch detection failed for %s:%s: %s", state.facility, tree_name, e
                )
                if on_progress:
                    on_progress(
                        f"epoch detection failed for {tree_name}",
                        state.discover_stats,
                        [{"tree_name": tree_name, "node_path": str(e)[:50]}],
                    )
                continue

            if not epochs:
                logger.info(
                    "No new epochs discovered for %s:%s", state.facility, tree_name
                )
                if on_progress:
                    on_progress(
                        f"no new epochs for {tree_name}",
                        state.discover_stats,
                        [{"tree_name": tree_name, "node_path": "structure unchanged"}],
                    )
                continue

            # Ingest epochs to graph
            epoch_count = await asyncio.to_thread(ingest_epochs, epochs)
            logger.info(
                "Ingested %d epochs for %s:%s", epoch_count, state.facility, tree_name
            )

            # Clean up checkpoint file after successful ingestion
            # (next run will skip via graph-based idempotency check)
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except Exception as e:
                    logger.debug(
                        "Could not remove checkpoint %s: %s", checkpoint_path, e
                    )

            # Create signals from the latest epoch's structure
            # Use the most recent epoch for signal discovery
            latest_epoch = max(epochs, key=lambda e: e["first_shot"])
            latest_shot = latest_epoch["first_shot"]
            latest_paths = structures.get(latest_shot, [])

            if latest_paths:
                if on_progress:
                    on_progress(
                        f"creating signals from {tree_name}",
                        state.discover_stats,
                        [
                            {
                                "tree_name": tree_name,
                                "node_path": f"{len(latest_paths)} paths",
                            }
                        ],
                    )

                signals = create_signals_from_epoch(
                    state.facility,
                    tree_name,
                    latest_epoch["id"],
                    latest_paths,
                    access_method_id,
                    state.reference_shot,
                )

                if signals:
                    count = await asyncio.to_thread(ingest_discovered_signals, signals)
                    state.discover_stats.processed += count

                    if on_progress:
                        results = [
                            {
                                "id": s["id"],
                                "tree_name": tree_name,
                                "node_path": s.get("node_path", ""),
                                "signals_in_tree": count,
                            }
                            for s in signals[:20]
                        ]
                        on_progress(
                            f"discovered {count} from {tree_name} ({len(epochs)} epochs)",
                            state.discover_stats,
                            results,
                        )

            await asyncio.sleep(0.1)

    # Scan TDI functions (if configured)
    if state.tdi_path and not state.should_stop_discovering():
        if on_progress:
            on_progress(
                f"scanning TDI {state.tdi_path}",
                state.discover_stats,
                [{"tree_name": "TDI", "node_path": state.tdi_path}],
            )

        signals = await asyncio.to_thread(
            discover_tdi_signals,
            state.facility,
            state.ssh_host,
            state.tdi_path,
            access_method_id,
        )

        if signals:
            count = await asyncio.to_thread(ingest_discovered_signals, signals)
            state.discover_stats.processed += count

            if on_progress:
                results = [
                    {
                        "id": s["id"],
                        "tree_name": "TDI",
                        "node_path": s.get("accessor", ""),
                        "signals_in_tree": count,
                    }
                    for s in signals[:20]
                ]
                on_progress(
                    f"discovered {count} TDI signals",
                    state.discover_stats,
                    results,
                )

    # Mark scan as complete
    state.discover_idle_count = 100  # Signal scan is fully done

    if on_progress:
        on_progress("scan complete", state.discover_stats)


# Retry configuration for rate limiting
_ENRICH_MAX_RETRIES = 5
_ENRICH_RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry


async def enrich_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that enriches signals with LLM classification.

    Uses batch processing for efficiency - processes 100 signals per LLM call.
    Uses Jinja2 prompt template with schema-injected physics domains.
    Uses centralized LLM access via get_model_for_task() with OpenRouter.
    """
    import os

    import litellm

    from imas_codex.agentic.agents import get_model_for_task
    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.data.models import SignalEnrichmentBatch

    # Suppress LiteLLM verbose output
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Get API key - same pattern as wiki/paths discovery
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it in .env or export it."
        )

    # Get model and ensure OpenRouter prefix
    # Using enrichment task - configured in pyproject.toml
    model = get_model_for_task("enrichment")
    model_id = model if model.startswith("openrouter/") else f"openrouter/{model}"

    # Render system prompt once (contains physics domains from schema)
    system_prompt = render_prompt("discovery/signal-enrichment")

    while not state.should_stop_enriching():
        # Claim batch of signals
        # Batch size 25 balances throughput vs LLM reliability
        # Larger batches can cause truncation with complex MDSplus paths
        signals = await asyncio.to_thread(
            claim_signals_for_enrichment,
            state.facility,
            batch_size=25,
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

        # Build user prompt with all signals
        user_lines = [
            f"Classify these {len(signals)} signals.\n",
            "Return results in the same order using signal_index (1-based).\n",
        ]
        for i, signal in enumerate(signals, 1):
            user_lines.append(f"\n## Signal {i}")
            user_lines.append(f"tree_name: {signal.get('tree_name', 'unknown')}")
            user_lines.append(f"node_path: {signal.get('node_path', 'unknown')}")
            user_lines.append(f"accessor: {signal['accessor']}")
            user_lines.append(f"units: {signal.get('units', '')}")
            user_lines.append(f"name: {signal.get('name', 'unknown')}")

        user_prompt = "\n".join(user_lines)

        # Retry loop for rate limiting / overloaded errors
        last_error = None
        response = None
        for attempt in range(_ENRICH_MAX_RETRIES):
            try:
                # Call LLM with structured output for batch
                # max_tokens=32000 supports up to ~200 signals per batch
                response = await litellm.acompletion(
                    model=model_id,
                    api_key=api_key,
                    max_tokens=32000,
                    response_format=SignalEnrichmentBatch,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                break  # Success
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if any(
                    x in error_msg
                    for x in ["overloaded", "rate", "429", "503", "timeout"]
                ):
                    delay = _ENRICH_RETRY_BASE_DELAY * (2**attempt)
                    logger.debug(
                        f"LLM rate limited (attempt {attempt + 1}/{_ENRICH_MAX_RETRIES}), "
                        f"waiting {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error - release all claims and continue
                    logger.warning("LLM error (non-retryable): %s", e)
                    for signal in signals:
                        await asyncio.to_thread(release_signal_claim, signal["id"])
                    break

        if response is None:
            if last_error:
                logger.warning(
                    "All LLM retries exhausted for batch of %d signals: %s",
                    len(signals),
                    last_error,
                )
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            continue

        # Log token usage for debugging truncation issues
        if hasattr(response, "usage"):
            output_tokens = response.usage.completion_tokens
            logger.debug(
                "LLM response: %d output tokens for %d signals",
                output_tokens,
                len(signals),
            )

        # Parse structured response
        raw_content = response.choices[0].message.content
        try:
            batch_result = SignalEnrichmentBatch.model_validate_json(raw_content)
        except Exception as e:
            # Log truncated content for debugging (first 500 chars)
            content_preview = raw_content[:500] if raw_content else "<empty>"
            logger.warning(
                "Failed to parse LLM response (len=%d, preview=%s...): %s",
                len(raw_content) if raw_content else 0,
                content_preview,
                e,
            )
            for signal in signals:
                await asyncio.to_thread(release_signal_claim, signal["id"])
            continue

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

        # Track cost
        if hasattr(response, "usage"):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = response._hidden_params["response_cost"]
            else:
                # Fallback: Gemini Flash rates via OpenRouter ($0.10/$0.40 per 1M tokens)
                cost = (input_tokens * 0.10 + output_tokens * 0.40) / 1_000_000

            state.enrich_stats.cost += cost

        # Update graph
        if enriched:
            await asyncio.to_thread(mark_signals_enriched, enriched)
            state.enrich_stats.processed += len(enriched)

            if on_progress:
                on_progress("enriched batch", state.enrich_stats, enriched)


async def validate_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that validates signals by testing data access."""
    while not state.should_stop_validating():
        # Claim batch of signals
        signals = await asyncio.to_thread(
            claim_signals_for_validation,
            state.facility,
            batch_size=3,
        )

        if not signals:
            state.validate_idle_count += 1
            if on_progress:
                on_progress("idle", state.validate_stats)
            await asyncio.sleep(1.0)
            continue

        state.validate_idle_count = 0

        if on_progress:
            on_progress("validating batch", state.validate_stats)

        # Validate each signal
        validated = []
        for signal in signals:
            shot = signal.get("example_shot") or state.reference_shot
            if not shot:
                await asyncio.to_thread(release_signal_claim, signal["id"])
                continue

            try:
                # Build validation script
                accessor = signal["accessor"]
                tree_name = signal.get("tree_name", "results")

                validation_script = f'''
import json
import MDSplus

try:
    tree = MDSplus.Tree("{tree_name}", {shot}, "readonly")
    data = tree.tdiExecute("{accessor}").data()
    result = {{"success": True, "shape": list(data.shape) if hasattr(data, "shape") else [len(data)]}}
except Exception as e:
    result = {{"success": False, "error": str(e)[:200]}}

print(json.dumps(result))
'''

                escaped = validation_script.replace("'", "'\"'\"'")
                cmd = ["ssh", state.ssh_host, f"python3 -c '{escaped}'"]

                proc_result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if proc_result.returncode == 0:
                    result = json.loads(proc_result.stdout)
                    if result.get("success"):
                        validated.append({"id": signal["id"]})
                    else:
                        await asyncio.to_thread(
                            mark_signal_failed,
                            signal["id"],
                            result.get("error", "validation failed"),
                            FacilitySignalStatus.enriched.value,
                        )
                else:
                    await asyncio.to_thread(
                        mark_signal_failed,
                        signal["id"],
                        proc_result.stderr[:200]
                        if proc_result.stderr
                        else "SSH failed",
                        FacilitySignalStatus.enriched.value,
                    )

            except Exception as e:
                logger.warning("Failed to validate signal %s: %s", signal["id"], e)
                await asyncio.to_thread(release_signal_claim, signal["id"])

        # Update graph
        if validated:
            await asyncio.to_thread(mark_signals_validated, validated)
            state.validate_stats.processed += len(validated)

            if on_progress:
                results = [
                    {"id": v["id"], "shot": shot, "success": True} for v in validated
                ]
                on_progress("validated batch", state.validate_stats, results)


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_parallel_data_discovery(
    facility: str,
    ssh_host: str | None = None,
    tree_names: list[str] | None = None,
    tdi_path: str | None = None,
    reference_shot: int | None = None,
    cost_limit: float = 10.0,
    signal_limit: int | None = None,
    focus: str | None = None,
    num_enrich_workers: int = 2,
    num_validate_workers: int = 1,
    discover_only: bool = False,
    enrich_only: bool = False,
    force: bool = False,
    on_discover_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_validate_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
) -> dict[str, Any]:
    """Run parallel data discovery with async workers.

    Args:
        facility: Facility ID (e.g., "tcv")
        ssh_host: SSH host for remote discovery
        tree_names: MDSplus tree names to discover
        tdi_path: Path to TDI function directory
        reference_shot: Reference shot for discovery/validation
        cost_limit: Maximum LLM cost in USD
        signal_limit: Maximum signals to process
        focus: Focus area for discovery
        num_enrich_workers: Number of enrich workers
        num_validate_workers: Number of validate workers
        discover_only: Only discover, don't enrich
        enrich_only: Only enrich discovered signals
        force: Re-scan trees even if epochs exist (merges, doesn't reset)
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
        tree_names=tree_names or [],
        reference_shot=reference_shot,
        tdi_path=tdi_path,
        force=force,
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
            "validated": 0,
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

    # Start validate workers (unless enrich_only)
    if not enrich_only:
        for i in range(num_validate_workers):
            worker_name = f"validate_worker_{i}"
            status = worker_group.create_status(worker_name)
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        validate_worker,
                        worker_name,
                        state,
                        state.should_stop_validating,
                        on_progress=on_validate_progress,
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
        "validated": state.validate_stats.processed,
        "cost": state.enrich_stats.cost,
        "elapsed_seconds": elapsed,
        "discover_rate": state.discover_stats.processed / elapsed if elapsed > 0 else 0,
        "enrich_rate": state.enrich_stats.processed / elapsed if elapsed > 0 else 0,
        "validate_rate": state.validate_stats.processed / elapsed if elapsed > 0 else 0,
    }
