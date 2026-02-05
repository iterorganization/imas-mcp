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
1. DISCOVER: Enumerate signals from data sources (MDSplus trees, TDI functions)
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
    """Mark signals as enriched with LLM-generated metadata."""
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


async def discover_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that discovers signals from data sources."""
    while not state.should_stop_discovering():
        # For now, discovery is done in bulk at startup
        # This worker handles incremental/continuous discovery
        state.discover_idle_count += 1

        if on_progress:
            on_progress("idle", state.discover_stats)

        await asyncio.sleep(2.0)


async def enrich_worker(
    state: DataDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Worker that enriches signals with LLM classification.

    Uses centralized LLM access via get_model_for_task() with OpenRouter.
    """
    import os

    import litellm

    from imas_codex.agentic.agents import get_model_for_task

    # Get API key - same pattern as wiki/paths discovery
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it in .env or export it."
        )

    # Get model and ensure OpenRouter prefix
    model = get_model_for_task(
        "enrichment"
    )  # Use enrichment model for physics classification
    model_id = model if model.startswith("openrouter/") else f"openrouter/{model}"

    while not state.should_stop_enriching():
        # Claim batch of signals
        signals = await asyncio.to_thread(
            claim_signals_for_enrichment,
            state.facility,
            batch_size=5,
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

        # Enrich each signal with LLM
        enriched = []
        for signal in signals:
            try:
                # Build prompt for physics domain classification
                prompt = f"""Classify this fusion data signal into a physics domain and provide a brief description.

Signal: {signal["accessor"]}
Name: {signal.get("name", "unknown")}
Units: {signal.get("units", "unknown")}
Tree: {signal.get("tree_name", "unknown")}
Node Path: {signal.get("node_path", "unknown")}

Respond in JSON format:
{{
    "physics_domain": "one of: equilibrium, transport, magnetohydrodynamics, turbulence, auxiliary_heating, current_drive, plasma_wall_interactions, divertor_physics, edge_plasma_physics, particle_measurement_diagnostics, electromagnetic_wave_diagnostics, radiation_measurement_diagnostics, magnetic_field_diagnostics, mechanical_measurement_diagnostics, plasma_control, machine_operations, magnetic_field_systems, structural_components, plant_systems, data_management, computational_workflow, general",
    "description": "brief physics description of what this signal measures",
    "name": "human-readable name for the signal"
}}"""

                response = await litellm.acompletion(
                    model=model_id,
                    api_key=api_key,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                enriched.append(
                    {
                        "id": signal["id"],
                        "physics_domain": result.get("physics_domain", "general"),
                        "description": result.get("description", ""),
                        "name": result.get("name", signal.get("name", "")),
                    }
                )

                # Track cost - try actual OpenRouter cost first, then fallback
                if hasattr(response, "usage"):
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

                    if (
                        hasattr(response, "_hidden_params")
                        and "response_cost" in response._hidden_params
                    ):
                        cost = response._hidden_params["response_cost"]
                    else:
                        # Fallback: Gemini Pro rates via OpenRouter ($1.25/$5 per 1M tokens)
                        cost = (input_tokens * 1.25 + output_tokens * 5) / 1_000_000

                    state.enrich_stats.cost += cost

            except Exception as e:
                logger.warning("Failed to enrich signal %s: %s", signal["id"], e)
                await asyncio.to_thread(release_signal_claim, signal["id"])

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
        cost_limit=cost_limit,
        signal_limit=signal_limit,
        focus=focus,
    )

    # Create worker group
    worker_group = SupervisedWorkerGroup()

    # Bulk discovery phase (if not enrich_only)
    bulk_discovered = 0
    if not enrich_only:
        # Get or create access method for MDSplus trees
        access_method_id = f"{facility}:mdsplus:tree"

        # Discover from MDSplus trees
        if tree_names and reference_shot:
            for tree_name in tree_names:
                logger.info("Discovering signals from %s:%s...", facility, tree_name)
                signals = await asyncio.to_thread(
                    discover_mdsplus_signals,
                    facility,
                    ssh_host,
                    tree_name,
                    reference_shot,
                    access_method_id,
                )
                if signals:
                    count = await asyncio.to_thread(ingest_discovered_signals, signals)
                    bulk_discovered += count
                    state.discover_stats.processed += count

                    if on_discover_progress:
                        on_discover_progress(
                            f"discovered {count} from {tree_name}",
                            state.discover_stats,
                            signals[:10],
                        )

        # Discover from TDI functions
        if tdi_path:
            logger.info("Discovering TDI signals from %s...", tdi_path)
            signals = await asyncio.to_thread(
                discover_tdi_signals,
                facility,
                ssh_host,
                tdi_path,
                access_method_id,
            )
            if signals:
                count = await asyncio.to_thread(ingest_discovered_signals, signals)
                bulk_discovered += count
                state.discover_stats.processed += count

                if on_discover_progress:
                    on_discover_progress(
                        f"discovered {count} TDI signals",
                        state.discover_stats,
                        signals[:10],
                    )

    if discover_only:
        return {
            "discovered": bulk_discovered,
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
