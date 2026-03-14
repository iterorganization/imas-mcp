"""Batch claim worker loop for the IMAS mapping pipeline.

Transforms the mapping pipeline from a single-shot LLM call into a
physics-domain-scoped, batch-claim worker loop with source-level
mapping and programmatic unit propagation.

Architecture follows the established discovery worker pattern from
``discovery/base/engine.py``:
- Independent async workers claim batches from the graph
- Graph + claimed_at timestamp for coordination
- Orphan recovery via timeout check in claim queries
- SupervisedWorkerGroup with automatic restart on crash

Workers:
- context_worker: Gathers IDS structure and section clusters (fast, one-shot)
- assign_worker: LLM assigns signal sources to IMAS sections
- map_worker: Per-source LLM mapping with semantic candidate context
- validate_worker: Programmatic validation of mappings
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.cli.logging import WorkerLogAdapter
from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase
from imas_codex.graph.client import GraphClient
from imas_codex.ids.mapping import PipelineCost

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


# =============================================================================
# Discovery State
# =============================================================================


@dataclass
class MappingDiscoveryState(DiscoveryStateBase):
    """Shared state for the mapping discovery pipeline."""

    target_ids: str = ""
    dd_version: str | None = None
    dd_major: int | None = None
    model: str | None = None

    # Context gathered in context phase
    context: dict[str, Any] = field(default_factory=dict)

    # Pipeline cost tracking
    cost: PipelineCost = field(default_factory=PipelineCost)

    # Results
    sources_found: int = 0
    sections_assigned: int = 0
    sections_mapped: int = 0
    bindings_total: int = 0
    bindings_passed: int = 0
    escalations: int = 0
    mapping_id: str | None = None

    # Worker stats
    context_stats: WorkerStats = field(default_factory=WorkerStats)
    assign_stats: WorkerStats = field(default_factory=WorkerStats)
    map_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    context_phase: PipelinePhase = field(init=False)
    assign_phase: PipelinePhase = field(init=False)
    map_phase: PipelinePhase = field(init=False)
    validate_phase: PipelinePhase = field(init=False)

    # Control
    persist: bool = True
    activate: bool = True

    def __post_init__(self) -> None:
        self.context_phase = PipelinePhase("context")
        self.assign_phase = PipelinePhase("assign")
        self.map_phase = PipelinePhase("map")
        self.validate_phase = PipelinePhase("validate")

    @property
    def total_cost(self) -> float:
        return self.cost.total_usd

    def should_stop(self) -> bool:
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        return False


# =============================================================================
# Claim Operations
# =============================================================================


@retry_on_deadlock()
def claim_sources_for_mapping(
    facility: str,
    ids_name: str,
    domains: list[str] | None = None,
    batch_size: int = 5,
) -> list[dict[str, Any]]:
    """Claim unmapped signal sources for the mapping pipeline.

    Uses the standard claim_token + ORDER BY rand() + @retry_on_deadlock
    pattern to prevent deadlocks.

    Args:
        facility: Facility identifier.
        ids_name: Target IDS name.
        domains: Physics domains to filter by.
        batch_size: Number of sources to claim per batch.

    Returns:
        List of claimed signal source dicts.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())

    domain_filter = ""
    params: dict[str, Any] = {
        "facility": facility,
        "batch_size": batch_size,
        "cutoff": cutoff,
        "token": claim_token,
    }

    if domains:
        domain_filter = "AND sg.physics_domain IN $domains"
        params["domains"] = domains

    with GraphClient() as gc:
        # Step 1: Claim with random ordering
        gc.query(
            f"""
            MATCH (sg:SignalSource {{facility_id: $facility}})
            WHERE sg.status = 'enriched'
              AND NOT EXISTS {{ (sg)-[:MAPS_TO_IMAS]->(:IMASNode) }}
              {domain_filter}
              AND (sg.claimed_at IS NULL
                   OR sg.claimed_at < datetime() - duration($cutoff))
            WITH sg ORDER BY rand() LIMIT $batch_size
            SET sg.claimed_at = datetime(), sg.claim_token = $token
            """,
            **params,
        )

        # Step 2: Read back claimed sources
        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility, claim_token: $token})
            OPTIONAL MATCH (m:FacilitySignal)-[:MEMBER_OF]->(sg)
            WITH sg, count(m) AS member_count,
                 collect(DISTINCT m.accessor)[..10] AS sample_accessors
            OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
            RETURN sg.id AS id, sg.group_key AS group_key,
                   sg.description AS description,
                   sg.keywords AS keywords,
                   sg.physics_domain AS physics_domain,
                   member_count,
                   sample_accessors,
                   rep.description AS rep_description,
                   rep.unit AS rep_unit,
                   rep.sign_convention AS rep_sign_convention,
                   rep.cocos AS rep_cocos
            """,
            facility=facility,
            token=claim_token,
        )

        return list(result)


def release_source_claim(source_id: str) -> None:
    """Release a claim on a signal source."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sg:SignalSource {id: $id})
            SET sg.claimed_at = null, sg.claim_token = null
            """,
            id=source_id,
        )


# =============================================================================
# Workers
# =============================================================================


async def context_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that gathers all context for the mapping pipeline.

    One-shot worker: gathers context and marks phase done.
    """
    wlog = WorkerLogAdapter(logger, worker_name="context_worker")

    if on_progress:
        on_progress("gathering context", state.context_stats)

    try:
        from imas_codex.ids.mapping import gather_context

        context = await asyncio.to_thread(
            gather_context,
            state.facility,
            state.target_ids,
            gc=GraphClient(),
            dd_version=state.dd_major,
        )
        state.context = context
        state.sources_found = len(context.get("groups", []))
        state.context_stats.processed = 1

        if on_progress:
            on_progress(
                f"{state.sources_found} sources found",
                state.context_stats,
            )
    except Exception as e:
        wlog.error("Context gathering failed: %s", e)
        state.context_stats.errors += 1
        raise

    state.context_phase.mark_done()


async def assign_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that assigns signal sources to IMAS sections via LLM."""
    wlog = WorkerLogAdapter(logger, worker_name="assign_worker")

    # Wait for context phase
    while not state.stop_requested:
        if state.context_phase.done and state.context:
            break
        await asyncio.sleep(0.5)

    if state.stop_requested or not state.context:
        return

    if on_progress:
        on_progress("assigning sections", state.assign_stats)

    try:
        from imas_codex.ids.mapping import assign_sections

        sections = await asyncio.to_thread(
            assign_sections,
            state.facility,
            state.target_ids,
            state.context,
            model=state.model,
            cost=state.cost,
        )
        state.context["sections"] = sections
        state.sections_assigned = len(sections.assignments)
        state.assign_stats.processed = state.sections_assigned

        if on_progress:
            on_progress(
                f"{state.sections_assigned} sections assigned",
                state.assign_stats,
            )
    except Exception as e:
        wlog.error("Section assignment failed: %s", e)
        state.assign_stats.errors += 1
        raise

    state.assign_phase.mark_done()


async def map_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that generates signal mappings per section via LLM."""
    wlog = WorkerLogAdapter(logger, worker_name="map_worker")

    # Wait for assign phase
    while not state.stop_requested:
        if state.assign_phase.done and state.context.get("sections"):
            break
        await asyncio.sleep(0.5)

    if state.stop_requested:
        return

    sections = state.context.get("sections")
    if not sections:
        state.map_phase.mark_done()
        return

    if on_progress:
        on_progress("generating mappings", state.map_stats)

    try:
        from imas_codex.ids.mapping import map_signals

        gc = GraphClient()
        batches = await asyncio.to_thread(
            map_signals,
            state.facility,
            state.target_ids,
            sections,
            state.context,
            gc=gc,
            model=state.model,
            cost=state.cost,
        )
        state.context["field_batches"] = batches
        state.sections_mapped = len(batches)

        # Count total bindings
        total = sum(len(b.mappings) for b in batches)
        state.bindings_total = total
        state.map_stats.processed = state.sections_mapped

        if on_progress:
            on_progress(
                f"{state.sections_mapped} sections, {total} bindings",
                state.map_stats,
            )
    except Exception as e:
        wlog.error("Signal mapping failed: %s", e)
        state.map_stats.errors += 1
        raise

    state.map_phase.mark_done()


async def validate_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Worker that validates and persists mappings."""
    wlog = WorkerLogAdapter(logger, worker_name="validate_worker")

    # Wait for map phase
    while not state.stop_requested:
        if state.map_phase.done and state.context.get("field_batches"):
            break
        await asyncio.sleep(0.5)

    if state.stop_requested:
        return

    sections = state.context.get("sections")
    field_batches = state.context.get("field_batches")
    if not sections or not field_batches:
        state.validate_phase.mark_done()
        return

    if on_progress:
        on_progress("validating mappings", state.validate_stats)

    try:
        from imas_codex.ids.mapping import (
            discover_assembly,
            validate_mappings,
        )
        from imas_codex.ids.models import persist_mapping_result

        gc = GraphClient()

        # Assembly discovery
        assembly = await asyncio.to_thread(
            discover_assembly,
            state.facility,
            state.target_ids,
            sections,
            field_batches,
            state.context,
            gc=gc,
            model=state.model,
            cost=state.cost,
        )

        # Validation
        dd_version = state.dd_version or ""
        validated = await asyncio.to_thread(
            validate_mappings,
            state.facility,
            state.target_ids,
            dd_version,
            sections,
            field_batches,
            gc=gc,
        )

        state.bindings_passed = sum(
            1 for c in validated.bindings
        )
        state.escalations = len(validated.escalations)
        state.validate_stats.processed = 1

        if on_progress:
            on_progress(
                f"{state.bindings_passed} bindings, {state.escalations} escalations",
                state.validate_stats,
            )

        # Persist
        if state.persist:
            status = "active" if state.activate else "generated"
            mapping_id = await asyncio.to_thread(
                persist_mapping_result,
                validated,
                assembly=assembly,
                gc=gc,
                status=status,
            )
            state.mapping_id = mapping_id
            wlog.info("Persisted mapping %s with status '%s'", mapping_id, status)

        state.context["validated"] = validated
        state.context["assembly"] = assembly

    except Exception as e:
        wlog.error("Validation/persistence failed: %s", e)
        state.validate_stats.errors += 1
        raise

    state.validate_phase.mark_done()


# =============================================================================
# Engine Entry Point
# =============================================================================


async def run_mapping_engine(
    state: MappingDiscoveryState,
    *,
    stop_event: asyncio.Event | None = None,
    on_progress: Callable | None = None,
) -> None:
    """Run the mapping pipeline as a discovery engine.

    Uses the standard WorkerSpec/run_discovery_engine pattern from
    ``discovery/base/engine.py``.
    """
    workers = [
        WorkerSpec(
            "context", "context_phase", context_worker,
            on_progress=on_progress,
        ),
        WorkerSpec(
            "assign", "assign_phase", assign_worker,
            on_progress=on_progress,
        ),
        WorkerSpec(
            "map", "map_phase", map_worker,
            on_progress=on_progress,
        ),
        WorkerSpec(
            "validate", "validate_phase", validate_worker,
            on_progress=on_progress,
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
    )
