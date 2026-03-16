"""Graph state machine workers for the IMAS mapping pipeline.

Source-centric pipeline: each SignalSource group is the work item,
processed through four phases tracked via ``mapping_status`` on the
graph node:

  (enriched, no mapping_status) → assigned → mapped → validated

Workers claim batches from the graph using ``mapping_claimed_at`` +
``mapping_claim_token`` for coordination with orphan recovery via timeout.

Architecture follows ``discovery/base/engine.py``:
- Independent async workers claim batches from the graph
- ``@retry_on_deadlock()`` + ``ORDER BY rand()`` + claim_token
- ``PipelinePhase.set_has_work_fn`` for phase completion detection
- ``OrphanRecoverySpec`` for automatic stale claim release
- ``run_discovery_engine`` with supervised worker group

Workers:
- context_worker: Gathers IDS structure + semantic context (one-shot per IDS)
- assign_worker: LLM assigns sources to IMAS sections (per-IDS batch)
- map_worker: LLM maps per-source fields (claim loop)
- validate_worker: Validates + persists per-IDS (once all mapped)
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
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
)
from imas_codex.graph.client import GraphClient
from imas_codex.ids.mapping import PipelineCost

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


# =============================================================================
# Discovery State — holds ALL IDS targets, not just one
# =============================================================================


@dataclass
class MappingDiscoveryState(DiscoveryStateBase):
    """Shared state for the mapping pipeline across all IDS targets."""

    # All IDS targets to map (resolved in pre-flight)
    target_ids_list: list[str] = field(default_factory=list)
    # Target info from discover_mappable_ids: [{ids_name, domains, source_count}]
    target_info: list[dict] = field(default_factory=list)

    dd_version: str | None = None
    dd_major: int | None = None
    model: str | None = None

    # Context cache: ids_name -> context dict (gathered by context_worker)
    contexts: dict[str, dict] = field(default_factory=dict)
    # Assignment results cache: ids_name -> TargetAssignmentBatch
    assignments: dict[str, Any] = field(default_factory=dict)
    # Mapping batches cache: ids_name -> list[(assignment, batch)]
    mapping_batches: dict[str, list] = field(default_factory=dict)

    # Pipeline cost tracking (cumulative across all IDS)
    cost: PipelineCost = field(default_factory=PipelineCost)

    # Aggregate counters (across all IDS)
    sources_total: int = 0
    sources_assigned: int = 0
    sources_mapped: int = 0
    sources_validated: int = 0
    bindings_total: int = 0
    bindings_passed: int = 0
    escalations: int = 0

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
    clear: bool = False

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
# Graph Claim Operations
# =============================================================================


@retry_on_deadlock()
def claim_sources_for_assignment(
    facility: str,
    domains: list[str] | None = None,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Claim enriched sources that have no mapping_status yet."""
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    token = str(_uuid.uuid4())
    params: dict[str, Any] = {
        "facility": facility,
        "batch_size": batch_size,
        "cutoff": cutoff,
        "token": token,
    }

    domain_filter = ""
    if domains:
        domain_filter = "AND sg.physics_domain IN $domains"
        params["domains"] = domains

    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (sg:SignalSource {{facility_id: $facility}})
            WHERE sg.status = 'enriched'
              AND sg.mapping_status IS NULL
              AND NOT EXISTS {{ (sg)-[:MAPS_TO_IMAS]->(:IMASNode) }}
              {domain_filter}
              AND (sg.mapping_claimed_at IS NULL
                   OR sg.mapping_claimed_at < datetime() - duration($cutoff))
            WITH sg ORDER BY rand() LIMIT $batch_size
            SET sg.mapping_claimed_at = datetime(),
                sg.mapping_claim_token = $token
            """,
            **params,
        )

        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility,
                                     mapping_claim_token: $token})
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
            token=token,
        )
        return list(result)


@retry_on_deadlock()
def claim_sources_for_mapping(
    facility: str,
    ids_name: str,
    batch_size: int = 3,
) -> list[dict[str, Any]]:
    """Claim sources where mapping_status = 'assigned' for a given IDS."""
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    token = str(_uuid.uuid4())

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
            WHERE sg.mapping_status = 'assigned'
              AND sg.mapping_target_ids = $ids_name
              AND (sg.mapping_claimed_at IS NULL
                   OR sg.mapping_claimed_at < datetime() - duration($cutoff))
            WITH sg ORDER BY rand() LIMIT $batch_size
            SET sg.mapping_claimed_at = datetime(),
                sg.mapping_claim_token = $token
            """,
            facility=facility,
            ids_name=ids_name,
            batch_size=batch_size,
            cutoff=cutoff,
            token=token,
        )

        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility,
                                     mapping_claim_token: $token})
            OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
            RETURN sg.id AS id, sg.group_key AS group_key,
                   sg.description AS description,
                   sg.keywords AS keywords,
                   sg.physics_domain AS physics_domain,
                   sg.mapping_target_ids AS target_ids,
                   sg.mapping_target_path AS target_path,
                   sg.mapping_target_type AS target_type,
                   rep.description AS rep_description,
                   rep.unit AS rep_unit,
                   rep.sign_convention AS rep_sign_convention,
                   rep.cocos AS rep_cocos
            """,
            facility=facility,
            token=token,
        )
        return list(result)


def set_mapping_status(source_id: str, status: str, **props: Any) -> None:
    """Set mapping_status on a source and clear the claim."""
    set_clauses = [
        "sg.mapping_status = $status",
        "sg.mapping_claimed_at = null",
        "sg.mapping_claim_token = null",
    ]
    params: dict[str, Any] = {"id": source_id, "status": status}

    for key, val in props.items():
        param_name = f"p_{key}"
        set_clauses.append(f"sg.mapping_{key} = ${param_name}")
        params[param_name] = val

    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (sg:SignalSource {{id: $id}})
            SET {', '.join(set_clauses)}
            """,
            **params,
        )


def release_mapping_claim(source_id: str) -> None:
    """Release a mapping claim without changing status."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sg:SignalSource {id: $id})
            SET sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null
            """,
            id=source_id,
        )


def release_mapping_claims_batch(source_ids: list[str]) -> None:
    """Release mapping claims on multiple sources."""
    if not source_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sg:SignalSource {id: sid})
            SET sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null
            """,
            ids=source_ids,
        )


def has_pending_assignment_work(
    facility: str,
    domains: list[str] | None = None,
) -> bool:
    """Check if enriched sources without mapping_status exist."""
    params: dict[str, Any] = {"facility": facility}
    domain_filter = ""
    if domains:
        domain_filter = "AND sg.physics_domain IN $domains"
        params["domains"] = domains

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sg:SignalSource {{facility_id: $facility}})
            WHERE sg.status = 'enriched'
              AND sg.mapping_status IS NULL
              AND NOT EXISTS {{ (sg)-[:MAPS_TO_IMAS]->(:IMASNode) }}
              {domain_filter}
            RETURN count(sg) > 0 AS has_work
            """,
            **params,
        )
        return result[0]["has_work"] if result else False


def has_pending_mapping_work(facility: str) -> bool:
    """Check if assigned-but-unmapped sources exist."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
            WHERE sg.mapping_status = 'assigned'
            RETURN count(sg) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_validation_work(facility: str) -> bool:
    """Check if mapped-but-unvalidated sources exist."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
            WHERE sg.mapping_status = 'mapped'
            RETURN count(sg) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def count_sources_by_mapping_status(facility: str) -> dict[str, int]:
    """Count signal sources grouped by mapping_status."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
            WHERE sg.status = 'enriched'
            RETURN coalesce(sg.mapping_status, 'pending') AS status,
                   count(sg) AS cnt
            """,
            facility=facility,
        )
        return {r["status"]: r["cnt"] for r in result}


def reset_mapping_state(
    facility: str, ids_names: list[str] | None = None,
) -> int:
    """Clear mapping_status on sources for fresh re-mapping."""
    params: dict[str, Any] = {"facility": facility}
    ids_filter = ""
    if ids_names:
        ids_filter = "AND sg.mapping_target_ids IN $ids_names"
        params["ids_names"] = ids_names

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sg:SignalSource {{facility_id: $facility}})
            WHERE sg.mapping_status IS NOT NULL
              {ids_filter}
            SET sg.mapping_status = null,
                sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null,
                sg.mapping_target_ids = null,
                sg.mapping_target_path = null,
                sg.mapping_target_type = null
            RETURN count(sg) AS cleared
            """,
            **params,
        )
        return result[0]["cleared"] if result else 0


# =============================================================================
# Workers
# =============================================================================


async def context_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Gather context for ALL IDS targets. One-shot per IDS, cached."""
    wlog = WorkerLogAdapter(logger, worker_name="context_worker")
    wlog.info(
        "Gathering context for %d IDS targets: %s",
        len(state.target_ids_list),
        state.target_ids_list,
    )

    from imas_codex.ids.mapping import gather_context

    total_sources = 0
    for ids_name in state.target_ids_list:
        if state.should_stop():
            break

        if on_progress:
            on_progress(f"gathering {ids_name}", state.context_stats, [
                {"detail": f"querying context for {ids_name}"}
            ])

        def _context_progress(detail: str) -> None:
            if on_progress:
                on_progress(
                    f"{ids_name}: {detail}",
                    state.context_stats,
                    [{"detail": f"{ids_name}: {detail}"}],
                )

        context = await asyncio.to_thread(
            gather_context,
            state.facility,
            ids_name,
            gc=GraphClient(),
            dd_version=state.dd_major,
            on_progress=_context_progress,
        )
        state.contexts[ids_name] = context
        ids_sources = len(context.get("groups", []))
        total_sources += ids_sources

        wlog.info(
            "Context for %s: %d sources, %d domains",
            ids_name,
            ids_sources,
            len(context.get("target_domains", [])),
        )

        if on_progress:
            on_progress(
                f"{ids_name}: {ids_sources} sources",
                state.context_stats,
                [{"detail": f"{ids_name}: {ids_sources} sources"}],
            )

    state.sources_total = total_sources
    state.context_stats.processed = len(state.target_ids_list)
    wlog.info(
        "Context complete: %d IDS, %d total sources",
        len(state.contexts), total_sources,
    )
    state.context_phase.mark_done()


async def assign_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Assign sources to IMAS target paths via LLM, per-IDS batch.

    For each IDS target, calls aassign_targets with all matching
    sources, then sets mapping_status='assigned' on each source.
    """
    wlog = WorkerLogAdapter(logger, worker_name="assign_worker")

    from imas_codex.ids.mapping import aassign_targets

    for ids_name in state.target_ids_list:
        if state.should_stop():
            break

        context = state.contexts.get(ids_name)
        if not context:
            wlog.warning("No context for %s, skipping", ids_name)
            continue

        groups = context.get("groups", [])
        if not groups:
            wlog.info("No sources for %s, skipping", ids_name)
            continue

        if on_progress:
            on_progress(f"assigning {ids_name}", state.assign_stats)

        wlog.info("Assigning %d sources for %s", len(groups), ids_name)

        try:
            sections = await aassign_targets(
                state.facility,
                ids_name,
                context,
                model=state.model,
                cost=state.cost,
            )
            state.assignments[ids_name] = sections
            assigned_count = len(sections.assignments)
            state.sources_assigned += assigned_count
            state.assign_stats.processed += assigned_count

            # Update graph: set mapping_status='assigned' on each source
            for a in sections.assignments:
                await asyncio.to_thread(
                    set_mapping_status,
                    a.source_id,
                    "assigned",
                    target_ids=ids_name,
                    target_path=a.imas_target_path,
                    target_type=a.target_type.value,
                )

            wlog.info(
                "Assigned %d sources for %s, cost $%.4f",
                assigned_count, ids_name, state.cost.total_usd,
            )

            if on_progress:
                stream_items = [
                    {
                        "source_id": a.source_id,
                        "target_path": a.imas_target_path,
                        "physics_domain": next(
                            (g.get("physics_domain", "")
                             for g in groups if g["id"] == a.source_id),
                            "",
                        ),
                    }
                    for a in sections.assignments
                ]
                on_progress(
                    f"{assigned_count} assigned for {ids_name}",
                    state.assign_stats,
                    stream_items,
                )

        except Exception as e:
            wlog.error("Assignment failed for %s: %s", ids_name, e)
            state.assign_stats.errors += 1
            raise

    state.assign_phase.mark_done()


async def map_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Claim assigned sources and generate field-level mappings.

    Claim loop: claims batches of assigned sources from the graph,
    generates mappings per source, and sets mapping_status='mapped'.
    """
    wlog = WorkerLogAdapter(logger, worker_name="map_worker")

    from imas_codex.ids.mapping import _acall_llm, _build_messages, _prepare_section_context
    from imas_codex.ids.models import SignalMappingBatch

    while not state.should_stop():
        found_any = False
        for ids_name in state.target_ids_list:
            if state.should_stop():
                break

            sources = await asyncio.to_thread(
                claim_sources_for_mapping,
                state.facility,
                ids_name,
                batch_size=3,
            )
            if not sources:
                continue

            found_any = True
            state.map_phase.record_activity(len(sources))

            context = state.contexts.get(ids_name, {})
            sections = state.assignments.get(ids_name)

            for source in sources:
                if state.should_stop():
                    release_mapping_claims_batch([s["id"] for s in sources])
                    return

                source_id = source["id"]
                target_path = source.get("target_path", "")

                # Find the assignment object
                assignment = None
                if sections:
                    assignment = next(
                        (a for a in sections.assignments
                         if a.source_id == source_id),
                        None,
                    )

                if not assignment:
                    wlog.warning(
                        "No assignment found for %s, releasing", source_id,
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                try:
                    prep = await asyncio.to_thread(
                        _prepare_section_context,
                        state.facility,
                        ids_name,
                        assignment,
                        context,
                        gc=GraphClient(),
                        dd_version=context.get("dd_version"),
                    )
                    messages = _build_messages(
                        "signal_mapping_system", prep["prompt"],
                    )
                    batch = await _acall_llm(
                        messages,
                        SignalMappingBatch,
                        model=state.model,
                        step_name=f"map_signals_{target_path}",
                        cost=state.cost,
                    )

                    state.mapping_batches.setdefault(ids_name, []).append(
                        (assignment, batch),
                    )
                    state.sources_mapped += 1
                    state.bindings_total += len(batch.mappings)
                    state.map_stats.processed += 1

                    await asyncio.to_thread(
                        set_mapping_status, source_id, "mapped",
                    )

                    wlog.info(
                        "Mapped %s -> %s: %d bindings",
                        source_id, target_path, len(batch.mappings),
                    )

                    if on_progress:
                        sg = next(
                            (g for g in context.get("groups", [])
                             if g["id"] == source_id),
                            {},
                        )
                        on_progress(
                            f"{source_id} -> {target_path}",
                            state.map_stats,
                            [{
                                "source_id": source_id,
                                "target_path": target_path,
                                "physics_domain": sg.get("physics_domain", ""),
                                "bindings": len(batch.mappings),
                            }],
                        )

                except Exception as e:
                    wlog.error("Mapping failed for %s: %s", source_id, e)
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    state.map_stats.errors += 1
                    raise

        if not found_any:
            state.map_phase.record_idle()
            if state.map_phase.done:
                break
            await asyncio.sleep(2.0)


async def validate_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Validate mappings and persist per-IDS once all sources are mapped."""
    wlog = WorkerLogAdapter(logger, worker_name="validate_worker")

    from imas_codex.ids.mapping import (
        AssemblyBatch,
        adiscover_assembly,
        validate_mappings,
    )
    from imas_codex.ids.models import persist_mapping_result

    for ids_name in state.target_ids_list:
        if state.should_stop():
            break

        batches_for_ids = state.mapping_batches.get(ids_name, [])
        sections = state.assignments.get(ids_name)
        context = state.contexts.get(ids_name, {})

        if not batches_for_ids or not sections:
            wlog.info("No mappings for %s, skipping validation", ids_name)
            continue

        if on_progress:
            on_progress(f"validating {ids_name}", state.validate_stats)

        gc = GraphClient()
        field_batches = [b for _, b in batches_for_ids]

        try:
            # Assembly discovery
            configs = []
            async for assignment, config in adiscover_assembly(
                state.facility,
                ids_name,
                sections,
                field_batches,
                context,
                gc=gc,
                model=state.model,
                cost=state.cost,
            ):
                configs.append(config)
                if on_progress:
                    on_progress(
                        f"assembly {len(configs)}/{len(sections.assignments)}",
                        state.validate_stats,
                        [{
                            "target_path": config.target_path,
                            "pattern": (
                                config.pattern.value
                                if hasattr(config.pattern, "value")
                                else str(config.pattern)
                            ),
                        }],
                    )

            assembly = AssemblyBatch(ids_name=ids_name, configs=configs)

            # Validation
            dd_version_str = state.dd_version or ""
            validated = await asyncio.to_thread(
                validate_mappings,
                state.facility,
                ids_name,
                dd_version_str,
                sections,
                field_batches,
                gc=gc,
            )

            ids_passed = len(validated.bindings)
            ids_escalations = len(validated.escalations)
            state.bindings_passed += ids_passed
            state.escalations += ids_escalations
            state.sources_validated += len(batches_for_ids)
            state.validate_stats.processed += 1

            if on_progress:
                on_progress(
                    f"{ids_name}: {ids_passed} passed, "
                    f"{ids_escalations} escalations",
                    state.validate_stats,
                    [{
                        "target_path": ids_name,
                        "passed": ids_passed,
                        "escalations": ids_escalations,
                    }],
                )

            # Persist
            mapping_id = None
            if state.persist:
                status = "active" if state.activate else "generated"
                mapping_id = await asyncio.to_thread(
                    persist_mapping_result,
                    validated,
                    assembly=assembly,
                    gc=gc,
                    status=status,
                )
                wlog.info(
                    "Persisted %s mapping %s (%s)",
                    ids_name, mapping_id, status,
                )

            # Mark sources as validated in graph
            for a, _ in batches_for_ids:
                await asyncio.to_thread(
                    set_mapping_status, a.source_id, "validated",
                )

            wlog.info(
                "Validated %s: %d passed, %d escalations, cost $%.4f",
                ids_name, ids_passed, ids_escalations, state.cost.total_usd,
            )

        except Exception as e:
            wlog.error("Validation failed for %s: %s", ids_name, e)
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

    Wires up graph-based phase completion checks, orphan recovery,
    and supervised workers via ``run_discovery_engine``.
    """
    all_domains = sorted({
        d for info in state.target_info
        for d in info.get("domains", [])
    })

    # Wire has_work_fn for phase completion detection
    state.assign_phase.set_has_work_fn(
        lambda: (
            has_pending_assignment_work(state.facility, all_domains)
            or not state.context_phase.done
        )
    )
    state.map_phase.set_has_work_fn(
        lambda: (
            has_pending_mapping_work(state.facility)
            or not state.assign_phase.done
        )
    )
    state.validate_phase.set_has_work_fn(
        lambda: (
            has_pending_validation_work(state.facility)
            or not state.map_phase.done
        )
    )

    # Clear previous mapping state if requested
    if state.clear:
        cleared = await asyncio.to_thread(
            reset_mapping_state,
            state.facility,
            state.target_ids_list,
        )
        if cleared:
            logger.info("Cleared mapping state for %d sources", cleared)

    workers = [
        WorkerSpec(
            "context", "context_phase", context_worker,
            on_progress=on_progress,
        ),
        WorkerSpec(
            "assign", "assign_phase", assign_worker,
            on_progress=on_progress,
            depends_on=["context_phase"],
        ),
        WorkerSpec(
            "map", "map_phase", map_worker,
            on_progress=on_progress,
            depends_on=["assign_phase"],
        ),
        WorkerSpec(
            "validate", "validate_phase", validate_worker,
            on_progress=on_progress,
            depends_on=["map_phase"],
        ),
    ]

    orphan_specs = [
        OrphanRecoverySpec(
            label="SignalSource",
            facility_field="facility_id",
            timeout_seconds=CLAIM_TIMEOUT_SECONDS,
            claimed_field="mapping_claimed_at",
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=orphan_specs,
    )
