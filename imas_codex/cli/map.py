"""CLI commands for IMAS signal mapping pipeline.

Usage:
    imas-codex imas map FACILITY                                # Map all achievable IDS
    imas-codex imas map FACILITY -d magnetic_field_systems      # Map IDS in a physics domain
    imas-codex imas map FACILITY -i pf_active                   # Map specific IDS
    imas-codex imas map status FACILITY [IDS_NAME]
    imas-codex imas map show FACILITY IDS_NAME
    imas-codex imas map validate FACILITY IDS_NAME
    imas-codex imas map clear FACILITY IDS_NAME
"""

import json
import logging

import click

from imas_codex.cli.logging import configure_cli_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group with default 'run' subcommand
# ---------------------------------------------------------------------------


class _DefaultRunGroup(click.Group):
    """Click group that defaults to 'run' when first arg is not a subcommand.

    Enables ``imas map jet`` as shorthand for ``imas map run jet``.
    """

    def resolve_command(self, ctx, args):
        if args:
            maybe_cmd = args[0]
            if maybe_cmd in self.commands:
                return super().resolve_command(ctx, args)
            # Not a known subcommand — prepend 'run'
            return super().resolve_command(ctx, ["run", *args])
        return super().resolve_command(ctx, args)


@click.group(cls=_DefaultRunGroup)
def map_cmd() -> None:
    """IMAS signal mapping pipeline.

    \b
    Maps facility signal sources to IMAS data structures. Sources are
    grouped by physics domain and matched to IDS targets via LLM.

    \b
    Running Mappings:
      imas-codex imas map FACILITY                              Map all achievable IDS
      imas-codex imas map FACILITY -d magnetic_field_systems     Map IDS in a domain
      imas-codex imas map FACILITY -i pf_active                  Map specific IDS
      imas-codex imas map FACILITY -d equilibrium -i magnetics   Union of filters

    \b
    Management Commands:
      status             Show mapping status
      show               Show detailed mapping JSON
      validate           Validate mapping paths, transforms, units
      clear              Remove mapping and relationships
      activate           Promote mapping to active status

    \b
    Options:
      --domain/-d        Filter by physics domain (repeatable)
      --ids/-i           Filter by IDS name (repeatable)
      --cost-limit/-c    Max total LLM spend in USD (default: $25)
      --time             Max total runtime in minutes
      --model/-m         Override LLM model identifier
      --clear            Clear existing mappings before generating
      --dry-run          Skip graph persistence (dry run)
      --no-activate      Persist as 'generated' without promoting to 'active'
    """


# ---------------------------------------------------------------------------
# Run command (default when no subcommand given)
# ---------------------------------------------------------------------------


@map_cmd.command("run")
@click.argument("facility")
@click.option(
    "--domain",
    "-d",
    "domains",
    multiple=True,
    help="Physics domain(s) to map. Repeatable. Without this flag, "
    "maps all domains with enriched signal sources.",
)
@click.option(
    "--ids",
    "-i",
    "ids_names",
    multiple=True,
    help="Specific IDS name(s) to map. Repeatable. "
    "Physics domains are derived from the IDS internally.",
)
@click.option("--model", "-m", help="Override LLM model identifier.")
@click.option("--dd-version", help="Override Data Dictionary version.")
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=25.0,
    help="Maximum total LLM spend in USD (default: $25).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Skip graph persistence (dry run).",
)
@click.option(
    "--no-activate",
    is_flag=True,
    help="Persist as 'generated' without promoting to 'active'.",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum total runtime in minutes.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--clear",
    is_flag=True,
    help="Clear existing mappings for targeted IDS before generating.",
)
def map_run(
    facility: str,
    domains: tuple[str, ...],
    ids_names: tuple[str, ...],
    model: str | None,
    dd_version: str | None,
    cost_limit: float,
    dry_run: bool,
    no_activate: bool,
    time_limit: int | None,
    verbose: bool,
    clear: bool,
) -> None:
    """Run the IMAS signal mapping pipeline.

    Maps facility signal sources to IMAS data structures. Signal sources
    are grouped by physics domain and matched to IDS targets.

    \b
    Scoping:
      No flags         Map all achievable IDS from all enriched signal sources
      --domain/-d      Map IDS whose IMASNodes touch these physics domains
      --ids/-i         Map specific IDS name(s) directly
      Both             Union of both result sets

    \b
    Examples:
      imas-codex imas map jet                                    Map all achievable IDS
      imas-codex imas map jet -d magnetic_field_systems           Map IDS in domain
      imas-codex imas map jet -i pf_active                        Map specific IDS
      imas-codex imas map jet -d equilibrium -i magnetics         Union of filters
      imas-codex imas map jet -c 2.0 --time 10                    Budget and time limits
      imas-codex imas map jet --clear -i pf_active                Clear and remap
    """
    import time as time_mod

    from imas_codex.cli.discover.common import (
        make_log_print,
        setup_logging,
        use_rich_output,
    )

    use_rich = use_rich_output()
    console = setup_logging("map", facility, use_rich, verbose=verbose)
    log_print = make_log_print("map", console)

    # -------------------------------------------------------------------
    # Pre-flight: discover mappable IDS from physics domain intersection
    # -------------------------------------------------------------------
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.tools import discover_mappable_ids

    gc = GraphClient()
    plan = discover_mappable_ids(
        facility,
        gc=gc,
        domains=list(domains) if domains else None,
        ids_filter=list(ids_names) if ids_names else None,
    )

    available = plan["available_domains"]
    targets = plan["ids_targets"]
    plan["total_sources"]

    if not available:
        click.echo(
            f"No enriched signal sources found for {facility}. "
            "Run signal discovery first: imas-codex discover signals "
            f"{facility}",
            err=True,
        )
        raise SystemExit(1)

    if not targets:
        if domains or ids_names:
            parts = []
            if domains:
                parts.append(f"domains={list(domains)}")
            if ids_names:
                parts.append(f"ids={list(ids_names)}")
            click.echo(
                f"No IDS found matching {', '.join(parts)} "
                f"for {facility}. Available domains: {available}",
                err=True,
            )
        else:
            click.echo(
                f"No mappable IDS found for {facility}.",
                err=True,
            )
        raise SystemExit(1)

    # -------------------------------------------------------------------
    # Compute global deadline
    # -------------------------------------------------------------------
    deadline: float | None = None
    if time_limit is not None:
        deadline = time_mod.time() + time_limit * 60

    ids_names_list = [t["ids_name"] for t in targets]

    # -------------------------------------------------------------------
    # Run mapping pipeline
    # -------------------------------------------------------------------
    if not use_rich:
        all_results = _run_plain_mode(
            facility=facility,
            targets=targets,
            model=model,
            dd_version=dd_version,
            cost_limit=cost_limit,
            dry_run=dry_run,
            no_activate=no_activate,
            clear=clear,
            deadline=deadline,
            log_print=log_print,
        )
    else:
        all_results = _run_rich_mode(
            facility=facility,
            ids_names=ids_names_list,
            targets=targets,
            model=model,
            dd_version=dd_version,
            cost_limit=cost_limit,
            dry_run=dry_run,
            no_activate=no_activate,
            clear=clear,
            deadline=deadline,
            verbose=verbose,
            console=console,
            log_print=log_print,
        )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    _print_summary(all_results, log_print)


def _run_plain_mode(
    *,
    facility: str,
    targets: list[dict],
    model: str | None,
    dd_version: str | None,
    cost_limit: float,
    dry_run: bool,
    no_activate: bool,
    clear: bool,
    deadline: float | None,
    log_print,
) -> list[dict]:
    """Run mapping for all IDS targets in plain (non-rich) mode."""
    import asyncio

    from imas_codex.ids.workers import MappingDiscoveryState, run_mapping_engine

    ids_names = [t["ids_name"] for t in targets]

    engine_state = MappingDiscoveryState(
        facility=facility,
        target_ids_list=ids_names,
        target_info=targets,
        dd_version=dd_version,
        model=model,
        cost_limit=cost_limit,
        persist=not dry_run,
        activate=not no_activate,
        clear=clear,
    )
    if deadline:
        engine_state.deadline = deadline

    def _on_progress(detail, stats, stream_items=None):
        log_print(f"  {detail}")

    try:
        asyncio.run(run_mapping_engine(engine_state, on_progress=_on_progress))
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")

    # Build results from engine state
    all_results: list[dict] = []
    for ids_name in ids_names:
        ids_result = engine_state.ids_results.get(ids_name, {})
        batches = engine_state.mapping_batches.get(ids_name, [])
        if batches or ids_result:
            all_results.append(
                {
                    "ids_name": ids_name,
                    "bindings": ids_result.get(
                        "bindings", sum(len(b.mappings) for _, b in batches)
                    ),
                    "escalations": ids_result.get("escalations", 0),
                    "persisted": engine_state.persist,
                }
            )

    return all_results


def _run_rich_mode(
    *,
    facility: str,
    ids_names: list[str],
    targets: list[dict],
    model: str | None,
    dd_version: str | None,
    cost_limit: float,
    dry_run: bool,
    no_activate: bool,
    clear: bool,
    deadline: float | None,
    verbose: bool,
    console,
    log_print,
) -> list[dict]:
    """Run mapping for all IDS targets with a single Rich progress display.

    No per-IDS stepping — the engine handles all IDS targets internally.
    Workers use the graph as a state machine with ``mapping_status`` and
    ``mapping_claimed_at`` on SignalSource nodes.
    """
    from imas_codex.cli.discover.common import DiscoveryConfig, run_discovery
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.ids.progress import MappingProgressDisplay
    from imas_codex.ids.workers import MappingDiscoveryState, run_mapping_engine

    try:
        facility_config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        return []

    display = MappingProgressDisplay(
        facility=facility,
        ids_targets=ids_names,
        cost_limit=cost_limit,
        model=model,
    )

    if deadline:
        display.state.deadline = deadline

    config = DiscoveryConfig(
        domain="mapping",
        facility=facility,
        facility_config=facility_config,
        display=display,
        check_graph=True,
        check_embed=True,
        check_ssh=False,
        check_model=True,
        model_section="reasoning",
        verbose=verbose,
    )

    all_results: list[dict] = []

    async def _run_mapping(stop_event, service_monitor):
        # Single engine call for ALL IDS targets
        engine_state = MappingDiscoveryState(
            facility=facility,
            target_ids_list=ids_names,
            target_info=targets,
            dd_version=dd_version,
            model=model,
            cost_limit=cost_limit,
            persist=not dry_run,
            activate=not no_activate,
            clear=clear,
        )
        if deadline:
            engine_state.deadline = deadline
        if service_monitor:
            engine_state.service_monitor = service_monitor

        def _update_display(detail, stats, stream_items=None):
            ds = display.state
            ds.sources_found = engine_state.sources_total
            ds.sections_assigned = engine_state.sources_assigned
            ds.sections_mapped = engine_state.sources_mapped
            ds.bindings_total = engine_state.bindings_total
            ds.bindings_passed = engine_state.bindings_passed
            ds.escalations = engine_state.escalations
            ds.sections_assembled = engine_state.sources_validated
            ds.cost = engine_state.cost
            ds.current_detail = str(detail)

            # Derive sections_total from all assignments
            total_assigned = sum(
                len(s.assignments)
                for s in engine_state.assignments.values()
                if hasattr(s, "assignments")
            )
            if total_assigned:
                ds.sections_total = total_assigned

            # Track current pipeline step from worker phases
            if engine_state.context_phase.done:
                if engine_state.assign_phase.done:
                    if engine_state.map_phase.done:
                        ds.current_step = "validate"
                    else:
                        ds.current_step = "mapping"
                else:
                    ds.current_step = "assign"
            else:
                ds.current_step = "context"

            # Push stream items to the appropriate queue
            if stream_items:
                step = ds.current_step
                if step == "context":
                    ds.context_queue.add(stream_items)
                elif step == "assign":
                    ds.assign_queue.add(stream_items)
                elif step == "mapping":
                    ds.map_queue.add(stream_items)
                elif step == "validate":
                    if stream_items and "pattern" in stream_items[0]:
                        ds.assembly_queue.add(stream_items)
                    else:
                        ds.validate_queue.add(stream_items)

        try:
            await run_mapping_engine(
                engine_state,
                stop_event=stop_event,
                on_progress=_update_display,
            )
        except ValueError as e:
            logger.warning("Mapping engine error: %s", e)

        # Build results from engine state
        for ids_name in ids_names:
            ids_result = engine_state.ids_results.get(ids_name, {})
            batches = engine_state.mapping_batches.get(ids_name, [])
            if batches or ids_result:
                result = {
                    "ids_name": ids_name,
                    "bindings": ids_result.get(
                        "bindings", sum(len(b.mappings) for _, b in batches)
                    ),
                    "escalations": ids_result.get("escalations", 0),
                    "persisted": engine_state.persist,
                }
                all_results.append(result)

        # Update display to show completion
        ds = display.state
        ds.current_step = "done"
        ds.completed_ids = all_results
        return {"completed": len(all_results), "total": len(ids_names)}

    try:
        run_discovery(config, _run_mapping)
    except SystemExit:
        raise
    except Exception as e:
        log_print(f"[red]Error in mapping pipeline: {e}[/red]")

    return all_results


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def _print_result(result) -> None:
    """Print mapping result summary for a single IDS."""
    click.echo(f"\nMapping: {result.mapping_id}")
    click.echo(f"Bindings: {len(result.validated.bindings)}")
    click.echo(f"Escalations: {len(result.validated.escalations)}")
    click.echo(f"Persisted: {result.persisted}")
    click.echo()

    # Cost breakdown
    click.echo("Cost breakdown:")
    for step, usd in result.cost.steps.items():
        click.echo(f"  {step}: ${usd:.4f}")
    click.echo(
        f"  Total: ${result.cost.total_usd:.4f} ({result.cost.total_tokens} tokens)"
    )

    # Escalations
    if result.validated.escalations:
        click.echo(f"\nEscalations ({len(result.validated.escalations)}):")
        for esc in result.validated.escalations:
            click.echo(
                f"  [{esc.severity.value}] {esc.source_id} \u2192 {esc.target_id}"
            )
            click.echo(f"    {esc.reason}")

    # Corrections
    if result.validated.corrections:
        click.echo(f"\nCorrections ({len(result.validated.corrections)}):")
        for c in result.validated.corrections:
            click.echo(f"  - {c}")

    # Unassigned groups
    if result.unassigned_groups:
        click.echo(f"\nUnassigned signal sources ({len(result.unassigned_groups)}):")
        for gid in result.unassigned_groups:
            click.echo(f"  - {gid}")


def _print_summary(results: list[dict], log_print) -> None:
    """Print summary across all mapped IDS."""
    if not results:
        log_print("[yellow]No IDS were successfully mapped.[/yellow]")
        return

    total_bindings = sum(r.get("bindings", 0) for r in results)
    total_escalations = sum(r.get("escalations", 0) for r in results)

    log_print("\n[bold]Mapping Summary[/bold]")
    log_print(f"  IDS mapped: {len(results)}")
    log_print(f"  Total bindings: {total_bindings}")
    log_print(f"  Total escalations: {total_escalations}")

    for r in results:
        log_print(
            f"    {r.get('ids_name', '?')}: "
            f"{r.get('bindings', 0)} bindings, "
            f"{r.get('escalations', 0)} escalations"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clear_mapping(facility: str, ids_name: str, log_print=None) -> int:
    """Remove a mapping and its relationships from the graph.

    Returns the number of mapping nodes deleted.
    """
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    # Delete MAPS_TO_IMAS from signal sources used by this mapping
    gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
              -[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
        MATCH (sg)-[r:MAPS_TO_IMAS]->(:IMASNode)
        DELETE r
        """,
        facility=facility,
        ids=ids_name,
    )

    # Delete evidence nodes
    gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
              -[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
        MATCH (sg)-[:HAS_EVIDENCE]->(ev:MappingEvidence)
        DETACH DELETE ev
        """,
        facility=facility,
        ids=ids_name,
    )

    # Delete mapping node and its relationships
    result = gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
        WITH m, m.id AS mid
        DETACH DELETE m
        RETURN count(*) AS deleted
        """,
        facility=facility,
        ids=ids_name,
    )

    deleted = result[0]["deleted"] if result else 0
    if log_print and deleted:
        log_print(f"Cleared previous mapping {facility}:{ids_name}")
    return deleted


# ---------------------------------------------------------------------------
# Status subcommand
# ---------------------------------------------------------------------------


@map_cmd.command("status")
@click.argument("facility")
@click.argument("ids_name", required=False)
def map_status(facility: str, ids_name: str | None) -> None:
    """Show mapping status for a facility.

    \b
    Without IDS_NAME, lists all mappings for the facility.
    With IDS_NAME, shows detailed status for that mapping.

    \b
    Examples:
      imas-codex imas map status jet
      imas-codex imas map status jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    if ids_name:
        from imas_codex.ids.tools import search_existing_mappings

        result = search_existing_mappings(facility, ids_name, gc=gc)
        if not result["mapping"]:
            click.echo(f"No mapping found for {facility}/{ids_name}.")
            return

        m = result["mapping"]
        click.echo(f"Mapping: {m['id']}")
        click.echo(f"Status: {m.get('status', 'unknown')}")
        click.echo(f"DD version: {m.get('dd_version', 'unknown')}")
        click.echo(f"Sections: {len(result['sections'])}")
        click.echo(f"Bindings: {len(result['bindings'])}")
    else:
        rows = gc.query(
            """
            MATCH (m:IMASMapping)
            WHERE m.facility_id = $facility
            RETURN m.id AS id, m.ids_name AS ids_name,
                   m.status AS status, m.dd_version AS dd_version
            ORDER BY m.ids_name
            """,
            facility=facility,
        )
        if not rows:
            click.echo(f"No mappings found for {facility}.")
            return

        click.echo(f"{'IDS':<20} {'Status':<12} {'DD Version':<12}")
        click.echo("-" * 44)
        for r in rows:
            click.echo(
                f"{r['ids_name']:<20} "
                f"{r.get('status', '?'):<12} "
                f"{r.get('dd_version', '?'):<12}"
            )


# ---------------------------------------------------------------------------
# Show subcommand
# ---------------------------------------------------------------------------


@map_cmd.command("show")
@click.argument("facility")
@click.argument("ids_name")
def map_show(facility: str, ids_name: str) -> None:
    """Show detailed mapping for a facility/IDS pair.

    \b
    Examples:
      imas-codex imas map show jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.ids.tools import search_existing_mappings

    result = search_existing_mappings(facility, ids_name)
    if not result["mapping"]:
        click.echo(f"No mapping found for {facility}/{ids_name}.")
        return

    click.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Validate subcommand
# ---------------------------------------------------------------------------


@map_cmd.command("validate")
@click.argument("facility")
@click.argument("ids_name")
def map_validate(facility: str, ids_name: str) -> None:
    """Validate existing mapping paths, transforms, units, and coverage.

    \b
    Checks source existence, target existence, transform execution,
    unit compatibility, duplicate targets, and field coverage.

    \b
    Examples:
      imas-codex imas map validate jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.tools import search_existing_mappings
    from imas_codex.ids.validation import (
        compute_assembly_coverage,
        compute_confidence_distribution,
        compute_coverage,
        compute_signal_coverage,
        compute_signal_source_coverage,
        validate_mapping,
    )

    gc = GraphClient()
    result = search_existing_mappings(facility, ids_name, gc=gc)
    if not result["mapping"]:
        click.echo(f"No mapping found for {facility}/{ids_name}.")
        return

    from imas_codex.ids.models import ValidatedSignalMapping

    bindings = [ValidatedSignalMapping(**b) for b in result["bindings"]]
    if not bindings:
        click.echo("No bindings to validate.")
        return

    report = validate_mapping(bindings, gc=gc)

    passed = sum(
        1
        for c in report.binding_checks
        if c.source_exists
        and c.target_exists
        and c.transform_executes
        and c.units_compatible
    )
    failed = len(report.binding_checks) - passed

    click.echo(f"Validated {len(bindings)} bindings:")
    click.echo(f"  Passed: {passed}")
    click.echo(f"  Failed: {failed}")

    for c in report.binding_checks:
        if c.error:
            click.echo(f"  \u2717 {c.source_id} \u2192 {c.target_id}")
            click.echo(f"    {c.error}")

    if report.duplicate_targets:
        click.echo(f"\nDuplicate targets ({len(report.duplicate_targets)}):")
        for dup in report.duplicate_targets:
            sources = [b.source_id for b in bindings if b.target_id == dup]
            click.echo(f"  {dup} \u2190 {', '.join(sources)}")

    # Coverage
    coverage = compute_coverage(ids_name, bindings, gc=gc)
    if coverage.total_leaf_fields > 0:
        click.echo(
            f"\nCoverage ({ids_name}): "
            f"{coverage.mapped_fields}/{coverage.total_leaf_fields} "
            f"leaf fields ({coverage.percentage:.1f}%)"
        )
        if coverage.mapped_paths:
            click.echo(f"  Mapped: {', '.join(coverage.mapped_paths[:10])}")
            if len(coverage.mapped_paths) > 10:
                click.echo(f"    ... and {len(coverage.mapped_paths) - 10} more")
        if coverage.unmapped_fields:
            shown = coverage.unmapped_fields[:10]
            click.echo(f"  Unmapped: {', '.join(shown)}")
            if len(coverage.unmapped_fields) > 10:
                click.echo(f"    ... and {len(coverage.unmapped_fields) - 10} more")

    # Signal source coverage
    sig_cov = compute_signal_coverage(facility, gc=gc)
    if sig_cov.total_enriched > 0:
        click.echo(
            f"\nSignal sources ({facility}): "
            f"{sig_cov.mapped}/{sig_cov.total_enriched} "
            f"enriched groups mapped ({sig_cov.percentage:.1f}%)"
        )
        if sig_cov.unmapped_groups:
            shown = sig_cov.unmapped_groups[:10]
            click.echo(f"  Unmapped: {', '.join(shown)}")
            if len(sig_cov.unmapped_groups) > 10:
                click.echo(f"    ... and {len(sig_cov.unmapped_groups) - 10} more")

    # Extended signal source coverage per IDS
    src_cov = compute_signal_source_coverage(facility, ids_name, gc=gc)
    if src_cov.total_enriched_matching > 0:
        click.echo(
            f"\nSignal source coverage ({facility}/{ids_name}): "
            f"{src_cov.mapped_to_ids}/{src_cov.total_enriched_matching} "
            f"enriched sources mapped ({src_cov.enriched_mapped_pct:.1f}%)"
        )
        if src_cov.unmapped_enriched:
            shown = src_cov.unmapped_enriched[:10]
            click.echo(f"  Unmapped enriched: {', '.join(shown)}")
            if len(src_cov.unmapped_enriched) > 10:
                click.echo(f"    ... and {len(src_cov.unmapped_enriched) - 10} more")
    if src_cov.discovered_sources > 0:
        click.echo(
            f"  Underspecified (discovered, not enriched): {src_cov.discovered_sources}"
        )
    if src_cov.multi_target_sources > 0:
        click.echo(f"  Multi-target sources: {src_cov.multi_target_sources}")

    # Assembly coverage
    asm_cov = compute_assembly_coverage(facility, ids_name, gc=gc)
    if asm_cov.total_sections > 0:
        click.echo(
            f"\nAssembly coverage ({ids_name}): "
            f"{asm_cov.sections_with_config}/{asm_cov.total_sections} "
            f"sections configured"
        )
        if asm_cov.sections_without_config:
            click.echo(f"  Unconfigured: {', '.join(asm_cov.sections_without_config)}")
        click.echo(
            f"  Patterns: {asm_cov.default_pattern_count} default "
            f"(array_per_node), {asm_cov.custom_pattern_count} custom"
        )
        click.echo(
            f"  init_arrays: {asm_cov.init_arrays_configured} configured, "
            f"{asm_cov.init_arrays_unconfigured} unconfigured"
        )

    # Confidence distribution
    conf_dist = compute_confidence_distribution(bindings)
    if conf_dist.total_bindings > 0:
        click.echo(
            f"\nConfidence distribution ({conf_dist.total_bindings} bindings, "
            f"avg {conf_dist.average_confidence:.2f}):"
        )
        click.echo(
            f"  High (>0.8): {conf_dist.high_count}  "
            f"Medium (0.5-0.8): {conf_dist.medium_count}  "
            f"Low (<0.5): {conf_dist.low_count}"
        )
        if conf_dist.low_bindings:
            click.echo("  Low-confidence (review needed):")
            for entry in conf_dist.low_bindings[:10]:
                click.echo(f"    {entry}")
            if len(conf_dist.low_bindings) > 10:
                click.echo(f"    ... and {len(conf_dist.low_bindings) - 10} more")


# ---------------------------------------------------------------------------
# Clear subcommand
# ---------------------------------------------------------------------------


@map_cmd.command("clear")
@click.argument("facility")
@click.argument("ids_name")
@click.confirmation_option(prompt="Delete this mapping and all its relationships?")
def map_clear(facility: str, ids_name: str) -> None:
    """Remove a mapping and its relationships from the graph.

    \b
    Examples:
      imas-codex imas map clear jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.cli.discover.common import make_log_print

    log_print = make_log_print("map")
    count = _clear_mapping(facility, ids_name, log_print)
    if count == 0:
        click.echo(f"No mapping found for {facility}/{ids_name}.")
    else:
        click.echo(f"Cleared mapping {facility}:{ids_name}.")


# ---------------------------------------------------------------------------
# Activate subcommand
# ---------------------------------------------------------------------------


@map_cmd.command("activate")
@click.argument("facility")
@click.argument("ids_name")
def map_activate(facility: str, ids_name: str) -> None:
    """Promote a mapping to active status for use by the assembler.

    \b
    Only mappings in 'generated' or 'validated' status can be activated.
    The assembler only loads mappings with status 'active'.

    \b
    Examples:
      imas-codex imas map activate jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    rows = gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
        RETURN m.status AS status
        """,
        facility=facility,
        ids=ids_name,
    )
    if not rows:
        click.echo(f"No mapping found for {facility}/{ids_name}.", err=True)
        raise SystemExit(1)

    current = rows[0].get("status")
    if current == "active":
        click.echo(f"Mapping {facility}:{ids_name} is already active.")
        return
    if current == "deprecated":
        click.echo(
            f"Cannot activate deprecated mapping {facility}:{ids_name}. "
            "Generate a new mapping first.",
            err=True,
        )
        raise SystemExit(1)

    gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
        SET m.status = 'active'
        """,
        facility=facility,
        ids=ids_name,
    )
    click.echo(f"Activated mapping {facility}:{ids_name} (was '{current}').")
