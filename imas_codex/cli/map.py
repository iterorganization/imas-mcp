"""CLI commands for IMAS mapping pipeline.

Usage:
    imas-codex imas map run FACILITY IDS_NAME
    imas-codex imas map status FACILITY [IDS_NAME]
    imas-codex imas map show FACILITY IDS_NAME
    imas-codex imas map validate FACILITY IDS_NAME
    imas-codex imas map validate-e2e FACILITY IDS_NAME --shot SHOT
    imas-codex imas map clear FACILITY IDS_NAME
"""

import json
import logging

import click

from imas_codex.cli.logging import configure_cli_logging

logger = logging.getLogger(__name__)


@click.group()
def map_cmd() -> None:
    """IMAS mapping pipeline commands."""


@map_cmd.command("run")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--model", "-m", help="Override LLM model identifier.")
@click.option("--dd-version", help="Override Data Dictionary version.")
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=5.0,
    help="Maximum LLM spend in USD (default: $5).",
)
@click.option("--no-persist", is_flag=True, help="Skip graph persistence.")
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
    help="Maximum runtime in minutes.",
)
@click.option("--plain", is_flag=True, help="Disable rich progress display.")
def map_run(
    facility: str,
    ids_name: str,
    model: str | None,
    dd_version: str | None,
    cost_limit: float,
    no_persist: bool,
    no_activate: bool,
    time_limit: int | None,
    plain: bool,
) -> None:
    """Run the full mapping pipeline.

    \b
    Examples:
      imas-codex imas map run jet pf_active
      imas-codex imas map run --no-persist jet pf_active
      imas-codex imas map run --cost-limit 2.0 --time 10 jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.ids.mapping import generate_mapping

    if plain:
        # Plain mode — simple synchronous execution
        click.echo(f"Generating mapping for {facility}/{ids_name}...")
        try:
            result = generate_mapping(
                facility,
                ids_name,
                model=model,
                dd_version=dd_version,
                persist=not no_persist,
                activate=not no_activate,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1) from None

        _print_result(result)
        return

    # Rich mode — progress display via run_discovery harness
    import asyncio
    import time

    from imas_codex.cli.discover.common import DiscoveryConfig, run_discovery
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.ids.progress import MappingProgressDisplay

    try:
        facility_config = get_facility(facility)
    except Exception:
        facility_config = {"id": facility}

    display = MappingProgressDisplay(
        facility=facility,
        ids_name=ids_name,
        cost_limit=cost_limit,
        model=model,
    )

    if time_limit:
        display.state.deadline = time.time() + time_limit * 60

    config = DiscoveryConfig(
        domain="mapping",
        facility=facility,
        facility_config=facility_config,
        display=display,
        check_graph=True,
        check_embed=False,
        check_ssh=False,
    )

    async def _run_mapping(stop_event, service_monitor):
        def _sync_run():
            return generate_mapping(
                facility,
                ids_name,
                model=model,
                dd_version=dd_version,
                persist=not no_persist,
                activate=not no_activate,
            )

        try:
            result = await asyncio.to_thread(_sync_run)
        except ValueError as e:
            click.echo(f"\nError: {e}", err=True)
            raise SystemExit(1) from None

        # Update display state for final render
        state = display.state
        state.current_step = "done"
        state.bindings_total = len(result.validated.bindings)
        state.bindings_passed = state.bindings_total
        state.escalations = len(result.validated.escalations)
        state.cost = result.cost

        return {"result": result}

    def _on_complete(results):
        result = results.get("result")
        if result:
            click.echo()
            _print_result(result)

    try:
        run_discovery(config, _run_mapping, on_complete=_on_complete)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None


def _print_result(result) -> None:
    """Print mapping result summary."""
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
                f"  [{esc.severity.value}] {esc.source_id} → {esc.target_id}"
            )
            click.echo(f"    {esc.reason}")

    # Corrections
    if result.validated.corrections:
        click.echo(f"\nCorrections ({len(result.validated.corrections)}):")
        for c in result.validated.corrections:
            click.echo(f"  - {c}")

    # Unassigned groups
    if result.unassigned_groups:
        click.echo(
            f"\nUnassigned signal groups ({len(result.unassigned_groups)}):"
        )
        for gid in result.unassigned_groups:
            click.echo(f"  - {gid}")


@map_cmd.command("status")
@click.argument("facility")
@click.argument("ids_name", required=False)
def map_status(facility: str, ids_name: str | None) -> None:
    """Show mapping status for a facility."""
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    if ids_name:
        # Show specific mapping status
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
        # List all mappings for facility
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


@map_cmd.command("show")
@click.argument("facility")
@click.argument("ids_name")
def map_show(facility: str, ids_name: str) -> None:
    """Show detailed mapping for a facility/IDS pair."""
    configure_cli_logging("map", facility=facility)

    from imas_codex.ids.tools import search_existing_mappings

    result = search_existing_mappings(facility, ids_name)
    if not result["mapping"]:
        click.echo(f"No mapping found for {facility}/{ids_name}.")
        return

    click.echo(json.dumps(result, indent=2, default=str))


@map_cmd.command("validate")
@click.argument("facility")
@click.argument("ids_name")
def map_validate(facility: str, ids_name: str) -> None:
    """Validate existing mapping paths, transforms, units, and coverage."""
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

    # Build lightweight binding objects from graph data
    from imas_codex.ids.models import ValidatedFieldMapping

    bindings = [ValidatedFieldMapping(**b) for b in result["bindings"]]
    if not bindings:
        click.echo("No bindings to validate.")
        return

    # Run validation
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
            click.echo(f"  ✗ {c.source_id} → {c.target_id}")
            click.echo(f"    {c.error}")

    if report.duplicate_targets:
        click.echo(f"\nDuplicate targets ({len(report.duplicate_targets)}):")
        for dup in report.duplicate_targets:
            sources = [b.source_id for b in bindings if b.target_id == dup]
            click.echo(f"  {dup} ← {', '.join(sources)}")

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
                click.echo(
                    f"    ... and {len(coverage.unmapped_fields) - 10} more"
                )

    # Signal group coverage
    sig_cov = compute_signal_coverage(facility, gc=gc)
    if sig_cov.total_enriched > 0:
        click.echo(
            f"\nSignal groups ({facility}): "
            f"{sig_cov.mapped}/{sig_cov.total_enriched} "
            f"enriched groups mapped ({sig_cov.percentage:.1f}%)"
        )
        if sig_cov.unmapped_groups:
            shown = sig_cov.unmapped_groups[:10]
            click.echo(f"  Unmapped: {', '.join(shown)}")
            if len(sig_cov.unmapped_groups) > 10:
                click.echo(
                    f"    ... and {len(sig_cov.unmapped_groups) - 10} more"
                )

    # Extended signal source coverage per IDS (8.1)
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
                click.echo(
                    f"    ... and {len(src_cov.unmapped_enriched) - 10} more"
                )
    if src_cov.discovered_sources > 0:
        click.echo(
            f"  Underspecified (discovered, not enriched): "
            f"{src_cov.discovered_sources}"
        )
    if src_cov.multi_target_sources > 0:
        click.echo(
            f"  Multi-target sources: {src_cov.multi_target_sources}"
        )

    # Assembly coverage (8.2)
    asm_cov = compute_assembly_coverage(facility, ids_name, gc=gc)
    if asm_cov.total_sections > 0:
        click.echo(
            f"\nAssembly coverage ({ids_name}): "
            f"{asm_cov.sections_with_config}/{asm_cov.total_sections} "
            f"sections configured"
        )
        if asm_cov.sections_without_config:
            click.echo(
                f"  Unconfigured: {', '.join(asm_cov.sections_without_config)}"
            )
        click.echo(
            f"  Patterns: {asm_cov.default_pattern_count} default "
            f"(array_per_node), {asm_cov.custom_pattern_count} custom"
        )
        click.echo(
            f"  init_arrays: {asm_cov.init_arrays_configured} configured, "
            f"{asm_cov.init_arrays_unconfigured} unconfigured"
        )

    # Confidence distribution (8.3)
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
                click.echo(
                    f"    ... and {len(conf_dist.low_bindings) - 10} more"
                )


@map_cmd.command("clear")
@click.argument("facility")
@click.argument("ids_name")
@click.confirmation_option(prompt="Delete this mapping and all its relationships?")
def map_clear(facility: str, ids_name: str) -> None:
    """Remove a mapping and its relationships from the graph."""
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    mapping_id = f"{facility}:{ids_name}"

    # Delete MAPS_TO_IMAS from signal groups used by this mapping
    gc.query(
        """
        MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
        MATCH (sg)-[r:MAPS_TO_IMAS]->(:IMASNode)
        DELETE r
        """,
        id=mapping_id,
    )

    # Delete evidence nodes
    gc.query(
        """
        MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
        MATCH (sg)-[:HAS_EVIDENCE]->(ev:MappingEvidence)
        DETACH DELETE ev
        """,
        id=mapping_id,
    )

    # Delete mapping node and its relationships
    gc.query(
        """
        MATCH (m:IMASMapping {id: $id})
        DETACH DELETE m
        """,
        id=mapping_id,
    )

    click.echo(f"Cleared mapping {mapping_id}.")


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
    mapping_id = f"{facility}:{ids_name}"

    rows = gc.query(
        """
        MATCH (m:IMASMapping {id: $id})
        RETURN m.status AS status
        """,
        id=mapping_id,
    )
    if not rows:
        click.echo(f"No mapping found for {mapping_id}.", err=True)
        raise SystemExit(1)

    current = rows[0].get("status")
    if current == "active":
        click.echo(f"Mapping {mapping_id} is already active.")
        return
    if current == "deprecated":
        click.echo(
            f"Cannot activate deprecated mapping {mapping_id}. "
            "Generate a new mapping first.",
            err=True,
        )
        raise SystemExit(1)

    gc.query(
        """
        MATCH (m:IMASMapping {id: $id})
        SET m.status = 'active'
        """,
        id=mapping_id,
    )
    click.echo(f"Activated mapping {mapping_id} (was '{current}').")


@map_cmd.command("validate-e2e")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--shot", type=int, required=True, help="Shot number to validate against.")
@click.option(
    "--strategy",
    type=click.Choice(["auto", "client", "remote"]),
    default="auto",
    help="Execution strategy (default: auto-detect).",
)
@click.option("--ssh-host", default=None, help="Override SSH host for extraction.")
def map_validate_e2e(
    facility: str,
    ids_name: str,
    shot: int,
    strategy: str,
    ssh_host: str | None,
) -> None:
    """Run end-to-end validation: extract → assemble → validate.

    \b
    Extracts signal data for a single shot from the facility,
    runs assembly, and validates the populated IDS.

    \b
    Examples:
      imas-codex imas map validate-e2e tcv pf_active --shot 80000
      imas-codex imas map validate-e2e jet magnetics --shot 99000 --strategy client
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.validation import validate_mapping_e2e

    gc = GraphClient()

    click.echo(f"E2E VALIDATION — {facility}/{ids_name} shot {shot}")
    click.echo("━" * 50)

    result = validate_mapping_e2e(
        facility,
        ids_name,
        shot,
        gc=gc,
        ssh_host=ssh_host,
        strategy=strategy,
    )

    # Strategy
    click.echo(f"STRATEGY  {result.strategy}")
    click.echo()

    # Extraction
    total = result.extraction_success + result.extraction_failed
    click.echo(
        f"EXTRACT   {result.extraction_success} success, "
        f"{result.extraction_failed} failed (of {total})"
    )
    for err in result.extraction_errors[:10]:
        click.echo(f"  ✗ {err}")
    if len(result.extraction_errors) > 10:
        click.echo(f"  ... and {len(result.extraction_errors) - 10} more")
    click.echo()

    # Assembly
    if result.assembly_error:
        click.echo(f"ASSEMBLE  ✗ {result.assembly_error}")
    elif result.assembly_success:
        click.echo("ASSEMBLE  ✓ Assembly completed")
    else:
        click.echo("ASSEMBLE  ✗ No data extracted")
    click.echo()

    # Field checks
    populated = sum(1 for f in result.field_checks if f.populated)
    click.echo(
        f"FIELDS    {populated}/{len(result.field_checks)} populated"
    )
    for fc in result.field_checks:
        if not fc.populated:
            click.echo(f"  ✗ {fc.target_path}")
        elif fc.value_range:
            click.echo(f"  ✓ {fc.target_path} {fc.value_range}")
    click.echo()

    # Time bases
    tb_mark = "✓" if result.time_base_consistent else "⚠"
    click.echo(f"TIMEBASES {tb_mark} {'consistent' if result.time_base_consistent else 'inconsistent'}")
    click.echo()

    # Overall
    overall = "✓ PASS" if result.all_passed else "✗ FAIL"
    click.echo(f"RESULT    {overall}")

    if not result.all_passed:
        raise SystemExit(1)
