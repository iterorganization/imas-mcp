"""CLI commands for IMAS mapping pipeline.

Usage:
    imas-codex imas map run FACILITY IDS_NAME
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


@click.group()
def map_cmd() -> None:
    """IMAS mapping pipeline commands."""


@map_cmd.command("run")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--model", "-m", help="Override LLM model identifier.")
@click.option("--dd-version", help="Override Data Dictionary version.")
@click.option("--no-persist", is_flag=True, help="Skip graph persistence.")
@click.option(
    "--no-activate",
    is_flag=True,
    help="Persist as 'generated' without promoting to 'active'.",
)
def map_run(
    facility: str,
    ids_name: str,
    model: str | None,
    dd_version: str | None,
    no_persist: bool,
    no_activate: bool,
) -> None:
    """Run the full mapping pipeline.

    \b
    Examples:
      imas-codex imas map run jet pf_active
      imas-codex imas map run --no-persist jet pf_active
    """
    configure_cli_logging("map", facility=facility)

    from imas_codex.ids.mapping import generate_mapping

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
        compute_coverage,
        compute_signal_coverage,
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
        MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_GROUP]->(sg:SignalGroup)
        MATCH (sg)-[r:MAPS_TO_IMAS]->(:IMASNode)
        DELETE r
        """,
        id=mapping_id,
    )

    # Delete evidence nodes
    gc.query(
        """
        MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_GROUP]->(sg:SignalGroup)
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
