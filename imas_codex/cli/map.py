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
def map_run(
    facility: str,
    ids_name: str,
    model: str | None,
    dd_version: str | None,
    no_persist: bool,
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
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None

    click.echo(f"\nMapping: {result.mapping_id}")
    click.echo(f"Field mappings: {len(result.validated.field_mappings)}")
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
                f"  [{esc.severity.value}] {esc.signal_group_id} → {esc.imas_path}"
            )
            click.echo(f"    {esc.reason}")

    # Corrections
    if result.validated.corrections:
        click.echo(f"\nCorrections ({len(result.validated.corrections)}):")
        for c in result.validated.corrections:
            click.echo(f"  - {c}")


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
        click.echo(f"Field mappings: {len(result['field_mappings'])}")
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
    """Validate existing mapping paths and consistency."""
    configure_cli_logging("map", facility=facility)

    from imas_codex.ids.tools import check_imas_paths, search_existing_mappings

    result = search_existing_mappings(facility, ids_name)
    if not result["mapping"]:
        click.echo(f"No mapping found for {facility}/{ids_name}.")
        return

    # Validate all target paths
    target_paths = [fm["imas_path"] for fm in result["field_mappings"]]
    if not target_paths:
        click.echo("No field mappings to validate.")
        return

    validation = check_imas_paths(target_paths)
    valid = sum(1 for v in validation if v.get("exists"))
    invalid = sum(1 for v in validation if not v.get("exists"))

    click.echo(f"Validated {len(target_paths)} paths:")
    click.echo(f"  Valid: {valid}")
    click.echo(f"  Invalid: {invalid}")

    for v in validation:
        if not v.get("exists"):
            suggestion = v.get("suggestion", "")
            msg = f"  ✗ {v['path']}"
            if suggestion:
                msg += f" (renamed → {suggestion})"
            click.echo(msg)


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
