"""CLI commands for IDS assembly and export.

Usage:
    imas-codex ids list [FACILITY]
    imas-codex ids show FACILITY IDS_NAME
    imas-codex ids export FACILITY IDS_NAME --epoch EPOCH [--output PATH]
"""

import logging

import click

from imas_codex.cli.logging import configure_cli_logging

logger = logging.getLogger(__name__)


@click.group()
def ids() -> None:
    """Assemble and export IMAS IDS from graph data."""


@ids.command("list")
@click.argument("facility", required=False)
def ids_list(facility: str | None) -> None:
    """List available IDS recipes."""
    from imas_codex.ids.assembler import list_recipes

    configure_cli_logging("ids", facility=facility or "all")

    recipes = list_recipes(facility)
    if not recipes:
        click.echo("No recipes found.")
        return

    click.echo(f"{'Facility':<12} {'IDS':<20} {'DD Version':<12}")
    click.echo("-" * 44)
    for r in recipes:
        click.echo(f"{r['facility']:<12} {r['ids_name']:<20} {r['dd_version']:<12}")


@ids.command("show")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--epoch", "-e", help="Epoch to show summary for (e.g., p68613).")
def ids_show(facility: str, ids_name: str, epoch: str | None) -> None:
    """Show IDS recipe details and available epochs."""
    from imas_codex.ids.assembler import IDSAssembler

    configure_cli_logging("ids", facility=facility)

    assembler = IDSAssembler(facility, ids_name)

    click.echo(f"IDS: {ids_name}")
    click.echo(f"Facility: {facility}")
    click.echo(f"DD version: {assembler.recipe['dd_version']}")
    if provider := assembler.recipe.get("provider"):
        click.echo(f"Provider: {provider}")
    click.echo()

    # List epochs
    epochs = assembler.list_epochs()
    if epochs:
        click.echo(f"Available epochs ({len(epochs)}):")
        for ep in epochs:
            shot_range = f"shots {ep['first_shot']}"
            if ep.get("last_shot"):
                shot_range += f"-{ep['last_shot']}"
            else:
                shot_range += "+"
            desc = ep.get("description", "")
            epoch_short = ep["id"].split(":")[-1]
            click.echo(f"  {epoch_short:<12} {shot_range:<20} {desc}")
    else:
        click.echo("No epochs found in graph.")

    # Show summary for specific epoch
    if epoch:
        click.echo()
        summary = assembler.summary(epoch)
        click.echo(f"Assembly summary for epoch {epoch}:")
        for array_name, stats in summary.get("arrays", {}).items():
            line = f"  {array_name}: {stats['count']} entries"
            if elements := stats.get("total_elements"):
                line += f" ({elements} total elements)"
            click.echo(line)


@ids.command("export")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--epoch", "-e", required=True, help="Epoch version (e.g., p68613).")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path (without extension). Default: {facility}_{ids_name}_{epoch}.",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["hdf5", "netcdf"]),
    default="hdf5",
    help="Storage backend.",
)
def ids_export(
    facility: str,
    ids_name: str,
    epoch: str,
    output: str | None,
    backend: str,
) -> None:
    """Assemble and export an IDS to file."""
    from pathlib import Path

    from imas_codex.ids.assembler import IDSAssembler

    configure_cli_logging("ids", facility=facility)

    if output is None:
        output = f"{facility}_{ids_name}_{epoch}"

    assembler = IDSAssembler(facility, ids_name)

    # Show summary first
    summary = assembler.summary(epoch)
    for array_name, stats in summary.get("arrays", {}).items():
        line = f"  {array_name}: {stats['count']} entries"
        if elements := stats.get("total_elements"):
            line += f" ({elements} total elements)"
        click.echo(line)

    # Export
    out_path = assembler.export(Path(output), epoch, backend=backend)
    click.echo(f"Exported {ids_name} to {out_path}")


@ids.command("epochs")
@click.argument("facility")
def ids_epochs(facility: str) -> None:
    """List structural epochs for a facility."""
    from imas_codex.graph.client import GraphClient

    configure_cli_logging("ids", facility=facility)

    with GraphClient() as gc:
        epochs = list(
            gc.query(
                """
                MATCH (se:StructuralEpoch {facility_id: $facility})
                OPTIONAL MATCH (se)<-[:INTRODUCED_IN]-(dn:DataNode)
                WITH se, count(dn) AS node_count
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.last_shot AS last_shot,
                       se.description AS description,
                       se.data_source_name AS source,
                       node_count
                ORDER BY se.first_shot
                """,
                facility=facility,
            )
        )

    if not epochs:
        click.echo(f"No epochs found for {facility}.")
        return

    click.echo(f"{'Epoch':<35} {'Shots':<20} {'Nodes':>6}  {'Description'}")
    click.echo("-" * 90)
    for ep in epochs:
        epoch_id = ep["id"]
        shot_range = str(ep.get("first_shot", "?"))
        if ep.get("last_shot"):
            shot_range += f"-{ep['last_shot']}"
        else:
            shot_range += "+"
        desc = ep.get("description", "")[:40]
        click.echo(f"{epoch_id:<35} {shot_range:<20} {ep['node_count']:>6}  {desc}")
