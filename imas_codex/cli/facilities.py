"""Facilities commands - facility configuration management."""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def facilities() -> None:
    """Manage facility configurations.

    \b
      imas-codex facilities list            List all registered facilities
      imas-codex facilities show <name>     Show detailed facility configuration
    """
    pass


@facilities.command("list")
def facilities_list() -> None:
    """List all registered facilities."""
    from imas_codex.discovery.facility import get_facility, list_facilities

    facility_names = list_facilities()

    table = Table(title="Registered Facilities")
    table.add_column("Name", style="cyan")
    table.add_column("SSH Target", style="green")
    table.add_column("Machine", style="yellow")

    for name in facility_names:
        config = get_facility(name)
        ssh_host = config.get("ssh_host", "-")
        ssh_user = config.get("ssh_user")
        ssh_target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
        machine = config.get("machine", "-")
        table.add_row(name, ssh_target, machine)

    if table.row_count == 0:
        console.print("[dim]No facilities registered.[/dim]")
    else:
        console.print(table)


@facilities.command("show")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def facilities_show(name: str, as_json: bool) -> None:
    """Show detailed configuration for a facility.

    This replaces the dynamic facility commands that were previously registered
    at the top level (e.g., `imas-codex tcv`).

    Examples:
        imas-codex facilities show tcv
        imas-codex facilities show iter --json
    """
    import json as json_module

    from imas_codex.discovery.facility import get_facility, list_facilities

    try:
        config = get_facility(name)
    except ValueError as e:
        available = ", ".join(list_facilities())
        raise click.ClickException(
            f"Facility '{name}' not found. Available: {available}"
        ) from e

    if as_json:
        # Output as JSON for scripting
        click.echo(json_module.dumps(config, indent=2, default=str))
    else:
        # Rich table output
        table = Table(title=f"Facility: {name}", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("ID", config.get("id", name))
        table.add_row("Machine", config.get("machine", "-"))
        table.add_row("ssh_host", config.get("ssh_host", "-"))
        table.add_row("ssh_user", config.get("ssh_user", "-"))

        paths = config.get("paths", {})
        if paths:
            for path_name, path_value in paths.items():
                table.add_row(f"Path: {path_name}", str(path_value))

        console.print(table)
        console.print()
        console.print(
            f"[dim]Use 'imas-codex discover sources {name}' to explore this facility.[/dim]"
        )
