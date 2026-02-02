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
    from imas_codex.remote.facilities import FacilityManager

    manager = FacilityManager()

    table = Table(title="Registered Facilities")
    table.add_column("Name", style="cyan")
    table.add_column("SSH Target", style="green")
    table.add_column("Status", style="yellow")

    for name, config in manager.list_facilities().items():
        ssh_target = (
            f"{config.ssh_user}@{config.ssh_host}"
            if config.ssh_user
            else config.ssh_host
        )
        status = "Active" if config.enabled else "Disabled"
        table.add_row(name, ssh_target, status)

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

    from imas_codex.remote.facilities import FacilityManager

    manager = FacilityManager()
    config = manager.get_facility(name)

    if config is None:
        available = ", ".join(manager.list_facilities().keys())
        raise click.ClickException(
            f"Facility '{name}' not found. Available: {available}"
        )

    if as_json:
        # Output as JSON for scripting
        output = {
            "name": name,
            "ssh_host": config.ssh_host,
            "ssh_user": config.ssh_user,
            "enabled": config.enabled,
            "paths": {k: str(v) for k, v in config.paths.items()}
            if config.paths
            else {},
            "infrastructure": config.infrastructure
            if hasattr(config, "infrastructure")
            else {},
        }
        click.echo(json_module.dumps(output, indent=2))
    else:
        # Rich table output
        table = Table(title=f"Facility: {name}", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("SSH Host", config.ssh_host or "-")
        table.add_row("SSH User", config.ssh_user or "-")
        table.add_row("Status", "Enabled" if config.enabled else "Disabled")

        if config.paths:
            for path_name, path_value in config.paths.items():
                table.add_row(f"Path: {path_name}", str(path_value))

        console.print(table)
        console.print()
        console.print(
            f"[dim]Use 'imas-codex discover sources {name}' to explore this facility.[/dim]"
        )
