"""Tools commands - manage development tools installation."""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def tools() -> None:
    """Manage development tools.

    \b
      imas-codex tools list               List available tools
      imas-codex tools check              Check tool installation status
      imas-codex tools install            Install tools
    """
    pass


@tools.command("list")
def tools_list() -> None:
    """List all available development tools."""
    from imas_codex.remote.tools import load_fast_tools

    config = load_fast_tools()

    table = Table(title="Available Development Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Required", style="yellow")

    for _key, tool in sorted(config.all_tools.items()):
        required = "✓" if tool.required else ""
        table.add_row(tool.name, tool.purpose, required)

    console.print(table)


@tools.command("check")
@click.argument("target", default="local")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tools_check(target: str, as_json: bool) -> None:
    """Check tool installation status on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    Examples:
        imas-codex tools check local
        imas-codex tools check tcv
    """
    import json as json_module

    from imas_codex.remote.tools import check_all_tools

    facility = None if target == "local" else target
    status = check_all_tools(facility=facility)

    if as_json:
        click.echo(json_module.dumps(status, indent=2))
        return

    table = Table(title=f"Tool Status: {target}")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Version", style="dim")

    for name, info in sorted(status.get("tools", {}).items()):
        if info.get("available", False):
            status_str = "[green]✓ Installed[/green]"
            version = info.get("version", "")
        else:
            status_str = "[red]✗ Missing[/red]"
            version = ""
        table.add_row(name, status_str, version)

    console.print(table)


@tools.command("install")
@click.argument("target", default="local")
@click.option("--tool", "tool_name", default=None, help="Specific tool to install")
@click.option("--force", is_flag=True, help="Reinstall even if already present")
def tools_install(target: str, tool_name: str | None, force: bool) -> None:
    """Install tools on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').
    If --tool is not specified, installs all required tools.

    Examples:
        imas-codex tools install local
        imas-codex tools install tcv --tool rg
        imas-codex tools install iter --force
    """
    from imas_codex.remote.tools import install_all_tools, install_tool

    facility = None if target == "local" else target

    console.print(f"[bold]Installing tools on {target}...[/bold]")

    if tool_name:
        # Install specific tool
        results = {tool_name: install_tool(tool_name, facility=facility, force=force)}
    else:
        # Install all required tools
        results = install_all_tools(facility=facility, force=force)

    # Display results
    success_count = sum(1 for r in results.values() if r.get("success", False))
    total_count = len(results)

    for name, result in results.items():
        if result.get("success", False):
            console.print(
                f"  [green]✓[/green] {name}: {result.get('message', 'Installed')}"
            )
        else:
            console.print(f"  [red]✗[/red] {name}: {result.get('error', 'Failed')}")

    console.print()
    console.print(f"[bold]Installed: {success_count}/{total_count}[/bold]")

    if success_count < total_count:
        raise SystemExit(1)
