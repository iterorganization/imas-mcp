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
    from imas_codex.agentic.tool_installer import ToolInstaller

    installer = ToolInstaller()
    tool_info = installer.get_all_tools()

    table = Table(title="Available Development Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Required", style="yellow")

    for name, info in sorted(tool_info.items()):
        required = "✓" if info.get("required", False) else ""
        table.add_row(name, info.get("description", ""), required)

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

    from imas_codex.agentic.tool_installer import ToolInstaller

    installer = ToolInstaller()

    if target == "local":
        status = installer.check_local()
    else:
        status = installer.check_remote(target)

    if as_json:
        click.echo(json_module.dumps(status, indent=2))
        return

    table = Table(title=f"Tool Status: {target}")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Version", style="dim")

    for name, info in sorted(status.items()):
        if info.get("installed", False):
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
    from imas_codex.agentic.tool_installer import ToolInstaller

    installer = ToolInstaller()

    console.print(f"[bold]Installing tools on {target}...[/bold]")

    if target == "local":
        results = installer.install_local(tool_name=tool_name, force=force)
    else:
        results = installer.install_remote(target, tool_name=tool_name, force=force)

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
