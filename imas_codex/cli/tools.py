"""Tools commands - manage development tools and Python environments."""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def tools() -> None:
    """Manage development tools and Python environments.

    \b
    Tool Management:
      imas-codex tools list               List available tools
      imas-codex tools check              Check tool installation status
      imas-codex tools install            Install tools (including uv)

    \b
    Python Environment:
      imas-codex tools python status      Show Python/uv status
      imas-codex tools python install     Install uv + modern Python
      imas-codex tools python venv        Create project venv
      imas-codex tools python setup       Complete environment setup
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


# === Python Environment Subcommands ===


@tools.group("python")
def python_group() -> None:
    """Manage Python environments on local and remote facilities.

    \b
    This enables modern Python (3.10+) development across facilities with
    different system Python versions, avoiding compatibility issues.

    \b
    Strategy:
      - Uses uv (astral-sh/uv) for Python version and venv management
      - uv downloads Python from python-build-standalone (GitHub), not PyPI
      - Works on airgapped facilities if Python is pre-cached

    \b
    Examples:
      imas-codex tools python status tcv
      imas-codex tools python install iter --version 3.12
      imas-codex tools python venv tcv
      imas-codex tools python setup jet  # Complete setup in one command
    """


@python_group.command("status")
@click.argument("target", default="local")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def python_status(target: str, as_json: bool) -> None:
    """Show Python environment status on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    Shows:
    - uv availability and version
    - System Python version
    - uv-managed Python installations
    - Existing imas-codex venv
    - Recommended next action
    """
    import json as json_module

    from imas_codex.remote.python import get_python_status

    facility = None if target == "local" else target
    status = get_python_status(facility=facility)

    if as_json:
        data = {
            "facility": status.facility,
            "uv_available": status.uv_available,
            "uv_version": status.uv_version,
            "system_python": status.system_python.version_string
            if status.system_python
            else None,
            "uv_pythons": [p.version_string for p in status.uv_pythons],
            "has_modern_python": status.has_modern_python,
            "venv_path": status.venv_path,
            "venv_python": status.venv_python.version_string
            if status.venv_python
            else None,
            "recommended_action": status.recommended_action,
        }
        click.echo(json_module.dumps(data, indent=2))
        return

    console.print(f"\n[bold]Python Environment: {target}[/bold]\n")

    # uv status
    if status.uv_available:
        console.print(f"  uv: [green]✓ v{status.uv_version}[/green]")
    else:
        console.print("  uv: [red]✗ Not installed[/red]")

    # System Python
    if status.system_python:
        meets = status.system_python.meets_minimum()
        color = "green" if meets else "yellow"
        console.print(
            f"  System Python: [{color}]{status.system_python.version_string}[/{color}]"
        )
    else:
        console.print("  System Python: [red]Not found[/red]")

    # uv-managed Pythons
    if status.uv_pythons:
        versions = ", ".join(p.version_string for p in status.uv_pythons)
        console.print(f"  uv Pythons: [green]{versions}[/green]")
    elif status.uv_available:
        console.print("  uv Pythons: [dim]None installed[/dim]")

    # venv status
    if status.venv_python:
        console.print(
            f"  imas-codex venv: [green]✓ Python {status.venv_python.version_string}[/green]"
        )
        console.print(f"    Path: [dim]{status.venv_path}[/dim]")
    else:
        console.print("  imas-codex venv: [dim]Not created[/dim]")

    # Recommendation
    console.print()
    action_messages = {
        "ready": "[green]✓ Ready for development[/green]",
        "install_uv": "[yellow]→ Run: imas-codex tools install {target} --tool uv[/yellow]",
        "install_python": "[yellow]→ Run: imas-codex tools python install {target}[/yellow]",
        "create_venv": "[yellow]→ Run: imas-codex tools python venv {target}[/yellow]",
    }
    msg = action_messages.get(
        status.recommended_action, f"[dim]{status.recommended_action}[/dim]"
    )
    console.print(f"  {msg.format(target=target)}")
    console.print()


@python_group.command("install")
@click.argument("target", default="local")
@click.option(
    "--version", "python_version", default="3.12", help="Python version to install"
)
@click.option("--include-uv", is_flag=True, help="Also install uv if missing")
def python_install(target: str, python_version: str, include_uv: bool) -> None:
    """Install Python via uv on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    uv downloads Python from python-build-standalone (GitHub releases),
    not PyPI, so this works on PyPI-airgapped facilities.

    Examples:
        imas-codex tools python install tcv
        imas-codex tools python install iter --version 3.13
        imas-codex tools python install jt60sa --include-uv
    """
    from imas_codex.remote.python import install_python, install_uv
    from imas_codex.remote.tools import check_tool

    facility = None if target == "local" else target

    console.print(f"\n[bold]Installing Python {python_version} on {target}...[/bold]\n")

    # Check/install uv if requested
    if include_uv:
        uv_status = check_tool("uv", facility=facility)
        if not uv_status.get("available"):
            console.print("  Installing uv...")
            uv_result = install_uv(facility=facility)
            if uv_result.get("success"):
                console.print(
                    f"  [green]✓[/green] uv installed: v{uv_result.get('version', 'unknown')}"
                )
            else:
                console.print(
                    f"  [red]✗[/red] uv installation failed: {uv_result.get('error')}"
                )
                raise SystemExit(1)
        else:
            console.print(
                f"  [green]✓[/green] uv already installed: v{uv_status.get('version')}"
            )

    # Install Python
    result = install_python(facility=facility, version=python_version)

    if result.get("success"):
        action = result.get("action", "installed")
        if action == "already_installed":
            console.print(
                f"  [green]✓[/green] Python {result.get('version')} already installed"
            )
        else:
            console.print(
                f"  [green]✓[/green] Python {result.get('version')} installed"
            )
    else:
        console.print(f"  [red]✗[/red] Installation failed: {result.get('error')}")
        raise SystemExit(1)

    console.print()


@python_group.command("venv")
@click.argument("target", default="local")
@click.option(
    "--python", "python_version", default="3.12", help="Python version for venv"
)
@click.option("--path", "venv_path", default=None, help="Custom venv path")
@click.option("--force", is_flag=True, help="Recreate even if exists")
def python_venv(
    target: str, python_version: str, venv_path: str | None, force: bool
) -> None:
    """Create an imas-codex venv on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    Creates a virtual environment using the best available Python
    (preferring uv-managed over system).

    Default path: ~/.local/share/imas-codex/venv

    Examples:
        imas-codex tools python venv tcv
        imas-codex tools python venv iter --python 3.13
        imas-codex tools python venv jet --force
    """
    from imas_codex.remote.python import DEFAULT_VENV_PATH, create_venv

    facility = None if target == "local" else target
    path = venv_path or DEFAULT_VENV_PATH

    console.print(f"\n[bold]Creating venv on {target}...[/bold]\n")

    result = create_venv(
        facility=facility,
        python_version=python_version,
        venv_path=path,
        force=force,
    )

    if result.get("success"):
        action = result.get("action", "created")
        if action == "already_exists":
            console.print(
                f"  [green]✓[/green] venv already exists: Python {result.get('python_version')}"
            )
        else:
            console.print(
                f"  [green]✓[/green] venv created: Python {result.get('python_version')}"
            )
        console.print(f"  Path: [dim]{result.get('venv_path')}[/dim]")
        console.print()
        console.print(f"  Activate with: source {result.get('venv_path')}/bin/activate")
    else:
        console.print(f"  [red]✗[/red] venv creation failed: {result.get('error')}")
        raise SystemExit(1)

    console.print()


@python_group.command("setup")
@click.argument("target", default="local")
@click.option("--python", "python_version", default="3.12", help="Python version")
@click.option("--force", is_flag=True, help="Force reinstall/recreate")
def python_setup(target: str, python_version: str, force: bool) -> None:
    """Complete Python environment setup on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    This performs all necessary steps:
    1. Install uv (if missing)
    2. Install Python via uv (if no modern Python available)
    3. Create imas-codex venv

    Examples:
        imas-codex tools python setup tcv
        imas-codex tools python setup iter --python 3.13
        imas-codex tools python setup jet --force
    """
    from imas_codex.remote.python import setup_python_env

    facility = None if target == "local" else target

    console.print(f"\n[bold]Setting up Python environment on {target}...[/bold]\n")

    result = setup_python_env(
        facility=facility,
        python_version=python_version,
        force=force,
    )

    # Display step results
    for step_info in result.get("steps", []):
        step = step_info.get("step", "unknown")
        step_result = step_info.get("result", {})

        step_names = {
            "install_uv": "uv",
            "install_python": "Python",
            "create_venv": "venv",
        }
        step_name = step_names.get(step, step)

        if step_result.get("success"):
            action = step_result.get("action", "")
            version = step_result.get("version", step_result.get("python_version", ""))
            if action in ("already_installed", "already_available", "already_exists"):
                console.print(
                    f"  [green]✓[/green] {step_name}: {version} (already present)"
                )
            else:
                console.print(f"  [green]✓[/green] {step_name}: {version}")
        else:
            console.print(
                f"  [red]✗[/red] {step_name}: {step_result.get('error', 'failed')}"
            )

    console.print()

    if result.get("success"):
        console.print("[bold green]✓ Setup complete![/bold green]")
        console.print(f"  venv path: {result.get('venv_path')}")
        console.print(f"  Python: {result.get('python_version')}")
        console.print()
        console.print(f"  Activate with: source {result.get('venv_path')}/bin/activate")
    else:
        console.print(f"[bold red]✗ Setup failed:[/bold red] {result.get('error')}")
        raise SystemExit(1)

    console.print()
