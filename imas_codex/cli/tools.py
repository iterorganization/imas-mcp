"""Tools commands - manage development tools and Python environment."""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group()
def tools() -> None:
    """Manage development tools and Python environment.

    \b
    Commands:
      imas-codex tools status [TARGET]   Show complete environment status
      imas-codex tools install [TARGET]  Install everything (tools + Python + venv)
      imas-codex tools list              List available tools

    \b
    Examples:
      imas-codex tools status tcv        # Check what's installed
      imas-codex tools install tcv       # Install everything needed
      imas-codex tools install tcv --tool rg  # Install just one tool
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


@tools.command("status")
@click.argument("target", default="local")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tools_status(target: str, as_json: bool) -> None:
    """Show complete environment status on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    Shows:
    - Fast CLI tools (rg, fd, git, gh, uv, etc.)
    - Python versions (system and uv-managed)
    - imas-codex venv status
    - Recommended next action

    Examples:
        imas-codex tools status local
        imas-codex tools status tcv
        imas-codex tools status jet --json
    """
    import json as json_module

    from imas_codex.remote.python import get_python_status
    from imas_codex.remote.tools import check_all_tools

    facility = None if target == "local" else target

    # Get tool status
    tool_status = check_all_tools(facility=facility)

    # Get Python status
    python_status = get_python_status(facility=facility)

    if as_json:
        data = {
            "facility": target,
            "tools": tool_status.get("tools", {}),
            "required_tools_ok": tool_status.get("required_ok", False),
            "missing_required": tool_status.get("missing_required", []),
            "python": {
                "uv_available": python_status.uv_available,
                "uv_version": python_status.uv_version,
                "system_python": python_status.system_python.version_string
                if python_status.system_python
                else None,
                "uv_pythons": [p.version_string for p in python_status.uv_pythons],
                "has_modern_python": python_status.has_modern_python,
                "venv_path": python_status.venv_path,
                "venv_python": python_status.venv_python.version_string
                if python_status.venv_python
                else None,
            },
            "ready": (
                tool_status.get("required_ok", False)
                and python_status.venv_python is not None
            ),
        }
        click.echo(json_module.dumps(data, indent=2))
        return

    console.print(f"\n[bold]Environment Status: {target}[/bold]\n")

    # === Fast Tools Section ===
    console.print("[bold]Fast Tools[/bold]")
    for name, info in sorted(tool_status.get("tools", {}).items()):
        if info.get("available", False):
            version = info.get("version", "")
            required_mark = (
                " [yellow](required)[/yellow]" if info.get("required") else ""
            )
            console.print(f"  [green]✓[/green] {name}: {version}{required_mark}")
        else:
            if info.get("required"):
                console.print(f"  [red]✗[/red] {name}: [red]missing (required)[/red]")
            else:
                console.print(f"  [dim]○[/dim] {name}: [dim]not installed[/dim]")

    console.print()

    # === Python Section ===
    console.print("[bold]Python Environment[/bold]")

    # uv status (already in tools, but show Python-specific info)
    if python_status.uv_available:
        console.print(f"  [green]✓[/green] uv: v{python_status.uv_version}")
    else:
        console.print("  [red]✗[/red] uv: [red]not installed[/red]")

    # System Python
    if python_status.system_python:
        meets = python_status.system_python.meets_minimum()
        color = "green" if meets else "yellow"
        status_note = "" if meets else " [yellow](< 3.10)[/yellow]"
        console.print(
            f"  [{color}]{'✓' if meets else '○'}[/{color}] System Python: "
            f"{python_status.system_python.version_string}{status_note}"
        )
    else:
        console.print("  [red]✗[/red] System Python: not found")

    # uv-managed Pythons
    if python_status.uv_pythons:
        versions = ", ".join(p.version_string for p in python_status.uv_pythons)
        console.print(f"  [green]✓[/green] uv Pythons: {versions}")

    # venv status
    if python_status.venv_python:
        console.print(
            f"  [green]✓[/green] imas-codex venv: Python {python_status.venv_python.version_string}"
        )
        console.print(f"      [dim]{python_status.venv_path}[/dim]")
    else:
        console.print("  [dim]○[/dim] imas-codex venv: not created")

    console.print()

    # === Summary ===
    all_required_ok = tool_status.get("required_ok", False)
    venv_ready = python_status.venv_python is not None

    if all_required_ok and venv_ready:
        console.print("[bold green]✓ Ready for development[/bold green]")
        console.print(f"  Activate venv: source {python_status.venv_path}/bin/activate")
    else:
        console.print(
            f"[bold yellow]→ Run: imas-codex tools install {target}[/bold yellow]"
        )
        if not all_required_ok:
            missing = ", ".join(tool_status.get("missing_required", []))
            console.print(f"  [dim]Missing tools: {missing}[/dim]")
        if not venv_ready:
            console.print("  [dim]venv not created[/dim]")

    console.print()


@tools.command("install")
@click.argument("target", default="local")
@click.option(
    "--tool", "tool_name", default=None, help="Install only this specific tool"
)
@click.option("--tools-only", is_flag=True, help="Skip Python/venv setup")
@click.option(
    "--python", "python_version", default="3.12", help="Python version for venv"
)
@click.option("--force", is_flag=True, help="Reinstall even if already present")
def tools_install(
    target: str,
    tool_name: str | None,
    tools_only: bool,
    python_version: str,
    force: bool,
) -> None:
    """Install development environment on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    By default, installs everything needed for development:
    1. Required fast tools (rg, fd, git, gh, uv)
    2. Python via uv (if system Python < 3.10)
    3. imas-codex venv

    Use --tool to install a specific tool, or --tools-only to skip Python setup.

    Examples:
        imas-codex tools install tcv          # Full setup
        imas-codex tools install tcv --tool rg  # Just ripgrep
        imas-codex tools install iter --tools-only  # Skip Python/venv
        imas-codex tools install jet --python 3.13  # Use Python 3.13
        imas-codex tools install jt60sa --force     # Reinstall everything
    """
    from imas_codex.remote.python import DEFAULT_VENV_PATH, setup_python_env
    from imas_codex.remote.tools import install_all_tools, install_tool, load_fast_tools

    facility = None if target == "local" else target
    has_failures = False
    config = load_fast_tools()

    console.print(f"\n[bold]Installing on {target}...[/bold]\n")

    # === Single tool mode ===
    if tool_name:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Installing {tool_name}...", total=None)
            result = install_tool(tool_name, facility=facility, force=force)

        if result.get("success"):
            action = result.get("action", "installed")
            version = result.get("version", "")
            if action in ("already_installed", "system_sufficient"):
                console.print(
                    f"[green]✓[/green] {tool_name}: {version} (already present)"
                )
            else:
                console.print(f"[green]✓[/green] {tool_name}: {version}")
        else:
            console.print(f"[red]✗[/red] {tool_name}: {result.get('error', 'failed')}")
            raise SystemExit(1)
        console.print()
        return

    # === Full installation mode ===
    # Step 1: Install fast tools with progress
    console.print("[bold]Fast Tools[/bold]")

    # Collect tool names for progress
    all_tools = list(config.all_tools.keys())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Installing {len(all_tools)} tools...", total=None)
        tool_results = install_all_tools(facility=facility, force=force)

    for name, result in sorted(tool_results.items()):
        if isinstance(result, dict):
            if result.get("success"):
                action = result.get("action", "installed")
                version = result.get("version", "")
                if action in ("already_installed", "system_sufficient"):
                    console.print(f"  [green]✓[/green] {name}: {version} (present)")
                else:
                    console.print(f"  [green]✓[/green] {name}: {version}")
            else:
                tool = config.get_tool(name)
                is_required = tool.required if tool else False

                if is_required:
                    console.print(
                        f"  [red]✗[/red] {name}: {result.get('error', 'failed')} [red](required)[/red]"
                    )
                    has_failures = True
                else:
                    console.print(
                        f"  [yellow]○[/yellow] {name}: {result.get('error', 'failed')} [dim](optional)[/dim]"
                    )

    console.print()

    # Step 2: Python environment (unless --tools-only)
    if tools_only:
        console.print("[dim]Skipping Python setup (--tools-only)[/dim]\n")
    else:
        console.print("[bold]Python Environment[/bold]")

        # Run with progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Setting up Python environment...", total=None)
            python_result = setup_python_env(
                facility=facility,
                python_version=python_version,
                force=force,
            )

        # Display step results
        for step_info in python_result.get("steps", []):
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
                version = step_result.get(
                    "version", step_result.get("python_version", "")
                )
                if action in (
                    "already_installed",
                    "already_available",
                    "already_exists",
                ):
                    console.print(
                        f"  [green]✓[/green] {step_name}: {version} (present)"
                    )
                else:
                    console.print(f"  [green]✓[/green] {step_name}: {version}")
            else:
                console.print(
                    f"  [red]✗[/red] {step_name}: {step_result.get('error', 'failed')}"
                )
                has_failures = True

        console.print()

    # === Summary ===
    if has_failures:
        console.print("[bold red]✗ Installation incomplete[/bold red]")
        console.print("  Some required components failed to install.")
        raise SystemExit(1)
    else:
        console.print("[bold green]✓ Installation complete[/bold green]")
        if not tools_only:
            console.print(f"  Activate venv: source {DEFAULT_VENV_PATH}/bin/activate")

    console.print()
