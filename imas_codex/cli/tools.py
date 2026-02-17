"""Tools commands - manage development tools and Python environment."""

import click
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

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
    from imas_codex.remote.tools import load_remote_tools

    config = load_remote_tools()

    table = Table(title="Available Development Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Binary", style="white")
    table.add_column("Version", style="white")
    table.add_column("Description", style="white")
    table.add_column("Required", style="yellow")

    for _key, tool in sorted(config.all_tools.items()):
        required = "✓" if tool.required else ""
        version = tool.releases.version or "-"
        table.add_row(tool.name, tool.binary, version, tool.purpose, required)

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

    # === Remote Tools Section ===
    console.print("[bold]Remote Tools[/bold]")
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
    "--python-only", is_flag=True, help="Skip remote tools, only setup Python/venv"
)
@click.option(
    "--python", "python_version", default="3.12", help="Python version for venv"
)
@click.option("--force", is_flag=True, help="Reinstall even if already present")
def tools_install(
    target: str,
    tool_name: str | None,
    tools_only: bool,
    python_only: bool,
    python_version: str,
    force: bool,
) -> None:
    """Install development environment on a target.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter').

    By default, installs everything needed for development:
    1. Required remote tools (rg, fd, git, gh, uv)
    2. Python via uv (if system Python < 3.10)
    3. imas-codex venv

    Use --tool to install a specific tool, --tools-only to skip Python setup,
    or --python-only to skip remote tools and only setup Python/venv.

    Examples:
        imas-codex tools install tcv            # Full setup
        imas-codex tools install tcv --tool rg  # Just ripgrep
        imas-codex tools install iter --tools-only   # Skip Python/venv
        imas-codex tools install iter --python-only  # Only Python/venv
        imas-codex tools install jet --python 3.13   # Use Python 3.13
        imas-codex tools install jt60sa --force      # Reinstall everything
    """
    from imas_codex.remote.python import DEFAULT_VENV_PATH, setup_python_env
    from imas_codex.remote.tools import install_tool, load_remote_tools

    # Validate conflicting options
    if tools_only and python_only:
        console.print(
            "[red]Error: --tools-only and --python-only are mutually exclusive[/red]"
        )
        raise SystemExit(1)

    if tool_name and (tools_only or python_only):
        console.print(
            "[red]Error: --tool cannot be used with --tools-only or --python-only[/red]"
        )
        raise SystemExit(1)

    facility = None if target == "local" else target
    has_failures = False
    config = load_remote_tools()

    console.print(f"\n[bold]Installing on {target}...[/bold]\n")

    def format_tool_result(name: str, result: dict, is_required: bool = False) -> Text:
        """Format a single tool result as rich Text."""
        if result.get("success"):
            action = result.get("action", "installed")
            version = result.get("version", "")
            if action in ("already_installed", "system_sufficient"):
                return Text.assemble(
                    ("  ✓ ", "green"),
                    (name, ""),
                    (f": {version} ", ""),
                    ("(present)", "dim"),
                )
            else:
                return Text.assemble(
                    ("  ✓ ", "green"),
                    (name, ""),
                    (f": {version}", ""),
                )
        else:
            error = result.get("error", "failed")
            if is_required:
                return Text.assemble(
                    ("  ✗ ", "red"),
                    (name, ""),
                    (f": {error} ", ""),
                    ("(required)", "red"),
                )
            else:
                return Text.assemble(
                    ("  ○ ", "yellow"),
                    (name, ""),
                    (f": {error} ", ""),
                    ("(optional)", "dim"),
                )

    # === Single tool mode ===
    if tool_name:
        spinner = Spinner("dots", text=f"Installing {tool_name}...")
        with Live(spinner, console=console, transient=True):
            result = install_tool(tool_name, facility=facility, force=force)

        tool = config.get_tool(tool_name)
        is_required = tool.required if tool else False
        console.print(format_tool_result(tool_name, result, is_required))

        if not result.get("success") and is_required:
            raise SystemExit(1)
        console.print()
        return

    # === Full installation mode ===
    # Step 1: Install remote tools with live progress (unless --python-only)
    if python_only:
        console.print("[dim]Skipping remote tools (--python-only)[/dim]\n")
    else:
        console.print("[bold]Remote Tools[/bold]")

        # Track completed tools for live display
        completed_tools: list[tuple[str, dict, bool]] = []
        current_tool: str | None = None
        all_tools = list(config.all_tools.keys())

        def render_tools_progress() -> Group:
            """Render current tools progress."""
            items = []
            # Show completed tools
            for name, result, is_required in completed_tools:
                items.append(format_tool_result(name, result, is_required))
            # Show current tool with spinner
            if current_tool:
                items.append(
                    Text.assemble(
                        ("  ⠋ ", "cyan"),
                        (current_tool, ""),
                        ("...", "dim"),
                    )
                )
            return Group(*items)

        def on_tool_progress(name: str, result: dict) -> None:
            """Callback called after each tool installation."""
            nonlocal current_tool, has_failures
            tool = config.get_tool(name)
            is_required = tool.required if tool else False
            completed_tools.append((name, result, is_required))
            if is_required and not result.get("success"):
                has_failures = True
            # Update current_tool for next iteration
            current_tool = None

        with Live(
            render_tools_progress(), console=console, refresh_per_second=10
        ) as live:
            for tool_key in all_tools:
                current_tool = tool_key
                live.update(render_tools_progress())
                result = install_tool(tool_key, facility=facility, force=force)
                on_tool_progress(tool_key, result)
                live.update(render_tools_progress())

        console.print()

    # Step 2: Python environment (unless --tools-only)
    if tools_only:
        console.print("[dim]Skipping Python setup (--tools-only)[/dim]\n")
    else:
        console.print("[bold]Python Environment[/bold]")

        # Track Python setup steps with live display
        python_steps: list[tuple[str, dict]] = []
        current_step: str | None = None

        def render_python_progress() -> Group:
            """Render Python setup progress."""
            step_names = {
                "install_uv": "uv",
                "install_python": "Python",
                "create_venv": "venv",
            }
            items = []
            for step, result in python_steps:
                step_name = step_names.get(step, step)
                if result.get("success"):
                    action = result.get("action", "")
                    version = result.get("version", result.get("python_version", ""))
                    if action in (
                        "already_installed",
                        "already_available",
                        "already_exists",
                    ):
                        items.append(
                            Text.assemble(
                                ("  ✓ ", "green"),
                                (step_name, ""),
                                (f": {version} ", ""),
                                ("(present)", "dim"),
                            )
                        )
                    else:
                        items.append(
                            Text.assemble(
                                ("  ✓ ", "green"),
                                (step_name, ""),
                                (f": {version}", ""),
                            )
                        )
                else:
                    items.append(
                        Text.assemble(
                            ("  ✗ ", "red"),
                            (step_name, ""),
                            (f": {result.get('error', 'failed')}", ""),
                        )
                    )
            if current_step:
                step_name = step_names.get(current_step, current_step)
                items.append(
                    Text.assemble(
                        ("  ⠋ ", "cyan"),
                        (step_name, ""),
                        ("...", "dim"),
                    )
                )
            return Group(*items)

        # Show initial spinner
        with Live(
            Text.assemble(("  ⠋ ", "cyan"), ("Setting up...", "dim")),
            console=console,
            refresh_per_second=10,
        ) as live:
            python_result = setup_python_env(
                facility=facility,
                python_version=python_version,
                force=force,
            )
            # Process results
            for step_info in python_result.get("steps", []):
                step = step_info.get("step", "unknown")
                step_result = step_info.get("result", {})
                python_steps.append((step, step_result))
                if not step_result.get("success"):
                    has_failures = True
            live.update(render_python_progress())

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


@tools.command("update")
@click.argument("target", default="local")
@click.option(
    "--tool", "tool_name", default=None, help="Update only this specific tool"
)
@click.option(
    "--all", "update_all", is_flag=True, help="Update all tools, not just outdated ones"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be updated without installing"
)
def tools_update(
    target: str,
    tool_name: str | None,
    update_all: bool,
    dry_run: bool,
) -> None:
    """Update outdated tools on a target.

    Compares installed versions against configured versions in
    remote_tools.yaml and upgrades any that are behind.

    TARGET can be 'local' or a facility name (e.g., 'tcv', 'iter', 'jet').

    \b
    Examples:
      imas-codex tools update local           # Update outdated tools locally
      imas-codex tools update jet             # Update outdated tools on JET
      imas-codex tools update iter --tool gh  # Update just gh on ITER
      imas-codex tools update local --all     # Force-update all installed tools
      imas-codex tools update jet --dry-run   # Show what would be updated
    """
    from imas_codex.remote.tools import (
        check_outdated_tools,
        check_tool,
        install_tool,
        load_remote_tools,
    )

    facility = None if target == "local" else target
    config = load_remote_tools()

    console.print(f"\n[bold]Checking tools on {target}...[/bold]\n")

    # Single tool mode
    if tool_name:
        tool = config.get_tool(tool_name)
        if not tool:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
            raise SystemExit(1)

        status = check_tool(tool_name, facility=facility)
        if not status.get("available"):
            console.print(
                f"  {tool_name} is not installed. Use 'tools install' instead."
            )
            raise SystemExit(1)

        installed = status.get("version", "?")
        configured = tool.releases.version

        if not update_all and installed == configured:
            console.print(f"  [green]✓[/green] {tool_name}: {installed} (up to date)")
            console.print()
            return

        console.print(f"  {tool_name}: {installed} → {configured}")

        if dry_run:
            console.print("\n[dim][DRY RUN] No changes made.[/dim]\n")
            return

        spinner = Spinner("dots", text=f"Updating {tool_name}...")
        with Live(spinner, console=console, transient=True):
            result = install_tool(tool_name, facility=facility, force=True)

        if result.get("success"):
            new_version = result.get("version", configured)
            console.print(
                f"  [green]✓[/green] {tool_name}: {installed} → {new_version}"
            )
        else:
            console.print(
                f"  [red]✗[/red] {tool_name}: {result.get('error', 'update failed')}"
            )
            raise SystemExit(1)

        console.print()
        return

    # Multi-tool mode: find outdated tools
    if update_all:
        # Force-update all installed non-system tools
        to_update = []
        for key, tool in config.all_tools.items():
            if tool.system_only or not tool.releases.version:
                continue
            status = check_tool(key, facility=facility)
            if status.get("available"):
                to_update.append(
                    {
                        "key": key,
                        "name": tool.name,
                        "installed": status.get("version", "?"),
                        "configured": tool.releases.version,
                        "required": tool.required,
                    }
                )
    else:
        to_update = check_outdated_tools(facility=facility)

    if not to_update:
        console.print("[green]All tools are up to date.[/green]\n")
        return

    # Display what will be updated
    table = Table(title="Tools to Update")
    table.add_column("Tool", style="cyan")
    table.add_column("Installed", style="yellow")
    table.add_column("Available", style="green")
    table.add_column("Required", style="white")

    for item in to_update:
        required = "✓" if item["required"] else ""
        table.add_row(item["key"], item["installed"], item["configured"], required)

    console.print(table)

    if dry_run:
        console.print("\n[dim][DRY RUN] No changes made.[/dim]\n")
        return

    console.print()

    # Perform updates
    updated = 0
    failed = 0
    for item in to_update:
        key = item["key"]
        spinner = Spinner("dots", text=f"Updating {key}...")
        with Live(spinner, console=console, transient=True):
            result = install_tool(key, facility=facility, force=True)

        if result.get("success"):
            new_version = result.get("version", item["configured"])
            console.print(
                f"  [green]✓[/green] {key}: {item['installed']} → {new_version}"
            )
            updated += 1
        else:
            console.print(
                f"  [red]✗[/red] {key}: {result.get('error', 'update failed')}"
            )
            failed += 1

    console.print()
    if failed:
        console.print(
            f"[bold yellow]Updated {updated}/{len(to_update)} tools "
            f"({failed} failed)[/bold yellow]\n"
        )
        raise SystemExit(1)
    else:
        console.print(f"[bold green]✓ Updated {updated} tool(s)[/bold green]\n")
