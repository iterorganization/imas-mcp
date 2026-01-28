"""CLI interface for IMAS Codex Server."""

import logging
import os
import warnings
from datetime import UTC
from typing import Literal, cast

# Suppress third-party deprecation warnings before importing other modules
# These are upstream issues in Pydantic, LlamaIndex, and Neo4j that we cannot fix
# Use simplefilter for broad suppression since Pydantic's custom warning classes
# (PydanticDeprecatedSince20, etc.) bypass message-based filterwarnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Relying on Driver's destructor.*")

import click  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from imas_codex import __version__, _get_dd_version  # noqa: E402

# Load environment variables from .env file
load_dotenv(override=True)

# Configure logging
logger = logging.getLogger(__name__)


# Create the main CLI group
@click.group(invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    help="Show the imas-codex version and exit.",
)
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """IMAS Codex - AI-enhanced MCP servers for fusion data.

    Use subcommands to start servers or manage data:

    \b
      imas-codex serve imas       Start the IMAS Data Dictionary MCP server
      imas-codex serve agents     Start the Agents MCP server
      imas-codex imas build       Build/update IMAS DD graph
      imas-codex imas status      Show DD graph statistics
      imas-codex facilities list  List configured facilities
    """
    if version:
        click.echo(__version__)
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ============================================================================
# Serve Command Group
# ============================================================================


@main.group()
def serve() -> None:
    """Start MCP servers.

    \b
      imas-codex serve imas     Start the IMAS Data Dictionary server
      imas-codex serve agents   Start the Agents server for facility exploration
    """
    pass


@serve.command("imas")
@click.option(
    "--transport",
    envvar="TRANSPORT",
    default="streamable-http",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol (env: TRANSPORT)",
)
@click.option(
    "--host",
    envvar="HOST",
    default="127.0.0.1",
    help="Host to bind (env: HOST) for HTTP transports",
)
@click.option(
    "--port",
    envvar="PORT",
    default=8000,
    type=int,
    help="Port to bind (env: PORT) for HTTP transports",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress output during server initialization",
)
@click.option(
    "--ids-filter",
    envvar="IDS_FILTER",
    type=str,
    help=(
        "Specific IDS names to include as a space-separated string (env: IDS_FILTER) "
        "e.g., 'core_profiles equilibrium'"
    ),
)
@click.option(
    "--dd-version",
    "dd_version_opt",
    envvar="IMAS_DD_VERSION",
    type=str,
    default=None,
    help=(
        "IMAS Data Dictionary version to use (env: IMAS_DD_VERSION). "
        "Defaults to [tool.imas-codex].default-dd-version in pyproject.toml."
    ),
)
def serve_imas(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
    dd_version_opt: str | None,
) -> None:
    """Start the IMAS Data Dictionary MCP server.

    Examples:
        # Run with default streamable-http transport
        imas-codex serve imas

        # Run with custom host/port
        imas-codex serve imas --host 0.0.0.0 --port 9000

        # Run with stdio transport (for MCP clients)
        imas-codex serve imas --transport stdio

        # Run with debug logging
        imas-codex serve imas --log-level DEBUG

        # Run with specific DD version
        imas-codex serve imas --dd-version 3.42.2
    """
    # Resolve DD version with CLI option taking precedence
    if dd_version_opt:
        os.environ["IMAS_DD_VERSION"] = dd_version_opt

    effective_dd_version = _get_dd_version(dd_version_opt)

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.debug(f"Set logging level to {log_level}")
    logger.debug(f"Starting IMAS MCP server with transport={transport}")

    # Log DD version with source indication
    if dd_version_opt:
        logger.info(f"IMAS DD version: {effective_dd_version} (--dd-version)")
    elif os.environ.get("IMAS_DD_VERSION"):
        logger.info(f"IMAS DD version: {effective_dd_version} (IMAS_DD_VERSION env)")
    else:
        logger.info(f"IMAS DD version: {effective_dd_version} (default)")

    # Parse ids_filter string into a set if provided
    ids_set: set | None = set(ids_filter.split()) if ids_filter else None
    if ids_set:
        logger.info(f"Starting server with IDS filter: {sorted(ids_set)}")
    else:
        logger.info("Starting server with all available IDS")

    # Log transport choice
    match transport:
        case "stdio":
            logger.debug("Using STDIO transport")
        case "streamable-http":
            logger.info(f"Using streamable-http transport on {host}:{port}")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    # For stdio transport, always disable rich output
    use_rich = not no_rich and transport != "stdio"
    if transport == "stdio" and not no_rich:
        logger.info(
            "Disabled rich output for stdio transport to prevent protocol interference"
        )

    from imas_codex.server import Server

    server_instance = Server(use_rich=use_rich, ids_set=ids_set)

    server_instance.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


@serve.command("agents")
@click.option(
    "--transport",
    envvar="AGENTS_TRANSPORT",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol (env: AGENTS_TRANSPORT)",
)
@click.option(
    "--host",
    envvar="AGENTS_HOST",
    default="127.0.0.1",
    help="Host to bind (env: AGENTS_HOST) for HTTP transports",
)
@click.option(
    "--port",
    envvar="AGENTS_PORT",
    default=8001,
    type=int,
    help="Port to bind (env: AGENTS_PORT) for HTTP transports",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
def serve_agents(
    transport: str,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Start the Agents MCP server for remote facility exploration.

    This server provides prompts for orchestrating subagents that explore
    remote fusion facilities via SSH.

    Examples:
        # Run with stdio transport (default, for Cursor)
        imas-codex serve agents

        # Run with HTTP transport
        imas-codex serve agents --transport streamable-http --port 8001
    """
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.info("Starting Agents MCP server")

    match transport:
        case "stdio":
            logger.debug("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    from imas_codex.agentic.server import AgentsServer

    server = AgentsServer()
    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


# ============================================================================
# Facilities Command Group
# ============================================================================


@main.group()
def facilities() -> None:
    """Manage remote facility configurations.

    \b
      imas-codex facilities list         List available facilities
      imas-codex facilities show <name>  Show facility configuration
    """
    pass


@facilities.command("list")
def facilities_list() -> None:
    """List available facility configurations."""
    from imas_codex.discovery import list_facilities

    facility_names = list_facilities()
    if facility_names:
        click.echo("Available facilities:")
        for name in sorted(facility_names):
            click.echo(f"  - {name}")
    else:
        click.echo("No facility configurations found.")


@facilities.command("show")
@click.argument("name")
@click.option(
    "--format", "-f", "fmt", default="yaml", type=click.Choice(["yaml", "json"])
)
@click.option("--public-only", is_flag=True, help="Show only public fields")
def facilities_show(name: str, fmt: str, public_only: bool) -> None:
    """Show configuration for a specific facility."""
    import json

    import yaml

    from imas_codex.discovery import get_facility, get_facility_metadata

    try:
        if public_only:
            data = get_facility_metadata(name)
        else:
            data = get_facility(name)

        if fmt == "json":
            click.echo(json.dumps(data, indent=2, default=str))
        else:
            click.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# ============================================================================
# Tools Command Group
# ============================================================================


@main.group()
def tools() -> None:
    """Manage fast CLI tools on local and remote facilities.

    \b
      imas-codex tools check <facility>    Check tool availability
      imas-codex tools install <facility>  Install missing tools
      imas-codex tools list                List available tools

    Tools are defined in imas_codex/config/fast_tools.yaml.
    Auto-detects local vs remote execution based on hostname.
    """
    pass


@tools.command("check")
@click.argument("facility", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tools_check(facility: str | None, as_json: bool) -> None:
    """Check availability of fast CLI tools.

    \b
    Examples:
      imas-codex tools check           # Check local tools
      imas-codex tools check iter      # Check on ITER (auto-detects local)
      imas-codex tools check tcv      # Check on TCV (via SSH)
    """
    import json as json_mod

    from imas_codex.remote.tools import check_all_tools, is_local_facility

    is_local = is_local_facility(facility)
    location = "locally" if is_local else f"via SSH to {facility}"
    click.echo(f"Checking tools {location}...")

    result = check_all_tools(facility=facility)

    if as_json:
        click.echo(json_mod.dumps(result, indent=2))
        return

    # Pretty print
    click.echo(f"\nFacility: {result['facility']}")
    click.echo(f"Required tools OK: {'✓' if result['required_ok'] else '✗'}")
    click.echo("\nTools:")

    for name, status in result["tools"].items():
        available = status.get("available", False)
        version = status.get("version", "")
        required = "required" if status.get("required") else "optional"
        icon = "✓" if available else "✗"
        version_str = f" ({version})" if version else ""
        click.echo(f"  {icon} {name}{version_str} [{required}]")

    if result.get("missing_required"):
        click.echo(f"\n⚠ Missing required: {', '.join(result['missing_required'])}")
        click.echo("  Run: imas-codex tools install " + (facility or ""))


@tools.command("install")
@click.argument("facility", required=False)
@click.option("--tool", "tool_name", help="Install a specific tool (e.g., gh, rg)")
@click.option(
    "--required-only", is_flag=True, help="Only install required tools (rg, fd, gh)"
)
@click.option("--force", is_flag=True, help="Reinstall even if already present")
@click.option("--dry-run", is_flag=True, help="Show what would be installed")
def tools_install(
    facility: str | None,
    tool_name: str | None,
    required_only: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Install fast CLI tools on a facility.

    \b
    Examples:
      imas-codex tools install              # Install locally
      imas-codex tools install iter         # Install on ITER (auto-detects local)
      imas-codex tools install tcv         # Install on TCV (via SSH)
      imas-codex tools install --tool gh    # Install only gh
      imas-codex tools install iter --tool gh  # Install gh on ITER
      imas-codex tools install --dry-run    # Show what would be installed
    """
    from imas_codex.remote.tools import (
        check_all_tools,
        detect_architecture,
        install_all_tools,
        install_tool,
        is_local_facility,
        load_fast_tools,
    )

    is_local = is_local_facility(facility)
    location = "locally" if is_local else f"via SSH to {facility}"

    # Single tool installation
    if tool_name:
        config = load_fast_tools()
        if tool_name not in config.all_tools:
            click.echo(f"Unknown tool: {tool_name}")
            click.echo(f"Available: {', '.join(config.all_tools.keys())}")
            raise SystemExit(1)

        if dry_run:
            tool = config.get_tool(tool_name)
            cmd = tool.get_install_command(detect_architecture(facility=facility))
            click.echo(f"Dry run - would install {tool_name} {location}:")
            click.echo(f"  {cmd}")
            return

        click.echo(f"Installing {tool_name} {location}...")
        result = install_tool(tool_name, facility=facility, force=force)

        if result.get("success"):
            if result.get("action") == "already_installed":
                click.echo(
                    f"• {tool_name} already installed (v{result.get('version')})"
                )
            else:
                click.echo(f"✓ Installed {tool_name} (v{result.get('version')})")
        else:
            click.echo(f"✗ Failed: {result.get('error')}")
            raise SystemExit(1)
        return

    if dry_run:
        click.echo(f"Dry run - would install tools {location}:")
        click.echo(f"Architecture: {detect_architecture(facility=facility)}")

        # Check what's missing
        status = check_all_tools(facility=facility)
        config = load_fast_tools()

        tools_to_check = config.required if required_only else config.all_tools
        for key, tool in tools_to_check.items():
            tool_status = status["tools"].get(key, {})
            if force or not tool_status.get("available"):
                cmd = tool.get_install_command(detect_architecture(facility=facility))
                click.echo(f"\n{key}:")
                click.echo(f"  {cmd}")
        return

    click.echo(f"Installing tools {location}...")
    result = install_all_tools(
        facility=facility, required_only=required_only, force=force
    )

    if result.get("installed"):
        click.echo(f"✓ Installed: {', '.join(result['installed'])}")
    if result.get("already_present"):
        click.echo(f"• Already present: {', '.join(result['already_present'])}")
    if result.get("failed"):
        click.echo("✗ Failed:")
        for fail in result["failed"]:
            click.echo(f"  - {fail['tool']}: {fail['error']}")

    if result.get("success"):
        click.echo("\n✓ All tools ready")
    else:
        click.echo("\n⚠ Some tools failed to install")
        raise SystemExit(1)


@tools.command("list")
def tools_list() -> None:
    """List available fast CLI tools."""
    from imas_codex.remote.tools import load_fast_tools

    config = load_fast_tools()

    click.echo("Required tools:")
    for key, tool in config.required.items():
        click.echo(f"  {key}: {tool.purpose}")
        if tool.fallback:
            click.echo(f"       fallback: {tool.fallback}")

    click.echo("\nOptional tools:")
    for key, tool in config.optional.items():
        click.echo(f"  {key}: {tool.purpose}")
        if tool.fallback:
            click.echo(f"       fallback: {tool.fallback}")


# ============================================================================
# Neo4j Command Group
# ============================================================================


@main.group()
def neo4j() -> None:
    """Manage Neo4j graph database for knowledge graph.

    \b
      imas-codex neo4j start   Start Neo4j server via Apptainer
      imas-codex neo4j stop    Stop Neo4j server
      imas-codex neo4j status  Check Neo4j server status
      imas-codex neo4j shell   Open Cypher shell
      imas-codex neo4j dump    Export graph to dump file
      imas-codex neo4j push    Push graph artifact to GHCR
      imas-codex neo4j pull    Pull graph artifact from GHCR
      imas-codex neo4j load    Load graph dump into database
      imas-codex neo4j service Manage as systemd user service
    """
    pass


@neo4j.command("start")
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Data directory (env: NEO4J_DATA)",
)
@click.option(
    "--password",
    envvar="NEO4J_PASSWORD",
    default="imas-codex",
    help="Neo4j password (env: NEO4J_PASSWORD)",
)
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
def neo4j_start(
    image: str | None,
    data_dir: str | None,
    password: str,
    foreground: bool,
) -> None:
    """Start Neo4j server via Apptainer.

    Examples:
        # Start with defaults
        imas-codex neo4j start

        # Run in foreground (for debugging)
        imas-codex neo4j start -f

        # Custom data directory
        imas-codex neo4j start --data-dir /path/to/data
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    # On Windows/Mac, use docker compose instead of Apptainer
    if platform.system() in ("Windows", "Darwin"):
        click.echo("This command uses Apptainer (for Linux HPC).", err=True)
        click.echo("On Windows/Mac, use Docker instead:", err=True)
        click.echo("  docker compose up -d neo4j", err=True)
        raise SystemExit(1)

    if not shutil.which("apptainer"):
        click.echo("Error: apptainer not found in PATH", err=True)
        raise SystemExit(1)

    # Resolve paths
    home = Path.home()
    image_path = (
        Path(image) if image else home / "apptainer" / "neo4j_2025.11-community.sif"
    )
    data_path = (
        Path(data_dir)
        if data_dir
        else home / ".local" / "share" / "imas-codex" / "neo4j"
    )

    if not image_path.exists():
        click.echo(f"Error: Neo4j image not found at {image_path}", err=True)
        click.echo(
            "Pull it with: apptainer pull docker://neo4j:2025.11-community", err=True
        )
        raise SystemExit(1)

    # Check if already running
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:7474/", timeout=2)
        click.echo("Neo4j is already running on port 7474")
        return
    except Exception:
        pass

    # Create data directories
    for subdir in ["data", "logs", "conf", "import"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_path}/data:/data",
        "--bind",
        f"{data_path}/logs:/logs",
        "--bind",
        f"{data_path}/import:/import",
        "--writable-tmpfs",
        "--env",
        f"NEO4J_AUTH=neo4j/{password}",
        str(image_path),
        "neo4j",
        "console",
    ]

    click.echo(f"Starting Neo4j from {image_path}")
    click.echo(f"Data directory: {data_path}")

    if foreground:
        # Run in foreground
        subprocess.run(cmd)
    else:
        # Run in background
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Write PID file
        pid_file = data_path / "neo4j.pid"
        pid_file.write_text(str(proc.pid))

        click.echo(f"Neo4j starting in background (PID: {proc.pid})")
        click.echo("Waiting for server...")

        import json as json_module
        import time
        import urllib.request

        for _ in range(30):
            try:
                urllib.request.urlopen("http://localhost:7474/", timeout=2)

                # Check if password needs to be changed (fresh install)
                try:
                    # Try with default password
                    req = urllib.request.Request(
                        "http://localhost:7474/db/system/tx/commit",
                        data=json_module.dumps(
                            {
                                "statements": [
                                    {
                                        "statement": f'ALTER CURRENT USER SET PASSWORD FROM "neo4j" TO "{password}"'
                                    }
                                ]
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    req.add_header(
                        "Authorization",
                        "Basic "
                        + __import__("base64").b64encode(b"neo4j:neo4j").decode(),
                    )
                    urllib.request.urlopen(req, timeout=5)
                    click.echo("Initial password changed successfully")
                except Exception:
                    # Password already changed or different auth
                    pass

                click.echo("Neo4j ready at http://localhost:7474")
                click.echo("Bolt: bolt://localhost:7687")
                click.echo(f"Credentials: neo4j / {password}")
                return
            except Exception:
                time.sleep(1)

        click.echo(
            "Warning: Neo4j may still be starting. Check with: imas-codex neo4j status"
        )


@neo4j.command("stop")
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Data directory (env: NEO4J_DATA)",
)
def neo4j_stop(data_dir: str | None) -> None:
    """Stop Neo4j server."""
    import signal
    import subprocess
    from pathlib import Path

    home = Path.home()
    data_path = (
        Path(data_dir)
        if data_dir
        else home / ".local" / "share" / "imas-codex" / "neo4j"
    )
    pid_file = data_path / "neo4j.pid"

    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Sent SIGTERM to Neo4j (PID: {pid})")
            pid_file.unlink()
        except ProcessLookupError:
            click.echo("Neo4j process not found (stale PID file)")
            pid_file.unlink()
    else:
        # Try pkill as fallback
        result = subprocess.run(
            ["pkill", "-f", "neo4j.*console"],
            capture_output=True,
        )
        if result.returncode == 0:
            click.echo("Neo4j stopped")
        else:
            click.echo("Neo4j not running")


@neo4j.command("status")
def neo4j_status() -> None:
    """Check Neo4j server status."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:7474/", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            click.echo("Neo4j is running")
            click.echo(f"  Version: {data.get('neo4j_version', 'unknown')}")
            click.echo(f"  Edition: {data.get('neo4j_edition', 'unknown')}")
            click.echo(f"  Bolt: {data.get('bolt_direct', 'unknown')}")
    except Exception:
        click.echo("Neo4j is not responding on port 7474")


@neo4j.command("shell")
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--password",
    envvar="NEO4J_PASSWORD",
    default="imas-codex",
    help="Neo4j password (env: NEO4J_PASSWORD)",
)
def neo4j_shell(image: str | None, password: str) -> None:
    """Open Cypher shell to Neo4j."""
    import subprocess
    from pathlib import Path

    home = Path.home()
    image_path = (
        Path(image) if image else home / "apptainer" / "neo4j_2025.11-community.sif"
    )

    if not image_path.exists():
        click.echo(f"Error: Neo4j image not found at {image_path}", err=True)
        raise SystemExit(1)

    cmd = [
        "apptainer",
        "exec",
        "--writable-tmpfs",
        str(image_path),
        "cypher-shell",
        "-u",
        "neo4j",
        "-p",
        password,
    ]

    subprocess.run(cmd)


@neo4j.command("dump")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: imas-codex-graph-{version}.dump)",
)
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Neo4j data directory (env: NEO4J_DATA)",
)
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
def neo4j_dump(output: str | None, data_dir: str | None, image: str | None) -> None:
    """Export graph database to a dump file.

    Creates a Neo4j database dump that can be pushed to GHCR as an artifact.
    Neo4j must be stopped before dumping.

    Examples:
        # Dump with auto-generated filename
        imas-codex neo4j dump

        # Dump to specific file
        imas-codex neo4j dump -o graph-v1.0.0.dump
    """
    import shutil
    import subprocess
    from pathlib import Path

    home = Path.home()
    data_path = (
        Path(data_dir)
        if data_dir
        else home / ".local" / "share" / "imas-codex" / "neo4j"
    )
    image_path = (
        Path(image) if image else home / "apptainer" / "neo4j_2025.11-community.sif"
    )

    if not image_path.exists():
        click.echo(f"Error: Neo4j image not found at {image_path}", err=True)
        raise SystemExit(1)

    if not shutil.which("apptainer"):
        click.echo("Error: apptainer not found in PATH", err=True)
        raise SystemExit(1)

    # Check if Neo4j is running
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:7474/", timeout=2)
        click.echo("Error: Neo4j is running. Stop it first: imas-codex neo4j stop")
        raise SystemExit(1)
    except Exception:
        pass

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"imas-codex-graph-{__version__}.dump")

    # Create dumps directory
    dumps_dir = data_path / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    # Run neo4j-admin dump
    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_path}/data:/data",
        "--bind",
        f"{dumps_dir}:/dumps",
        "--writable-tmpfs",
        str(image_path),
        "neo4j-admin",
        "database",
        "dump",
        "neo4j",
        "--to-path=/dumps",
        "--overwrite-destination=true",
    ]

    click.echo("Dumping graph database...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        click.echo(f"Error dumping database: {result.stderr}", err=True)
        raise SystemExit(1)

    # Move dump to output location
    dump_file = dumps_dir / "neo4j.dump"
    if dump_file.exists():
        import shutil as shutil_mod

        shutil_mod.move(str(dump_file), str(output_path))
        click.echo(f"Graph dumped to: {output_path}")
        click.echo(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        click.echo("Error: Dump file not created", err=True)
        raise SystemExit(1)


@neo4j.command("push")
@click.argument("version")
@click.option(
    "--dump-file",
    "-f",
    type=click.Path(exists=True),
    help="Dump file to push (default: imas-codex-graph-{version}.dump)",
)
@click.option(
    "--registry",
    default="ghcr.io/iterorganization",
    help="Container registry (default: ghcr.io/iterorganization)",
)
@click.option(
    "--token",
    envvar="GHCR_TOKEN",
    help="GHCR authentication token (env: GHCR_TOKEN)",
)
def neo4j_push(
    version: str,
    dump_file: str | None,
    registry: str,
    token: str | None,
) -> None:
    """Push graph dump to GHCR as an OCI artifact.

    VERSION should match the release tag (e.g., v1.0.0).

    Examples:
        # Push with version tag
        imas-codex neo4j push v1.0.0

        # Push specific dump file
        imas-codex neo4j push v1.0.0 -f custom-graph.dump
    """
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("oras"):
        click.echo("Error: oras not found in PATH", err=True)
        click.echo(
            "Install with: curl -LO https://github.com/oras-project/oras/releases/download/v1.2.0/oras_1.2.0_linux_amd64.tar.gz"
        )
        raise SystemExit(1)

    # Determine dump file
    if dump_file:
        dump_path = Path(dump_file)
    else:
        # Try version-specific file first, then generic
        dump_path = Path(f"imas-codex-graph-{version.lstrip('v')}.dump")
        if not dump_path.exists():
            dump_path = Path(f"imas-codex-graph-{__version__}.dump")

    if not dump_path.exists():
        click.echo(f"Error: Dump file not found: {dump_path}", err=True)
        click.echo("Run 'imas-codex neo4j dump' first")
        raise SystemExit(1)

    # Login to GHCR if token provided
    if token:
        login_cmd = ["oras", "login", "ghcr.io", "-u", "token", "--password-stdin"]
        login_proc = subprocess.run(
            login_cmd,
            input=token,
            text=True,
            capture_output=True,
        )
        if login_proc.returncode != 0:
            click.echo(f"Error logging in to GHCR: {login_proc.stderr}", err=True)
            raise SystemExit(1)
        click.echo("Logged in to GHCR")

    # Push artifact
    artifact_ref = f"{registry}/imas-codex-graph:{version}"
    push_cmd = [
        "oras",
        "push",
        artifact_ref,
        f"{dump_path}:application/vnd.neo4j.dump",
        "--annotation",
        f"org.opencontainers.image.version={version}",
        "--annotation",
        "org.opencontainers.image.source=https://github.com/iterorganization/imas-codex",
        "--annotation",
        f"io.imas-codex.schema-version={__version__}",
    ]

    click.echo(f"Pushing {dump_path} to {artifact_ref}...")
    result = subprocess.run(push_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        click.echo(f"Error pushing artifact: {result.stderr}", err=True)
        raise SystemExit(1)

    click.echo(f"Successfully pushed: {artifact_ref}")

    # Also tag as latest
    tag_cmd = ["oras", "tag", artifact_ref, f"{registry}/imas-codex-graph:latest"]
    subprocess.run(tag_cmd, capture_output=True)
    click.echo(f"Tagged as: {registry}/imas-codex-graph:latest")


@neo4j.command("pull")
@click.option(
    "--version",
    "-v",
    "version",
    default="latest",
    help="Version tag to pull (default: latest)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: imas-codex-graph.dump)",
)
@click.option(
    "--registry",
    default="ghcr.io/iterorganization",
    help="Container registry (default: ghcr.io/iterorganization)",
)
@click.option(
    "--token",
    envvar="GHCR_TOKEN",
    help="GHCR authentication token (env: GHCR_TOKEN)",
)
def neo4j_pull(
    version: str,
    output: str | None,
    registry: str,
    token: str | None,
) -> None:
    """Pull graph dump from GHCR.

    Examples:
        # Pull latest
        imas-codex neo4j pull

        # Pull specific version
        imas-codex neo4j pull -v v1.0.0

        # Pull to specific file
        imas-codex neo4j pull -o graph.dump
    """
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("oras"):
        click.echo("Error: oras not found in PATH", err=True)
        raise SystemExit(1)

    # Login to GHCR if token provided
    if token:
        login_cmd = ["oras", "login", "ghcr.io", "-u", "token", "--password-stdin"]
        login_proc = subprocess.run(
            login_cmd,
            input=token,
            text=True,
            capture_output=True,
        )
        if login_proc.returncode != 0:
            click.echo(f"Error logging in to GHCR: {login_proc.stderr}", err=True)
            raise SystemExit(1)

    # Determine output path
    output_path = Path(output) if output else Path("imas-codex-graph.dump")

    # Pull artifact
    artifact_ref = f"{registry}/imas-codex-graph:{version}"

    click.echo(f"Pulling {artifact_ref}...")
    pull_cmd = ["oras", "pull", artifact_ref, "-o", str(output_path.parent)]
    result = subprocess.run(pull_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        click.echo(f"Error pulling artifact: {result.stderr}", err=True)
        raise SystemExit(1)

    click.echo(f"Successfully pulled to: {output_path}")
    click.echo(f"To load: imas-codex neo4j load {output_path}")


@neo4j.command("load")
@click.argument("dump_file", type=click.Path(exists=True))
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Neo4j data directory (env: NEO4J_DATA)",
)
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing database",
)
def neo4j_load(
    dump_file: str,
    data_dir: str | None,
    image: str | None,
    force: bool,
) -> None:
    """Load graph dump into Neo4j.

    Neo4j must be stopped before loading.

    Examples:
        # Load dump file
        imas-codex neo4j load graph.dump

        # Force overwrite existing data
        imas-codex neo4j load graph.dump --force
    """
    import shutil
    import subprocess
    from pathlib import Path

    home = Path.home()
    data_path = (
        Path(data_dir)
        if data_dir
        else home / ".local" / "share" / "imas-codex" / "neo4j"
    )
    image_path = (
        Path(image) if image else home / "apptainer" / "neo4j_2025.11-community.sif"
    )
    dump_path = Path(dump_file)

    if not image_path.exists():
        click.echo(f"Error: Neo4j image not found at {image_path}", err=True)
        raise SystemExit(1)

    if not shutil.which("apptainer"):
        click.echo("Error: apptainer not found in PATH", err=True)
        raise SystemExit(1)

    # Check if Neo4j is running
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:7474/", timeout=2)
        click.echo("Error: Neo4j is running. Stop it first: imas-codex neo4j stop")
        raise SystemExit(1)
    except Exception:
        pass

    # Create data directories
    for subdir in ["data", "logs", "dumps"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    # Copy dump to dumps directory
    dumps_dir = data_path / "dumps"
    target_dump = dumps_dir / "neo4j.dump"
    import shutil as shutil_mod

    shutil_mod.copy(str(dump_path), str(target_dump))

    # Build load command
    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_path}/data:/data",
        "--bind",
        f"{dumps_dir}:/dumps",
        "--writable-tmpfs",
        str(image_path),
        "neo4j-admin",
        "database",
        "load",
        "neo4j",
        "--from-path=/dumps",
    ]

    if force:
        cmd.append("--overwrite-destination=true")

    click.echo(f"Loading {dump_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        click.echo(f"Error loading database: {result.stderr}", err=True)
        raise SystemExit(1)

    click.echo("Graph loaded successfully")
    click.echo("Start Neo4j with: imas-codex neo4j start")


@neo4j.command("service")
@click.argument("action", type=click.Choice(["install", "uninstall", "status"]))
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Data directory (env: NEO4J_DATA)",
)
@click.option(
    "--password",
    envvar="NEO4J_PASSWORD",
    default="imas-codex",
    help="Neo4j password (env: NEO4J_PASSWORD)",
)
def neo4j_service(
    action: str,
    image: str | None,
    data_dir: str | None,
    password: str,
) -> None:
    """Manage Neo4j as a systemd user service.

    This creates a persistent user-level systemd service that starts
    Neo4j on login and survives reboots.

    Examples:
        # Install and enable the service
        imas-codex neo4j service install

        # Check service status
        imas-codex neo4j service status

        # Remove the service
        imas-codex neo4j service uninstall

    After install, use systemctl to control:
        systemctl --user start imas-codex-neo4j
        systemctl --user stop imas-codex-neo4j
        systemctl --user restart imas-codex-neo4j
        journalctl --user -u imas-codex-neo4j -f
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    # Check platform
    if platform.system() != "Linux":
        click.echo("Error: systemd services only supported on Linux", err=True)
        click.echo("On Windows/Mac, use Docker instead:", err=True)
        click.echo("  docker compose up -d neo4j", err=True)
        raise SystemExit(1)

    # Check systemctl
    if not shutil.which("systemctl"):
        click.echo("Error: systemctl not found", err=True)
        click.echo("systemd is required for the service command", err=True)
        raise SystemExit(1)

    # Check apptainer
    if not shutil.which("apptainer"):
        click.echo("Error: apptainer not found in PATH", err=True)
        raise SystemExit(1)

    home = Path.home()
    service_dir = home / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-neo4j.service"

    image_path = (
        Path(image) if image else home / "apptainer" / "neo4j_2025.11-community.sif"
    )
    data_path = (
        Path(data_dir)
        if data_dir
        else home / ".local" / "share" / "imas-codex" / "neo4j"
    )
    apptainer_path = shutil.which("apptainer")

    if action == "install":
        if not image_path.exists():
            click.echo(f"Error: Neo4j image not found at {image_path}", err=True)
            click.echo(
                "Pull it with: apptainer pull docker://neo4j:2025.11-community",
                err=True,
            )
            raise SystemExit(1)

        # Create directories
        service_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "logs", "conf", "import"]:
            (data_path / subdir).mkdir(parents=True, exist_ok=True)

        # Create service file
        service_content = f"""[Unit]
Description=Neo4j Graph Database (IMAS Codex)
After=network.target

[Service]
Type=simple
ExecStart={apptainer_path} exec \\
    --bind {data_path}/data:/data \\
    --bind {data_path}/logs:/logs \\
    --bind {data_path}/import:/import \\
    --writable-tmpfs \\
    --env NEO4J_AUTH=neo4j/{password} \\
    {image_path} \\
    neo4j console
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        click.echo(f"Created {service_file}")

        # Reload and enable
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "imas-codex-neo4j"], check=True
        )

        click.echo("Service installed and enabled")
        click.echo()
        click.echo("Control the service with:")
        click.echo("  systemctl --user start imas-codex-neo4j")
        click.echo("  systemctl --user stop imas-codex-neo4j")
        click.echo("  systemctl --user status imas-codex-neo4j")
        click.echo("  journalctl --user -u imas-codex-neo4j -f")
        click.echo()
        click.echo("The service will auto-start on login.")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return

        # Stop and disable
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-neo4j"],
            capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-neo4j"],
            capture_output=True,
        )

        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

        click.echo("Service uninstalled")
        click.echo("Data remains at:", data_path)

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            click.echo("Install with: imas-codex neo4j service install")
            return

        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-neo4j"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        if result.returncode != 0 and result.stderr:
            click.echo(result.stderr, err=True)


# ============================================================================
# Ingest Command Group
# ============================================================================


@main.group()
def ingest() -> None:
    """Ingest code examples from remote facilities.

    \b
      imas-codex ingest run <facility>   Process discovered SourceFile nodes
      imas-codex ingest status <facility> Show queue statistics
      imas-codex ingest list <facility>   List discovered files
    """
    pass


@ingest.command("run")
@click.argument("facility")
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Maximum files to process (default: all discovered files)",
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimum interest score threshold (default: 0.0)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-ingest files even if already present",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without ingesting",
)
def ingest_run(
    facility: str,
    limit: int | None,
    min_score: float,
    force: bool,
    dry_run: bool,
) -> None:
    """Process discovered SourceFile nodes for a facility.

    Scouts discover files for ingestion using the queue_source_files MCP tool.
    This command fetches those files, generates embeddings, and creates
    CodeExample nodes with searchable chunks.

    Examples:
        # Process all discovered files
        imas-codex ingest run tcv

        # Process only high-priority files
        imas-codex ingest run tcv --min-score 0.7

        # Limit to 100 files
        imas-codex ingest run tcv -n 100

        # Preview what would be processed
        imas-codex ingest run tcv --dry-run
    """
    import asyncio

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from imas_codex.code_examples import get_pending_files, ingest_code_files

    console = Console()

    # Get pending files (discovered, not yet ingested)
    console.print(f"[cyan]Fetching discovered files for {facility}...[/cyan]")
    query_limit = limit if limit is not None else 10000  # Large number for "all"
    pending = get_pending_files(
        facility, limit=query_limit, min_interest_score=min_score
    )

    if not pending:
        console.print("[yellow]No discovered files awaiting ingestion.[/yellow]")
        console.print(
            "Scouts can discover files using the queue_source_files MCP tool."
        )
        return

    console.print(f"[green]Found {len(pending)} discovered files[/green]")

    if dry_run:
        console.print("\n[cyan]Files that would be processed:[/cyan]")
        for i, f in enumerate(pending[:20], 1):
            score = f.get("interest_score", 0.5)
            console.print(f"  {i}. [{score:.2f}] {f['path']}")
        if len(pending) > 20:
            console.print(f"  ... and {len(pending) - 20} more")
        return

    # Run ingestion with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(pending))

        def progress_callback(current: int, total: int, message: str) -> None:
            # Update total if pipeline reports a different count (e.g., after skip)
            progress.update(
                task, completed=current, total=total, description=message[:50]
            )

        try:
            stats = asyncio.run(
                ingest_code_files(
                    facility=facility,
                    remote_paths=None,  # Use graph queue
                    progress_callback=progress_callback,
                    force=force,
                    limit=limit,
                )
            )

            # Final update to ensure 100%
            progress.update(task, completed=stats["files"] + stats["skipped"])

        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]")
            raise SystemExit(1) from e

    # Print summary
    console.print("\n[green]Ingestion complete![/green]")
    console.print(f"  Files processed: {stats['files']}")
    console.print(f"  Chunks created:  {stats['chunks']}")
    console.print(f"  IDS references:  {stats['ids_found']}")
    console.print(f"  MDSplus paths:   {stats['mdsplus_paths']}")
    console.print(f"  TreeNodes linked: {stats['tree_nodes_linked']}")
    console.print(f"  Skipped:         {stats['skipped']}")


@ingest.command("status")
@click.argument("facility")
def ingest_status(facility: str) -> None:
    """Show queue statistics for a facility.

    Examples:
        imas-codex ingest status tcv
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.code_examples import get_queue_stats

    console = Console()
    stats = get_queue_stats(facility)

    if not stats:
        console.print(f"[yellow]No SourceFile nodes for {facility}[/yellow]")
        return

    table = Table(title=f"SourceFile Queue: {facility}")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="green")

    total = 0
    for status, count in sorted(stats.items()):
        table.add_row(status, str(count))
        total += count

    table.add_row("─" * 10, "─" * 5)
    table.add_row("Total", str(total), style="bold")

    console.print(table)


@ingest.command("queue")
@click.argument("facility")
@click.argument("paths", nargs=-1)
@click.option(
    "--from-file",
    "-f",
    "from_file",
    type=click.Path(exists=True),
    help="Read file paths from a text file (one per line)",
)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read file paths from stdin",
)
@click.option(
    "--interest-score",
    "-s",
    default=0.5,
    type=float,
    help="Interest score for all files (0.0-1.0, default: 0.5)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be queued without making changes",
)
def ingest_queue(
    facility: str,
    paths: tuple[str, ...],
    from_file: str | None,
    stdin: bool,
    interest_score: float,
    dry_run: bool,
) -> None:
    """Discover source files for ingestion.

    Accepts paths as arguments, from a file, or from stdin. Creates
    SourceFile nodes with status='discovered'. Already-discovered or ingested
    files are skipped automatically.

    Examples:
        # Discover paths directly (LLM-friendly)
        imas-codex ingest queue tcv /path/a.py /path/b.py /path/c.py

        # Discover from file (for large batches)
        imas-codex ingest queue tcv -f files.txt

        # Discover from stdin (pipe from rg)
        ssh tcv 'rg -l "IMAS" /home' | imas-codex ingest queue tcv --stdin

        # Set priority score
        imas-codex ingest queue tcv /path/a.py -s 0.9

        # Preview
        imas-codex ingest queue tcv /path/a.py --dry-run
    """
    import sys
    from pathlib import Path

    from rich.console import Console

    from imas_codex.code_examples import queue_source_files

    console = Console()

    # Read file paths from arguments, file, or stdin
    path_list: list[str] = []
    if paths:
        path_list = list(paths)
    elif from_file:
        path_list = Path(from_file).read_text().strip().splitlines()
    elif stdin:
        path_list = sys.stdin.read().strip().splitlines()
    else:
        console.print(
            "[red]Error: Provide paths as arguments, --from-file, or --stdin[/red]"
        )
        raise SystemExit(1)

    # Filter empty lines and comments
    path_list = [
        p.strip() for p in path_list if p.strip() and not p.strip().startswith("#")
    ]

    if not path_list:
        console.print("[yellow]No file paths provided[/yellow]")
        return

    console.print(f"[cyan]Discovering {len(path_list)} files for {facility}...[/cyan]")

    if dry_run:
        console.print("\n[cyan]Files that would be discovered:[/cyan]")
        for i, path in enumerate(path_list[:20], 1):
            console.print(f"  {i}. {path}")
        if len(path_list) > 20:
            console.print(f"  ... and {len(path_list) - 20} more")
        console.print(f"\n[dim]Interest score: {interest_score}[/dim]")
        return

    result = queue_source_files(
        facility=facility,
        file_paths=path_list,
        interest_score=interest_score,
        discovered_by="cli",
    )

    console.print(f"[green]✓ Discovered: {result['discovered']}[/green]")
    console.print(
        f"[yellow]↷ Skipped: {result['skipped']} (already discovered/ingested)[/yellow]"
    )
    if result["errors"]:
        for err in result["errors"]:
            console.print(f"[red]✗ Error: {err}[/red]")

    console.print(
        f"\n[dim]Run ingestion: imas-codex ingest run {facility} -n {min(result['discovered'], 500)}[/dim]"
    )


@ingest.command("list")
@click.argument("facility")
@click.option(
    "--status",
    "-s",
    default="discovered",
    type=click.Choice(["discovered", "ingested", "failed", "stale", "all"]),
    help="Filter by status (default: discovered)",
)
@click.option(
    "--limit",
    "-n",
    default=50,
    type=int,
    help="Maximum files to show (default: 50)",
)
def ingest_list(facility: str, status: str, limit: int) -> None:
    """List SourceFile nodes for a facility.

    Examples:
        # List discovered files
        imas-codex ingest list tcv

        # List failed files
        imas-codex ingest list tcv -s failed

        # List all files
        imas-codex ingest list tcv -s all
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient

    console = Console()

    with GraphClient() as client:
        if status == "all":
            result = client.query(
                """
                MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
                RETURN sf.path AS path, sf.status AS status,
                       sf.interest_score AS score, sf.error AS error
                ORDER BY sf.interest_score DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=limit,
            )
        else:
            result = client.query(
                """
                MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE sf.status = $status
                RETURN sf.path AS path, sf.status AS status,
                       sf.interest_score AS score, sf.error AS error
                ORDER BY sf.interest_score DESC
                LIMIT $limit
                """,
                facility=facility,
                status=status,
                limit=limit,
            )

    if not result:
        console.print(f"[yellow]No SourceFile nodes with status '{status}'[/yellow]")
        return

    table = Table(title=f"SourceFiles ({status}): {facility}")
    table.add_column("Path", style="cyan", max_width=60)
    table.add_column("Status", style="green")
    table.add_column("Score", justify="right")
    if status == "failed":
        table.add_column("Error", style="red", max_width=30)

    for row in result:
        score = f"{row['score']:.2f}" if row["score"] is not None else "-"
        if status == "failed":
            table.add_row(row["path"], row["status"], score, row["error"] or "")
        else:
            table.add_row(row["path"], row["status"], score)

    console.print(table)
    console.print(f"\n[dim]Showing {len(result)} of possibly more files[/dim]")


# ============================================================================
# Release Command
# ============================================================================


@main.command("release")
@click.argument("version")
@click.option(
    "-m",
    "--message",
    required=True,
    help="Release message (used for git tag annotation)",
)
@click.option(
    "--remote",
    type=click.Choice(["origin", "upstream"]),
    default="upstream",
    help="Target remote: 'origin' prepares PR, 'upstream' finalizes release",
)
@click.option(
    "--skip-graph",
    is_flag=True,
    help="Skip graph dump and push (upstream mode only)",
)
@click.option(
    "--skip-git",
    is_flag=True,
    help="Skip git tag creation and push",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
def release(
    version: str,
    message: str,
    remote: str,
    skip_graph: bool,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """Create a new release with two modes based on remote.

    VERSION should be a semantic version with 'v' prefix (e.g., v1.0.0).
    The project version is derived from git tags via hatch-vcs.

    MODE: --remote origin (prepare PR)
    - Creates and pushes tag to origin
    - No graph operations (graph is local-only data)

    MODE: --remote upstream (finalize release - default)
    - Pre-flight: clean tree, synced with upstream
    - Updates _GraphMeta node with version
    - Dumps and pushes graph to GHCR
    - Creates and pushes tag to upstream (triggers CI)

    Workflow:
    1. imas-codex release vX.Y.Z -m 'message' --remote origin
    2. Create PR on GitHub, merge to upstream
    3. git pull upstream main
    4. imas-codex release vX.Y.Z -m 'message' --remote upstream

    Examples:
        # Prepare PR (pushes tag to fork)
        imas-codex release v1.0.0 -m 'Add EPFL' --remote origin

        # Finalize release (graph to GHCR, tag to upstream)
        imas-codex release v1.0.0 -m 'Add EPFL' --remote upstream

        # Dry run
        imas-codex release v1.0.0 -m 'Test' --dry-run
    """
    import re
    import subprocess

    # Validate version format
    if not re.match(r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", version):
        click.echo(f"Error: Invalid version format: {version}", err=True)
        click.echo("Expected format: v1.0.0 or v1.0.0-rc1")
        raise SystemExit(1)

    version_number = version.lstrip("v")

    # Determine mode
    is_origin_mode = remote == "origin"
    mode_desc = "PR preparation" if is_origin_mode else "release finalization"

    click.echo(f"{'[DRY RUN] ' if dry_run else ''}Release {version} ({mode_desc})")
    click.echo(f"Message: {message}")
    click.echo(f"Remote: {remote}")
    click.echo()

    # Pre-flight checks
    click.echo("Pre-flight checks...")

    # Check 1: On main branch
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
    )
    current_branch = branch_result.stdout.strip()
    if current_branch != "main":
        click.echo(f"  ✗ Not on main branch (current: {current_branch})", err=True)
        click.echo("    Switch to main: git checkout main")
        raise SystemExit(1)
    click.echo("  ✓ On main branch")

    # Check 2: Remote exists
    remote_result = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if remote_result.returncode != 0:
        click.echo(f"  ✗ Remote '{remote}' not found", err=True)
        click.echo(f"    Add it: git remote add {remote} <url>")
        raise SystemExit(1)
    click.echo(f"  ✓ Remote '{remote}' exists")

    # For upstream mode: stricter checks
    if not is_origin_mode:
        # Check 3: Clean working tree (required for upstream)
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if status_result.stdout.strip():
            click.echo("  ✗ Working tree has uncommitted changes", err=True)
            click.echo("    Commit or stash changes first")
            if not dry_run:
                raise SystemExit(1)
        else:
            click.echo("  ✓ Working tree is clean")

        # Check 4: Synced with upstream
        subprocess.run(["git", "fetch", remote, "main"], capture_output=True)
        ahead_behind = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", f"main...{remote}/main"],
            capture_output=True,
            text=True,
        )
        if ahead_behind.returncode == 0:
            parts = ahead_behind.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
                if behind > 0:
                    click.echo(
                        f"  ✗ Local is {behind} commits behind {remote}/main", err=True
                    )
                    click.echo(f"    Pull first: git pull {remote} main")
                    if not dry_run:
                        raise SystemExit(1)
                if ahead > 0:
                    click.echo(
                        f"  ✗ Local is {ahead} commits ahead of {remote}/main",
                        err=True,
                    )
                    click.echo("    Ensure PR is merged first")
                    if not dry_run:
                        raise SystemExit(1)
                if ahead == 0 and behind == 0:
                    click.echo(f"  ✓ Synced with {remote}/main")

    click.echo()

    # =========================================================================
    # ORIGIN MODE: Push branch + tag
    # =========================================================================
    if is_origin_mode:
        # Step 1: Push branch
        click.echo("Step 1: Pushing branch to origin...")
        if not dry_run:
            result = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing branch: {result.stderr}", err=True)
                raise SystemExit(1)
            click.echo("  Pushed to origin/main")
        else:
            click.echo("  [would push to origin/main]")

        # Step 2: Create and push tag
        if not skip_git:
            click.echo("\nStep 2: Creating and pushing tag...")
            if not dry_run:
                # Create tag
                result = subprocess.run(
                    ["git", "tag", "-a", version, "-m", message],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    if "already exists" in result.stderr:
                        click.echo(f"  Warning: Tag {version} already exists")
                    else:
                        click.echo(f"Error creating tag: {result.stderr}", err=True)
                        raise SystemExit(1)
                else:
                    click.echo(f"  Created tag: {version}")

                # Push tag to origin
                result = subprocess.run(
                    ["git", "push", "origin", version],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    click.echo(f"Error pushing tag: {result.stderr}", err=True)
                    raise SystemExit(1)
                click.echo("  Pushed tag to origin")
            else:
                click.echo(f"  [would create and push tag {version} to origin]")
        else:
            click.echo("\nStep 2: Skipped (--skip-git)")

        click.echo()
        if dry_run:
            click.echo("[DRY RUN] No changes made.")
        else:
            click.echo(f"PR preparation complete for {version}!")
            click.echo("\nNext steps:")
            click.echo("  1. Create PR on GitHub from origin/main to upstream/main")
            click.echo("  2. After merge: git pull upstream main")
            click.echo(f"  3. Run: imas-codex release {version} -m '{message}'")

    # =========================================================================
    # UPSTREAM MODE: Graph operations, push tag
    # =========================================================================
    else:
        # Step 1: Validate no private fields in graph
        click.echo("Step 1: Validating graph contains no private fields...")
        if not dry_run:
            try:
                from imas_codex.graph import GraphClient, get_schema

                schema = get_schema()
                private_slots = schema.get_private_slots("Facility")

                if private_slots:
                    with GraphClient() as client:
                        # Check Facility nodes for private fields
                        for slot in private_slots:
                            result = client.query(
                                f"MATCH (f:Facility) WHERE f.{slot} IS NOT NULL "
                                f"RETURN f.id AS id, f.{slot} AS value LIMIT 5"
                            )
                            if result:
                                click.echo(
                                    f"  ✗ Private field '{slot}' found in graph!",
                                    err=True,
                                )
                                for r in result:
                                    click.echo(
                                        f"    - Facility {r['id']}: {slot}={r['value']}"
                                    )
                                click.echo(
                                    "\nPrivate data must not be in graph before OCI push."
                                )
                                click.echo(
                                    "Remove with: MATCH (f:Facility) REMOVE f.<field>"
                                )
                                raise SystemExit(1)

                    click.echo(
                        f"  ✓ No private fields found (checked: {private_slots})"
                    )
                else:
                    click.echo("  ✓ No private slots defined in schema")
            except SystemExit:
                raise
            except Exception as e:
                click.echo(f"Warning: Could not validate graph: {e}", err=True)
                click.echo("  Is Neo4j running? Check with: imas-codex neo4j status")
        else:
            click.echo("  [would validate no private fields in graph]")

        # Step 2: Update _GraphMeta node
        if not skip_graph:
            click.echo("\nStep 2: Updating graph metadata...")
            if not dry_run:
                try:
                    from imas_codex.graph import GraphClient

                    with GraphClient() as client:
                        facilities_result = client.query(
                            "MATCH (f:Facility) RETURN collect(f.id) as facilities"
                        )
                        facilities = (
                            facilities_result[0]["facilities"]
                            if facilities_result
                            else []
                        )
                        client.query(
                            """
                            MERGE (m:_GraphMeta {id: 'meta'})
                            SET m.version = $version,
                                m.message = $message,
                                m.updated_at = datetime(),
                                m.facilities = $facilities
                            """,
                            version=version_number,
                            message=message,
                            facilities=facilities,
                        )
                        click.echo(f"  _GraphMeta updated: version={version_number}")
                        click.echo(f"  Facilities: {', '.join(facilities)}")
                except Exception as e:
                    click.echo(
                        f"Warning: Could not update graph metadata: {e}", err=True
                    )
                    click.echo(
                        "  Is Neo4j running? Check with: imas-codex neo4j status"
                    )
            else:
                click.echo("  [would update _GraphMeta node in graph]")

            # Step 3: Dump graph
            click.echo("\nStep 3: Dumping graph...")
            if not dry_run:
                click.echo("  Stopping Neo4j for dump...")
                subprocess.run(
                    ["uv", "run", "imas-codex", "neo4j", "stop"],
                    capture_output=True,
                )
                dump_file = f"imas-codex-graph-{version_number}.dump"
                result = subprocess.run(
                    ["uv", "run", "imas-codex", "neo4j", "dump", "-o", dump_file],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    click.echo(f"Error dumping graph: {result.stderr}", err=True)
                    raise SystemExit(1)
                click.echo(f"  Dumped to: {dump_file}")
            else:
                click.echo(f"  [would dump to: imas-codex-graph-{version_number}.dump]")

            # Step 4: Push to GHCR
            click.echo("\nStep 4: Pushing graph to GHCR...")
            if not dry_run:
                result = subprocess.run(
                    ["uv", "run", "imas-codex", "neo4j", "push", version],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    click.echo(f"Error pushing graph: {result.stderr}", err=True)
                    click.echo("  You may need to set GHCR_TOKEN")
                    raise SystemExit(1)
                click.echo(
                    f"  Pushed to ghcr.io/iterorganization/imas-codex-graph:{version}"
                )
            else:
                click.echo(
                    f"  [would push to ghcr.io/iterorganization/imas-codex-graph:{version}]"
                )
        else:
            click.echo("\nStep 2-4: Skipped (--skip-graph)")

        # Step 5: Git tag
        if not skip_git:
            click.echo("\nStep 5: Create and push git tag...")
            if not dry_run:
                result = subprocess.run(
                    ["git", "tag", "-a", version, "-m", message],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    if "already exists" in result.stderr:
                        click.echo(f"  Warning: Tag {version} already exists")
                    else:
                        click.echo(f"Error creating tag: {result.stderr}", err=True)
                        raise SystemExit(1)
                else:
                    click.echo(f"  Created tag: {version}")

                result = subprocess.run(
                    ["git", "push", "upstream", version],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    click.echo(f"Error pushing tag: {result.stderr}", err=True)
                    raise SystemExit(1)
                click.echo("  Pushed tag to upstream")
            else:
                click.echo(f"  [would create tag: {version}]")
                click.echo("  [would push tag to: upstream]")
        else:
            click.echo("\nStep 5: Skipped (--skip-git)")

        click.echo()
        if dry_run:
            click.echo("[DRY RUN] No changes made.")
        else:
            click.echo(f"Release {version} complete!")
            click.echo("Tag pushed to upstream. CI will build and publish the package.")


# ============================================================================
# Wiki Commands
# ============================================================================


@main.group(deprecated=True)
def wiki() -> None:
    """[DEPRECATED] Wiki commands - use 'discover docs' instead.

    This command group is deprecated and will be removed in a future version.
    Please migrate to the new discovery pipeline:

    \b
    OLD                              NEW
    imas-codex wiki discover         imas-codex discover docs
    imas-codex wiki crawl            imas-codex discover docs (integrated)
    imas-codex wiki score            imas-codex discover docs (integrated)
    imas-codex wiki ingest           imas-codex ingest docs
    imas-codex wiki status           imas-codex discover status --domain docs
    imas-codex wiki sites            imas-codex discover sources list
    imas-codex wiki credentials      (credentials still stored in keyring)
    """
    import warnings

    warnings.warn(
        "The 'wiki' command group is deprecated. Use 'discover docs' instead.",
        DeprecationWarning,
        stacklevel=2,
    )


# -----------------------------------------------------------------------------
# Wiki Credentials Subcommands
# -----------------------------------------------------------------------------


@wiki.group("credentials")
def wiki_credentials() -> None:
    """Manage wiki site credentials.

    Credentials are stored securely in your system keyring
    (GNOME Keyring on Linux, Keychain on macOS).

    \b
      imas-codex wiki credentials list [<facility>]  List sites and status
      imas-codex wiki credentials set <site>         Store credentials
      imas-codex wiki credentials get <site>         Check if credentials exist
      imas-codex wiki credentials delete <site>      Remove credentials
    """
    pass


@wiki_credentials.command("set")
@click.argument("site")
def wiki_credentials_set(site: str) -> None:
    """Store credentials for a wiki site.

    Prompts for username and password, then stores them securely
    in your system keyring.

    Examples:
        imas-codex wiki credentials set iter-confluence
    """
    import getpass

    from imas_codex.wiki.auth import CredentialManager

    creds = CredentialManager()

    if not creds._keyring_available:
        click.echo("❌ System keyring not available.", err=True)
        click.echo("\nKeyring requires a running D-Bus session.", err=True)
        click.echo("On headless systems, you may need to:", err=True)
        click.echo("  1. Install: pip install keyrings.alt", err=True)
        click.echo("  2. Or use environment variables instead:", err=True)
        env_user = creds._env_var_name(site, "username")
        env_pass = creds._env_var_name(site, "password")
        click.echo(f"     export {env_user}=your_username", err=True)
        click.echo(f"     export {env_pass}=your_password", err=True)
        raise SystemExit(1)

    click.echo(f"Setting credentials for: {site}")
    click.echo("(Stored securely in system keyring)\n")

    username = click.prompt("Username")
    password = getpass.getpass("Password: ")

    if creds.set_credentials(site, username, password):
        click.echo(f"\n✓ Credentials stored for {site}")
    else:
        click.echo("\n❌ Failed to store credentials", err=True)
        click.echo("\nKeyring backend not configured. Setup options:", err=True)
        click.echo("", err=True)
        click.echo(
            "Option 1: Install a keyring backend (recommended for desktop)", err=True
        )
        click.echo(
            "  Linux:   sudo apt install gnome-keyring  # or libsecret", err=True
        )
        click.echo("  macOS:   Built-in Keychain should work automatically", err=True)
        click.echo(
            "  Windows: Built-in Credential Locker should work automatically", err=True
        )
        click.echo("", err=True)
        click.echo(
            "Option 2: Use file-based backend (for servers/containers)", err=True
        )
        click.echo("  pip install keyrings.alt", err=True)
        click.echo("  Then create ~/.config/python_keyring/keyringrc.cfg:", err=True)
        click.echo("    [backend]", err=True)
        click.echo("    default-keyring=keyrings.alt.file.PlaintextKeyring", err=True)
        click.echo("", err=True)
        click.echo("Option 3: Use environment variables (for CI/automation)", err=True)
        env_user = creds._env_var_name(site, "username")
        env_pass = creds._env_var_name(site, "password")
        click.echo(f"  export {env_user}=your_username", err=True)
        click.echo(f"  export {env_pass}=your_password", err=True)
        raise SystemExit(1)


@wiki_credentials.command("get")
@click.argument("site")
def wiki_credentials_get(site: str) -> None:
    """Check if credentials exist for a wiki site.

    Does not display the actual credentials, only confirms existence.

    Examples:
        imas-codex wiki credentials get iter-confluence
    """
    from imas_codex.wiki.auth import CredentialManager

    creds = CredentialManager()

    if creds.has_credentials(site):
        click.echo(f"✓ Credentials found for {site}")

        # Check for valid session
        session = creds.get_session(site)
        if session:
            click.echo("✓ Valid session cached")
        else:
            click.echo("○ No cached session")
    else:
        click.echo(f"○ No credentials found for {site}")
        click.echo("\nTo set credentials:")
        click.echo(f"  imas-codex wiki credentials set {site}")


@wiki_credentials.command("delete")
@click.argument("site")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def wiki_credentials_delete(site: str, yes: bool) -> None:
    """Delete stored credentials for a wiki site.

    Also removes any cached session cookies.

    Examples:
        imas-codex wiki credentials delete iter-confluence
    """
    from imas_codex.wiki.auth import CredentialManager

    creds = CredentialManager()

    if not yes:
        click.confirm(f"Delete credentials for {site}?", abort=True)

    deleted_creds = creds.delete_credentials(site)
    deleted_session = creds.delete_session(site)

    if deleted_creds:
        click.echo(f"✓ Deleted credentials for {site}")
    else:
        click.echo(f"○ No credentials found for {site}")

    if deleted_session:
        click.echo("✓ Deleted cached session")


@wiki_credentials.command("list")
@click.argument("facility", required=False)
def wiki_credentials_list(facility: str | None) -> None:
    """List wiki sites and their credential status.

    Shows all wiki sites configured for a facility (or all facilities if
    not specified), including which require authentication and whether
    credentials are configured.

    \b
    Examples:
        imas-codex wiki credentials list            List all facilities
        imas-codex wiki credentials list iter       List ITER sites only
    """
    from imas_codex.discovery.facility import list_facilities
    from imas_codex.wiki.auth import CredentialManager
    from imas_codex.wiki.discovery import WikiConfig

    creds = CredentialManager()

    # Determine which facilities to show
    if facility:
        facilities = [facility]
    else:
        facilities = list_facilities()

    if not facilities:
        click.echo("No facilities configured.")
        return

    for fac in facilities:
        try:
            sites = WikiConfig.list_sites(fac)
        except Exception as e:
            click.echo(f"\n{fac}: Error loading config - {e}", err=True)
            continue

        if not sites:
            click.echo(f"\n{fac}: No wiki sites configured")
            continue

        click.echo(f"\n{fac}:")
        for site in sites:
            # Determine credential service name
            cred_service = site.credential_service
            auth_info = f"Auth: {site.auth_type}"

            if site.requires_auth and cred_service:
                has_creds = creds.has_credentials(cred_service)
                status = "✓" if has_creds else "○"
                cred_info = f"  [{status}] {cred_service}"
            else:
                cred_info = ""

            # Build display line
            click.echo(f"  • {site.base_url}")
            click.echo(f"      Type: {site.site_type}, {auth_info}")
            if cred_info:
                click.echo(f"      Credentials: {cred_info}")
            if site.requires_ssh:
                click.echo(f"      SSH: {site.ssh_host}")

    # Show summary
    click.echo("\n---")
    click.echo("Legend: ✓ = credentials configured, ○ = not set")
    click.echo("To set credentials: imas-codex wiki credentials set <site>")


@wiki.command("sites")
@click.argument("facility")
def wiki_sites(facility: str) -> None:
    """List configured wiki sites for a facility.

    Shows all wiki/documentation sites configured in the facility's
    YAML configuration, including authentication requirements.

    Examples:
        imas-codex wiki sites iter
        imas-codex wiki sites tcv
    """
    from imas_codex.wiki.auth import CredentialManager
    from imas_codex.wiki.discovery import WikiConfig

    sites = WikiConfig.list_sites(facility)
    creds = CredentialManager()

    if not sites:
        click.echo(f"No wiki sites configured for facility: {facility}")
        return

    click.echo(f"Wiki sites for {facility}:\n")

    for i, site in enumerate(sites):
        click.echo(f"  [{i}] {site.base_url}")
        click.echo(f"      Type: {site.site_type}")
        click.echo(f"      Auth: {site.auth_type}")

        if site.portal_page:
            click.echo(f"      Portal: {site.portal_page}")

        if site.requires_auth and site.credential_service:
            has_creds = creds.has_credentials(site.credential_service)
            status = "✓ configured" if has_creds else "○ not set"
            click.echo(f"      Credentials ({site.credential_service}): {status}")

        if site.requires_ssh:
            click.echo(f"      SSH host: {site.ssh_host}")

        click.echo()


@wiki.command("discover")
@click.argument("facility")
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Focus discovery (e.g., 'equilibrium reconstruction', 'diagnostics')",
)
@click.option(
    "--start-page",
    default="Portal:TCV",
    help="Page to start discovery from (default: Portal:TCV)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=10.00,
    type=float,
    help="Maximum cost budget in USD (default: 10.00)",
)
@click.option(
    "--max-pages",
    "-n",
    default=None,
    type=int,
    help="Maximum pages to crawl (default: unlimited)",
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    help="Maximum link depth from portal (default: unlimited)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model to use (default: from config for 'discovery' task)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show agent reasoning",
)
def wiki_discover(
    facility: str,
    prompt: str | None,
    start_page: str,
    cost_limit: float,
    max_pages: int | None,
    max_depth: int | None,
    model: str | None,
    verbose: bool,
) -> None:
    """Discover wiki pages using integrated pipeline.

    Runs complete workflow automatically:
    1. CRAWL: Fast link extraction, builds wiki graph structure
    2. PREFETCH: Fetch page content and generate summaries
    3. SCORE: Content-aware LLM evaluation, assigns interest scores
    4. INGEST: Fetch and chunk high-score pages for search

    Graph-driven: restarts resume from existing state.

    Examples:
        # Full discovery with default settings
        imas-codex wiki discover tcv

        # Focus on equilibrium topics
        imas-codex wiki discover tcv -p "equilibrium reconstruction"

        # Limit crawl scope
        imas-codex wiki discover tcv -n 500 --max-depth 3

        # Verbose mode to see agent reasoning
        imas-codex wiki discover tcv -v
    """
    import asyncio

    from imas_codex.wiki.discovery import run_wiki_discovery

    # Run discovery
    asyncio.run(
        run_wiki_discovery(
            facility=facility,
            cost_limit_usd=cost_limit,
            max_pages=max_pages,
            max_depth=max_depth,
            verbose=verbose,
            model=model,
            focus=prompt,
        )
    )


@wiki.command("crawl")
@click.argument("facility")
@click.option(
    "--max-pages",
    "-n",
    default=None,
    type=int,
    help="Maximum pages to crawl this session (default: unlimited)",
)
@click.option(
    "--max-depth",
    default=10,
    type=int,
    help="Maximum link depth from portal (default: 10)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show progress details",
)
def wiki_crawl(
    facility: str,
    max_pages: int | None,
    max_depth: int,
    verbose: bool,
) -> None:
    """Crawl wiki links without scoring (no LLM).

    Fast breadth-first crawl that extracts links and builds
    the wiki graph structure. Use 'wiki score' afterwards
    to evaluate pages.

    Graph-driven: restarts resume from existing state. Already-crawled
    pages are skipped, pending pages form the frontier.

    Examples:
        # Full crawl to completion
        imas-codex wiki crawl tcv

        # Crawl up to 500 pages this session
        imas-codex wiki crawl tcv -n 500

        # Shallow crawl
        imas-codex wiki crawl tcv --max-depth 3
    """
    from rich.console import Console

    from imas_codex.wiki.discovery import WikiDiscovery
    from imas_codex.wiki.progress import CrawlProgressMonitor

    console = Console()
    discovery = WikiDiscovery(
        facility=facility,
        max_pages=max_pages,
        max_depth=max_depth,
        verbose=verbose,
    )

    try:
        with CrawlProgressMonitor(facility=facility) as monitor:
            discovery.crawl(monitor)

        console.print(f"  Links found: {discovery.stats.links_found}")
        console.print(f"  Max depth: {discovery.stats.max_depth_reached}")
        console.print(f"  Frontier: {discovery.stats.frontier_size} pages pending")
        console.print("\nRun 'imas-codex wiki score tcv' to evaluate pages")
    finally:
        discovery.close()


@wiki.command("prefetch")
@click.argument("facility")
@click.option(
    "--batch-size",
    "-b",
    default=50,
    type=int,
    help="Pages per batch (default: 50)",
)
@click.option(
    "--max-pages",
    "-n",
    default=None,
    type=int,
    help="Maximum pages to process (default: unlimited)",
)
@click.option(
    "--include-scored",
    is_flag=True,
    help="Also prefetch already-scored pages",
)
def wiki_prefetch(
    facility: str, batch_size: int, max_pages: int | None, include_scored: bool
) -> None:
    """Prefetch and summarize page previews before scoring.

    Fetches page content and generates LLM summaries for content-aware
    scoring. This is an optional step that improves scoring accuracy,
    especially for ITER Confluence pages.

    Examples:
        # Prefetch 100 pages for pilot testing
        imas-codex wiki prefetch iter --max-pages 100

        # Prefetch all discovered pages
        imas-codex wiki prefetch tcv

        # Prefetch including already-scored pages
        imas-codex wiki prefetch iter --include-scored
    """
    import asyncio

    from rich.console import Console

    from imas_codex.wiki.prefetch import prefetch_pages

    console = Console()
    console.print(f"[cyan]Starting prefetch for {facility}...[/cyan]")
    stats = asyncio.run(
        prefetch_pages(
            facility_id=facility,
            batch_size=batch_size,
            max_pages=max_pages,
            include_scored=include_scored,
        )
    )

    console.print("\n[green]Prefetch complete:[/green]")
    console.print(f"  Fetched: {stats['fetched']}")
    console.print(f"  Summarized: {stats['summarized']}")
    console.print(f"  Failed: {stats['failed']}")


@wiki.command("score")
@click.argument("facility")
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Focus scoring (e.g., 'equilibrium reconstruction', 'diagnostics')",
)
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Maximum pages to score (default: unlimited)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=20.0,
    type=float,
    help="Maximum cost budget in USD (default: 20.0)",
)
@click.option(
    "--batch-size",
    "-b",
    default=100,
    type=int,
    help="Pages per agent batch (default: 100)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model to use (default: from config for 'discovery' task)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show agent reasoning",
)
def wiki_score(
    facility: str,
    prompt: str | None,
    limit: int | None,
    cost_limit: float,
    batch_size: int,
    model: str | None,
    verbose: bool,
) -> None:
    """Score crawled wiki pages using content-aware LLM evaluation.

    Evaluates pages based on graph metrics (in_degree, out_degree,
    link_depth) AND content summaries (if prefetched) to assign
    interest_score (0.0-1.0).

    For best results, run 'wiki prefetch' first to enable content-aware
    scoring. Otherwise falls back to metric-based scoring.

    Uses CLI-orchestrated batching with fresh agents per batch to avoid
    context overflow. Continues until all pages are scored, page limit
    is reached, or cost limit is exceeded.

    Examples:
        # Score all crawled pages (up to $20 cost)
        imas-codex wiki score tcv

        # Focus on equilibrium topics
        imas-codex wiki score tcv -p "equilibrium reconstruction"

        # Score with verbose agent output
        imas-codex wiki score tcv -v

        # Limit to 500 pages
        imas-codex wiki score tcv -n 500

        # Limit cost to $5
        imas-codex wiki score tcv --cost-limit 5.0
    """
    import asyncio

    from imas_codex.wiki.discovery import WikiDiscovery
    from imas_codex.wiki.progress import ScoreProgressMonitor

    discovery = WikiDiscovery(
        facility=facility,
        cost_limit_usd=cost_limit,
        max_pages=limit,
        verbose=verbose,
        model=model,
        focus=prompt,
    )

    try:
        with ScoreProgressMonitor(cost_limit=cost_limit, facility=facility) as monitor:
            scored = asyncio.run(
                discovery.score(monitor=monitor, batch_size=batch_size)
            )

        if scored == 0:
            monitor._console.print("[yellow]No pages to score[/yellow]")
    finally:
        discovery.close()


@wiki.command("ingest")
@click.argument("facility")
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Maximum items to ingest (default: all)",
)
@click.option(
    "--min-score",
    default=0.5,
    type=float,
    help="Minimum interest score threshold (default: 0.5)",
)
@click.option(
    "--pages",
    "-p",
    multiple=True,
    help="Specific page names to ingest (bypasses graph queue)",
)
@click.option(
    "--rate-limit",
    default=0.5,
    type=float,
    help="Seconds between requests (default: 0.5)",
)
@click.option(
    "--type",
    "content_type",
    default="all",
    type=click.Choice(["all", "pages", "artifacts"]),
    help="Content type to ingest (default: all)",
)
@click.option(
    "--max-size-mb",
    default=5.0,
    type=float,
    help="Maximum artifact size in MB (default: 5.0). Larger files are deferred.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview without ingesting",
)
def wiki_ingest(
    facility: str,
    limit: int | None,
    min_score: float,
    pages: tuple[str, ...],
    rate_limit: float,
    content_type: str,
    max_size_mb: float,
    dry_run: bool,
) -> None:
    """Ingest wiki pages and artifacts from the graph queue.

    Processes WikiPage and WikiArtifact nodes with status='scored'.
    Use 'wiki score' first to evaluate and queue content.

    Artifacts larger than --max-size-mb are marked as 'deferred' with
    their size stored. They remain searchable via metadata but don't
    flood the graph with heavy embeddings.

    Examples:
        # Ingest all scored content (pages + artifacts)
        imas-codex wiki ingest tcv

        # Ingest only pages
        imas-codex wiki ingest tcv --type pages

        # Ingest only artifacts (PDFs, etc.)
        imas-codex wiki ingest tcv --type artifacts

        # Ingest only high-score content
        imas-codex wiki ingest tcv --min-score 0.7

        # Allow larger artifacts (10 MB)
        imas-codex wiki ingest tcv --max-size-mb 10

        # Ingest specific pages (bypasses queue)
        imas-codex wiki ingest tcv -p Thomson -p Ion_Temperature_Nodes

        # Preview without saving
        imas-codex wiki ingest tcv --dry-run
    """
    import asyncio

    from rich.console import Console
    from rich.table import Table

    from imas_codex.wiki import (
        WikiArtifactPipeline,
        WikiIngestionPipeline,
        get_pending_wiki_artifacts,
        get_pending_wiki_pages,
    )
    from imas_codex.wiki.pipeline import create_wiki_vector_index

    console = Console()

    # Track combined stats
    total_stats = {
        "pages": 0,
        "pages_failed": 0,
        "artifacts": 0,
        "artifacts_deferred": 0,
        "artifacts_failed": 0,
        "chunks": 0,
        "tree_nodes_linked": 0,
        "imas_paths_linked": 0,
        "conventions": 0,
    }

    # Handle explicit page list (bypasses type filter)
    if pages:
        page_list = list(pages)
        console.print(f"[cyan]Ingesting {len(page_list)} specified pages...[/cyan]")

        if dry_run:
            console.print("\n[yellow][DRY RUN] Pages that would be ingested:[/yellow]")
            for i, page in enumerate(page_list[:20], 1):
                console.print(f"  {i:2}. {page}")
            if len(page_list) > 20:
                console.print(f"  ... and {len(page_list) - 20} more")
            return

        try:
            create_wiki_vector_index()
        except Exception as e:
            console.print(f"[dim]Vector index: {e}[/dim]")

        pipeline = WikiIngestionPipeline(facility_id=facility, use_rich=True)
        try:
            stats = asyncio.run(pipeline.ingest_pages(page_list, rate_limit=rate_limit))
            total_stats.update(stats)
        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]")
            raise SystemExit(1) from e

    else:
        # Graph-driven ingestion based on type
        console.print(f"[cyan]Checking graph queue for {facility}...[/cyan]")

        # Create vector index once
        try:
            create_wiki_vector_index()
        except Exception as e:
            console.print(f"[dim]Vector index: {e}[/dim]")

        # Process pages if requested
        if content_type in ("all", "pages"):
            pending_pages = get_pending_wiki_pages(
                facility_id=facility,
                limit=limit,
                min_interest_score=min_score,
            )

            if pending_pages:
                console.print(
                    f"[green]Found {len(pending_pages)} pending pages[/green]"
                )

                if dry_run:
                    table = Table(title="[DRY RUN] Pages that would be ingested")
                    table.add_column("#", style="dim", width=4)
                    table.add_column("Page Name", style="cyan")
                    table.add_column("Score", style="dim", width=6)

                    for i, page in enumerate(pending_pages[:20], 1):
                        score = page.get("interest_score", 0.5)
                        table.add_row(str(i), page["title"], f"{score:.2f}")
                    if len(pending_pages) > 20:
                        table.add_row(
                            "...", f"[dim]and {len(pending_pages) - 20} more[/dim]", ""
                        )
                    console.print(table)
                else:
                    pipeline = WikiIngestionPipeline(
                        facility_id=facility, use_rich=True
                    )
                    try:
                        stats = asyncio.run(
                            pipeline.ingest_from_graph(
                                limit=limit,
                                min_interest_score=min_score,
                                rate_limit=rate_limit,
                            )
                        )
                        for k, v in stats.items():
                            if k in total_stats:
                                total_stats[k] += v
                    except Exception as e:
                        console.print(f"[red]Error during page ingestion: {e}[/red]")
                        raise SystemExit(1) from e
            else:
                console.print(f"[dim]No pending pages for {facility}[/dim]")

        # Process artifacts if requested
        if content_type in ("all", "artifacts"):
            pending_artifacts = get_pending_wiki_artifacts(
                facility_id=facility,
                limit=limit,
                min_interest_score=min_score,
            )

            if pending_artifacts:
                console.print(
                    f"[green]Found {len(pending_artifacts)} pending artifacts[/green]"
                )

                # Show type breakdown
                type_counts: dict[str, int] = {}
                for a in pending_artifacts:
                    t = a.get("artifact_type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1
                console.print(
                    "[dim]By type:[/dim] "
                    + ", ".join(f"{t}: {c}" for t, c in type_counts.items())
                )

                if dry_run:
                    table = Table(title="[DRY RUN] Artifacts that would be ingested")
                    table.add_column("#", style="dim", width=4)
                    table.add_column("Filename", style="cyan")
                    table.add_column("Type", width=8)
                    table.add_column("Score", style="dim", width=6)

                    for i, artifact in enumerate(pending_artifacts[:20], 1):
                        score = artifact.get("interest_score", 0.5)
                        table.add_row(
                            str(i),
                            artifact["filename"],
                            artifact.get("artifact_type", "?"),
                            f"{score:.2f}",
                        )
                    if len(pending_artifacts) > 20:
                        table.add_row(
                            "...",
                            f"[dim]and {len(pending_artifacts) - 20} more[/dim]",
                            "",
                            "",
                        )
                    console.print(table)
                else:
                    artifact_pipeline = WikiArtifactPipeline(
                        facility_id=facility,
                        use_rich=True,
                        max_size_mb=max_size_mb,
                    )
                    try:
                        stats = asyncio.run(
                            artifact_pipeline.ingest_from_graph(
                                limit=limit,
                                min_interest_score=min_score,
                            )
                        )
                        for k, v in stats.items():
                            if k in total_stats:
                                total_stats[k] += v
                        # Track oversized separately if available
                        if "artifacts_oversized" in stats:
                            total_stats["artifacts_oversized"] = (
                                total_stats.get("artifacts_oversized", 0)
                                + stats["artifacts_oversized"]
                            )
                    except Exception as e:
                        console.print(
                            f"[red]Error during artifact ingestion: {e}[/red]"
                        )
                        raise SystemExit(1) from e
            else:
                console.print(f"[dim]No pending artifacts for {facility}[/dim]")

        if dry_run:
            return

    # Print combined summary
    console.print("\n[green]Ingestion complete![/green]")
    if total_stats["pages"] > 0 or content_type in ("all", "pages"):
        console.print(f"  Pages ingested:      {total_stats['pages']}")
        console.print(f"  Pages failed:        {total_stats['pages_failed']}")
    if total_stats["artifacts"] > 0 or content_type in ("all", "artifacts"):
        console.print(f"  Artifacts ingested:  {total_stats['artifacts']}")
        deferred = total_stats["artifacts_deferred"]
        oversized = total_stats.get("artifacts_oversized", 0)
        if oversized > 0:
            console.print(f"  Artifacts deferred:  {deferred} ({oversized} oversized)")
        else:
            console.print(f"  Artifacts deferred:  {deferred}")
        console.print(f"  Artifacts failed:    {total_stats['artifacts_failed']}")
    console.print(f"  Chunks created:      {total_stats['chunks']}")
    console.print(f"  TreeNodes linked:    {total_stats['tree_nodes_linked']}")
    console.print(f"  IMAS paths linked:   {total_stats['imas_paths_linked']}")
    console.print(f"  Conventions found:   {total_stats['conventions']}")


@wiki.command("status")
@click.argument("facility")
def wiki_status(facility: str) -> None:
    """Show wiki queue and ingestion statistics.

    Shows the queue status (discovered pages pending ingestion) and
    statistics about already-ingested pages.

    Examples:
        imas-codex wiki status tcv
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient
    from imas_codex.wiki.pipeline import get_wiki_queue_stats, get_wiki_stats

    console = Console()

    # Get queue stats
    queue_stats = get_wiki_queue_stats(facility)

    # Get ingestion stats
    ingestion_stats = get_wiki_stats(facility)

    if queue_stats["total"] == 0 and ingestion_stats["pages"] == 0:
        console.print(f"[yellow]No wiki pages for {facility}[/yellow]")
        console.print(
            f"[dim]Discover pages with: imas-codex wiki discover {facility}[/dim]"
        )
        return

    # Display queue summary
    queue_table = Table(title=f"Wiki Queue: {facility}")
    queue_table.add_column("Status", style="cyan")
    queue_table.add_column("Count", justify="right")

    queue_table.add_row("Crawled (pending score)", str(queue_stats["crawled"]))
    queue_table.add_row("Scored (pending ingest)", str(queue_stats["scored"]))
    queue_table.add_row("Skipped", str(queue_stats["skipped"]))
    queue_table.add_row("Ingested", str(queue_stats["ingested"]))
    queue_table.add_row("Failed", str(queue_stats["failed"]))
    queue_table.add_row("─" * 20, "─" * 6)
    queue_table.add_row("[bold]Total[/bold]", f"[bold]{queue_stats['total']}[/bold]")

    console.print(queue_table)

    if queue_stats["crawled"] > 0:
        console.print(
            f"\n[dim]Score pending pages with: imas-codex wiki score {facility}[/dim]"
        )
    if queue_stats["scored"] > 0:
        console.print(
            f"\n[dim]Process scored pages with: imas-codex wiki ingest {facility}[/dim]"
        )

    if ingestion_stats["pages"] == 0:
        return

    # Display ingestion summary
    console.print()
    summary = Table(title="Ingestion Statistics")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Pages ingested", str(ingestion_stats["pages"]))
    summary.add_row("Chunks created", str(ingestion_stats["chunks"]))
    summary.add_row("TreeNodes linked", str(ingestion_stats["tree_nodes_linked"]))
    summary.add_row("IMAS paths linked", str(ingestion_stats["imas_paths_linked"]))

    console.print(summary)

    # Get detailed breakdown
    with GraphClient() as gc:
        # Top linked pages
        top_pages = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            WHERE wp.status = 'ingested'
            RETURN wp.title AS title, wp.chunk_count AS chunks,
                   wp.link_count AS links
            ORDER BY wp.link_count DESC
            LIMIT 10
            """,
            facility_id=facility,
        )

        # Convention summary
        conventions = gc.query(
            """
            MATCH (sc:SignConvention {facility_id: $facility_id})
            RETURN sc.convention_type AS type, sc.name AS name
            LIMIT 10
            """,
            facility_id=facility,
        )

    # Top pages
    if top_pages:
        top_table = Table(title="Top Linked Pages")
        top_table.add_column("Page", style="cyan")
        top_table.add_column("Chunks", justify="right")
        top_table.add_column("Links", justify="right")

        for row in top_pages:
            top_table.add_row(
                row["title"] or "?",
                str(row["chunks"] or 0),
                str(row["links"] or 0),
            )

        console.print()
        console.print(top_table)

    # Conventions
    if conventions:
        console.print("\n[bold]Sign Conventions Found:[/bold]")
        for row in conventions:
            console.print(f"  [{row['type']}] {row['name']}")


# ============================================================================
# Scout Commands
# ============================================================================


def _run_exploration_agent(
    facility: str,
    resource: str,
    prompt: str | None,
    model: str | None,
    cost_limit: float,
    max_steps: int,
    dry_run: bool,
    verbose: bool,
    base_prompt: str,
) -> None:
    """Common exploration agent runner for non-wiki resources."""
    import asyncio

    from rich.console import Console
    from rich.panel import Panel

    from imas_codex.agentic.agents import get_model_for_task
    from imas_codex.agentic.explore import ExplorationAgent
    from imas_codex.discovery import get_facility as get_facility_config

    console = Console()

    # Validate facility exists
    try:
        get_facility_config(facility)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from e

    # Get model
    model_id = model or get_model_for_task("exploration")

    # Build full prompt
    full_prompt = base_prompt
    if prompt:
        full_prompt = f"{base_prompt}. Focus: {prompt}"

    # Show configuration
    console.print(
        Panel(
            f"[bold]Facility:[/bold] {facility}\n"
            f"[bold]Resource:[/bold] {resource}\n"
            f"[bold]Model:[/bold] {model_id}\n"
            f"[bold]Cost Limit:[/bold] ${cost_limit:.2f}\n"
            f"[bold]Guidance:[/bold] {prompt or '(none)'}\n"
            f"[bold]Max Steps:[/bold] {max_steps}",
            title="Scout Configuration",
        )
    )

    if dry_run:
        console.print("\n[dim]Dry run - agent will not execute[/dim]")
        return

    agent = ExplorationAgent(
        facility=facility,
        model=model_id,
        cost_limit_usd=cost_limit,
        verbose=verbose,
        max_steps=max_steps,
    )

    try:
        console.print("\n[bold]Starting exploration agent...[/bold]")
        result = asyncio.run(agent.explore(prompt=full_prompt))

        # Show results
        console.print("\n[bold green]Discovery Complete[/bold green]")
        console.print(f"  Files queued: {result.files_queued}")
        console.print(f"  Paths discovered: {result.paths_discovered}")
        console.print(f"  Notes added: {result.notes_added}")
        console.print(f"  Cost: ${result.cost_usd:.4f}")
        console.print(f"  Duration: {result.progress.elapsed_seconds:.1f}s")

        if result.summary:
            console.print("\n[bold]Summary:[/bold]")
            console.print(result.summary[:500])

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@main.group(deprecated=True)
def scout() -> None:
    """[DEPRECATED] Scout commands - use 'discover' instead.

    This command group is deprecated and will be removed in a future version.
    Please migrate to the new discovery pipeline:

    \b
    OLD                              NEW
    scout files <facility>           discover code <facility>
    scout wiki <facility>            discover docs <facility>
    scout codes <facility>           discover code <facility>
    scout data <facility>            discover data <facility>
    scout paths <facility>           discover paths <facility>
    scout status <facility>          discover status <facility>

    The new 'discover' group provides a unified, graph-led discovery pipeline.
    """
    import warnings

    warnings.warn(
        "The 'scout' command group is deprecated. Use 'discover' instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@scout.command("files")
@click.argument("facility")
@click.option(
    "--steps", "-n", default=20, help="Number of exploration steps (default: 20)"
)
@click.option("--root-path", "-p", multiple=True, help="Root paths to explore")
@click.option(
    "--focus", "-f", default="general", help="Focus: general, imas, physics, data"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run LLM exploration but don't persist to graph (validation mode)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def scout_files(
    facility: str,
    steps: int,
    root_path: tuple[str, ...],
    focus: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Discover files using graph-first stateless exploration.

    Each step:
    1. Queries graph for frontier (discovered paths)
    2. LLM decides ONE action based on frontier
    3. Executes and persists to graph
    4. Repeats fresh from graph state

    The graph IS the state - no context carried between steps.

    \b
    EXAMPLES:
        # Default exploration (20 steps)
        imas-codex scout files tcv

        # More steps with specific focus
        imas-codex scout files tcv -n 50 --focus imas

        # Start from specific paths
        imas-codex scout files tcv -p /home/codes -p /work/imas

        # Dry-run to test LLM behavior without graph persistence
        imas-codex scout files tcv --dry-run -n 5
    """
    from rich.console import Console

    from imas_codex.agentic.scout import (
        ScoutConfig,
        StatelessScout,
        get_exploration_summary,
    )
    from imas_codex.agentic.scout.display import ScoutDisplay
    from imas_codex.agentic.scout.tools import set_command_callback

    console = Console()

    # Create configuration - dry_run validates schema but doesn't persist
    config = ScoutConfig(
        facility=facility,
        max_steps=steps,
        exploration_focus=focus,
        root_paths=list(root_path) if root_path else [],
        dry_run=dry_run,
        verbose=verbose,
    )

    scout = StatelessScout(config)

    # Seed frontier if empty
    seed_result = scout.seed_frontier()
    seeded = (
        seed_result.get("paths_added", 0)
        if seed_result.get("status") == "seeded"
        else 0
    )

    # Get initial summary
    summary = get_exploration_summary(facility)

    # Use Rich Live display
    display = ScoutDisplay(facility=facility, max_steps=steps, console=console)
    display.start(dry_run=dry_run)

    # Hook command callback to update display when commands run
    def on_command(cmd: str, output: str) -> None:
        """Update display when a command executes."""
        cmd_short = cmd if len(cmd) <= 60 else cmd[:57] + "..."
        display.update_step(display._current_step, action=cmd_short)

    set_command_callback(on_command)

    try:
        # Show initial state
        display.update_stats(
            total_paths=summary.get("total_paths", 0),
            remaining=summary.get("remaining", 0),
            explored=summary.get("explored", 0),
            files_queued=summary.get("files_queued", 0),
        )
        if seeded:
            display.add_history(f"Seeded {seeded} root paths")

        # Run exploration
        for i in range(steps):
            display.update_step(i + 1, action="Running agent...")

            result = scout.step()
            status = result.get("status", "unknown")
            path = result.get("path", "")
            action = result.get("action", "")
            commands = result.get("commands", [])

            # Build action display with commands
            action_display = action
            if commands:
                # Show first command as summary
                cmd_summary = commands[0]
                if len(cmd_summary) > 60:
                    cmd_summary = cmd_summary[:57] + "..."
                action_display = f"{cmd_summary}"
                if len(commands) > 1:
                    action_display += f" (+{len(commands) - 1} more)"

            # Update display
            display.update_step(i + 1, path=path, action=action_display)

            # Add to history with path and action summary
            if path:
                history_entry = path
                if action:
                    history_entry += f" → {action}"
                display.add_history(history_entry)

            # Refresh stats after each step
            updated = get_exploration_summary(facility)
            display.update_stats(
                total_paths=updated.get("total_paths", 0),
                remaining=updated.get("remaining", 0),
                explored=updated.get("explored", 0),
                files_queued=updated.get("files_queued", 0),
            )

            if status in ("complete", "max_steps_reached"):
                display.show_result("success", status)
                break

            if status == "error":
                display.show_result("error", result.get("error", "Unknown error"))
                break

    except KeyboardInterrupt:
        display.show_result("warning", "Interrupted")

    finally:
        set_command_callback(None)  # Clear callback
        display.stop()

    # Show final summary
    final_summary = get_exploration_summary(facility)
    console.print("\n[bold green]Exploration Complete[/bold green]")
    console.print(f"  Steps taken: {scout.steps_taken}")
    console.print(f"  Total paths: {final_summary.get('total_paths', 0)}")
    console.print(
        f"  Explored: {final_summary.get('explored', 0)} ({final_summary.get('coverage', 0):.1%})"
    )
    console.print(f"  Remaining: {final_summary.get('remaining', 0)}")
    console.print(f"  Files queued: {final_summary.get('files_queued', 0)}")

    # Show what was discovered
    if final_summary.get("status_counts"):
        console.print("\n[bold]Discovery Summary[/bold]")
        for status, count in sorted(final_summary["status_counts"].items()):
            console.print(f"  {status}: {count}")


@scout.command("status")
@click.argument("facility")
@click.option("--paths", "-p", is_flag=True, help="Show unexplored paths")
@click.option("--skipped", is_flag=True, help="Show skipped (dead-end) paths")
@click.option("--limit", "-n", default=20, help="Number of items to show")
def scout_status(
    facility: str,
    paths: bool,
    skipped: bool,
    limit: int,
) -> None:
    """Show scout exploration progress for a facility.

    Displays the "moving frontier" status showing what's been explored
    vs what remains to be discovered.

    \b
    EXAMPLES:
        # Show frontier summary
        imas-codex scout status tcv

        # Show unexplored high-priority paths
        imas-codex scout status tcv --paths

        # Show what was skipped as dead-ends
        imas-codex scout status tcv --skipped
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.agentic.scout import get_exploration_summary, get_frontier

    console = Console()

    # Get exploration summary
    summary = get_exploration_summary(facility)

    console.print(f"\n[bold blue]Scout Status: {facility}[/bold blue]\n")

    if "error" in summary:
        console.print(f"[red]Error: {summary['error']}[/red]")
        return

    # Summary table
    summary_table = Table(title="Exploration Summary", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")

    summary_table.add_row("Total paths", str(summary.get("total_paths", 0)))
    summary_table.add_row("Coverage", f"{summary.get('coverage', 0):.1%}")
    summary_table.add_row("Explored", str(summary.get("explored", 0)))
    summary_table.add_row("Remaining", str(summary.get("remaining", 0)))
    summary_table.add_row("Files queued", str(summary.get("files_queued", 0)))
    summary_table.add_row("Files ingested", str(summary.get("files_ingested", 0)))
    console.print(summary_table)

    # Status breakdown
    status_counts = summary.get("status_counts", {})
    if status_counts:
        status_table = Table(title="\nStatus Breakdown")
        status_table.add_column("Status")
        status_table.add_column("Count", justify="right")

        for status, count in sorted(status_counts.items()):
            status_table.add_row(status, str(count))
        console.print(status_table)

    # Unexplored paths
    if paths:
        console.print("\n")
        frontier = get_frontier(facility, limit=limit)
        if frontier:
            paths_table = Table(title="High-Priority Unexplored Paths")
            paths_table.add_column("Path")
            paths_table.add_column("Score", justify="right")
            paths_table.add_column("Reason")

            for p in frontier:
                paths_table.add_row(
                    p.get("path", "?"),
                    f"{p.get('interest_score', 0.5):.2f}",
                    (p.get("interest_reason") or "")[:40],
                )
            console.print(paths_table)
        else:
            console.print("[dim]No unexplored paths found[/dim]")

    # Skipped paths
    if skipped:
        console.print("\n")
        from imas_codex.graph import GraphClient

        try:
            with GraphClient() as client:
                result = client.query(
                    """
                    MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                    WHERE p.status = 'skipped'
                    RETURN p.path AS path, p.skip_reason AS reason
                    ORDER BY p.skipped_at DESC
                    LIMIT $limit
                    """,
                    facility=facility,
                    limit=limit,
                )
                if result:
                    skip_table = Table(title="Skipped Paths (Dead-Ends)")
                    skip_table.add_column("Path")
                    skip_table.add_column("Reason")

                    for p in result:
                        skip_table.add_row(
                            p.get("path", "?"),
                            p.get("reason", "?"),
                        )
                    console.print(skip_table)
                else:
                    console.print("[dim]No skipped paths found[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@scout.command("wiki")
@click.argument("facility")
@click.option(
    "--portal",
    "-p",
    default=None,
    help="Portal page to start from (default: from facility config)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model (default: anthropic/claude-sonnet-4-20250514)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=10.0,
    type=float,
    help="Maximum cost budget in USD (default: 10.0)",
)
@click.option(
    "--max-pages",
    "-n",
    default=None,
    type=int,
    help="Maximum pages to crawl (default: unlimited)",
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    help="Maximum link depth from portal (default: unlimited)",
)
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def scout_wiki(
    facility: str,
    portal: str | None,
    model: str | None,
    cost_limit: float,
    max_pages: int | None,
    max_depth: int | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Discover wiki pages and documentation.

    Crawls wiki starting from a portal page, evaluates page value using LLM,
    and queues high-value pages for ingestion.

    Credentials are prompted interactively if not already stored.
    Use 'imas-codex wiki credentials set <site>' to pre-configure.

    \\b
    EXAMPLES:
        # Discover ITER wiki pages
        imas-codex scout wiki iter

        # Start from specific portal
        imas-codex scout wiki tcv --portal Portal:TCV

        # Limit scope
        imas-codex scout wiki iter --max-pages 500 --max-depth 3

        # Preview configuration
        imas-codex scout wiki iter --dry-run
    """
    import asyncio

    from rich.console import Console
    from rich.panel import Panel

    from imas_codex.discovery import get_facility as get_facility_config
    from imas_codex.wiki.discovery import run_wiki_discovery

    console = Console()

    # Validate facility exists
    try:
        facility_config = get_facility_config(facility)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from e

    # Get portal from config if not specified
    wiki_sites = facility_config.get("wiki_sites", [])
    default_portal = None
    if wiki_sites and not portal:
        default_portal = wiki_sites[0].get("portal_page")

    # Show configuration
    console.print(
        Panel(
            f"[bold]Facility:[/bold] {facility}\n"
            f"[bold]Portal:[/bold] {portal or default_portal or '(auto-detect)'}\n"
            f"[bold]Model:[/bold] {model or 'anthropic/claude-sonnet-4-20250514'}\n"
            f"[bold]Cost Limit:[/bold] ${cost_limit:.2f}\n"
            f"[bold]Max Pages:[/bold] {max_pages or 'unlimited'}\n"
            f"[bold]Max Depth:[/bold] {max_depth or 'unlimited'}",
            title="Scout Configuration",
        )
    )

    if dry_run:
        console.print("\n[dim]Dry run - agent will not execute[/dim]")
        return

    console.print("\n[bold]Starting wiki discovery...[/bold]")
    asyncio.run(
        run_wiki_discovery(
            facility=facility,
            cost_limit_usd=cost_limit,
            max_pages=max_pages,
            max_depth=max_depth,
            verbose=verbose,
        )
    )


@scout.command("codes")
@click.argument("facility")
@click.option(
    "--focus",
    "-f",
    default=None,
    help="Specific codes to find (e.g., 'CHEASE LIUQE ASTRA')",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model (default: from config)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=10.0,
    type=float,
    help="Maximum cost in USD (default: 10.0)",
)
@click.option(
    "--max-steps",
    "-n",
    default=30,
    type=int,
    help="Maximum agent iterations (default: 30)",
)
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def scout_codes(
    facility: str,
    focus: str | None,
    model: str | None,
    cost_limit: float,
    max_steps: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Discover physics simulation codes.

    Finds physics codes (equilibrium solvers, transport codes, etc.)
    and their documentation, input files, and integration points.

    \\b
    EXAMPLES:
        # Discover all codes at EPFL
        imas-codex scout codes tcv

        # Find specific codes
        imas-codex scout codes tcv --focus "CHEASE LIUQE"

        # Search ITER for transport codes
        imas-codex scout codes iter --focus "transport JINTRAC"
    """
    _run_exploration_agent(
        facility=facility,
        resource="codes",
        prompt=focus,
        model=model,
        cost_limit=cost_limit,
        max_steps=max_steps,
        dry_run=dry_run,
        verbose=verbose,
        base_prompt="Find physics simulation codes and their documentation",
    )


@scout.command("data")
@click.argument("facility")
@click.option(
    "--focus",
    "-f",
    default=None,
    help="Data type focus (e.g., 'MDSplus', 'HDF5', 'IMAS')",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model (default: from config)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=10.0,
    type=float,
    help="Maximum cost in USD (default: 10.0)",
)
@click.option(
    "--max-steps",
    "-n",
    default=30,
    type=int,
    help="Maximum agent iterations (default: 30)",
)
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def scout_data(
    facility: str,
    focus: str | None,
    model: str | None,
    cost_limit: float,
    max_steps: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Discover data formats and locations.

    Finds MDSplus trees, HDF5 files, IMAS databases, UDA endpoints,
    and other data storage systems.

    \\b
    EXAMPLES:
        # Discover all data sources at ITER
        imas-codex scout data iter

        # Focus on MDSplus trees
        imas-codex scout data tcv --focus "MDSplus tree structure"

        # Find IMAS databases
        imas-codex scout data iter --focus "IMAS IDS locations"
    """
    _run_exploration_agent(
        facility=facility,
        resource="data",
        prompt=focus,
        model=model,
        cost_limit=cost_limit,
        max_steps=max_steps,
        dry_run=dry_run,
        verbose=verbose,
        base_prompt="Discover data formats and locations (MDSplus trees, HDF5 files, IMAS databases, UDA endpoints)",
    )


@scout.command("paths")
@click.argument("facility")
@click.option(
    "--focus",
    "-f",
    default=None,
    help="Path focus (e.g., 'home directories', 'shared codes')",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model (default: from config)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=10.0,
    type=float,
    help="Maximum cost in USD (default: 10.0)",
)
@click.option(
    "--max-steps",
    "-n",
    default=30,
    type=int,
    help="Maximum agent iterations (default: 30)",
)
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def scout_paths(
    facility: str,
    focus: str | None,
    model: str | None,
    cost_limit: float,
    max_steps: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Discover directory structure and key locations.

    Maps the filesystem to identify important directories:
    home areas, shared code locations, data archives, etc.

    \\b
    EXAMPLES:
        # Map EPFL directory structure
        imas-codex scout paths tcv

        # Find shared code locations
        imas-codex scout paths iter --focus "shared modules"
    """
    _run_exploration_agent(
        facility=facility,
        resource="paths",
        prompt=focus,
        model=model,
        cost_limit=cost_limit,
        max_steps=max_steps,
        dry_run=dry_run,
        verbose=verbose,
        base_prompt="Map directory structure and identify key locations",
    )


# ============================================================================
# IMAS DD Commands
# ============================================================================


@main.group()
def imas() -> None:
    """Manage IMAS Data Dictionary graph.

    \b
      imas-codex imas build    Build/update DD graph from imas-python
      imas-codex imas status   Show DD graph statistics
      imas-codex imas search   Semantic search for paths
      imas-codex imas versions List available DD versions
    """
    pass


@imas.command("build")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "-c",
    "--current-only",
    is_flag=True,
    help="Process only current DD version (default: all versions)",
)
@click.option(
    "--from-version",
    type=str,
    help="Start from a specific version (for incremental updates)",
)
@click.option(
    "-f", "--force", is_flag=True, help="Force regenerate all embeddings (ignore cache)"
)
@click.option(
    "--skip-clusters", is_flag=True, help="Skip importing semantic clusters into graph"
)
@click.option(
    "--skip-embeddings",
    is_flag=True,
    help="Skip embedding generation for current version paths",
)
@click.option(
    "--embedding-model",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence transformer model for embeddings",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Filter to specific IDS (space-separated, for testing)",
)
@click.option(
    "--dry-run", is_flag=True, help="Preview changes without writing to graph"
)
def imas_build(
    verbose: bool,
    quiet: bool,
    current_only: bool,
    from_version: str | None,
    force: bool,
    skip_clusters: bool,
    skip_embeddings: bool,
    embedding_model: str,
    ids_filter: str | None,
    dry_run: bool,
) -> None:
    """Build the IMAS Data Dictionary Knowledge Graph.

    Populates Neo4j with complete IMAS DD structure including:

    \b
    - DDVersion nodes with version tracking (PREDECESSOR relationships)
    - IDS nodes for top-level structures (core_profiles, equilibrium, etc.)
    - IMASPath nodes with hierarchical relationships (PARENT, IDS)
    - Unit nodes with HAS_UNIT relationships
    - CoordinateSpec nodes with HAS_COORDINATE relationships
    - PathChange nodes for metadata evolution between versions
    - RENAMED_TO relationships for path migrations
    - HAS_ERROR relationships linking data paths to error fields
    - Vector embeddings for semantic search (current version only)
    - SemanticCluster nodes with centroids for cluster-based search

    Embeddings are generated only for current version paths to avoid noise
    from deprecated/renamed paths. Version history is queryable via graph
    relationships, not vector search.

    \b
    Examples:
        imas-codex imas build                  # Build all DD versions (default)
        imas-codex imas build --current-only   # Build current version only
        imas-codex imas build --from-version 4.0.0  # Incremental from 4.0.0
        imas-codex imas build --force          # Regenerate all embeddings
        imas-codex imas build --dry-run -v     # Preview without writing
        imas-codex imas build --ids-filter "core_profiles equilibrium"  # Test subset
    """
    # Import and call the standalone build script's logic
    from imas_codex import dd_version as current_dd_version
    from imas_codex.graph.build_dd import build_dd_graph, get_all_dd_versions

    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress imas library's verbose logging
    logging.getLogger("imas").setLevel(logging.WARNING)

    try:
        # Determine versions to process
        available_versions = get_all_dd_versions()

        if current_only:
            versions = [current_dd_version]
        elif from_version:
            try:
                start_idx = available_versions.index(from_version)
                versions = available_versions[start_idx:]
            except ValueError as e:
                click.echo(f"Error: Unknown version {from_version}", err=True)
                click.echo(
                    f"Available: {', '.join(available_versions[:5])}...", err=True
                )
                raise SystemExit(1) from e
        else:
            # Default: all versions
            versions = available_versions

        logger.debug(f"Processing {len(versions)} DD versions")
        if len(versions) > 1:
            logger.debug(f"Versions: {versions[0]} → {versions[-1]}")

        # Parse IDS filter
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.debug(f"Filtering to IDS: {sorted(ids_set)}")

        if dry_run:
            click.echo("DRY RUN - no changes will be written to graph")

        # Build graph
        from imas_codex.graph import GraphClient

        with GraphClient() as client:
            stats = build_dd_graph(
                client=client,
                versions=versions,
                ids_filter=ids_set,
                include_clusters=not skip_clusters,
                include_embeddings=not skip_embeddings,
                dry_run=dry_run,
                embedding_model=embedding_model,
                force_embeddings=force,
            )

        # Report results
        click.echo("\n=== Build Complete ===")
        click.echo(f"Versions processed: {stats['versions_processed']}")
        click.echo(f"IDS nodes: {stats['ids_created']}")
        click.echo(f"IMASPath nodes created: {stats['paths_created']}")
        click.echo(f"Unit nodes: {stats['units_created']}")
        click.echo(f"PathChange nodes: {stats['path_changes_created']}")
        if not skip_embeddings:
            click.echo(f"Paths filtered (error/metadata): {stats['paths_filtered']}")
            click.echo(f"HAS_ERROR relationships: {stats['error_relationships']}")
            click.echo(f"Embeddings updated: {stats['embeddings_updated']}")
            click.echo(f"Embeddings cached: {stats['embeddings_cached']}")
            if stats.get("definitions_changed", 0) > 0:
                click.echo(
                    f"Definitions changed: {stats['definitions_changed']} "
                    "(deprecated paths cleaned)"
                )
        if not skip_clusters:
            click.echo(f"Cluster nodes: {stats['clusters_created']}")

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error building DD graph: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@imas.command("status")
@click.option(
    "--version", "-v", "version_filter", help="Show details for specific version"
)
def imas_status(version_filter: str | None) -> None:
    """Show IMAS DD graph statistics.

    Displays summary of DD graph content including version coverage,
    path counts, relationship statistics, and embedding status.

    \b
    Examples:
        imas-codex imas status             # Overall summary
        imas-codex imas status -v 4.1.0    # Details for specific version
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient

    console = Console()

    with GraphClient() as gc:
        # Get version summary
        versions = gc.query("""
            MATCH (v:DDVersion)
            OPTIONAL MATCH (v)-[:PREDECESSOR]->(prev:DDVersion)
            RETURN v.id AS version, v.is_current AS is_current, prev.id AS predecessor
            ORDER BY v.id
        """)

        if not versions:
            console.print("[yellow]No DD versions in graph.[/yellow]")
            console.print("Build with: imas-codex imas build")
            return

        # Version table
        version_table = Table(title="DD Versions in Graph")
        version_table.add_column("Version", style="cyan")
        version_table.add_column("Current", justify="center")
        version_table.add_column("Predecessor")

        for v in versions:
            current = "✓" if v["is_current"] else ""
            version_table.add_row(v["version"], current, v["predecessor"] or "—")

        console.print(version_table)

        # Get detailed stats for specific version or overall
        if version_filter:
            stats = gc.query(
                """
                MATCH (p:IMASPath)-[:INTRODUCED_IN]->(v:DDVersion {id: $version})
                WITH count(p) AS paths
                OPTIONAL MATCH (p2:IMASPath)-[:INTRODUCED_IN]->(:DDVersion {id: $version})
                WHERE p2.embedding IS NOT NULL
                RETURN paths, count(p2) AS with_embeddings
            """,
                version=version_filter,
            )

            if stats:
                console.print(f"\n[bold]Version {version_filter}:[/bold]")
                console.print(f"  Paths introduced: {stats[0]['paths']}")
                console.print(f"  With embeddings: {stats[0]['with_embeddings']}")
        else:
            # Overall stats
            overall = gc.query("""
                MATCH (p:IMASPath) WITH count(p) AS total_paths
                MATCH (i:IDS) WITH total_paths, count(i) AS ids_count
                MATCH (u:Unit) WITH total_paths, ids_count, count(u) AS unit_count
                MATCH (c:CoordinateSpec) WITH total_paths, ids_count, unit_count, count(c) AS coord_count
                MATCH (p2:IMASPath) WHERE p2.embedding IS NOT NULL
                WITH total_paths, ids_count, unit_count, coord_count, count(p2) AS with_embeddings
                OPTIONAL MATCH (:IMASPath)-[r:HAS_UNIT]->(:Unit)
                WITH total_paths, ids_count, unit_count, coord_count, with_embeddings, count(r) AS unit_rels
                OPTIONAL MATCH (:IMASPath)-[r2:HAS_COORDINATE]->()
                RETURN total_paths, ids_count, unit_count, coord_count, with_embeddings, unit_rels, count(r2) AS coord_rels
            """)

            if overall:
                s = overall[0]
                stats_table = Table(title="Graph Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Count", justify="right")

                stats_table.add_row("IMASPath nodes", str(s["total_paths"]))
                stats_table.add_row("IDS nodes", str(s["ids_count"]))
                stats_table.add_row("Unit nodes", str(s["unit_count"]))
                stats_table.add_row("CoordinateSpec nodes", str(s["coord_count"]))
                stats_table.add_row("Paths with embeddings", str(s["with_embeddings"]))
                stats_table.add_row("HAS_UNIT relationships", str(s["unit_rels"]))
                stats_table.add_row(
                    "HAS_COORDINATE relationships", str(s["coord_rels"])
                )

                console.print()
                console.print(stats_table)

            # Cluster stats
            clusters = gc.query("MATCH (c:SemanticCluster) RETURN count(c) AS count")
            if clusters and clusters[0]["count"] > 0:
                console.print(f"\nSemanticCluster nodes: {clusters[0]['count']}")


@imas.command("search")
@click.argument("query")
@click.option("-n", "--limit", default=10, help="Max results (default: 10)")
@click.option("--ids", help="Filter to specific IDS")
@click.option("--version", "-v", "version_filter", help="Filter to DD version")
def imas_search(
    query: str, limit: int, ids: str | None, version_filter: str | None
) -> None:
    """Semantic search for IMAS paths.

    Uses vector embeddings to find paths matching natural language queries.

    \b
    Examples:
        imas-codex imas search "electron temperature"
        imas-codex imas search "magnetic field boundary" --ids equilibrium
        imas-codex imas search "plasma current" -n 20
    """
    from rich.console import Console
    from rich.table import Table
    from sentence_transformers import SentenceTransformer

    from imas_codex.graph import GraphClient

    console = Console()

    # Generate query embedding
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(query).tolist()

    # Build filter clause
    where_clauses = []
    if ids:
        where_clauses.append(f"node.id STARTS WITH '{ids}/'")
    if version_filter:
        where_clauses.append(f"node.dd_version = '{version_filter}'")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    with GraphClient() as gc:
        results = gc.query(
            f"""
            CALL db.index.vector.queryNodes("imas_path_embedding", $limit * 2, $embedding)
            YIELD node, score
            {where_clause}
            RETURN node.id AS path, score, node.units AS units, node.documentation AS doc
            LIMIT $limit
        """,
            embedding=embedding,
            limit=limit,
        )

    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        return

    table = Table(title=f"Search: '{query}'")
    table.add_column("Score", style="dim", width=6)
    table.add_column("Path", style="cyan")
    table.add_column("Units", width=8)

    for r in results:
        units = r["units"] or ""
        table.add_row(f"{r['score']:.3f}", r["path"], units)

    console.print(table)


def _resolve_version(version_spec: str) -> str:
    """Resolve incomplete version specification to a full version.

    Args:
        version_spec: Version spec like "4", "4.1", or "4.1.0"

    Returns:
        Full version string like "4.1.0"

    Raises:
        ValueError: If no matching version found
    """
    from imas_codex.graph.build_dd import get_all_dd_versions

    all_versions = get_all_dd_versions()

    # If exact match, return it
    if version_spec in all_versions:
        return version_spec

    # Parse the version spec
    parts = version_spec.split(".")
    if len(parts) == 1:
        # Major only (e.g., "4") - find latest in that major
        major = parts[0]
        matching = [v for v in all_versions if v.startswith(f"{major}.")]
        if matching:
            return matching[-1]  # sorted, so last is highest
    elif len(parts) == 2:
        # Major.minor (e.g., "4.1") - find latest patch in that minor
        prefix = f"{parts[0]}.{parts[1]}."
        matching = [v for v in all_versions if v.startswith(prefix)]
        if matching:
            return matching[-1]

    raise ValueError(f"No DD version matching '{version_spec}'")


@imas.command("version")
@click.argument("version", required=False)
@click.option(
    "--available",
    "-a",
    is_flag=True,
    help="Show all available versions from imas-python",
)
@click.option(
    "--list",
    "-l",
    "list_versions",
    is_flag=True,
    help="List all versions in the graph",
)
def imas_version(version: str | None, available: bool, list_versions: bool) -> None:
    """Show DD version info (defaults to latest/current version).

    Without arguments, shows details for the current (latest) DD version.
    Incomplete version numbers are resolved to the latest matching version:
    - "4" resolves to the latest 4.x.x (e.g., 4.1.0)
    - "4.0" resolves to the latest 4.0.x (e.g., 4.0.0)

    \b
    Examples:
        imas-codex imas version              # Show current version details
        imas-codex imas version 4            # Show latest 4.x version
        imas-codex imas version 4.0.0        # Show specific version
        imas-codex imas version --list       # List all versions in graph
        imas-codex imas version --available  # All available versions
    """
    from rich.console import Console

    console = Console()

    if available:
        from imas_codex.graph.build_dd import get_all_dd_versions

        versions_list = get_all_dd_versions()
        console.print(f"[bold]Available DD versions ({len(versions_list)}):[/bold]")
        # Group by major version
        major_groups: dict[str, list[str]] = {}
        for v in versions_list:
            major = v.split(".")[0]
            major_groups.setdefault(major, []).append(v)

        for major, vers in sorted(major_groups.items()):
            console.print(f"  {major}.x: {', '.join(vers)}")
        return

    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        if list_versions:
            # Summary view of all versions
            _show_versions_summary(gc, console)
        elif version:
            # Resolve incomplete version and show details
            try:
                resolved = _resolve_version(version)
                if resolved != version:
                    console.print(f"[dim]Resolved '{version}' → {resolved}[/dim]")
                _show_version_details(gc, console, resolved)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
        else:
            # Default: show details for current version
            current = gc.query(
                "MATCH (v:DDVersion {is_current: true}) RETURN v.id AS version"
            )
            if current:
                _show_version_details(gc, console, current[0]["version"])
            else:
                console.print("[yellow]No current version set in graph.[/yellow]")
                console.print("Build with: imas-codex imas build")


def _show_version_details(gc, console, version: str) -> None:
    """Show detailed statistics for a specific DD version."""
    from rich.table import Table

    # Check version exists
    check = gc.query(
        "MATCH (v:DDVersion {id: $version}) RETURN v",
        version=version,
    )
    if not check:
        console.print(f"[red]Version {version} not found in graph.[/red]")
        return

    # Get version metadata
    meta = gc.query(
        """
        MATCH (v:DDVersion {id: $version})
        RETURN v.is_current AS is_current,
               v.embeddings_built_at AS embeddings_built_at,
               v.embeddings_model AS embeddings_model,
               v.embeddings_count AS embeddings_count
        """,
        version=version,
    )[0]

    # Get path statistics - simplified query to avoid timeout
    stats = gc.query(
        """
        MATCH (v:DDVersion {id: $version})
        OPTIONAL MATCH (introduced:IMASPath)-[:INTRODUCED_IN]->(v)
        WITH v, count(introduced) AS paths_introduced
        OPTIONAL MATCH (deprecated:IMASPath)-[:DEPRECATED_IN]->(v)
        WITH v, paths_introduced, count(deprecated) AS paths_deprecated
        OPTIONAL MATCH (embedded:IMASPath)-[:INTRODUCED_IN]->(v)
        WHERE embedded.embedding IS NOT NULL
        RETURN paths_introduced, paths_deprecated, count(embedded) AS paths_embedded
        """,
        version=version,
    )[0]

    # Get PathChange statistics (count changes involving this version)
    path_changes = gc.query(
        """
        MATCH (pc:PathChange)-[:VERSION]->(v:DDVersion {id: $version})
        RETURN pc.change_type AS change_type, count(*) AS count
        ORDER BY count DESC
        LIMIT 10
        """,
        version=version,
    )

    # Display version header
    current_marker = " [green](current)[/green]" if meta["is_current"] else ""
    console.print(f"\n[bold]DD Version {version}{current_marker}[/bold]\n")

    # Path statistics table
    table = Table(title="Path Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Paths introduced", str(stats["paths_introduced"]))
    table.add_row("Paths deprecated", str(stats["paths_deprecated"]))
    table.add_row("Paths embedded", str(stats["paths_embedded"]))

    console.print(table)

    # PathChange statistics if any
    if path_changes:
        pc_table = Table(title="Metadata Changes", show_header=True)
        pc_table.add_column("Change Type", style="magenta")
        pc_table.add_column("Count", justify="right")

        for pc in path_changes:
            pc_table.add_row(pc["change_type"], str(pc["count"]))

        console.print(pc_table)

    # Embeddings metadata
    if meta["embeddings_built_at"]:
        console.print(
            f"\n[dim]Embeddings: {meta['embeddings_count']} paths, "
            f"model: {meta['embeddings_model']}, "
            f"built: {meta['embeddings_built_at']}[/dim]"
        )


def _show_versions_summary(gc, console) -> None:
    """Show summary of all DD versions in graph."""
    # Count paths INTRODUCED_IN each version (not deprecated paths)
    versions = gc.query("""
        MATCH (v:DDVersion)
        OPTIONAL MATCH (p:IMASPath)-[:INTRODUCED_IN]->(v)
        WITH v, count(p) AS introduced
        OPTIONAL MATCH (p2:IMASPath)-[:INTRODUCED_IN]->(v) WHERE p2.embedding IS NOT NULL
        RETURN v.id AS version, v.is_current AS is_current, introduced, count(p2) AS embedded
        ORDER BY v.id
    """)

    if not versions:
        console.print("[yellow]No versions in graph.[/yellow]")
        console.print("Build with: imas-codex imas build")
        return

    console.print("[bold]DD versions in graph:[/bold]")
    for v in versions:
        current = " [green](current)[/green]" if v["is_current"] else ""
        embedded = f", {v['embedded']} embedded" if v["embedded"] else ""
        console.print(
            f"  {v['version']}: {v['introduced']} paths introduced{embedded}{current}"
        )


# Clusters Commands - Semantic Clustering of IMAS Paths
# ============================================================================


@main.group()
def clusters() -> None:
    """Manage semantic clusters of IMAS data paths.

    \b
      imas-codex clusters build   Build HDBSCAN clusters from embeddings
      imas-codex clusters label   Generate LLM labels for clusters
      imas-codex clusters sync    Sync clusters to Neo4j graph
      imas-codex clusters status  Show cluster statistics
    """
    pass


@clusters.command("build")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("-f", "--force", is_flag=True, help="Force rebuild even if files exist")
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    help="Minimum cluster size for HDBSCAN (default: 2)",
)
@click.option(
    "--min-samples",
    type=int,
    default=2,
    help="Minimum samples for HDBSCAN core points (default: 2)",
)
@click.option(
    "--cluster-method",
    type=click.Choice(["eom", "leaf"]),
    default="eom",
    help="HDBSCAN cluster selection method: 'eom' for broader, 'leaf' for finer",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include (space-separated)",
)
def clusters_build(
    verbose: bool,
    quiet: bool,
    force: bool,
    min_cluster_size: int,
    min_samples: int,
    cluster_method: str,
    ids_filter: str | None,
) -> None:
    """Build semantic clusters of IMAS data paths using HDBSCAN.

    This command creates clusters based on semantic embeddings of path
    documentation. It does NOT generate LLM labels - use 'clusters label'
    for that step.

    \b
    Examples:
      imas-codex clusters build                    # Build with defaults
      imas-codex clusters build -v -f              # Force rebuild, verbose
      imas-codex clusters build --ids-filter "core_profiles equilibrium"
    """
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from imas_codex.core.clusters import Clusters
    from imas_codex.embeddings.config import EncoderConfig

    ids_set = set(ids_filter.split()) if ids_filter else None

    if ids_set:
        click.echo(f"Building clusters for IDS: {sorted(ids_set)}")
    else:
        click.echo("Building clusters for all IDS...")

    try:
        encoder_config = EncoderConfig(
            ids_set=ids_set,
            use_rich=not quiet,
        )
        clusters_manager = Clusters(encoder_config=encoder_config)
        output_file = clusters_manager.file_path

        should_build = force or not output_file.exists()
        if not should_build and clusters_manager.needs_rebuild():
            should_build = True
            click.echo("Dependencies changed, rebuilding...")

        if should_build:
            config_overrides = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_method,
                "use_rich": not quiet,
            }
            clusters_manager.build(force=force, **config_overrides)
            click.echo(f"✓ Built clusters: {output_file}")
        else:
            click.echo(f"Clusters already exist: {output_file}")

    except Exception as e:
        click.echo(f"Error building clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("label")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("-f", "--force", is_flag=True, help="Force regenerate all labels")
@click.option(
    "--cost-limit",
    type=float,
    default=10.0,
    help="Maximum cost in USD for LLM requests (default: $10)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of clusters per LLM batch (default: from settings)",
)
@click.option(
    "--export/--no-export",
    default=True,
    help="Export labels to JSON for version control (default: True)",
)
def clusters_label(
    verbose: bool,
    quiet: bool,
    force: bool,
    cost_limit: float,
    batch_size: int | None,
    export: bool,
) -> None:
    """Generate LLM labels for semantic clusters.

    Uses the configured language model to generate human-readable labels
    and descriptions for each cluster. Labels are cached to avoid
    regenerating existing ones unless --force is used.

    \b
    Examples:
      imas-codex clusters label                # Label unlabeled clusters
      imas-codex clusters label -f             # Force regenerate all labels
      imas-codex clusters label --cost-limit 5 # Limit to $5 USD
    """
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from imas_codex.clusters.label_cache import LabelCache
    from imas_codex.clusters.labeler import ClusterLabeler
    from imas_codex.core.clusters import Clusters
    from imas_codex.embeddings.config import EncoderConfig

    try:
        # Load clusters
        encoder_config = EncoderConfig(use_rich=not quiet)
        clusters_manager = Clusters(encoder_config=encoder_config)

        if not clusters_manager.is_available():
            click.echo("No clusters found. Run 'clusters build' first.", err=True)
            raise SystemExit(1)

        cluster_data = clusters_manager.get_clusters()
        click.echo(f"Found {len(cluster_data)} clusters")

        # Initialize cache and labeler
        label_cache = LabelCache()
        labeler = ClusterLabeler(batch_size=batch_size)

        # Get cached and uncached clusters
        if force:
            uncached = cluster_data
            cached = {}
            click.echo("Force mode: regenerating all labels")
        else:
            cached, uncached = label_cache.get_many(cluster_data)
            click.echo(f"Cached: {len(cached)}, Need labeling: {len(uncached)}")

        if not uncached:
            click.echo("All clusters already labeled")
            if export:
                exported = label_cache.export_labels()
                click.echo(f"Exported {len(exported)} labels to definitions")
            return

        # Generate labels with cost tracking
        click.echo(f"Generating labels (cost limit: ${cost_limit:.2f})...")
        labels = labeler.generate_labels(uncached)

        # Store in cache
        label_tuples = [
            (c.get("paths", []), lbl.label, lbl.description)
            for c, lbl in zip(uncached, labels, strict=False)
            if lbl
        ]
        stored = label_cache.set_many(label_tuples)
        click.echo(f"✓ Cached {stored} new labels")

        # Export to JSON for version control
        if export:
            exported = label_cache.export_labels()
            click.echo(f"✓ Exported {len(exported)} labels to definitions")

    except Exception as e:
        click.echo(f"Error labeling clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("sync")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
def clusters_sync(verbose: bool, quiet: bool, dry_run: bool) -> None:
    """Sync semantic clusters to Neo4j knowledge graph.

    Creates/updates SemanticCluster nodes and IN_CLUSTER relationships
    linking IMASPath nodes to their clusters.

    \b
    Examples:
      imas-codex clusters sync              # Sync clusters to graph
      imas-codex clusters sync --dry-run    # Preview without changes
    """
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from imas_codex.graph.build_dd import import_semantic_clusters
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()

        if dry_run:
            click.echo("Dry run - previewing cluster sync...")
        else:
            click.echo("Syncing clusters to graph...")

        count = import_semantic_clusters(client, dry_run=dry_run)

        if dry_run:
            click.echo(f"Would sync {count} clusters")
        else:
            click.echo(f"✓ Synced {count} clusters to graph")

    except Exception as e:
        click.echo(f"Error syncing clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("status")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed statistics")
def clusters_status(verbose: bool) -> None:
    """Show cluster statistics and cache status.

    \b
    Examples:
      imas-codex clusters status     # Basic stats
      imas-codex clusters status -v  # Detailed stats
    """
    from imas_codex.clusters.label_cache import LabelCache
    from imas_codex.core.clusters import Clusters
    from imas_codex.embeddings.config import EncoderConfig

    try:
        # Cluster file status
        encoder_config = EncoderConfig()
        clusters_manager = Clusters(encoder_config=encoder_config)

        click.echo("=== Cluster Status ===")
        if clusters_manager.is_available():
            cluster_data = clusters_manager.get_clusters()
            click.echo(f"Clusters file: {clusters_manager.file_path}")
            click.echo(f"Total clusters: {len(cluster_data)}")

            if verbose:
                # Count cross-IDS vs intra-IDS
                cross_ids = sum(1 for c in cluster_data if c.get("cross_ids", False))
                click.echo(f"Cross-IDS clusters: {cross_ids}")
                click.echo(f"Intra-IDS clusters: {len(cluster_data) - cross_ids}")
        else:
            click.echo("No clusters file found")

        # Label cache status
        click.echo("\n=== Label Cache ===")
        label_cache = LabelCache()
        stats = label_cache.get_stats()
        click.echo(f"Cache file: {stats['cache_file']}")
        click.echo(f"Total labels: {stats['total_labels']}")
        click.echo(f"Cache size: {stats['cache_size_mb']:.2f} MB")

        if verbose and stats["by_model"]:
            click.echo("Labels by model:")
            for model, count in stats["by_model"].items():
                click.echo(f"  {model}: {count}")

    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        raise SystemExit(1) from e


# ============================================================================
# Enrich Commands - AI-Assisted Metadata Generation
# ============================================================================


@main.group()
def enrich() -> None:
    """Enrich graph nodes with AI-generated metadata.

    Uses CodeAgent to analyze and describe data from multiple sources:
    - TreeNodes from MDSplus/HDF5 trees
    - Wiki pages from facility documentation
    - Code files from ingested source

    \b
      imas-codex enrich nodes     Enrich TreeNode metadata
      imas-codex enrich run       Run a custom enrichment task
    """
    pass


@enrich.command("run")
@click.argument("task")
@click.option(
    "--type",
    "agent_type",
    default="enrichment",
    type=click.Choice(["enrichment", "mapping", "exploration"]),
    help="Agent type to use",
)
@click.option(
    "--cost-limit",
    "-c",
    default=None,
    type=float,
    help="Maximum cost budget in USD",
)
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def agent_run(
    task: str, agent_type: str, cost_limit: float | None, verbose: bool
) -> None:
    """Run an agent with a task using smolagents CodeAgent.

    The agent generates Python code to autonomously:
    - Query the Neo4j knowledge graph
    - Search code examples and IMAS paths
    - Adapt and self-debug to solve problems

    Examples:
        imas-codex agent run "Describe what \\RESULTS::ASTRA is used for"

        imas-codex agent run "Find IMAS paths for electron temperature" --type mapping

        imas-codex agent run "Explore EPFL for equilibrium codes" --type exploration -c 1.0
    """
    from imas_codex.agentic import quick_task_sync

    click.echo(f"Running {agent_type} agent (CodeAgent)...")
    if verbose:
        click.echo(f"Task: {task}")
    if cost_limit:
        click.echo(f"Cost limit: ${cost_limit:.2f}")
    click.echo()

    try:
        result = quick_task_sync(task, agent_type, verbose, cost_limit)
        click.echo("\n=== Agent Response ===")
        click.echo(result)
    except Exception as e:
        click.echo(f"Agent error: {e}", err=True)
        raise SystemExit(1) from None


@enrich.command("nodes")
@click.argument("paths", nargs=-1)
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Guidance for enrichment (e.g., 'Focus on equilibrium signals')",
)
@click.option(
    "--limit", "-n", default=None, type=int, help="Max nodes to enrich (default: all)"
)
@click.option("--tree", default=None, help="Filter to specific tree name")
@click.option(
    "--status", default="pending", help="Target status (pending, enriched, stale)"
)
@click.option("--force", is_flag=True, help="Include all nodes regardless of status")
@click.option(
    "--linked",
    is_flag=True,
    help="Only nodes with code context (more reliable enrichment)",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help="Paths per batch (auto-selected if not set: 100 for Flash, 200 for Pro)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model to use (default: from config for 'enrichment' task)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=None,
    type=float,
    help="Maximum cost budget in USD (default: no limit)",
)
@click.option("--dry-run", is_flag=True, help="Preview without persisting to graph")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def enrich_nodes(
    paths: tuple[str, ...],
    prompt: str | None,
    limit: int | None,
    tree: str | None,
    status: str,
    force: bool,
    linked: bool,
    batch_size: int | None,
    model: str | None,
    cost_limit: float | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Enrich TreeNode metadata using CodeAgent.

    The agent generates Python code to gather context from the
    knowledge graph and code examples, then produces physics-accurate
    descriptions. Uses adaptive problem-solving and self-debugging.

    Paths are grouped by parent node for efficient batch processing.
    A Rich progress display shows current batch, tree, and statistics.

    By default, discovers and processes ALL nodes with status='pending'.
    Use --tree to filter to a specific tree.
    Use --limit to cap the number of nodes processed.

    \b
    EXAMPLES:
        # Enrich all pending nodes in the results tree
        imas-codex enrich nodes --tree results

        # Focus on specific physics with guidance
        imas-codex enrich nodes --tree results -p "Focus on equilibrium signals"

        # Enrich all pending nodes across all trees
        imas-codex enrich nodes

        # Limit to first 100 nodes
        imas-codex enrich nodes --tree tcv_shot --limit 100

        # Process stale nodes (marked for re-enrichment)
        imas-codex enrich nodes --status stale

        # Only enrich nodes with code context (more reliable)
        imas-codex enrich nodes --linked

        # Include ALL nodes (pending + enriched) for (re-)enrichment
        imas-codex enrich nodes --force

        # Re-enrich only already processed nodes
        imas-codex enrich nodes --status enriched

        # Enrich specific paths
        imas-codex enrich nodes "\\RESULTS::IBS" "\\RESULTS::LIUQE"

        # Use Pro model for higher quality
        imas-codex enrich nodes --model google/gemini-3-pro-preview -b 200

        # Preview without saving
        imas-codex enrich nodes --dry-run
    """
    import asyncio
    import logging

    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    from imas_codex.agentic import (
        BatchProgress,
        batch_enrich_paths,
        compose_batches,
        discover_nodes_to_enrich,
        estimate_enrichment_cost,
        get_model_for_task,
        get_parent_path,
    )

    console = Console()

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Resolve model from config if not specified
    effective_model = model or get_model_for_task("enrichment")

    # Determine target status
    # --force means "include all statuses" (both pending and already enriched)
    # --status specifies a specific status to target
    target_status = "all" if force else status

    # Get paths - either from args or discover from graph
    if paths:
        path_list = list(paths)
        console.print(f"[cyan]Enriching {len(path_list)} specified paths...[/cyan]")
        tree_name = tree or "unknown"
    else:
        if force:
            filter_desc = "all statuses (--force)"
        else:
            filter_desc = f"status='{target_status}'"
        if linked:
            filter_desc += ", with code context"
        console.print(f"[cyan]Discovering nodes with {filter_desc}...[/cyan]")
        nodes = discover_nodes_to_enrich(
            tree_name=tree,
            status=target_status,
            with_context_only=linked,
            limit=limit,
        )
        if not nodes:
            console.print(f"[yellow]No nodes found with {filter_desc}.[/yellow]")
            return
        path_list = [n["path"] for n in nodes]
        with_ctx = sum(1 for n in nodes if n["has_context"])
        console.print(
            f"[green]Found {len(path_list)} nodes[/green] "
            f"([dim]{with_ctx} with code context[/dim])"
        )
        tree_name = tree or nodes[0].get("tree", "unknown") if nodes else "unknown"

    # Auto-select batch size based on model
    # Benchmarked optimal values:
    # - Flash: batch 100 = 2.5 paths/sec, 100% success, 71% high confidence
    # - Pro: batch 200 = 0.4 paths/sec, 100% success, 100% high confidence
    effective_batch_size = batch_size
    if effective_batch_size is None:
        if "pro" in effective_model.lower():
            effective_batch_size = 200
        else:
            effective_batch_size = 100

    # Compose smart batches grouped by parent (for preview)
    batches = compose_batches(
        path_list, batch_size=effective_batch_size, group_by_parent=True
    )

    # Show cost estimate
    cost_est = estimate_enrichment_cost(len(path_list), effective_batch_size)
    cost_info = (
        f"[dim]Batches: {len(batches)} | "
        f"Est. time: {cost_est['estimated_hours'] * 60:.0f}min | "
        f"Est. cost: ${cost_est['estimated_cost']:.2f}"
    )
    if cost_limit is not None:
        cost_info += f" | Limit: ${cost_limit:.2f}"
    cost_info += "[/dim]"
    console.print()
    console.print(cost_info)
    console.print(f"[dim]Model: {effective_model} (smolagents CodeAgent)[/dim]")

    if dry_run:
        console.print("\n[yellow][DRY RUN] Will not persist to graph[/yellow]")
        console.print("\n[cyan]Batch preview:[/cyan]")
        for i, batch in enumerate(batches[:5], 1):
            parent = get_parent_path(batch[0]) if batch else "?"
            console.print(f"  Batch {i}: {len(batch)} paths from [bold]{parent}[/bold]")
            for p in batch[:3]:
                console.print(f"    {p}")
            if len(batch) > 3:
                console.print(f"    [dim]... and {len(batch) - 3} more[/dim]")
        if len(batches) > 5:
            console.print(f"\n  [dim]... and {len(batches) - 5} more batches[/dim]")
        return

    # State for progress display (updated by callback)
    class ProgressState:
        def __init__(self) -> None:
            self.batch_num = 0
            self.total_batches = len(batches)
            self.parent_path = ""
            self.paths_processed = 0
            self.paths_total = len(path_list)
            self.enriched = 0
            self.errors = 0
            self.high_conf = 0
            self.elapsed = 0.0

        def rate(self) -> float:
            return self.paths_processed / self.elapsed if self.elapsed > 0 else 0

    state = ProgressState()

    def create_progress_display() -> Group:
        """Create the rich progress display."""
        # Overall progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Enriching"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("", total=state.paths_total)
        progress.update(task, completed=state.paths_processed)

        # Current batch info
        batch_info = Table.grid(padding=(0, 2))
        batch_info.add_column(style="dim")
        batch_info.add_column(style="bold")
        batch_info.add_row(
            "Batch:",
            f"{state.batch_num}/{state.total_batches}",
        )
        batch_info.add_row("Tree:", tree_name)
        batch_info.add_row("Group:", state.parent_path or "—")

        # Statistics
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="dim")
        stats.add_column(justify="right")
        stats.add_row("Enriched:", f"[green]{state.enriched}[/green]")
        stats.add_row("Errors:", f"[red]{state.errors}[/red]")
        stats.add_row("High conf:", f"[cyan]{state.high_conf}[/cyan]")
        stats.add_row("Rate:", f"{state.rate():.1f} paths/sec")

        # Combine into panels
        info_panel = Panel(
            batch_info,
            title="[bold]Current Batch[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
        stats_panel = Panel(
            stats,
            title="[bold]Statistics[/bold]",
            border_style="green",
            padding=(0, 1),
        )

        # Layout with side-by-side panels
        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_column(ratio=1)
        layout.add_row(info_panel, stats_panel)

        return Group(progress, layout)

    # Progress callback that updates state
    live_display: Live | None = None

    def on_progress(p: BatchProgress) -> None:
        state.batch_num = p.batch_num
        state.total_batches = p.total_batches
        state.parent_path = p.parent_path
        state.paths_processed = p.paths_processed
        state.enriched = p.enriched
        state.errors = p.errors
        state.high_conf = p.high_confidence
        state.elapsed = p.elapsed_seconds
        if live_display:
            live_display.update(create_progress_display())

    async def run_with_progress() -> list:
        nonlocal live_display
        with Live(
            create_progress_display(), console=console, refresh_per_second=4
        ) as live:
            live_display = live
            return await batch_enrich_paths(
                paths=path_list,
                tree_name=tree_name,
                batch_size=effective_batch_size,
                verbose=verbose,
                dry_run=False,  # We handle dry_run above
                model=effective_model,
                progress_callback=on_progress,
            )

    console.print()
    results = asyncio.run(run_with_progress())
    console.print()

    # Final summary
    enriched_count = sum(1 for r in results if r.description)
    error_count = sum(1 for r in results if r.error)
    high_conf_count = sum(1 for r in results if r.confidence == "high")

    summary = Table(title="Enrichment Summary", show_header=False, box=None)
    summary.add_column(style="dim")
    summary.add_column(justify="right")
    summary.add_row("Total paths:", str(len(results)))
    summary.add_row("Enriched:", f"[green]{enriched_count}[/green]")
    summary.add_row("Errors:", f"[red]{error_count}[/red]")
    summary.add_row("High confidence:", f"[cyan]{high_conf_count}[/cyan]")
    summary.add_row("Time:", f"{state.elapsed:.1f}s")
    if state.elapsed > 0:
        summary.add_row("Rate:", f"{len(results) / state.elapsed:.1f} paths/sec")
    summary.add_row("", "[green]✓ Persisted to graph[/green]")

    console.print(Panel(summary, border_style="green"))


@enrich.command("mark-stale")
@click.argument("pattern")
@click.option("--tree", default=None, help="Filter to specific tree name")
@click.option("--dry-run", is_flag=True, help="Preview without updating")
def enrich_mark_stale(
    pattern: str,
    tree: str | None,
    dry_run: bool,
) -> None:
    """Mark TreeNodes as stale for re-enrichment.

    Matches nodes by path pattern and sets enrichment_status='stale'.
    Use this when new context is available and you want to re-process nodes.

    \b
    EXAMPLES:
        # Mark all LIUQE nodes as stale
        imas-codex enrich mark-stale "LIUQE"

        # Mark nodes in results tree matching pattern
        imas-codex enrich mark-stale "THOMSON" --tree results

        # Preview what would be marked
        imas-codex enrich mark-stale "BOLO" --dry-run
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        where_clauses = [
            "t.enrichment_status = 'enriched'",
            f"t.path CONTAINS '{pattern}'",
        ]
        if tree:
            where_clauses.append(f't.tree_name = "{tree}"')

        # Count matching nodes
        count_query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            RETURN count(t) AS count
        """
        result = gc.query(count_query)
        count = result[0]["count"] if result else 0

        if count == 0:
            click.echo(f"No enriched nodes matching '{pattern}' found.")
            return

        if dry_run:
            click.echo(f"[DRY RUN] Would mark {count} nodes as stale")
            # Show sample
            sample_query = f"""
                MATCH (t:TreeNode)
                WHERE {" AND ".join(where_clauses)}
                RETURN t.path AS path LIMIT 10
            """
            samples = gc.query(sample_query)
            for s in samples:
                click.echo(f"  {s['path']}")
            if count > 10:
                click.echo(f"  ... and {count - 10} more")
            return

        # Mark as stale
        update_query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            SET t.enrichment_status = 'stale'
            RETURN count(t) AS updated
        """
        result = gc.query(update_query)
        updated = result[0]["updated"] if result else 0
        click.echo(f"✓ Marked {updated} nodes as stale")


# ============================================================================
# Discovery Commands - Graph-Led Facility Exploration
# ============================================================================


@main.group()
def discover():
    """Discover facility resources with graph-led exploration.

    \b
    Discovery Pipeline:
      1. discover paths → Directory structure + LLM scoring
      2. discover code  → Source files in scored paths
         discover docs  → Wiki pages + filesystem artifacts
         discover data  → MDSplus trees, HDF5, IMAS DBs

    \b
    Commands:
      discover paths <facility>    Scan and score directory structure
      discover code <facility>     Find source files (placeholder)
      discover docs <facility>     Find documentation (placeholder)
      discover data <facility>     Find data sources (placeholder)

    \b
    Management:
      discover status <facility>   Show discovery statistics
      discover inspect <facility>  Debug view of scanned/scored paths
      discover clear <facility>    Clear paths (reset discovery)
      discover sources             Manage documentation sources

    The graph is the single source of truth. All discovery operations
    are idempotent and resume from the current graph state.
    """
    pass


@discover.command("paths")
@click.argument("facility")
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=10.0,
    help="Maximum LLM spend in USD (default: $10)",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Maximum paths to process (for debugging)",
)
@click.option(
    "--focus",
    "-f",
    type=str,
    help="Natural language focus (e.g., 'equilibrium codes')",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=float,
    help="Minimum score to expand paths",
)
@click.option(
    "--scan-workers",
    default=1,
    type=int,
    help="Number of scan workers (default: 1, single SSH connection)",
)
@click.option(
    "--score-workers",
    default=4,
    type=int,
    help="Number of score workers (default: 4, parallel LLM calls)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    default=False,
    help="SSH scan only, no LLM scoring (fast, requires SSH access)",
)
@click.option(
    "--score-only",
    is_flag=True,
    default=False,
    help="LLM scoring only, no SSH scanning (offline, graph-only)",
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
def discover_paths(
    facility: str,
    cost_limit: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    scan_workers: int,
    score_workers: int,
    scan_only: bool,
    score_only: bool,
    no_rich: bool,
) -> None:
    """Discover and score directory structure at a facility.

    \b
    Examples:
      imas-codex discover paths <facility>              # Default $10 limit
      imas-codex discover paths <facility> -c 20.0      # $20 limit
      imas-codex discover paths iter --focus "equilibrium codes"
      imas-codex discover paths iter --scan-only        # SSH only, no LLM
      imas-codex discover paths iter --score-only       # LLM only, no SSH

    Parallel scan workers enumerate directories via SSH while score workers
    classify paths using LLM. Both run concurrently with the graph as
    coordination. Discovery is idempotent - rerun to continue from current state.

    \b
    Phase separation:
      --scan-only   Fast enumeration requiring SSH access. Populates graph
                    with directory listings but no LLM scoring.
      --score-only  Offline scoring using existing graph data. No SSH required.
                    Only expands paths already scored above threshold.
    """
    from rich.console import Console

    console = Console()

    # Validate mutually exclusive flags
    if scan_only and score_only:
        console.print(
            "[red]Error: --scan-only and --score-only are mutually exclusive[/red]"
        )
        raise SystemExit(1)

    _run_iterative_discovery(
        facility=facility,
        budget=cost_limit,
        limit=limit,
        focus=focus,
        threshold=threshold,
        num_scan_workers=scan_workers,
        num_score_workers=score_workers,
        scan_only=scan_only,
        score_only=score_only,
        no_rich=no_rich,
    )


def _run_iterative_discovery(
    facility: str,
    budget: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    num_scan_workers: int = 1,
    num_score_workers: int = 4,
    scan_only: bool = False,
    score_only: bool = False,
    no_rich: bool = False,
) -> None:
    """Run parallel scan/score discovery.

    Args:
        facility: Facility ID
        budget: Maximum LLM cost in dollars
        limit: Maximum paths to process
        focus: Natural language focus for scoring
        threshold: Minimum score to expand paths
        num_scan_workers: Number of parallel scan workers
        num_score_workers: Number of parallel score workers
        scan_only: If True, only run SSH scanning (no LLM scoring)
        score_only: If True, only run LLM scoring (no SSH scanning)
        no_rich: If True, use logging output instead of rich progress
    """
    import asyncio
    import sys

    from imas_codex.agentic.agents import get_model_for_task
    from imas_codex.discovery import get_discovery_stats, seed_facility_roots

    # Auto-detect if rich can run (TTY check) or use no_rich flag
    use_rich = not no_rich and sys.stdout.isatty()

    if use_rich:
        from rich.console import Console

        console = Console()
    else:
        console = None  # Will use logging instead
        # Configure logging output for non-rich mode
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    # Setup logger for non-rich mode
    disc_logger = logging.getLogger("imas_codex.discovery")
    if not use_rich:
        disc_logger.setLevel(logging.INFO)

    def log_print(msg: str, style: str = "") -> None:
        """Print to console or log, stripping rich markup."""
        import re

        # Strip rich markup for logging: [bold], [red], etc
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            disc_logger.info(clean_msg)

    # Check if we have any paths, seed if not
    stats = get_discovery_stats(facility)
    if stats["total"] == 0:
        if score_only:
            log_print(
                "[red]Error: --score-only requires existing paths in the graph.[/red]"
            )
            log_print(
                f"[yellow]Run 'imas-codex discover paths {facility}' or "
                "'--scan-only' first to populate the graph.[/yellow]"
            )
            raise SystemExit(1)
        log_print(f"[cyan]Seeding root paths for {facility}...[/cyan]")
        seed_facility_roots(facility)
        stats = get_discovery_stats(facility)

    # For score_only, check we have listed (scannable) paths
    if score_only and stats.get("listed", 0) == 0:
        log_print("[yellow]Warning: No 'listed' paths available for scoring.[/yellow]")
        log_print(
            "Paths must be scanned before they can be scored. "
            "Checking for already-scored paths to expand..."
        )

    # Adjust worker counts based on mode flags
    effective_scan_workers = 0 if score_only else num_scan_workers
    effective_score_workers = 0 if scan_only else num_score_workers

    # Get model name for display
    model_name = get_model_for_task("score")
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/") :]

    # Display mode
    mode_str = ""
    if scan_only:
        mode_str = " [bold cyan](SCAN ONLY)[/bold cyan]"
    elif score_only:
        mode_str = " [bold green](SCORE ONLY)[/bold green]"

    log_print(
        f"[bold]Starting parallel discovery for {facility.upper()}[/bold]{mode_str}"
    )
    if not scan_only:
        log_print(f"Cost limit: ${budget:.2f}")
    if limit:
        log_print(f"Path limit: {limit}")
    if not scan_only:
        log_print(f"Model: {model_name}")
    log_print(
        f"Workers: {effective_scan_workers} scan, {effective_score_workers} score"
    )
    if focus and not scan_only:
        log_print(f"Focus: {focus}")

    # Run the async discovery loop
    try:
        result, scored_this_run = asyncio.run(
            _async_discovery_loop(
                facility=facility,
                budget=budget,
                limit=limit,
                focus=focus,
                threshold=threshold,
                console=console,
                num_scan_workers=effective_scan_workers,
                num_score_workers=effective_score_workers,
                scan_only=scan_only,
                score_only=score_only,
                use_rich=use_rich,
            )
        )

        # Print detailed summary with paths scored this run
        _print_discovery_summary(
            console, facility, result, scored_this_run, scan_only=scan_only
        )

    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"\n[red]Error: {e}[/red]")
        raise SystemExit(1) from e


def _print_discovery_summary(
    console,
    facility: str,
    result: dict,
    scored_this_run: set[str] | None = None,
    scan_only: bool = False,
) -> None:
    """Print detailed discovery summary with statistics.

    Args:
        console: Rich console (or None for logging mode)
        facility: Facility ID
        result: Discovery result dict
        scored_this_run: Set of paths scored in this discovery run
        scan_only: If True, show scan-focused summary
    """
    from imas_codex.discovery import get_discovery_stats
    from imas_codex.discovery.frontier import get_high_value_paths

    disc_logger = logging.getLogger("imas_codex.discovery")

    # Get final graph stats
    stats = get_discovery_stats(facility)
    coverage = stats["scored"] / stats["total"] * 100 if stats["total"] > 0 else 0
    elapsed = result.get("elapsed_seconds", 0)
    # Use consistent time format: Xh Ym Zs
    if elapsed >= 3600:
        hours, rem = divmod(int(elapsed), 3600)
        mins = rem // 60
        elapsed_str = f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    elif elapsed >= 60:
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if secs else f"{mins}m"
    else:
        elapsed_str = f"{int(elapsed)}s"
    scan_rate = result.get("scan_rate")
    score_rate = result.get("score_rate")

    # Non-rich mode: log simple summary
    if console is None:
        disc_logger.info(
            f"Discovery complete: scanned={result['scanned']}, "
            f"scored={result['scored']}, cost=${result['cost']:.3f}, "
            f"elapsed={elapsed_str}"
        )
        disc_logger.info(
            f"Graph state: total={stats['total']}, scored={stats['scored']} "
            f"({coverage:.1f}%), pending={stats.get('pending', 0)}"
        )
        return

    # Rich mode: use panels
    from rich.panel import Panel
    from rich.text import Text

    console.print()

    # Build compact summary - width=100 to match progress display
    facility_upper = facility.upper()
    summary = Text()

    # Row 1: This Run stats (adjust based on scan_only mode)
    summary.append("This Run  ", style="bold cyan")
    summary.append(f"scanned {result['scanned']:,}", style="white")
    if not scan_only:
        summary.append(" · ", style="dim")
        summary.append(f"scored {result['scored']:,}", style="white")
        summary.append(" · ", style="dim")
        summary.append(f"cost ${result['cost']:.3f}", style="yellow")
    summary.append(" · ", style="dim")
    summary.append(f"{elapsed_str}", style="cyan")
    summary.append("\n")

    # Row 2: Rates
    summary.append("Rates     ", style="bold blue")
    if scan_rate:
        summary.append(f"scan {scan_rate:.1f}/s", style="white")
    else:
        summary.append("scan -", style="dim")
    if not scan_only:
        summary.append(" · ", style="dim")
        if score_rate:
            summary.append(f"score {score_rate:.1f}/s", style="white")
        else:
            summary.append("score -", style="dim")
    summary.append("\n")

    # Row 3: Graph State
    summary.append("Graph     ", style="bold green")
    summary.append(f"total {stats['total']:,}", style="white")
    summary.append(" · ", style="dim")
    if not scan_only:
        summary.append(
            f"coverage {coverage:.1f}%", style="green" if coverage > 50 else "yellow"
        )
        summary.append(" · ", style="dim")
    frontier = stats.get("discovered", 0) + stats.get("listed", 0)
    summary.append(f"frontier {frontier:,}", style="cyan")
    summary.append(" · ", style="dim")
    summary.append(f"depth {stats.get('max_depth', 0)}", style="cyan")

    # Title based on mode
    if scan_only:
        title = f"[bold blue]{facility_upper} Scan Complete[/bold blue]"
        border = "blue"
    else:
        title = f"[bold green]{facility_upper} Discovery Complete[/bold green]"
        border = "green"

    console.print(
        Panel(
            summary,
            title=title,
            border_style=border,
            width=100,  # Match progress display width
        )
    )

    # Show high-value paths found IN THIS RUN only (skip in scan_only mode)
    if scan_only:
        # Show next step hint
        console.print()
        console.print(
            f"[dim]Next step: Run 'imas-codex discover paths {facility} --score-only' "
            "to score listed paths.[/dim]"
        )
        return

    all_high_value = get_high_value_paths(facility, min_score=0.7, limit=50)

    # Filter to paths scored in this run
    if scored_this_run:
        high_value = [p for p in all_high_value if p["path"] in scored_this_run]
    else:
        high_value = all_high_value

    if high_value:
        console.print()
        console.print(
            f"[bold]High-value paths discovered this run ({len(high_value)}):[/bold]"
        )
        for p in high_value[:5]:
            purpose = p.get("path_purpose", "unknown")
            desc = p.get("description", "")[:55]
            if len(p.get("description", "")) > 55:
                desc += "..."
            console.print(f"  [{p['score']:.2f}] [cyan]{p['path']}[/cyan]")
            if desc:
                console.print(f"         {purpose}: {desc}")
        if len(high_value) > 5:
            console.print(f"  ... and {len(high_value) - 5} more high-value paths")


async def _async_discovery_loop(
    facility: str,
    budget: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    console,
    num_scan_workers: int = 2,
    num_score_workers: int = 4,
    scan_only: bool = False,
    score_only: bool = False,
    use_rich: bool = True,
) -> tuple[dict, set[str]]:
    """Async discovery loop with parallel scan/score workers.

    Args:
        facility: Facility ID
        budget: Maximum LLM cost in dollars
        limit: Maximum paths to process
        focus: Natural language focus for scoring
        threshold: Minimum score to expand paths
        console: Rich console for output (None for logging mode)
        num_scan_workers: Number of parallel scan workers
        num_score_workers: Number of parallel score workers
        scan_only: If True, skip scoring (scan workers only)
        score_only: If True, skip scanning (score workers only)
        use_rich: If True, use rich display; otherwise use logging

    Returns:
        Tuple of (result dict, set of paths scored in this run)
    """
    import logging

    from imas_codex.discovery.parallel import run_parallel_discovery

    disc_logger = logging.getLogger("imas_codex.discovery")

    disc_logger = logging.getLogger("imas_codex.discovery")
    scored_this_run: set[str] = set()

    if use_rich:
        # Rich-based visual progress display
        from imas_codex.agentic.agents import get_model_for_task
        from imas_codex.discovery.parallel_progress import ParallelProgressDisplay

        model_name = get_model_for_task("score")

        with ParallelProgressDisplay(
            facility=facility,
            cost_limit=budget,
            model=model_name,
            console=console,
            focus=focus or "",
            scan_only=scan_only,
            score_only=score_only,
        ) as display:
            # Periodic graph state refresh
            async def refresh_graph_state():
                while True:
                    display.refresh_from_graph(facility)
                    await asyncio.sleep(0.5)

            async def queue_ticker():
                while True:
                    display.tick()
                    await asyncio.sleep(0.15)

            import asyncio

            refresh_task = asyncio.create_task(refresh_graph_state())
            ticker_task = asyncio.create_task(queue_ticker())

            def on_scan(msg, stats, paths=None, scan_results=None):
                display.update_scan(msg, stats, paths=paths, scan_results=scan_results)

            def on_score(msg, stats, results=None):
                display.update_score(msg, stats, results=results)

            try:
                result = await run_parallel_discovery(
                    facility=facility,
                    cost_limit=budget,
                    path_limit=limit,
                    focus=focus,
                    threshold=threshold,
                    num_scan_workers=num_scan_workers,
                    num_score_workers=num_score_workers,
                    on_scan_progress=on_scan,
                    on_score_progress=on_score,
                )
            finally:
                refresh_task.cancel()
                ticker_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
                try:
                    await ticker_task
                except asyncio.CancelledError:
                    pass

            display.refresh_from_graph(facility)
            scored_this_run = display.get_paths_scored_this_run()

    else:
        # Logging-based progress - report only on batch completions
        def on_scan_log(msg: str, stats, paths=None, scan_results=None):
            """Log scan progress only on significant batches."""
            # Only log when batch completed (not idle/waiting messages)
            # Debug: always log to confirm callback is working
            disc_logger.debug(
                f"scan callback: msg={msg}, results={len(scan_results) if scan_results else 0}"
            )
            if scan_results and len(scan_results) > 0:
                disc_logger.info(
                    f"SCAN batch: {len(scan_results)} paths, "
                    f"total: {stats.processed}, rate: {stats.rate:.1f}/s"
                )

        def on_score_log(msg: str, stats, results=None):
            """Log score progress only on batch completion."""
            if results and len(results) > 0:
                # Track scored paths
                for r in results:
                    if r.get("path"):
                        scored_this_run.add(r["path"])

                disc_logger.info(
                    f"SCORE batch: {len(results)} paths, "
                    f"total: {stats.processed}, cost: ${stats.cost:.3f}"
                )

        result = await run_parallel_discovery(
            facility=facility,
            cost_limit=budget,
            path_limit=limit,
            focus=focus,
            threshold=threshold,
            num_scan_workers=num_scan_workers,
            num_score_workers=num_score_workers,
            on_scan_progress=on_scan_log,
            on_score_progress=on_score_log,
        )

    # Return result for summary (box printed by caller, no duplicate text)
    return (
        {
            "cycles": 1,  # Continuous operation, not cycle-based
            "scanned": result["scanned"],
            "scored": result["scored"],
            "expanded": result.get("expanded", 0),
            "cost": result["cost"],
            "elapsed_seconds": result["elapsed_seconds"],
            "scan_rate": result.get("scan_rate"),
            "score_rate": result.get("score_rate"),
        },
        scored_this_run,
    )


# Status and management commands under discover group


@discover.command("status")
@click.argument("facility")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["paths", "code", "docs", "data"]),
    help="Show detailed status for specific domain",
)
def discover_status(facility: str, as_json: bool, domain: str | None) -> None:
    """Show discovery statistics for a facility.

    Displays:
    - Path counts by status (pending, scanned, scored, skipped)
    - Coverage percentage
    - Frontier size (paths awaiting scan)
    - High-value paths (score > 0.7)

    Examples:
        imas-codex discover status iter
        imas-codex discover status iter --json
        imas-codex discover status iter --domain paths
    """
    import json as json_module

    from imas_codex.discovery import get_discovery_stats, get_high_value_paths
    from imas_codex.discovery.progress import print_discovery_status

    try:
        if as_json:
            stats = get_discovery_stats(facility)
            high_value = get_high_value_paths(facility, min_score=0.7, limit=20)
            output = {
                "facility": facility,
                "stats": stats,
                "high_value_paths": high_value,
            }
            click.echo(json_module.dumps(output, indent=2))
        else:
            print_discovery_status(facility)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover.command("inspect")
@click.argument("facility")
@click.option(
    "--scanned",
    "-s",
    default=5,
    type=int,
    help="Number of scanned paths to show",
)
@click.option(
    "--scored",
    "-r",
    default=5,
    type=int,
    help="Number of scored paths to show",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def discover_inspect(facility: str, scanned: int, scored: int, as_json: bool) -> None:
    """Inspect scanned and scored paths from the graph.

    Displays sample paths with their attributes to assess how well
    the scan and score processes are functioning.

    Examples:
        imas-codex discover inspect iter
        imas-codex discover inspect iter --scanned 10 --scored 10
        imas-codex discover inspect iter --json
    """
    import json

    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient

    console = Console()

    try:
        with GraphClient() as gc:
            # Get sample scanned paths
            scanned_paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = 'scanned'
                RETURN p.path AS path, p.total_files AS total_files,
                       p.total_dirs AS total_dirs, p.has_readme AS has_readme,
                       p.has_makefile AS has_makefile, p.has_git AS has_git,
                       p.depth AS depth, p.scanned_at AS scanned_at
                ORDER BY p.scanned_at DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=scanned,
            )

            # Get sample scored paths
            scored_paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = 'scored' AND p.score IS NOT NULL
                RETURN p.path AS path, p.score AS score,
                       p.score_code AS score_code, p.score_data AS score_data,
                       p.score_imas AS score_imas, p.path_purpose AS path_purpose,
                       p.description AS description, p.total_files AS total_files,
                       p.scored_at AS scored_at
                ORDER BY p.score DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=scored,
            )

        if as_json:
            output = {
                "facility": facility,
                "scanned_paths": list(scanned_paths),
                "scored_paths": list(scored_paths),
            }
            console.print_json(json.dumps(output, default=str))
            return

        # Print scanned paths table
        console.print(f"\n[bold cyan]Scanned Paths ({len(scanned_paths)})[/bold cyan]")
        if scanned_paths:
            scan_table = Table(show_header=True, header_style="bold")
            scan_table.add_column("Path", style="cyan", no_wrap=True, max_width=40)
            scan_table.add_column("Files", justify="right")
            scan_table.add_column("Dirs", justify="right")
            scan_table.add_column("README", justify="center")
            scan_table.add_column("Makefile", justify="center")
            scan_table.add_column("Git", justify="center")
            scan_table.add_column("Depth", justify="right")

            for p in scanned_paths:
                path_display = p["path"]
                if len(path_display) > 40:
                    path_display = "..." + path_display[-37:]
                scan_table.add_row(
                    path_display,
                    str(p.get("total_files", 0) or 0),
                    str(p.get("total_dirs", 0) or 0),
                    "✓" if p.get("has_readme") else "",
                    "✓" if p.get("has_makefile") else "",
                    "✓" if p.get("has_git") else "",
                    str(p.get("depth", 0) or 0),
                )
            console.print(scan_table)
        else:
            console.print("  (no scanned paths found)")

        # Print scored paths table
        console.print(f"\n[bold green]Scored Paths ({len(scored_paths)})[/bold green]")
        if scored_paths:
            score_table = Table(show_header=True, header_style="bold")
            score_table.add_column("Path", style="cyan", no_wrap=True, max_width=35)
            score_table.add_column("Score", justify="right", style="bold")
            score_table.add_column("Code", justify="right")
            score_table.add_column("Data", justify="right")
            score_table.add_column("IMAS", justify="right")
            score_table.add_column("Purpose", max_width=15)
            score_table.add_column("Description", max_width=30)

            for p in scored_paths:
                path_display = p["path"]
                if len(path_display) > 35:
                    path_display = "..." + path_display[-32:]

                # Color score
                score_val = p.get("score", 0) or 0
                if score_val >= 0.7:
                    score_str = f"[green]{score_val:.2f}[/green]"
                elif score_val >= 0.4:
                    score_str = f"[yellow]{score_val:.2f}[/yellow]"
                else:
                    score_str = f"[red]{score_val:.2f}[/red]"

                desc = p.get("description", "") or ""
                if len(desc) > 30:
                    desc = desc[:27] + "..."

                score_table.add_row(
                    path_display,
                    score_str,
                    f"{p.get('score_code', 0) or 0:.2f}",
                    f"{p.get('score_data', 0) or 0:.2f}",
                    f"{p.get('score_imas', 0) or 0:.2f}",
                    p.get("path_purpose", "") or "",
                    desc,
                )
            console.print(score_table)
        else:
            console.print("  (no scored paths found)")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover.command("clear")
@click.argument("facility")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt",
)
def discover_clear(facility: str, force: bool) -> None:
    """Clear all discovered paths for a facility.

    This is a destructive operation that removes all FacilityPath nodes
    for the specified facility. Use with caution.

    Examples:
        imas-codex discover clear iter
        imas-codex discover clear iter --force
    """
    from imas_codex.discovery import clear_facility_paths, get_discovery_stats

    try:
        stats = get_discovery_stats(facility)
        total = stats.get("total", 0)

        if total == 0:
            click.echo(f"No paths to clear for {facility}")
            return

        if not force:
            click.confirm(
                f"This will delete {total} paths for {facility}. Continue?",
                abort=True,
            )

        deleted = clear_facility_paths(facility)
        click.echo(f"✓ Deleted {deleted} paths for {facility}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover.command("seed")
@click.argument("facility")
@click.option(
    "--path",
    "-p",
    multiple=True,
    help="Additional root paths to seed",
)
def discover_seed(facility: str, path: tuple[str, ...]) -> None:
    """Seed facility root paths without scanning.

    Creates initial FacilityPath nodes for the facility's actionable paths
    and any additional paths specified. Useful for testing graph setup.

    Examples:
        imas-codex discover seed iter
        imas-codex discover seed iter -p /home/custom/path
    """
    from imas_codex.discovery import seed_facility_roots

    try:
        additional_paths = list(path) if path else None
        created = seed_facility_roots(facility, additional_paths)
        click.echo(f"✓ Created {created} root path(s) for {facility}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# ============================================================================
# Placeholder Discovery Commands (Future Implementation)
# ============================================================================


@discover.command("code")
@click.argument("facility")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be discovered without making changes",
)
def discover_code(facility: str, dry_run: bool) -> None:
    """Discover source files in scored paths.

    NOT YET IMPLEMENTED.

    This command will scan high-value paths (score >= 0.7) and create
    SourceFile nodes for Python, Fortran, MATLAB, and other code files.

    Prerequisites:
        Run 'discover paths <facility>' first to identify high-value directories.

    Examples:
        imas-codex discover code iter
        imas-codex discover code iter --dry-run
    """
    from rich.console import Console

    from imas_codex.discovery import get_discovery_stats

    console = Console()
    stats = get_discovery_stats(facility)

    if stats.get("scored", 0) == 0:
        console.print(
            f"[yellow]⚠ No scored paths found for {facility.upper()}[/yellow]\n"
        )
        console.print("Discovery pipeline:")
        console.print(
            "  1. [bold]discover paths[/bold] → 2. discover code → 3. ingest code"
        )
        console.print(f"\nNext step: [cyan]imas-codex discover paths {facility}[/cyan]")
        raise SystemExit(1)

    console.print("[yellow]discover code: Not yet implemented[/yellow]")
    console.print(f"\nCurrent paths status for {facility.upper()}:")
    console.print(f"  Scored paths: {stats.get('scored', 0)}")
    console.print(f"  High-value (≥0.7): {stats.get('high_value', 'unknown')}")
    console.print("\nThis feature will scan scored paths for source files.")
    raise SystemExit(1)


@discover.command("docs")
@click.argument("facility")
@click.option(
    "--source",
    "-s",
    help="Specific DocSource ID to crawl (otherwise all for facility)",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=5.0,
    help="Maximum LLM spend in USD (default: $5)",
)
def discover_docs(facility: str, source: str | None, cost_limit: float) -> None:
    """Discover documentation: wiki pages and filesystem artifacts.

    NOT YET IMPLEMENTED.

    This command crawls configured documentation sources (wiki sites,
    readthedocs, etc.) and scans scored paths for document artifacts
    (PDFs, READMEs, etc.).

    Prerequisites:
        - Configure doc sources: 'discover sources add --facility <facility> ...'
        - Or run 'discover paths' first for filesystem artifacts

    Examples:
        imas-codex discover docs iter
        imas-codex discover docs iter --source iter-confluence
        imas-codex discover docs iter -c 10.0
    """
    from rich.console import Console

    console = Console()
    console.print("[yellow]discover docs: Not yet implemented[/yellow]")
    console.print("\nThis feature will:")
    console.print("  1. Crawl configured DocSource sites (wikis, readthedocs)")
    console.print("  2. Scan scored paths for document artifacts (PDFs, READMEs)")
    console.print("  3. Score and prioritize discovered documentation")
    console.print(
        f"\nUse 'discover sources' to manage documentation sources for {facility}."
    )
    raise SystemExit(1)


@discover.command("data")
@click.argument("facility")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be discovered without making changes",
)
def discover_data(facility: str, dry_run: bool) -> None:
    """Discover data sources: MDSplus trees, HDF5 files, IMAS databases.

    NOT YET IMPLEMENTED.

    This command scans for data infrastructure and creates nodes for
    MDSplus servers/trees, HDF5 datasets, and IMAS database entries.

    Prerequisites:
        Run 'discover paths <facility>' first to identify data directories.

    Examples:
        imas-codex discover data iter
        imas-codex discover data iter --dry-run
    """
    from rich.console import Console

    console = Console()
    console.print("[yellow]discover data: Not yet implemented[/yellow]")
    console.print("\nThis feature will discover:")
    console.print("  - MDSplus servers and tree structures")
    console.print("  - HDF5 datasets and schemas")
    console.print("  - IMAS database entries")
    raise SystemExit(1)


# ============================================================================
# Documentation Sources Management
# ============================================================================


@discover.group("sources")
def discover_sources():
    """Manage documentation sources for discovery.

    \b
    Commands:
      list     List all configured sources
      add      Add a new documentation source
      rm       Remove a documentation source
      enable   Enable a paused source
      disable  Disable a source

    Documentation sources are stored in the graph and can be queried.
    Credentials are stored securely in the system keyring.

    Examples:
        imas-codex discover sources list
        imas-codex discover sources add --name "ITER Wiki" --url https://...
    """
    pass


@discover_sources.command("list")
@click.option("--facility", "-f", help="Filter by facility")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def sources_list(facility: str | None, as_json: bool) -> None:
    """List all configured documentation sources.

    Examples:
        imas-codex discover sources list
        imas-codex discover sources list --facility iter
        imas-codex discover sources list --json
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient

    console = Console()

    try:
        with GraphClient() as gc:
            if facility:
                sources = gc.query(
                    """
                    MATCH (s:DocSource)-[:FACILITY_ID]->(f:Facility {id: $facility})
                    RETURN s.id AS id, s.name AS name, s.url AS url,
                           s.source_type AS type, s.status AS status,
                           s.page_count AS pages
                    ORDER BY s.name
                    """,
                    facility=facility,
                )
            else:
                sources = gc.query(
                    """
                    MATCH (s:DocSource)
                    OPTIONAL MATCH (s)-[:FACILITY_ID]->(f:Facility)
                    RETURN s.id AS id, s.name AS name, s.url AS url,
                           s.source_type AS type, s.status AS status,
                           s.page_count AS pages, f.id AS facility
                    ORDER BY s.name
                    """
                )

        if as_json:
            import json

            console.print_json(json.dumps(list(sources), default=str))
            return

        if not sources:
            console.print("[dim]No documentation sources configured.[/dim]")
            console.print("\nAdd a source with:")
            console.print("  imas-codex discover sources add --name '...' --url '...'")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="dim")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Pages", justify="right")
        if not facility:
            table.add_column("Facility")

        for idx, src in enumerate(sources):
            status = src.get("status", "active")
            status_style = "green" if status == "active" else "yellow"
            row = [
                str(idx),
                src["id"],
                src.get("name", ""),
                src.get("type", ""),
                f"[{status_style}]{status}[/{status_style}]",
                str(src.get("pages", 0) or 0),
            ]
            if not facility:
                row.append(src.get("facility", "-"))
            table.add_row(*row)

        console.print(table)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover_sources.command("add")
@click.option("--name", "-n", required=True, help="Human-readable name")
@click.option("--url", "-u", required=True, help="Base URL of the source")
@click.option("--portal", "-p", help="Portal/starting page (relative to URL)")
@click.option(
    "--type",
    "-t",
    "source_type",
    type=click.Choice(
        [
            "mediawiki",
            "confluence",
            "readthedocs",
            "github_wiki",
            "sphinx",
            "generic_html",
        ]
    ),
    default="generic_html",
    help="Type of documentation site",
)
@click.option(
    "--auth",
    type=click.Choice(["none", "ssh_proxy", "basic", "session"]),
    default="none",
    help="Authentication method",
)
@click.option("--facility", "-f", help="Link to facility (creates if doesn't exist)")
@click.option("--credential-service", help="Keyring service name for credentials")
def sources_add(
    name: str,
    url: str,
    portal: str | None,
    source_type: str,
    auth: str,
    facility: str | None,
    credential_service: str | None,
) -> None:
    """Add a new documentation source.

    Examples:
        imas-codex discover sources add -n "ITER Wiki" -u https://wiki.iter.org
        imas-codex discover sources add -n "TCV Wiki" -u https://spcwiki.epfl.ch \\
            --facility tcv --type mediawiki --auth ssh_proxy
        imas-codex discover sources add -n "CHEASE Docs" -u https://chease.readthedocs.io \\
            --type readthedocs
    """
    import re
    from datetime import datetime

    from imas_codex.graph import GraphClient

    # Generate ID from name
    source_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    try:
        with GraphClient() as gc:
            # Check if source already exists
            existing = gc.query(
                "MATCH (s:DocSource {id: $id}) RETURN s.id", id=source_id
            )
            if existing:
                click.echo(f"Error: Source '{source_id}' already exists", err=True)
                raise SystemExit(1)

            # Create facility if specified and doesn't exist
            if facility:
                fac_exists = gc.query(
                    "MATCH (f:Facility {id: $id}) RETURN f.id", id=facility
                )
                if not fac_exists:
                    gc.query(
                        """
                        CREATE (f:Facility {id: $id, name: $name})
                        """,
                        id=facility,
                        name=facility.upper(),
                    )
                    click.echo(f"Created facility: {facility}")

            # Create DocSource
            now = datetime.now(UTC).isoformat()
            gc.query(
                """
                CREATE (s:DocSource {
                    id: $id,
                    name: $name,
                    url: $url,
                    portal_page: $portal,
                    source_type: $source_type,
                    auth_type: $auth,
                    credential_service: $credential_service,
                    status: 'active',
                    created_at: datetime($now),
                    page_count: 0,
                    artifact_count: 0
                })
                """,
                id=source_id,
                name=name,
                url=url,
                portal=portal,
                source_type=source_type,
                auth=auth,
                credential_service=credential_service,
                now=now,
            )

            # Link to facility if specified
            if facility:
                gc.query(
                    """
                    MATCH (s:DocSource {id: $source_id})
                    MATCH (f:Facility {id: $facility_id})
                    CREATE (s)-[:FACILITY_ID]->(f)
                    """,
                    source_id=source_id,
                    facility_id=facility,
                )

        click.echo(f"✓ Created documentation source: {source_id}")
        if facility:
            click.echo(f"  Linked to facility: {facility}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover_sources.command("rm")
@click.argument("source_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def sources_rm(source_id: str, force: bool) -> None:
    """Remove a documentation source.

    Examples:
        imas-codex discover sources rm iter-wiki
        imas-codex discover sources rm iter-wiki --force
    """
    from imas_codex.graph import GraphClient

    try:
        with GraphClient() as gc:
            # Check if exists and get stats
            result = gc.query(
                """
                MATCH (s:DocSource {id: $id})
                OPTIONAL MATCH (s)<-[:SOURCE]-(p:WikiPage)
                RETURN s.name AS name, count(p) AS page_count
                """,
                id=source_id,
            )

            if not result:
                click.echo(f"Error: Source '{source_id}' not found", err=True)
                raise SystemExit(1)

            name = result[0]["name"]
            page_count = result[0]["page_count"]

            if not force:
                msg = f"Delete source '{name}'"
                if page_count:
                    msg += f" and {page_count} associated pages"
                msg += "?"
                click.confirm(msg, abort=True)

            # Delete source and associated pages
            gc.query(
                """
                MATCH (s:DocSource {id: $id})
                OPTIONAL MATCH (s)<-[:SOURCE]-(p:WikiPage)
                DETACH DELETE s, p
                """,
                id=source_id,
            )

        click.echo(f"✓ Deleted source: {source_id}")
        if page_count:
            click.echo(f"  Removed {page_count} associated pages")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover_sources.command("enable")
@click.argument("source_id")
def sources_enable(source_id: str) -> None:
    """Enable a paused documentation source.

    Examples:
        imas-codex discover sources enable iter-wiki
    """
    from imas_codex.graph import GraphClient

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:DocSource {id: $id})
                SET s.status = 'active'
                RETURN s.name AS name
                """,
                id=source_id,
            )

            if not result:
                click.echo(f"Error: Source '{source_id}' not found", err=True)
                raise SystemExit(1)

        click.echo(f"✓ Enabled: {source_id}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover_sources.command("disable")
@click.argument("source_id")
def sources_disable(source_id: str) -> None:
    """Disable a documentation source.

    Examples:
        imas-codex discover sources disable iter-wiki
    """
    from imas_codex.graph import GraphClient

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (s:DocSource {id: $id})
                SET s.status = 'disabled'
                RETURN s.name AS name
                """,
                id=source_id,
            )

            if not result:
                click.echo(f"Error: Source '{source_id}' not found", err=True)
                raise SystemExit(1)

        click.echo(f"✓ Disabled: {source_id}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# ============================================================================
# Dynamic Facility Commands
# ============================================================================


def _create_facility_command(facility_name: str, description: str) -> click.Command:
    """Create a CLI command for a specific facility."""

    @click.command(name=facility_name, help=f"Show configuration for {description}")
    @click.option("--config", "show_config", is_flag=True, help="Show full config")
    def facility_cmd(show_config: bool) -> None:
        """Show facility configuration.

        Facility knowledge is now stored in the graph database.
        Use SSH directly for exploration:
            ssh tcv "which python3; python3 --version"

        See imas_codex/config/README.md for comprehensive exploration guidance.

        Examples:
            # Show facility info
            imas-codex tcv

            # Show full config
            imas-codex tcv --config
        """
        from imas_codex.discovery import get_facility

        data = get_facility(facility_name)

        if show_config:
            import json

            click.echo(json.dumps(data, indent=2, default=str))
        else:
            click.echo(f"Facility: {data.get('id', facility_name)}")
            click.echo(f"Name: {data.get('name', '')}")
            click.echo(f"Machine: {data.get('machine', '')}")
            click.echo(f"Description: {data.get('description', '')}")
            if data.get("ssh_host"):
                click.echo(f"SSH Host: {data['ssh_host']}")
            click.echo("\nUse --config for full configuration")

    return facility_cmd


def _register_facility_commands() -> None:
    """Register a CLI command for each configured facility."""
    try:
        from imas_codex.discovery import get_facility, list_facilities

        for facility_name in list_facilities():
            try:
                data = get_facility(facility_name)
                cmd = _create_facility_command(
                    facility_name, data.get("description", "")
                )
                main.add_command(cmd)
            except Exception:
                # Skip facilities with invalid configs
                pass
    except Exception:
        # Don't fail CLI startup if facility loading fails
        pass


# Register facility commands at import time
_register_facility_commands()


if __name__ == "__main__":
    main()
