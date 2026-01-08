"""CLI interface for IMAS Codex Server."""

import logging
import os
import warnings
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

    Use subcommands to start servers or manage facilities:

    \b
      imas-codex serve imas       Start the IMAS Data Dictionary MCP server
      imas-codex serve agents     Start the Agents MCP server
      imas-codex facilities list  List configured facilities
      imas-codex facilities show  Show facility configuration
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

    from imas_codex.agents.server import AgentsServer

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

    from imas_codex.discovery import get_facility, get_facility_public

    try:
        if public_only:
            data = get_facility_public(name)
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
        imas-codex ingest run epfl

        # Process only high-priority files
        imas-codex ingest run epfl --min-score 0.7

        # Limit to 100 files
        imas-codex ingest run epfl -n 100

        # Preview what would be processed
        imas-codex ingest run epfl --dry-run
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
        imas-codex ingest status epfl
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
        imas-codex ingest queue epfl /path/a.py /path/b.py /path/c.py

        # Discover from file (for large batches)
        imas-codex ingest queue epfl -f files.txt

        # Discover from stdin (pipe from rg)
        ssh epfl 'rg -l "IMAS" /home' | imas-codex ingest queue epfl --stdin

        # Set priority score
        imas-codex ingest queue epfl /path/a.py -s 0.9

        # Preview
        imas-codex ingest queue epfl /path/a.py --dry-run
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
        imas-codex ingest list epfl

        # List failed files
        imas-codex ingest list epfl -s failed

        # List all files
        imas-codex ingest list epfl -s all
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


@main.group()
def wiki() -> None:
    """Ingest wiki documentation from remote facilities.

    \b
      imas-codex wiki discover <facility>  Discover wiki pages
      imas-codex wiki ingest <facility>    Ingest wiki pages
      imas-codex wiki status <facility>    Show ingestion statistics
    """
    pass


@wiki.command("discover")
@click.argument("facility")
@click.option(
    "--start-page",
    default="Portal:TCV",
    help="Page to start crawling from (default: Portal:TCV)",
)
@click.option(
    "--limit",
    "-n",
    default=100,
    type=int,
    help="Maximum pages to discover (default: 100)",
)
@click.option(
    "--priority-only",
    is_flag=True,
    help="Only show priority pages (known high-value pages)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview pages without queuing in graph",
)
def wiki_discover(
    facility: str,
    start_page: str,
    limit: int,
    priority_only: bool,
    dry_run: bool,
) -> None:
    """Discover wiki pages and queue them for ingestion.

    Starts from a portal page, finds all internal wiki links, and creates
    WikiPage nodes with status='discovered' in the graph. The ingest command
    then processes pages from the graph queue.

    Examples:
        # Discover and queue pages
        imas-codex wiki discover epfl

        # Queue priority pages (known high-value pages)
        imas-codex wiki discover epfl --priority-only

        # Preview without queuing
        imas-codex wiki discover epfl --dry-run
    """
    from rich.console import Console
    from rich.table import Table

    from imas_codex.graph import GraphClient
    from imas_codex.wiki import discover_wiki_pages, queue_wiki_pages
    from imas_codex.wiki.scraper import get_priority_pages

    console = Console()

    # Show current graph state
    if not priority_only:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility_id})
                RETURN count(wp) AS total
                """,
                facility_id=facility,
            )
            existing_count = result[0]["total"] if result else 0
            if existing_count > 0:
                console.print(
                    f"[dim]Graph already has {existing_count} pages. "
                    f"Discovery will find up to {limit} NEW pages.[/dim]\n"
                )

    # Get pages to queue
    if priority_only:
        pages = get_priority_pages()
        interest_score = 0.9  # Priority pages are high-value
        is_priority = True
        console.print(
            f"[cyan]Queueing {len(pages)} priority wiki pages for {facility}...[/cyan]"
        )
    else:
        console.print(f"[cyan]Discovering wiki pages for {facility}...[/cyan]")
        console.print(f"[dim]Starting from: {start_page}, limit: {limit}[/dim]\n")
        pages = discover_wiki_pages(
            start_page=start_page,
            facility=facility,
            max_pages=limit,
        )
        interest_score = 0.5  # Default score for discovered pages
        is_priority = False

    if not pages:
        console.print("[yellow]No new pages discovered[/yellow]")
        if not priority_only:
            console.print(
                "[dim]All reachable pages from this start point are already in the graph.[/dim]"
            )
            console.print(
                "[dim]Try a different --start-page or increase --limit to explore deeper.[/dim]"
            )
        return

    # Display in table
    table = Table(
        title=f"{'[DRY RUN] ' if dry_run else ''}New Wiki Pages ({len(pages)})"
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Page Name", style="cyan")

    for i, page in enumerate(pages[:50], 1):
        table.add_row(str(i), page)

    if len(pages) > 50:
        table.add_row("...", f"[dim]and {len(pages) - 50} more[/dim]")

    console.print(table)

    if dry_run:
        console.print(f"\n[yellow][DRY RUN] Would queue {len(pages)} pages[/yellow]")
        console.print("[dim]Run without --dry-run to queue pages[/dim]")
        return

    # Queue pages in graph
    result = queue_wiki_pages(
        facility_id=facility,
        page_names=pages,
        interest_score=interest_score,
        is_priority=is_priority,
    )

    console.print(f"\n[green]Queued {result['queued']} pages[/green]")
    if result["skipped"] > 0:
        console.print(f"[dim]Skipped {result['skipped']} already-queued pages[/dim]")
    console.print(f"\n[dim]Process with: imas-codex wiki ingest {facility}[/dim]")


@wiki.command("ingest")
@click.argument("facility")
@click.option(
    "--limit",
    "-n",
    default=50,
    type=int,
    help="Maximum pages to ingest (default: 50)",
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimum interest score threshold (default: 0.0)",
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
    "--dry-run",
    is_flag=True,
    help="Preview without ingesting",
)
def wiki_ingest(
    facility: str,
    limit: int,
    min_score: float,
    pages: tuple[str, ...],
    rate_limit: float,
    dry_run: bool,
) -> None:
    """Ingest wiki pages from the graph queue.

    Processes WikiPage nodes with status='discovered' from the graph.
    Use 'wiki discover' first to queue pages.

    Examples:
        # Ingest queued pages
        imas-codex wiki ingest epfl

        # Ingest only high-score pages
        imas-codex wiki ingest epfl --min-score 0.7

        # Ingest specific pages (bypasses queue)
        imas-codex wiki ingest epfl -p Thomson -p Ion_Temperature_Nodes

        # Preview without saving
        imas-codex wiki ingest epfl --dry-run
    """
    import asyncio

    from rich.console import Console
    from rich.table import Table

    from imas_codex.wiki import WikiIngestionPipeline, get_pending_wiki_pages
    from imas_codex.wiki.pipeline import create_wiki_vector_index

    console = Console()

    # Determine which pages to ingest
    if pages:
        # Direct page list bypasses graph queue
        page_list = list(pages)
        console.print(f"[cyan]Ingesting {len(page_list)} specified pages...[/cyan]")

        if dry_run:
            console.print("\n[yellow][DRY RUN] Pages that would be ingested:[/yellow]")
            for i, page in enumerate(page_list[:20], 1):
                console.print(f"  {i:2}. {page}")
            if len(page_list) > 20:
                console.print(f"  ... and {len(page_list) - 20} more")
            return

        # Create vector index if needed
        try:
            create_wiki_vector_index()
        except Exception as e:
            console.print(f"[dim]Vector index: {e}[/dim]")

        # Run ingestion with explicit page list
        pipeline = WikiIngestionPipeline(facility_id=facility, use_rich=True)

        try:
            stats = asyncio.run(pipeline.ingest_pages(page_list, rate_limit=rate_limit))
        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]")
            raise SystemExit(1) from e
    else:
        # Graph-driven: get pending pages from queue
        console.print(f"[cyan]Checking graph queue for {facility}...[/cyan]")
        pending = get_pending_wiki_pages(
            facility_id=facility,
            limit=limit,
            min_interest_score=min_score,
        )

        if not pending:
            console.print(f"[yellow]No pending wiki pages for {facility}[/yellow]")
            console.print(
                f"[dim]Queue pages with: imas-codex wiki discover {facility}[/dim]"
            )
            return

        console.print(f"[green]Found {len(pending)} pending pages[/green]")

        if dry_run:
            table = Table(title="[DRY RUN] Pages that would be ingested")
            table.add_column("#", style="dim", width=4)
            table.add_column("Page Name", style="cyan")
            table.add_column("Score", style="dim", width=6)

            for i, page in enumerate(pending[:20], 1):
                score = page.get("interest_score", 0.5)
                table.add_row(str(i), page["title"], f"{score:.2f}")
            if len(pending) > 20:
                table.add_row("...", f"[dim]and {len(pending) - 20} more[/dim]", "")

            console.print(table)
            return

        # Create vector index if needed
        try:
            create_wiki_vector_index()
        except Exception as e:
            console.print(f"[dim]Vector index: {e}[/dim]")

        # Run graph-driven ingestion
        pipeline = WikiIngestionPipeline(facility_id=facility, use_rich=True)

        try:
            stats = asyncio.run(
                pipeline.ingest_from_graph(
                    limit=limit,
                    min_interest_score=min_score,
                    rate_limit=rate_limit,
                )
            )
        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]")
            raise SystemExit(1) from e

    console.print("\n[green]Ingestion complete![/green]")
    console.print(f"  Pages ingested:    {stats['pages']}")
    console.print(f"  Pages failed:      {stats['pages_failed']}")
    console.print(f"  Chunks created:    {stats['chunks']}")
    console.print(f"  TreeNodes linked:  {stats['tree_nodes_linked']}")
    console.print(f"  IMAS paths linked: {stats['imas_paths_linked']}")
    console.print(f"  Conventions found: {stats['conventions']}")


@wiki.command("status")
@click.argument("facility")
def wiki_status(facility: str) -> None:
    """Show wiki queue and ingestion statistics.

    Shows the queue status (discovered pages pending ingestion) and
    statistics about already-ingested pages.

    Examples:
        imas-codex wiki status epfl
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

    queue_table.add_row("Discovered (pending)", str(queue_stats["discovered"]))
    queue_table.add_row("Scraped", str(queue_stats["scraped"]))
    queue_table.add_row("Chunked", str(queue_stats["chunked"]))
    queue_table.add_row("Linked (complete)", str(queue_stats["linked"]))
    queue_table.add_row("Failed", str(queue_stats["failed"]))
    queue_table.add_row("─" * 12, "─" * 6)
    queue_table.add_row("[bold]Total[/bold]", f"[bold]{queue_stats['total']}[/bold]")

    console.print(queue_table)

    if queue_stats["discovered"] > 0:
        console.print(
            f"\n[dim]Process pending pages with: imas-codex wiki ingest {facility}[/dim]"
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
# Agent Commands
# ============================================================================


@main.group()
def agent() -> None:
    """Run LlamaIndex agents for exploration and enrichment.

    \b
      imas-codex agent run       Run an agent with a task
      imas-codex agent enrich    Enrich TreeNode metadata
    """
    pass


@agent.command("run")
@click.argument("task")
@click.option(
    "--type",
    "agent_type",
    default="enrichment",
    type=click.Choice(["enrichment", "mapping", "exploration"]),
    help="Agent type to use",
)
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def agent_run(task: str, agent_type: str, verbose: bool) -> None:
    """Run an agent with a task.

    The agent can autonomously use tools to:
    - Query the Neo4j knowledge graph
    - Execute SSH commands on remote facilities
    - Search code examples and IMAS paths

    Examples:
        imas-codex agent run "Describe what \\RESULTS::ASTRA is used for"

        imas-codex agent run "Find IMAS paths for electron temperature" --type mapping
    """
    import asyncio

    from imas_codex.agents import quick_agent_task

    click.echo(f"Running {agent_type} agent...")
    if verbose:
        click.echo(f"Task: {task}\n")

    try:
        result = asyncio.run(quick_agent_task(task, agent_type, verbose))
        click.echo("\n=== Agent Response ===")
        click.echo(result)
    except Exception as e:
        click.echo(f"Agent error: {e}", err=True)
        raise SystemExit(1) from None


@agent.command("enrich")
@click.argument("paths", nargs=-1)
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
    default="google/gemini-3-flash-preview",
    help="LLM model to use (default: google/gemini-3-flash-preview)",
)
@click.option("--dry-run", is_flag=True, help="Preview without persisting to graph")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def agent_enrich(
    paths: tuple[str, ...],
    limit: int | None,
    tree: str | None,
    status: str,
    force: bool,
    linked: bool,
    batch_size: int | None,
    model: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Enrich TreeNode metadata using ReAct agent with tool access.

    The agent gathers context from the knowledge graph, code examples,
    and IMAS DD before generating physics-accurate descriptions.

    Paths are grouped by parent node for efficient batch processing.
    A Rich progress display shows current batch, tree, and statistics.

    By default, discovers and processes ALL nodes with status='pending'.
    Use --tree to filter to a specific tree.
    Use --limit to cap the number of nodes processed.

    \b
    EXAMPLES:
        # Enrich all pending nodes in the results tree
        imas-codex agent enrich --tree results

        # Enrich all pending nodes across all trees
        imas-codex agent enrich

        # Limit to first 100 nodes
        imas-codex agent enrich --tree tcv_shot --limit 100

        # Process stale nodes (marked for re-enrichment)
        imas-codex agent enrich --status stale

        # Only enrich nodes with code context (more reliable)
        imas-codex agent enrich --linked

        # Include ALL nodes (pending + enriched) for (re-)enrichment
        imas-codex agent enrich --force

        # Re-enrich only already processed nodes
        imas-codex agent enrich --status enriched

        # Enrich specific paths
        imas-codex agent enrich "\\RESULTS::IBS" "\\RESULTS::LIUQE"

        # Use Pro model for higher quality
        imas-codex agent enrich --model google/gemini-3-pro-preview -b 200

        # Preview without saving
        imas-codex agent enrich --dry-run
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

    from imas_codex.agents import (
        BatchProgress,
        compose_batches,
        discover_nodes_to_enrich,
        estimate_enrichment_cost,
        get_parent_path,
        react_batch_enrich_paths,
    )

    console = Console()

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        if "pro" in model.lower():
            effective_batch_size = 200
        else:
            effective_batch_size = 100

    # Compose smart batches grouped by parent (for preview)
    batches = compose_batches(
        path_list, batch_size=effective_batch_size, group_by_parent=True
    )

    # Show cost estimate
    cost_est = estimate_enrichment_cost(len(path_list), effective_batch_size)
    console.print()
    console.print(
        f"[dim]Batches: {len(batches)} | "
        f"Est. time: {cost_est['estimated_hours'] * 60:.0f}min | "
        f"Est. cost: ${cost_est['estimated_cost']:.2f}[/dim]"
    )

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
            return await react_batch_enrich_paths(
                paths=path_list,
                tree_name=tree_name,
                batch_size=effective_batch_size,
                verbose=verbose,
                dry_run=False,  # We handle dry_run above
                model=model,
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


@agent.command("mark-stale")
@click.argument("pattern")
@click.option("--tree", default=None, help="Filter to specific tree name")
@click.option("--dry-run", is_flag=True, help="Preview without updating")
def agent_mark_stale(
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
        imas-codex agent mark-stale "LIUQE"

        # Mark nodes in results tree matching pattern
        imas-codex agent mark-stale "THOMSON" --tree results

        # Preview what would be marked
        imas-codex agent mark-stale "BOLO" --dry-run
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
            ssh epfl "which python3; python3 --version"

        See imas_codex/config/README.md for comprehensive exploration guidance.

        Examples:
            # Show facility info
            imas-codex epfl

            # Show full config
            imas-codex epfl --config
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
