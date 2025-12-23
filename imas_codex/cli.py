"""CLI interface for IMAS Codex Server."""

import logging
import os
from typing import Literal, cast

import click
from dotenv import load_dotenv

from imas_codex import __version__, _get_dd_version

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
def facilities_show(name: str, fmt: str) -> None:
    """Show configuration for a specific facility."""
    import json

    import yaml

    from imas_codex.discovery import get_config

    try:
        config = get_config(name)
        context = config.to_context()

        if fmt == "json":
            click.echo(json.dumps(context, indent=2))
        else:
            click.echo(yaml.dump(context, default_flow_style=False, sort_keys=False))

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
    import shutil
    import subprocess
    from pathlib import Path

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

    if not shutil.which("apptainer"):
        click.echo("Error: apptainer not found in PATH", err=True)
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
@click.option(
    "--username",
    envvar="GHCR_USERNAME",
    help="GHCR username (env: GHCR_USERNAME)",
)
def neo4j_push(
    version: str,
    dump_file: str | None,
    registry: str,
    token: str | None,
    username: str | None,
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

    # Login to GHCR if credentials provided
    if token and username:
        login_cmd = ["oras", "login", "ghcr.io", "-u", username, "--password-stdin"]
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
@click.option(
    "--username",
    envvar="GHCR_USERNAME",
    help="GHCR username (env: GHCR_USERNAME)",
)
def neo4j_pull(
    version: str,
    output: str | None,
    registry: str,
    token: str | None,
    username: str | None,
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

    # Login to GHCR if credentials provided
    if token and username:
        login_cmd = ["oras", "login", "ghcr.io", "-u", username, "--password-stdin"]
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
    "--skip-graph",
    is_flag=True,
    help="Skip graph dump and push",
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
    skip_graph: bool,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """Create a new release with synced schema, graph, and git tag.

    VERSION should be a semantic version with 'v' prefix (e.g., v1.0.0).

    This command:
    1. Updates all LinkML schema versions in schemas/
    2. Regenerates Pydantic models from schemas
    3. Adds _GraphMeta node with version info
    4. Dumps and pushes graph to GHCR
    5. Creates annotated git tag
    6. Pushes tag to trigger CI release

    Examples:
        # Full release
        imas-codex release v1.0.0 -m 'Initial EPFL facility knowledge'

        # Release without graph (schema-only changes)
        imas-codex release v1.0.1 -m 'Fix schema typo' --skip-graph

        # Dry run to preview changes
        imas-codex release v1.0.0 -m 'Test' --dry-run
    """
    import re
    import subprocess
    from pathlib import Path

    # Validate version format
    if not re.match(r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", version):
        click.echo(f"Error: Invalid version format: {version}", err=True)
        click.echo("Expected format: v1.0.0 or v1.0.0-rc1")
        raise SystemExit(1)

    version_number = version.lstrip("v")
    schemas_dir = Path("imas_codex/schemas")

    click.echo(f"{'[DRY RUN] ' if dry_run else ''}Preparing release {version}")
    click.echo(f"Message: {message}")
    click.echo()

    # Step 1: Update schema versions
    click.echo("Step 1: Updating schema versions...")
    schema_files = list(schemas_dir.glob("*.yaml"))
    for schema_file in schema_files:
        if schema_file.name.startswith("_"):
            continue
        click.echo(f"  - {schema_file.name}")
        if not dry_run:
            content = schema_file.read_text()
            # Update version field
            updated = re.sub(
                r"^version:\s*['\"]?[\d.]+['\"]?",
                f"version: {version_number}",
                content,
                flags=re.MULTILINE,
            )
            schema_file.write_text(updated)

    # Step 2: Regenerate models
    click.echo("\nStep 2: Regenerating Pydantic models...")
    if not dry_run:
        result = subprocess.run(
            ["uv", "run", "build-models", "--force"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(f"Error regenerating models: {result.stderr}", err=True)
            raise SystemExit(1)
        click.echo("  Models regenerated")
    else:
        click.echo("  [would run: uv run build-models --force]")

    # Step 3: Update _GraphMeta node
    if not skip_graph:
        click.echo("\nStep 3: Updating graph metadata...")
        if not dry_run:
            try:
                from imas_codex.graph import GraphClient

                with GraphClient() as client:
                    # Get list of facilities in graph
                    facilities_result = client.query(
                        "MATCH (f:Facility) RETURN collect(f.id) as facilities"
                    )
                    facilities = (
                        facilities_result[0]["facilities"] if facilities_result else []
                    )

                    # Update _GraphMeta node
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
                click.echo(f"Warning: Could not update graph metadata: {e}", err=True)
                click.echo("  Is Neo4j running? Check with: imas-codex neo4j status")
        else:
            click.echo("  [would update _GraphMeta node in graph]")

        # Step 4: Dump graph
        click.echo("\nStep 4: Dumping graph...")
        if not dry_run:
            # First stop Neo4j for dump
            click.echo("  Stopping Neo4j for dump...")
            subprocess.run(
                ["uv", "run", "imas-codex", "neo4j", "stop"],
                capture_output=True,
            )

            # Dump
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

        # Step 5: Push to GHCR
        click.echo("\nStep 5: Pushing graph to GHCR...")
        if not dry_run:
            result = subprocess.run(
                ["uv", "run", "imas-codex", "neo4j", "push", version],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing graph: {result.stderr}", err=True)
                click.echo("  You may need to set GHCR_TOKEN and GHCR_USERNAME")
                raise SystemExit(1)
            click.echo(
                f"  Pushed to ghcr.io/iterorganization/imas-codex-graph:{version}"
            )
        else:
            click.echo(
                f"  [would push to ghcr.io/iterorganization/imas-codex-graph:{version}]"
            )
    else:
        click.echo("\nStep 3-5: Skipped (--skip-graph)")

    # Step 6: Git operations
    if not skip_git:
        click.echo("\nStep 6: Git tag and push...")
        if not dry_run:
            # Stage schema changes
            subprocess.run(
                ["git", "add", "imas_codex/schemas/", "imas_codex/graph/models.py"],
                capture_output=True,
            )

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True,
            )
            if status.returncode != 0:
                # There are staged changes
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"chore: bump schema version to {version_number}",
                    ],
                    capture_output=True,
                )
                click.echo("  Committed schema version bump")

            # Create annotated tag
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

            # Push tag
            result = subprocess.run(
                ["git", "push", "origin", version],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing tag: {result.stderr}", err=True)
                raise SystemExit(1)
            click.echo("  Pushed tag to origin")

            # Push branch (with schema changes)
            subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True,
            )
        else:
            click.echo(f"  [would create and push tag: {version}]")
    else:
        click.echo("\nStep 6: Skipped (--skip-git)")

    click.echo()
    if dry_run:
        click.echo("[DRY RUN] No changes made. Run without --dry-run to execute.")
    else:
        click.echo(f"Release {version} complete!")
        click.echo("CI will now build and publish the package.")


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
        from imas_codex.discovery import get_config

        config = get_config(facility_name)

        if show_config:
            import json

            click.echo(json.dumps(config.model_dump(), indent=2))
        else:
            click.echo(f"Facility: {config.facility}")
            click.echo(f"Description: {config.description}")
            click.echo(f"SSH Host: {config.ssh_host}")
            if config.paths.data:
                click.echo(f"Data paths: {', '.join(config.paths.data)}")
            if config.known_systems.diagnostics:
                click.echo(
                    f"Diagnostics: {', '.join(config.known_systems.diagnostics)}"
                )
            click.echo("\nUse --config for full configuration")

    return facility_cmd


def _register_facility_commands() -> None:
    """Register a CLI command for each configured facility."""
    try:
        from imas_codex.discovery import get_config, list_facilities

        for facility_name in list_facilities():
            try:
                config = get_config(facility_name)
                cmd = _create_facility_command(facility_name, config.description)
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
