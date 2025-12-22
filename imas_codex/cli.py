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
# Dynamic Facility Commands
# ============================================================================


def _create_facility_command(facility_name: str, description: str) -> click.Command:
    """Create a CLI command for a specific facility."""

    @click.command(name=facility_name, help=f"Explore {description}")
    @click.argument("command", required=False)
    @click.option("--status", is_flag=True, help="Show session history")
    @click.option(
        "--finish",
        "finish_arg",
        nargs=1,
        default=None,
        required=False,
        help=(
            "Persist learnings. Optionally specify artifact type "
            "(environment, tools, filesystem, data) or use - to read from stdin."
        ),
    )
    @click.option("--discard", is_flag=True, help="Clear session without persisting")
    @click.option(
        "--artifacts", is_flag=True, help="List all artifacts for this facility"
    )
    @click.option(
        "--artifact",
        "artifact_view",
        type=str,
        default=None,
        help="View a specific artifact (environment, tools, filesystem, data)",
    )
    def facility_cmd(
        command: str | None,
        status: bool,
        finish_arg: str | None,
        discard: bool,
        artifacts: bool,
        artifact_view: str | None,
    ) -> None:
        """Execute commands on a remote facility or manage session.

        Examples:
            # Single command
            imas-codex epfl "python3 --version"

            # Batch commands with chaining
            imas-codex epfl "which python3; python3 --version; pip list | head"

            # Multi-line script via heredoc
            imas-codex epfl << 'EOF'
            echo "=== Python ==="
            python3 --version
            pip list | head -10
            EOF

            # Persist typed artifact (environment, tools, filesystem, data)
            imas-codex epfl --finish environment - << 'EOF'
            python:
              version: "3.9.21"
              path: "/usr/bin/python3"
            os:
              name: "RHEL"
              version: "9.6"
            EOF

            # Auto-detect artifact type from keys
            imas-codex epfl --finish - << 'EOF'
            python:
              version: "3.9.21"
            EOF

            # View artifacts
            imas-codex epfl --artifacts
            imas-codex epfl --artifact environment
        """
        import json
        import sys

        from imas_codex.remote import (
            discard_session,
            get_session_status,
            run_command,
            run_script,
        )
        from imas_codex.remote.finish import (
            finish_session,
            list_artifacts,
            load_artifact,
        )

        # Handle --artifacts
        if artifacts:
            manifest = list_artifacts(facility_name)
            if manifest.get("artifacts"):
                click.echo(f"Artifacts for {facility_name}:")
                for name, info in manifest["artifacts"].items():
                    status_str = info.get("status", "unknown")
                    updated = info.get("updated", "")
                    click.echo(f"  - {name}: {status_str} (updated: {updated})")
            else:
                click.echo(f"No artifacts found for {facility_name}")
            return

        # Handle --artifact TYPE
        if artifact_view:
            artifact_data = load_artifact(facility_name, artifact_view)
            if artifact_data:
                click.echo(json.dumps(artifact_data, indent=2))
            else:
                click.echo(
                    f"No {artifact_view} artifact found for {facility_name}", err=True
                )
                raise SystemExit(1)
            return

        # Handle --status
        if status:
            session_status = get_session_status(facility_name)
            click.echo(session_status.format())
            return

        # Handle --discard
        if discard:
            if discard_session(facility_name):
                click.echo(f"Session cleared for {facility_name}")
            else:
                click.echo(f"No active session for {facility_name}")
            return

        # Handle --finish [TYPE] [LEARNINGS]
        if finish_arg is not None:
            artifact_type: str | None = None
            learnings_input: str | None = None

            # Check if finish_arg is an artifact type
            valid_types = {"environment", "tools", "filesystem", "data"}
            if finish_arg in valid_types:
                # Artifact type specified, read learnings from stdin
                artifact_type = finish_arg
                if sys.stdin.isatty():
                    click.echo(
                        f"Error: --finish {artifact_type} requires input via stdin",
                        err=True,
                    )
                    raise SystemExit(1)
                learnings_input = sys.stdin.read()
            elif finish_arg == "-":
                # Read from stdin, auto-detect type
                if sys.stdin.isatty():
                    click.echo("Error: --finish - requires input via stdin", err=True)
                    raise SystemExit(1)
                learnings_input = sys.stdin.read()
            else:
                # Inline YAML/JSON
                learnings_input = finish_arg

            if not learnings_input or not learnings_input.strip():
                click.echo("Error: --finish requires learnings", err=True)
                raise SystemExit(1)

            success, message = finish_session(
                facility_name,
                artifact_type=artifact_type,
                learnings=learnings_input,
            )
            if success:
                click.echo(message)
            else:
                click.echo(f"Error: {message}", err=True)
                raise SystemExit(1)
            return

        # Handle command execution (argument or stdin)
        script_content = None

        if command:
            # Single command from argument
            script_content = command
        elif not sys.stdin.isatty():
            # Multi-line script from stdin (heredoc)
            script_content = sys.stdin.read()

        if script_content and script_content.strip():
            try:
                # Use run_script for multi-line, run_command for single-line
                if "\n" in script_content.strip():
                    result = run_script(facility_name, script_content)
                else:
                    result = run_command(facility_name, script_content)

                # Print stdout
                if result.stdout:
                    click.echo(result.stdout, nl=False)
                    if not result.stdout.endswith("\n"):
                        click.echo()
                # Print stderr to stderr
                if result.stderr:
                    click.echo(result.stderr, err=True, nl=False)
                    if not result.stderr.endswith("\n"):
                        click.echo(err=True)
                # Exit with command's exit code
                if result.exit_code != 0:
                    raise SystemExit(result.exit_code)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                raise SystemExit(1) from e
            except Exception as e:
                click.echo(f"Connection error: {e}", err=True)
                raise SystemExit(1) from e
            return

        # No command or option - show help
        ctx = click.get_current_context()
        click.echo(ctx.get_help())

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
