"""CLI interface for IMAS MCP Server."""

import logging
import os
from pathlib import Path
from typing import Literal, cast

import click
from dotenv import load_dotenv

from imas_mcp import __version__, dd_version
from imas_mcp.server import Server

# Load environment variables from .env file
load_dotenv(override=True)

# Configure logging
logger = logging.getLogger(__name__)


def _print_version(
    ctx: click.Context, param: click.Parameter, value: bool
) -> None:  # pragma: no cover - simple utility
    """Callback to print only the raw version and exit early."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(__version__)
    ctx.exit()


@click.command()
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show the imas-mcp version and exit (raw version only).",
)
@click.option(
    "--transport",
    envvar="TRANSPORT",
    default="streamable-http",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol (env: TRANSPORT) (stdio, sse, or streamable-http)",
)
@click.option(
    "--host",
    envvar="HOST",
    default="127.0.0.1",
    help="Host to bind (env: HOST) for sse and streamable-http transports",
)
@click.option(
    "--port",
    envvar="PORT",
    default=8000,
    type=int,
    help="Port to bind (env: PORT) for sse and streamable-http transports",
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
    "--docs-server-port",
    envvar="DOCS_SERVER_PORT",
    default=6280,
    type=int,
    help="Port for docs-mcp-server (env: DOCS_SERVER_PORT) (default: 6280)",
)
@click.option(
    "--docs-store-path",
    envvar="DOCS_MCP_STORE_PATH",
    type=click.Path(path_type=Path),
    help="Path for docs-mcp-server data storage (env: DOCS_MCP_STORE_PATH) (default: ./docs-data)",
)
@click.option(
    "--disable-docs-server",
    is_flag=True,
    help="Disable automatic startup of docs-mcp-server",
)
def main(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
    docs_server_port: int,
    docs_store_path: Path | None,
    disable_docs_server: bool,
) -> None:
    """Run the AI-enhanced MCP server with configurable transport options.

    Examples:
        # Run with default streamable-http transport
        imas-mcp

        # Run with custom host/port
        imas-mcp --host 0.0.0.0 --port 9000

        # Run with stdio transport (for MCP clients)
        imas-mcp --transport stdio

        # Run with debug logging
        imas-mcp --log-level DEBUG

        # Run without rich progress output
        imas-mcp --no-rich

    Note: streamable-http transport (default) uses stateful mode to support
    MCP sampling functionality for enhanced AI interactions.

    To set DD version, use the IMAS_DD_VERSION environment variable.

    For documentation management, use the separate 'docs' command:
        docs list
        docs add <library> <url>
        docs remove <library> --force
        docs clear-all --force
    """
    # Configure logging based on the provided level
    # Force reconfigure logging by getting the root logger and setting its level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Also update all existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.debug(f"Set logging level to {log_level}")
    logger.debug(f"Starting MCP server with transport={transport}")

    dd_version_env = os.environ.get("IMAS_DD_VERSION")
    ids_filter_env = os.environ.get("IDS_FILTER")

    if dd_version_env:
        logger.info(f"IMAS DD version: {dd_version} (IMAS_DD_VERSION={dd_version_env})")
    else:
        logger.info(f"IMAS DD version: {dd_version}")

    if ids_filter_env:
        logger.info(f"IDS filter: {ids_filter_env}")

    # Configure docs server
    if disable_docs_server:
        logger.info("Docs server auto-start disabled")

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

    # For stdio transport, always disable rich output to prevent protocol interference
    use_rich = not no_rich and transport != "stdio"
    if transport == "stdio" and not no_rich:
        logger.info(
            "Disabled rich output for stdio transport to prevent protocol interference"
        )

    server = Server(use_rich=use_rich, ids_set=ids_set)

    # Configure docs server if not disabled
    if not disable_docs_server:
        server.docs_manager.default_port = docs_server_port
        # Override store path if provided via CLI
        if docs_store_path:
            server.docs_manager.store_path = docs_store_path
            server.docs_manager.store_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Using CLI-specified docs store path: {docs_store_path.absolute()}"
            )

    # Setup signal handlers for clean shutdown
    server.setup_signal_handlers()

    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
