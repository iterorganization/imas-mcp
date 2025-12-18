"""CLI interface for IMAS Codex Server."""

import logging
import os
from typing import Literal, cast

import click
from dotenv import load_dotenv

from imas_codex import __version__, _get_dd_version
from imas_codex.server import Server

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
    help="Show the imas-codex version and exit (raw version only).",
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
def main(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
    dd_version_opt: str | None,
) -> None:
    """Run the AI-enhanced MCP server with configurable transport options.

    Examples:
        # Run with default streamable-http transport
        imas-codex

        # Run with custom host/port
        imas-codex --host 0.0.0.0 --port 9000

        # Run with stdio transport (for MCP clients)
        imas-codex --transport stdio

        # Run with debug logging
        imas-codex --log-level DEBUG

        # Run without rich progress output
        imas-codex --no-rich

        # Run with specific DD version
        imas-codex --dd-version 3.42.2

    DD Version Priority (highest to lowest):
        1. --dd-version CLI option
        2. IMAS_DD_VERSION environment variable
        3. [tool.imas-codex].default-dd-version in pyproject.toml

    Note: streamable-http transport (default) uses stateful mode to support
    MCP sampling functionality for enhanced AI interactions.
    """
    # Resolve DD version with CLI option taking precedence
    if dd_version_opt:
        # Set env var so _get_dd_version picks it up (CLI overrides env)
        os.environ["IMAS_DD_VERSION"] = dd_version_opt

    # Re-resolve dd_version with the updated env (if CLI option was provided)
    effective_dd_version = _get_dd_version(dd_version_opt)

    # Configure logging based on the provided level
    # Force reconfigure logging by getting the root logger and setting its level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Also update all existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.debug(f"Set logging level to {log_level}")
    logger.debug(f"Starting MCP server with transport={transport}")

    # Log DD version with source indication
    if dd_version_opt:
        logger.info(f"IMAS DD version: {effective_dd_version} (--dd-version)")
    elif os.environ.get("IMAS_DD_VERSION"):
        logger.info(f"IMAS DD version: {effective_dd_version} (IMAS_DD_VERSION env)")
    else:
        logger.info(f"IMAS DD version: {effective_dd_version} (default)")

    ids_filter_env = os.environ.get("IDS_FILTER")

    if ids_filter_env:
        logger.info(f"IDS filter: {ids_filter_env}")

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

    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
