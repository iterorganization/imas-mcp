#!/usr/bin/env python3
"""CLI script to run the IMAS MCP server with configurable options."""

import logging
from typing import Optional

import click

from imas_mcp.server import Server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol to use (stdio, sse, or streamable-http)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (for sse and streamable-http transports)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind to (for sse and streamable-http transports)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specify which IDS to include in the index as a space-separated string (e.g., 'core_profiles equilibrium'). If not specified, all IDS will be indexed.",
)
@click.option(
    "--auto-build/--no-auto-build",
    default=False,
    help="Automatically build the search index if it doesn't exist or is empty (default: False)",
)
def run_server(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    ids_filter: Optional[str],
    auto_build: bool,
) -> None:
    """Run the MCP server with configurable transport options.

    Examples:
        # Run with default STDIO transport
        python -m scripts.run_server

        # Run with SSE transport on custom host/port
        python -m scripts.run_server --transport sse --host 0.0.0.0 --port 9000

        # Run with debug logging
        python -m scripts.run_server --log-level DEBUG

        # Run with specific IDS filter
        python -m scripts.run_server --ids-filter "core_profiles equilibrium"

        # Run with auto-build enabled
        python -m scripts.run_server --auto-build

        # Run with streamable-http transport
        python -m scripts.run_server --transport streamable-http --port 8080
    """
    # Configure logging based on the provided level
    logging.basicConfig(level=getattr(logging, log_level))
    logger.info(f"Starting MCP server with transport={transport}")

    # Parse ids_filter string into a set, or None if empty
    ids_set = set(ids_filter.split()) if ids_filter else None
    if ids_set:
        logger.info(f"Filtering to IDS set: {ids_set}")

    if auto_build:
        logger.info("Auto-build enabled: will build index if empty or missing")

    match transport:
        case "stdio":
            logger.info("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    # Create and run the server with the specified IDS filter and auto_build option
    server = Server(ids_set=ids_set, auto_build=auto_build)
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
