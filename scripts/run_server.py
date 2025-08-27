#!/usr/bin/env python3
"""CLI script to run the AI-enhanced IMAS MCP server with configurable options."""

import logging

import click

from imas_mcp.server import Server

# Configure logging
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
    "--no-rich",
    is_flag=True,
    help="Disable rich progress output during server initialization",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
def run_server(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
) -> None:
    """Run the AI-enhanced MCP server with configurable transport options.

    Examples:
        # Run with default STDIO transport
        python -m scripts.run_server

        # Run with HTTP transport on custom host/port
        python -m scripts.run_server --transport http --host 0.0.0.0 --port 9000

        # Run with debug logging
        python -m scripts.run_server --log-level DEBUG

        # Run with HTTP transport on specific port
        python -m scripts.run_server --transport http --port 8080

        # Run without rich progress output
        python -m scripts.run_server --no-rich
    """
    # Configure logging based on the provided level
    # For stdio transport, default to WARNING to prevent INFO logs appearing as warnings in MCP clients
    if transport == "stdio" and log_level == "INFO":
        log_level = "WARNING"
        logger.debug(
            "Adjusted log level to WARNING for stdio transport to prevent client warnings"
        )

    # Force reconfigure logging by getting the root logger and setting its level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Also update all existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.debug(f"Set logging level to {log_level}")
    logger.debug(f"Starting MCP server with transport={transport}")

    # Parse ids_filter string into a set if provided
    ids_set: set | None = set(ids_filter.split()) if ids_filter else None
    if ids_set:
        logger.info(f"Starting server with IDS filter: {sorted(ids_set)}")
    else:
        logger.info("Starting server with all available IDS")

    match transport:
        case "stdio":
            logger.debug("Using STDIO transport")
        case "http":
            logger.info(f"Using HTTP transport on {host}:{port}")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    # Create and run the AI-enhanced server
    server = Server(use_rich=not no_rich, ids_set=ids_set)
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
