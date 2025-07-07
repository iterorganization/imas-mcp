#!/usr/bin/env python3
"""CLI script to run the AI-enhanced IMAS MCP server with configurable options."""

import logging

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
def run_server(
    transport: str,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Run the AI-enhanced MCP server with configurable transport options.

    Examples:
        # Run with default STDIO transport
        python -m scripts.run_server

        # Run with SSE transport on custom host/port
        python -m scripts.run_server --transport sse --host 0.0.0.0 --port 9000

        # Run with debug logging
        python -m scripts.run_server --log-level DEBUG

        # Run with streamable-http transport
        python -m scripts.run_server --transport streamable-http --port 8080
    """
    # Configure logging based on the provided level
    logging.basicConfig(level=getattr(logging, log_level))
    logger.info(f"Starting MCP server with transport={transport}")

    match transport:
        case "stdio":
            logger.info("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    # Create and run the AI-enhanced server
    server = Server()
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
