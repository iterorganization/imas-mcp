"""Serve command - unified MCP server."""

import logging
import os
from typing import Literal, cast

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--transport",
    envvar="TRANSPORT",
    default="stdio",
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
    "--read-only",
    is_flag=True,
    default=False,
    help="Suppress write tools and Python REPL (for container deployments)",
)
def serve(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    read_only: bool,
) -> None:
    """Start the IMAS Codex MCP server.

    Provides tools for IMAS data dictionary exploration, facility signal
    search, documentation search, code search, and graph operations.

    \b
    Examples:
        # Run with stdio transport (default, for Copilot CLI / VS Code)
        imas-codex serve

        # Run with HTTP transport
        imas-codex serve --transport streamable-http --port 8000

        # Run in read-only mode (for containers)
        imas-codex serve --read-only --transport streamable-http
    """
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.info("Starting IMAS Codex MCP server")

    if read_only:
        logger.info("Read-only mode: write tools and Python REPL suppressed")

    match transport:
        case "stdio":
            os.environ["IMAS_CODEX_RICH"] = "0"
            logger.debug("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    from imas_codex.llm.server import AgentsServer

    server = AgentsServer(read_only=read_only)
    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )
