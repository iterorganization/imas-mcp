"""Serve commands - MCP servers."""

import logging
import os
from typing import Literal, cast

import click

from imas_codex import _get_dd_version

logger = logging.getLogger(__name__)


@click.group()
def serve() -> None:
    """MCP servers for IMAS data and agent orchestration.

    \b
      imas-codex serve imas         Start the IMAS DD MCP server
      imas-codex serve agents       Start the Agents MCP server

    \b
    Services are managed via dedicated CLIs:
      imas-codex graph ...          Neo4j graph database
      imas-codex embed ...          GPU embedding server
      imas-codex llm ...            LiteLLM proxy
      imas-codex tunnel ...         SSH tunnels
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
        "Defaults to [tool.imas-codex.data-dictionary].version in pyproject.toml."
    ),
)
def serve_imas(
    transport: str,
    host: str,
    port: int,
    log_level: str,
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

    # For stdio transport, disable rich output automatically
    if transport == "stdio":
        os.environ["IMAS_CODEX_RICH"] = "0"
        logger.info(
            "Disabled rich output for stdio transport to prevent protocol interference"
        )

    from imas_codex.server import Server

    server_instance = Server(ids_set=ids_set)

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

    from imas_codex.llm.server import AgentsServer

    server = AgentsServer()
    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )
