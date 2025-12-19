"""CLI interface for IMAS Codex Server."""

import asyncio
import json
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
    """IMAS Codex - AI-enhanced MCP server for fusion data.

    Run without a command to start the server, or use subcommands:

    \b
      imas-codex serve          Start the MCP server
      imas-codex explore        Explore a remote facility
      imas-codex survey         Survey a directory structure
      imas-codex search         Search for code patterns
      imas-codex ask            Ask a question about a facility
    """
    if version:
        click.echo(__version__)
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
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
def serve(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
    dd_version_opt: str | None,
) -> None:
    """Start the AI-enhanced MCP server with configurable transport options.

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

    from imas_codex.server import Server

    server_instance = Server(use_rich=use_rich, ids_set=ids_set)

    server_instance.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


# ============================================================================
# Discovery Commands
# ============================================================================


@main.command()
@click.argument("facility")
@click.option(
    "--max-iter",
    default=5,
    type=int,
    help="Maximum iterations in the agentic loop",
)
@click.option(
    "--model",
    default=None,
    help="LLM model to use (default: anthropic/claude-opus-4.5)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for results (JSON)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def explore(
    facility: str,
    max_iter: int,
    model: str | None,
    output: str | None,
    verbose: bool,
) -> None:
    """Explore a remote facility's environment.

    Connects to FACILITY via SSH and discovers available tools,
    Python libraries, and system capabilities.

    Examples:
        imas-codex explore epfl
        imas-codex explore epfl --verbose
        imas-codex explore epfl -o results.json
    """
    from imas_codex.discovery import Investigator

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    click.echo(f"Exploring facility: {facility}")

    inv = Investigator(facility, model=model)
    result = asyncio.run(
        inv.run(
            task="Discover the system environment",
            prompt_name="explore_environment",
            max_iterations=max_iter,
        )
    )

    if result.success:
        click.echo(click.style("✓ Exploration complete", fg="green"))
        click.echo(f"Iterations: {result.iterations}")

        if output:
            with open(output, "w") as f:
                json.dump(result.findings, f, indent=2)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(result.findings, indent=2))
    else:
        click.echo(click.style("✗ Exploration incomplete", fg="yellow"))
        click.echo(f"Iterations: {result.iterations}")
        if result.errors:
            click.echo("Errors:")
            for err in result.errors:
                click.echo(f"  - {err}")


@main.command()
@click.argument("facility")
@click.argument("path")
@click.option("--depth", default=3, type=int, help="Maximum directory depth")
@click.option("--max-iter", default=5, type=int, help="Maximum iterations")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def survey(
    facility: str,
    path: str,
    depth: int,
    max_iter: int,
    output: str | None,
) -> None:
    """Survey a directory structure on a remote facility.

    Examples:
        imas-codex survey epfl /common/tcv/codes
        imas-codex survey epfl /home/user --depth 2
    """
    from imas_codex.discovery import Investigator

    click.echo(f"Surveying {path} on {facility}...")

    inv = Investigator(facility)
    result = asyncio.run(
        inv.run(
            task=f"Survey directory: {path}",
            prompt_name="directory_survey",
            max_iterations=max_iter,
            target_path=path,
            max_depth=depth,
        )
    )

    if result.success:
        click.echo(click.style("✓ Survey complete", fg="green"))
        if output:
            with open(output, "w") as f:
                json.dump(result.findings, f, indent=2)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(result.findings, indent=2))
    else:
        click.echo(click.style("✗ Survey incomplete", fg="yellow"))


@main.command()
@click.argument("facility")
@click.argument("pattern")
@click.option(
    "--path",
    "-p",
    multiple=True,
    help="Paths to search (can specify multiple)",
)
@click.option(
    "--type",
    "-t",
    "file_type",
    default="*.py",
    help="File type pattern (default: *.py)",
)
@click.option("--max-iter", default=5, type=int, help="Maximum iterations")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def search(
    facility: str,
    pattern: str,
    path: tuple[str, ...],
    file_type: str,
    max_iter: int,
    output: str | None,
) -> None:
    """Search for code patterns on a remote facility.

    Examples:
        imas-codex search epfl "import MDSplus"
        imas-codex search epfl "class.*Diagnostic" -t "*.py"
        imas-codex search epfl "Connection" -p /common/tcv/codes
    """
    from imas_codex.discovery import Investigator

    click.echo(f"Searching for '{pattern}' on {facility}...")

    inv = Investigator(facility)
    result = asyncio.run(
        inv.run(
            task=f"Search for: {pattern}",
            prompt_name="code_search",
            max_iterations=max_iter,
            pattern=pattern,
            file_types=[file_type],
            search_paths=list(path) if path else None,
        )
    )

    if result.success:
        click.echo(click.style("✓ Search complete", fg="green"))
        if output:
            with open(output, "w") as f:
                json.dump(result.findings, f, indent=2)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(result.findings, indent=2))
    else:
        click.echo(click.style("✗ Search incomplete", fg="yellow"))


@main.command()
@click.argument("facility")
@click.argument("question")
@click.option("--max-iter", default=5, type=int, help="Maximum iterations")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def ask(
    facility: str,
    question: str,
    max_iter: int,
    output: str | None,
) -> None:
    """Ask a freeform question about a facility.

    Examples:
        imas-codex ask epfl "What MDSplus trees are available?"
        imas-codex ask epfl "How do I access TCV shot data?"
    """
    from imas_codex.discovery import Investigator

    click.echo(f"Asking: {question}")

    inv = Investigator(facility)
    result = asyncio.run(
        inv.run(
            task=question,
            max_iterations=max_iter,
        )
    )

    if result.success:
        click.echo(click.style("✓ Investigation complete", fg="green"))
        if output:
            with open(output, "w") as f:
                json.dump(result.findings, f, indent=2)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(result.findings, indent=2))
    else:
        click.echo(click.style("✗ Investigation incomplete", fg="yellow"))


@main.command("list-facilities")
def list_facilities_cmd() -> None:
    """List available facility configurations."""
    from imas_codex.discovery import list_facilities

    facilities = list_facilities()
    if facilities:
        click.echo("Available facilities:")
        for f in sorted(facilities):
            click.echo(f"  - {f}")
    else:
        click.echo("No facility configurations found.")


if __name__ == "__main__":
    main()
