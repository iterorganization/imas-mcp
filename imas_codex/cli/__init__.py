"""CLI interface for IMAS Codex Server.

Modular CLI structure with command groups split by functionality.
"""

import logging
import warnings

# Suppress third-party deprecation warnings before importing other modules
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Relying on Driver's destructor.*")

import click  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from imas_codex import __version__  # noqa: E402

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

    Use subcommands to start servers or manage data:

    \b
      imas-codex serve imas       Start the IMAS Data Dictionary MCP server
      imas-codex serve agents     Start the Agents MCP server
      imas-codex serve embed      Start GPU embedding server
      imas-codex imas build       Build/update IMAS DD graph
      imas-codex imas status      Show DD graph statistics
      imas-codex facilities list  List configured facilities
    """
    if version:
        click.echo(__version__)
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def register_commands() -> None:
    """Register all command groups with the main CLI."""
    from imas_codex.cli.clusters import clusters
    from imas_codex.cli.data import data
    from imas_codex.cli.discover import discover
    from imas_codex.cli.enrich import enrich
    from imas_codex.cli.facilities import facilities
    from imas_codex.cli.hosts import hosts
    from imas_codex.cli.imas_dd import imas
    from imas_codex.cli.ingest import ingest
    from imas_codex.cli.release import release
    from imas_codex.cli.serve import serve
    from imas_codex.cli.tools import tools
    from imas_codex.cli.utils import setup_age

    main.add_command(serve)
    main.add_command(data)
    main.add_command(discover)
    main.add_command(imas)
    main.add_command(clusters)
    main.add_command(enrich)
    main.add_command(ingest)
    main.add_command(tools)
    main.add_command(hosts)
    main.add_command(facilities)
    main.add_command(release)
    main.add_command(setup_age)


# Register commands at import time
register_commands()

__all__ = ["main"]
