"""CLI interface for IMAS Codex Server.

Modular CLI structure with command groups split by functionality.
"""

# Warning filters are set in imas_codex/__init__.py at package import time
# to ensure they're applied before any aiohttp imports.

import logging

import click
from dotenv import load_dotenv

from imas_codex import __version__

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
      imas-codex graph push       Push graph archive to GHCR
      imas-codex graph db start   Start Neo4j database
      imas-codex config private   Manage private facility YAML
      imas-codex imas build       Build/update IMAS DD graph
      imas-codex facilities list  List configured facilities
    """
    if version:
        click.echo(__version__)
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


class _DeprecatedDataGroup(click.MultiCommand):
    """Deprecated 'data' command that forwards to 'graph' and 'config'."""

    def __init__(self, graph_group: click.Group, config_group: click.Group) -> None:
        super().__init__(
            name="data",
            help="[DEPRECATED] Use 'imas-codex graph' or 'imas-codex config' instead.",
        )
        self._graph = graph_group
        self._config = config_group

    def list_commands(self, ctx: click.Context) -> list[str]:
        graph_cmds = self._graph.list_commands(ctx)
        config_cmds = self._config.list_commands(ctx)
        return sorted(set(graph_cmds + config_cmds))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        click.echo(
            "Warning: 'imas-codex data' is deprecated. "
            "Use 'imas-codex graph' or 'imas-codex config' instead.",
            err=True,
        )
        # Try graph first, then config
        result = self._graph.get_command(ctx, cmd_name)
        if result is None:
            result = self._config.get_command(ctx, cmd_name)
        return result


def register_commands() -> None:
    """Register all command groups with the main CLI."""
    from imas_codex.cli.compute import hpc
    from imas_codex.cli.config_cli import config
    from imas_codex.cli.credentials import credentials
    from imas_codex.cli.discover import discover
    from imas_codex.cli.embed import embed
    from imas_codex.cli.enrich import enrich
    from imas_codex.cli.facilities import facilities
    from imas_codex.cli.graph_cli import graph
    from imas_codex.cli.hosts import hosts
    from imas_codex.cli.imas_dd import imas
    from imas_codex.cli.ingest import ingest
    from imas_codex.cli.release import release
    from imas_codex.cli.serve import serve
    from imas_codex.cli.tools import tools
    from imas_codex.cli.utils import setup_age

    main.add_command(serve)
    main.add_command(graph)
    main.add_command(config)
    main.add_command(hpc)
    main.add_command(discover)
    main.add_command(embed)
    main.add_command(imas)
    main.add_command(enrich)
    main.add_command(ingest)
    main.add_command(tools)
    main.add_command(hosts)
    main.add_command(facilities)
    main.add_command(release)
    main.add_command(setup_age)
    main.add_command(credentials)

    # Deprecated alias: 'data' forwards to 'graph' and 'config'
    main.add_command(_DeprecatedDataGroup(graph, config))


# Register commands at import time
register_commands()

__all__ = ["main"]
