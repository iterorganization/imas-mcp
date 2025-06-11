#!/usr/bin/env python3
"""
Get the exact index name for the LexicographicSearch class.
This script prints the exact index name that should be copied.
"""

import sys
from typing import Optional

import click
from imas_mcp_server.lexicographic_search import LexicographicSearch


@click.command()
@click.option(
    "--ids-filter",
    multiple=True,
    help="Specific IDS names to include in the index name calculation (can be used multiple times)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show additional information about the index"
)
def main(ids_filter: tuple, verbose: bool) -> int:
    """Get the exact index name for the LexicographicSearch class.

    This command prints the exact index name that would be used for the given
    configuration. Useful for CI/CD scripts that need to know the index name
    without building the index.

    Examples:
        get-index-name                                      # Get default index name
        get-index-name --ids-filter core_profiles           # Get name for specific IDS
        get-index-name -v                                   # Show additional info"""
    try:  # Convert ids_filter tuple to set if provided
        ids_set: Optional[set] = set(ids_filter) if ids_filter else None

        # Initialize the search class (without building)
        search = LexicographicSearch(ids_set=ids_set, build=False)

        if verbose:
            click.echo(f"Data Dictionary version: {search.dd_version}", err=True)
            if ids_set:
                click.echo(f"IDS set: {sorted(ids_set)}", err=True)
            else:
                click.echo("IDS set: All IDS (no filter)", err=True)
            click.echo(f"Index directory: {search.dirname}", err=True)
            click.echo(f"Index name: {search.indexname}", err=True)
        else:
            click.echo(search.indexname)

        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
