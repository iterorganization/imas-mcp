#!/usr/bin/env python3
"""
CLI script for adding documentation libraries to the IMAS MCP server.

This script provides a simple wrapper around docs-mcp-server with proper
environment variable loading.
"""

import os
import subprocess
import sys

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


@click.command()
@click.argument("library", required=True)
@click.argument("url", required=True)
@click.option(
    "--version",
    default="",
    help="Version string (e.g., '1.0.0', '2.1'). Leave empty for unversioned documentation",
)
@click.option(
    "--max-pages", default=250, type=int, help="Maximum pages to scrape (default: 250)"
)
@click.option(
    "--max-depth", default=5, type=int, help="Maximum depth for crawling (default: 5)"
)
@click.option(
    "--ignore-errors/--no-ignore-errors",
    default=True,
    help="Ignore errors during scraping (default: enabled)",
)
def add_docs(
    library: str,
    url: str,
    version: str,
    max_pages: int,
    max_depth: int,
    ignore_errors: bool,
):
    """
    Add documentation library to the IMAS MCP server.

    LIBRARY: Name of the library to add (e.g., 'udunits', 'pandas')

    URL: Documentation URL to scrape (e.g., 'https://docs.unidata.ucar.edu/udunits/current/')

    Examples:
        add-docs udunits https://docs.unidata.ucar.edu/udunits/current/
        add-docs pandas https://pandas.pydata.org/docs/ --version 2.0.1
        add-docs numpy https://numpy.org/doc/stable/ --max-pages 500 --max-depth 3
        add-docs imas-python https://imas-python.readthedocs.io/en/stable/ --no-ignore-errors
    """
    # Basic validation
    if not library or not library.strip():
        click.echo("Error: Library name cannot be empty", err=True)
        sys.exit(1)

    if not url or not url.strip():
        click.echo("Error: URL cannot be empty", err=True)
        sys.exit(1)

    if not (url.startswith("http://") or url.startswith("https://")):
        click.echo("Error: URL must start with http:// or https://", err=True)
        sys.exit(1)

    # Build the npx command
    cmd = ["npx", "@arabold/docs-mcp-server@latest", "scrape", library, url]

    # Only add version if specified
    if version and version.strip():
        cmd.extend(["--version", version])

    cmd.extend(
        [
            "--max-pages",
            str(max_pages),
            "--max-depth",
            str(max_depth),
        ]
    )

    # Add ignore-errors flag if specified
    if ignore_errors:
        cmd.append("--ignore-errors")

    try:
        # Run the command with shell=True and pass the current environment
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)

        result = subprocess.run(cmd_str, shell=True, check=True, env=os.environ)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running docs-mcp-server: {e}", err=True)
        sys.exit(e.returncode)
    except FileNotFoundError:
        click.echo(
            "Error: npx command not found. Please ensure Node.js and npm are installed.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    add_docs()
