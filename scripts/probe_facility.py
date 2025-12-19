#!/usr/bin/env python3
"""
Probe a remote facility to discover available tools and environment.

This script connects to a remote facility via SSH and catalogs:
- Available command-line tools (search, tree, code analysis, etc.)
- Python version and data libraries
- OS and shell information

The results are saved as JSON for later use in configuring the Discovery Engine.
"""

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from imas_codex.discovery.connection import FacilityConnection
from imas_codex.discovery.explorer import RemoteExplorer


def get_output_dir() -> Path:
    """Get the output directory for probe results."""
    return Path(__file__).parent.parent / "imas_codex" / "config"


def serialize_datetime(obj: object) -> str:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@click.command()
@click.argument("host", default="epfl")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress logging except errors")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: config/{host}_environment.json)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "text"]),
    default="json",
    help="Output format",
)
@click.option("--no-save", is_flag=True, help="Don't save to file, just print")
def probe_facility(
    host: str,
    verbose: bool,
    quiet: bool,
    output: Path | None,
    output_format: str,
    no_save: bool,
) -> int:
    """
    Probe a remote facility to discover its environment.

    HOST is the SSH host alias (default: epfl).

    Examples:
        probe-facility                    # Probe epfl (default)
        probe-facility epfl -v            # Verbose output
        probe-facility jet --format text  # Human-readable output
        probe-facility tcv --no-save      # Just print, don't save
    """
    # Configure logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Connecting to {host}...")

        # Create connection and explorer
        conn = FacilityConnection(host=host)
        explorer = RemoteExplorer(connection=conn)

        with conn.session():
            logger.info(f"Connected to {host}, probing environment...")

            # Probe the environment
            env = explorer.probe_environment()

            # Count available tools
            available_count = sum(1 for t in env.available_tools if t.available)
            total_count = len(env.available_tools)

            logger.info(
                f"Probe complete: {available_count}/{total_count} tools available"
            )

        # Convert to dict for serialization
        env_dict = asdict(env)

        # Format output
        if output_format == "json":
            output_str = json.dumps(env_dict, indent=2, default=serialize_datetime)
        elif output_format == "yaml":
            import yaml

            output_str = yaml.dump(env_dict, default_flow_style=False, sort_keys=False)
        else:  # text
            output_str = format_text_output(env)

        # Print or save
        if no_save:
            click.echo(output_str)
        else:
            # Determine output path
            if output is None:
                output_dir = get_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                ext = "json" if output_format == "json" else output_format
                output = output_dir / f"{host}_environment.{ext}"

            output.write_text(output_str)
            click.echo(f"Saved environment probe to: {output}")

        return 0

    except Exception as e:
        logger.error(f"Error probing facility: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


def format_text_output(env) -> str:
    """Format environment as human-readable text."""
    lines = [
        f"Remote Environment: {env.host}",
        f"Probed at: {env.probe_timestamp}",
        f"OS: {env.os_info or 'Unknown'}",
        f"Shell: {env.shell or 'Unknown'}",
        f"Python: {env.python_version or 'Not found'}",
        "",
        "Data Libraries:",
    ]

    if env.data_libraries:
        for lib in env.data_libraries:
            lines.append(f"  - {lib}")
    else:
        lines.append("  (none found)")

    lines.append("")
    lines.append("Available Tools:")

    # Group tools by availability
    available = [t for t in env.available_tools if t.available]
    unavailable = [t for t in env.available_tools if not t.available]

    if available:
        lines.append(f"  Found ({len(available)}):")
        for tool in sorted(available, key=lambda t: t.name):
            version_str = f" ({tool.version})" if tool.version else ""
            lines.append(f"    - {tool.name}{version_str}")

    if unavailable:
        lines.append(f"  Not found ({len(unavailable)}):")
        names = sorted(t.name for t in unavailable)
        lines.append(f"    {', '.join(names)}")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(probe_facility())
