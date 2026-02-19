"""Common CLI options and utilities for discovery commands.

This module provides reusable option decorators for discovery CLI commands,
ensuring consistent behavior across all discovery domains (paths, wiki,
signals, files).

Usage:
    from imas_codex.cli.discover.common import (
        cost_options,
        output_options,
        phase_options,
        facility_argument,
        create_discovery_monitor,
    )

    @paths.command()
    @facility_argument
    @cost_options
    @output_options
    @phase_options
    def paths_command(...):
        pass
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import click
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from imas_codex.discovery.base.services import ServiceMonitor

console = Console()
logger = logging.getLogger(__name__)

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Valid discovery domains
DISCOVERY_DOMAINS = ("paths", "wiki", "signals", "files")


# =============================================================================
# Option Decorators - Composable option groups for discovery commands
# =============================================================================


def facility_argument[F: Callable[..., Any]](func: F) -> F:
    """Add facility argument to a command."""
    return click.argument("facility")(func)


def cost_options[F: Callable[..., Any]](func: F) -> F:
    """Add cost and time control options.

    Options:
        --cost-limit/-c: Maximum LLM spend in USD (default: $10)
        --time/-t: Maximum runtime in minutes
    """

    @click.option(
        "--cost-limit",
        "-c",
        type=float,
        default=10.0,
        help="Maximum LLM spend in USD (default: $10)",
    )
    @click.option(
        "--time",
        "-t",
        "time_limit",
        type=int,
        default=None,
        help="Maximum runtime in minutes",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def output_options[F: Callable[..., Any]](func: F) -> F:
    """Add output control options.

    Options:
        --verbose/-v: Show detailed progress
    """

    @click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        help="Show detailed progress",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def phase_options[F: Callable[..., Any]](func: F) -> F:
    """Add scan/score phase control options.

    Options:
        --scan-only: Only scan, skip scoring/enrichment
        --score-only: Only score already-discovered items
    """

    @click.option(
        "--scan-only",
        is_flag=True,
        default=False,
        help="Only scan, skip scoring/enrichment",
    )
    @click.option(
        "--score-only",
        is_flag=True,
        default=False,
        help="Only score already-discovered items",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def worker_options(scan_default: int = 1, score_default: int = 2) -> Callable[[F], F]:
    """Add worker count options.

    Args:
        scan_default: Default number of scan workers
        score_default: Default number of score workers

    Options:
        --scan-workers: Number of parallel scan workers
        --score-workers: Number of parallel score workers
    """

    def decorator(func: F) -> F:
        @click.option(
            "--scan-workers",
            type=int,
            default=scan_default,
            help=f"Number of parallel scan workers (default: {scan_default})",
        )
        @click.option(
            "--score-workers",
            type=int,
            default=score_default,
            help=f"Number of parallel score workers (default: {score_default})",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def focus_option[F: Callable[..., Any]](func: F) -> F:
    """Add focus option for natural language filtering.

    Options:
        --focus/-f: Natural language focus (e.g., 'equilibrium codes')
    """
    return click.option(
        "--focus",
        "-f",
        type=str,
        default=None,
        help="Natural language focus (e.g., 'equilibrium codes')",
    )(func)


def limit_option(name: str = "limit", default: int | None = None) -> Callable[[F], F]:
    """Add a limit option for debugging/testing.

    Args:
        name: Option name (limit, max-pages, max-paths, etc.)
        default: Default value (None = no limit)
    """

    def decorator(func: F) -> F:
        return click.option(
            f"--{name}",
            "-l",
            type=int,
            default=default,
            help=f"Maximum items to process (default: {'unlimited' if default is None else default})",
        )(func)

    return decorator


def domain_option(
    required: bool = False, default: str | None = None
) -> Callable[[F], F]:
    """Add domain selection option.

    Args:
        required: Whether domain selection is required
        default: Default domain if not specified

    Options:
        --domain/-d: Discovery domain (paths, wiki, signals, files)
    """

    def decorator(func: F) -> F:
        return click.option(
            "--domain",
            "-d",
            type=click.Choice(DISCOVERY_DOMAINS),
            required=required,
            default=default,
            help="Discovery domain to target",
        )(func)

    return decorator


# =============================================================================
# Utility Functions - Shared helpers for discovery commands
# =============================================================================


def get_facility_or_fail(facility: str) -> dict:
    """Get facility config or raise ClickException."""
    from imas_codex.discovery.base.facility import get_facility, list_facilities

    try:
        return get_facility(facility)
    except ValueError as e:
        available = ", ".join(list_facilities())
        raise click.ClickException(
            f"Unknown facility '{facility}'. Available: {available}"
        ) from e


def use_rich_output() -> bool:
    """Determine whether to use rich output via auto-detection."""
    from imas_codex.cli.rich_output import should_use_rich

    return should_use_rich()


def log_print(message: str) -> None:
    """Print message, stripping rich markup when rich is disabled."""
    if not use_rich_output():
        import re

        plain = re.sub(r"\[/?[^\]]+\]", "", message)
        click.echo(plain)
    else:
        console.print(message)


def format_cost(cost: float) -> str:
    """Format cost as USD string."""
    return f"${cost:.4f}"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_summary_table(
    title: str,
    rows: list[tuple[str, str | int | float]],
) -> None:
    """Print a summary table with key-value pairs."""
    if not use_rich_output():
        click.echo(f"\n{title}")
        click.echo("-" * len(title))
        for key, value in rows:
            click.echo(f"  {key}: {value}")
    else:
        table = Table(title=title, show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")
        for key, value in rows:
            table.add_row(key, str(value))
        console.print(table)


def print_cost_summary(
    cost: float,
    time_seconds: float,
    *,
    cost_limit: float | None = None,
    time_limit: int | None = None,
) -> None:
    """Print cost and time summary with optional limit comparison."""
    rows = [
        ("Cost", format_cost(cost)),
        ("Time", format_duration(time_seconds)),
    ]

    if cost_limit is not None:
        pct = (cost / cost_limit * 100) if cost_limit > 0 else 0
        rows.append(("Cost limit", f"{format_cost(cost_limit)} ({pct:.0f}% used)"))

    if time_limit is not None:
        pct = (time_seconds / (time_limit * 60) * 100) if time_limit > 0 else 0
        rows.append(("Time limit", f"{time_limit}m ({pct:.0f}% used)"))

    print_summary_table("Resource Usage", rows)


# =============================================================================
# Service Monitor Factory
# =============================================================================


def create_discovery_monitor(
    facility_config: dict,
    *,
    check_graph: bool = True,
    check_embed: bool = True,
    check_ssh: bool = True,
    check_auth: bool = True,
    check_model: bool = False,
    model_section: str = "language",
    poll_interval: float = 15.0,
) -> ServiceMonitor:
    """Create a service monitor from a facility config dict.

    Extracts SSH host, access method, auth type, and primary wiki URL
    from the facility config and delegates to
    :func:`~imas_codex.discovery.base.services.create_service_monitor`.

    This is the **single factory** that all ``discover`` CLI commands
    should use — it avoids duplicating config extraction logic across
    wiki.py, signals.py, paths.py, and files.py.

    Args:
        facility_config: Dict returned by ``get_facility(name)``
        check_graph: Include Neo4j check
        check_embed: Include embedding server check
        check_ssh: Include SSH connectivity check
        check_auth: Include wiki page reachability check
        poll_interval: Polling interval for live checks

    Returns:
        Configured (but not started) ServiceMonitor
    """
    from imas_codex.discovery.base.services import create_service_monitor

    # Derive SSH host — first wiki site with ssh_available, else facility id
    facility_id = facility_config.get("id", "")
    ssh_host: str | None = None
    access_method: str | None = None
    auth_type: str | None = None
    wiki_url: str | None = None

    wiki_sites = facility_config.get("wiki_sites") or []
    for site in wiki_sites:
        site_cfg = site if isinstance(site, dict) else {}
        if site_cfg.get("ssh_available"):
            ssh_host = ssh_host or facility_id
            access_method = access_method or site_cfg.get("access_method")
            auth_type = auth_type or site_cfg.get("auth_type")
            wiki_url = wiki_url or site_cfg.get("url")

    # Fall back to facility-level SSH host
    if not ssh_host and facility_config.get("ssh_available", False):
        ssh_host = facility_id

    if not wiki_url and wiki_sites:
        first = wiki_sites[0] if isinstance(wiki_sites[0], dict) else {}
        wiki_url = first.get("url")
        access_method = access_method or first.get("access_method")
        auth_type = auth_type or first.get("auth_type")

    logger.debug(
        "create_discovery_monitor: facility=%s ssh=%s access=%s auth=%s url=%s",
        facility_id,
        ssh_host,
        access_method,
        auth_type,
        wiki_url,
    )

    return create_service_monitor(
        facility=facility_id,
        ssh_host=ssh_host,
        access_method=access_method,
        auth_type=auth_type,
        wiki_url=wiki_url,
        check_graph=check_graph,
        check_embed=check_embed,
        check_ssh=check_ssh,
        check_auth=check_auth,
        check_model=check_model,
        model_section=model_section,
        poll_interval=poll_interval,
    )


class CostTimeTracker:
    """Track cost and time for discovery operations.

    Usage:
        tracker = CostTimeTracker(cost_limit=10.0, time_limit=30)
        while tracker.can_continue():
            # Do work
            tracker.add_cost(0.01)
        tracker.print_summary()
    """

    def __init__(
        self,
        cost_limit: float | None = None,
        time_limit: int | None = None,  # minutes
    ):
        import time

        self.cost_limit = cost_limit
        self.time_limit = time_limit
        self.cost = 0.0
        self.start_time = time.time()

    def add_cost(self, cost: float) -> None:
        """Add cost to running total."""
        self.cost += cost

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        import time

        return time.time() - self.start_time

    @property
    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return self.elapsed_seconds / 60

    def cost_exceeded(self) -> bool:
        """Check if cost limit has been exceeded."""
        if self.cost_limit is None:
            return False
        return self.cost >= self.cost_limit

    def time_exceeded(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.time_limit is None:
            return False
        return self.elapsed_minutes >= self.time_limit

    def can_continue(self) -> bool:
        """Check if we can continue (neither limit exceeded)."""
        return not self.cost_exceeded() and not self.time_exceeded()

    def stop_reason(self) -> str | None:
        """Get reason for stopping, or None if can continue."""
        if self.cost_exceeded():
            return f"cost limit (${self.cost_limit:.2f})"
        if self.time_exceeded():
            return f"time limit ({self.time_limit}m)"
        return None

    def print_summary(self) -> None:
        """Print cost and time summary."""
        print_cost_summary(
            self.cost,
            self.elapsed_seconds,
            cost_limit=self.cost_limit,
            time_limit=self.time_limit,
        )
