"""Common CLI utilities for discovery commands.

Provides shared infrastructure used by all discovery domains:
- CLI harness for async discovery with rich/plain output
- Service monitor factory
- Domain option decorator
- Rich output detection
- Discovery domain enumeration
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import BaseProgressDisplay
    from imas_codex.discovery.base.services import ServiceMonitor

logger = logging.getLogger(__name__)

# Valid discovery domains
DISCOVERY_DOMAINS = ("paths", "wiki", "signals", "code", "documents")


# =============================================================================
# Shared --reset-to option
# =============================================================================

# Human-readable target descriptions per domain
_RESET_TARGET_HELP: dict[str, dict[str, str]] = {
    "signals": {
        "discovered": "re-enrich all enriched/checked signals",
        "enriched": "re-check all checked signals",
    },
    "paths": {
        "triaged": "re-score all scored paths",
        "scanned": "re-triage and re-score all triaged/scored paths",
    },
    "wiki": {
        "scanned": "re-score all scored/ingested pages",
        "scored": "re-ingest all ingested pages",
    },
    "code": {
        "discovered": "re-triage all triaged/scored/ingested files",
        "triaged": "re-enrich and re-score all scored/ingested files",
        "scored": "re-ingest all ingested files",
    },
    "documents": {
        "discovered": "re-score all scored/ingested documents",
        "scored": "re-ingest all ingested documents",
    },
}


def reset_to_option(domain: str) -> Callable:
    """Create a ``--reset-to`` Click option for a specific discovery domain.

    Usage::

        @click.command()
        @reset_to_option("signals")
        def signals(facility, reset_to, ...):
            ...

    The option accepts the valid target states for the domain and provides
    domain-specific help text listing valid values and what they do.
    """
    from imas_codex.discovery.base.reset import get_valid_targets

    targets = get_valid_targets(domain)
    help_parts = _RESET_TARGET_HELP.get(domain, {})

    # Build help text
    lines = ["Reset nodes to a target state for reprocessing. Valid targets:"]
    for t in targets:
        desc = help_parts.get(t, "")
        lines.append(f"  {t}: {desc}" if desc else f"  {t}")

    help_text = "\n".join(lines)

    return click.option(
        "--reset-to",
        type=click.Choice(targets, case_sensitive=False),
        default=None,
        help=help_text,
    )


# =============================================================================
# CLI Harness — eliminates async boilerplate across all discovery commands
# =============================================================================


@dataclass
class DiscoveryConfig:
    """Configuration for the discovery CLI harness.

    Each CLI command constructs this config to declare its requirements,
    then passes it to :func:`run_discovery` which handles all async
    setup/teardown, display lifecycle, and service monitoring.
    """

    domain: str
    """Domain name for logging (e.g. 'paths', 'wiki', 'signals')."""

    facility: str
    """Facility identifier."""

    facility_config: dict
    """Full facility config dict from get_facility()."""

    # Service monitor configuration
    check_graph: bool = True
    check_embed: bool = True
    check_ssh: bool = True
    check_auth: bool = False
    check_model: bool = False
    model_section: str = "language"

    # Display configuration
    display: BaseProgressDisplay | None = None
    """Progress display instance (created by caller). If None, runs in plain mode."""

    graph_refresh_interval: float = 0.5
    """Interval for graph state refresh task (seconds)."""

    ticker_interval: float = 0.15
    """Interval for display tick task (seconds)."""

    # Custom graph refresh (overrides default display.refresh_from_graph)
    graph_refresh_fn: Callable[[], Awaitable[None]] | None = None
    """Async function called each refresh cycle instead of display.refresh_from_graph().

    Use when the display needs explicit kwargs (e.g., signals.py's update_from_graph).
    """

    # Logging suppression during rich display
    suppress_loggers: list[str] = field(default_factory=list)
    """Logger names to suppress to WARNING during rich display."""

    verbose: bool = False
    """Whether verbose output was requested."""

    force_exit_on_complete: bool = False
    """Force a hard process exit after summary output is flushed."""


def make_log_print(domain: str, console: Any | None = None) -> Callable[[str], None]:
    """Create a log_print closure for a discovery domain.

    Prints to Rich console when available, otherwise logs with markup stripped.
    """
    domain_logger = logging.getLogger(f"imas_codex.discovery.{domain}")

    def log_print(msg: str, style: str = "") -> None:
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            domain_logger.info(clean_msg)

    return log_print


def setup_logging(
    domain: str,
    facility: str,
    use_rich: bool,
    verbose: bool = False,
) -> Any | None:
    """Set up CLI logging and return a Console (if rich) or None.

    Handles:
    - File logging via configure_cli_logging()
    - Console creation for rich mode
    - logging.basicConfig for plain mode
    """
    from rich.console import Console

    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging(domain, facility=facility, verbose=verbose)

    if use_rich:
        return Console()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return None


def run_discovery(
    config: DiscoveryConfig,
    async_main: Callable[..., Awaitable[dict]],
    *,
    on_complete: Callable[[dict], None] | None = None,
) -> dict:
    """Run a discovery pipeline with full async lifecycle management.

    This is the single harness that all discovery CLI commands use. It handles:
    - Rich vs. plain mode detection and setup
    - Service monitor lifecycle (start/stop)
    - Shutdown handler installation (3-press protocol)
    - Background refresh and ticker tasks
    - Graceful cleanup on completion or interruption

    Args:
        config: Discovery configuration declaring domain, display, etc.
        async_main: Async function receiving (stop_event, service_monitor)
            that runs the actual discovery pipeline and returns a results dict.
            service_monitor is None in plain mode.
        on_complete: Optional callback invoked after pipeline completes
            (within the display context, before summary).

    Returns:
        Results dict from async_main.
    """
    from imas_codex.cli.shutdown import safe_asyncio_run

    use_rich = config.display is not None

    if not use_rich:
        # Plain mode: just run with shutdown handlers
        async def _run_plain():
            from imas_codex.cli.shutdown import install_shutdown_handlers

            stop_event = asyncio.Event()
            install_shutdown_handlers(stop_event=stop_event)
            return await async_main(stop_event, None)

        result = safe_asyncio_run(_run_plain())
        _record_session_time(config, result)
        return result

    # Rich mode: full lifecycle with display, service monitor, refresh tasks
    display = config.display
    service_monitor = create_discovery_monitor(
        config.facility_config,
        check_graph=config.check_graph,
        check_embed=config.check_embed,
        check_ssh=config.check_ssh,
        check_auth=config.check_auth,
        check_model=config.check_model,
        model_section=config.model_section,
    )

    # Suppress noisy loggers during rich display: raise console handler
    # levels to ERROR so chatty WARNING-level messages don't leak past the
    # rich-display canvas.  Logger levels are left unchanged so that file
    # handlers (attached by configure_cli_logging()) continue to receive
    # all messages.
    for mod in config.suppress_loggers:
        lg = logging.getLogger(mod)
        for handler in lg.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.ERROR)

    # Detach any console StreamHandler attached to the root or imas_codex
    # logger so that nothing prints to stderr while the rich display is live.
    # Keeps RotatingFileHandler / FileHandler (file logging is preserved).
    for logger_name in ("", "imas_codex"):
        lg = logging.getLogger(logger_name)
        for handler in list(lg.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                lg.removeHandler(handler)

    display.service_monitor = service_monitor

    with display:

        async def _run_rich():
            from imas_codex.cli.shutdown import install_shutdown_handlers

            stop_event = asyncio.Event()
            await service_monitor.__aenter__()
            install_shutdown_handlers(stop_event=stop_event, display=display)

            async def _refresh_graph():
                while True:
                    try:
                        if not stop_event.is_set():
                            if config.graph_refresh_fn:
                                await config.graph_refresh_fn()
                            elif hasattr(display, "refresh_from_graph"):
                                # Run in thread pool to avoid blocking the
                                # event loop — sync Neo4j queries prevent
                                # signal delivery (Ctrl+C) and display updates.
                                await asyncio.to_thread(
                                    display.refresh_from_graph, config.facility
                                )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    await asyncio.sleep(config.graph_refresh_interval)

            async def _tick():
                while True:
                    try:
                        display.tick()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    await asyncio.sleep(config.ticker_interval)

            refresh_task = asyncio.create_task(_refresh_graph())
            ticker_task = asyncio.create_task(_tick())

            try:
                result = await async_main(stop_event, service_monitor)
            finally:
                # Signal background tasks to stop before cancelling,
                # so they don't start a new to_thread graph query that
                # would block cancellation for seconds.
                stop_event.set()
                refresh_task.cancel()
                ticker_task.cancel()
                # Bounded wait so a blocked thread (e.g. Neo4j query
                # inside to_thread) can't prevent shutdown.
                try:
                    await asyncio.wait({refresh_task, ticker_task}, timeout=3.0)
                except asyncio.CancelledError:
                    pass
                try:
                    await asyncio.wait_for(
                        service_monitor.__aexit__(None, None, None),
                        timeout=3.0,
                    )
                except (asyncio.CancelledError, TimeoutError):
                    pass

            return result

        result = safe_asyncio_run(_run_rich())

        # Final graph refresh for accurate summary
        if on_complete:
            on_complete(result)
        elif hasattr(display, "refresh_from_graph"):
            try:
                display.refresh_from_graph(config.facility)
            except Exception:
                pass

        if hasattr(display, "print_summary"):
            display.print_summary()

    # Record session wall-clock time to the graph for accumulated display
    _record_session_time(config, result)

    if config.force_exit_on_complete and "PYTEST_CURRENT_TEST" not in os.environ:
        from imas_codex.cli.shutdown import force_exit

        force_exit()

    return result


def _record_session_time(config: DiscoveryConfig, result: dict) -> None:
    """Record session wall-clock time to the Facility node.

    Best-effort — failures are silently ignored.
    """
    elapsed = result.get("elapsed_seconds", 0)
    if elapsed <= 0:
        return
    try:
        from imas_codex.discovery.base.progress import record_session_time

        record_session_time(config.facility, config.domain, elapsed)
    except Exception:
        pass


def domain_option(required: bool = False, default: str | None = None) -> Callable:
    """Add domain selection option.

    Args:
        required: Whether domain selection is required
        default: Default domain if not specified

    Options:
        --domain/-d: Discovery domain (paths, wiki, signals, files)
    """

    def decorator[F: Callable[..., Any]](func: F) -> F:
        return click.option(
            "--domain",
            "-d",
            type=click.Choice(DISCOVERY_DOMAINS),
            required=required,
            default=default,
            help="Discovery domain to target",
        )(func)

    return decorator


def use_rich_output() -> bool:
    """Determine whether to use rich output via auto-detection."""
    from imas_codex.cli.rich_output import should_use_rich

    return should_use_rich()


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
