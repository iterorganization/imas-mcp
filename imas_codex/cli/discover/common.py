"""Common CLI utilities for discovery commands.

Provides shared infrastructure used by all discovery domains:
- Service monitor factory
- Domain option decorator
- Rich output detection
- Discovery domain enumeration
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from imas_codex.discovery.base.services import ServiceMonitor

logger = logging.getLogger(__name__)

# Valid discovery domains
DISCOVERY_DOMAINS = ("paths", "wiki", "signals", "code", "documents")


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
