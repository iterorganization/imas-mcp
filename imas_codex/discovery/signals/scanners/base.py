"""Base scanner protocol and registry for data source plugins.

Defines the interface all scanners must implement and provides a registry
for dispatching scanners by data source type. Scanner types correspond to
keys in the facility config data_sources section.

Architecture:
    The scanner system is facility-agnostic. Each facility declares its
    data_sources in facility YAML, and the registry dispatches to the
    matching scanner plugin. Scanners handle facility-specific enumeration
    while shared infrastructure handles enrichment and validation.

    Scanner types:
    - tdi: MDSplus TDI function files (.fun) — TCV
    - ppf: JET Processed Pulse Files — JET
    - edas: Experiment Data Access System — JT-60SA
    - mdsplus: Direct MDSplus tree traversal — any MDSplus facility
    - imas: IMAS IDS signal enumeration — ITER, JET, JT-60SA
    - wiki: Wiki-documented signal extraction — any facility with wiki

    The wiki scanner is special: it runs for ALL facilities that have
    wiki_sites configured, complementing the primary data source scanners
    with pre-documented signal metadata (descriptions, units, paths).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from imas_codex.graph.models import DataAccess, FacilitySignal

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result from a scanner's scan() method.

    Attributes:
        signals: Discovered FacilitySignal nodes (status=discovered).
        data_access: DataAccess node created/found for this data source.
        metadata: Scanner-specific metadata (e.g., TDI function list, PPF DDA catalog).
            Stored for use by enrichment and check phases.
        stats: Summary statistics for logging/progress.
        wiki_context: Wiki-derived context keyed by signal accessor or path.
            Used by the enrichment pipeline to inject high-quality descriptions
            instead of relying solely on LLM generation.
    """

    signals: list[FacilitySignal] = field(default_factory=list)
    data_access: DataAccess | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    wiki_context: dict[str, dict[str, str]] = field(default_factory=dict)


@runtime_checkable
class DataSourceScanner(Protocol):
    """Interface for data source-specific signal discovery.

    Each scanner handles one data source type (TDI, PPF, EDAS, MDSplus, IMAS).
    The scanner_type class attribute must match the key in facility config
    data_sources section.

    Lifecycle:
        1. scan() — discover signals from data source
        2. (shared) enrichment via LLM classification, with wiki context injection
        3. check() — validate signals return data

    Scanners are stateless — all state flows through arguments.
    """

    scanner_type: str
    """Data source type key matching data_sources config (e.g., 'tdi', 'ppf')."""

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from this data source.

        Args:
            facility: Facility ID (e.g., "tcv", "jet")
            ssh_host: SSH host for remote access
            config: Data source config from facility YAML (e.g., data_sources.tdi)
            reference_shot: Reference shot for validation context

        Returns:
            ScanResult with discovered signals and metadata
        """
        ...

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate signals return data for reference shot.

        Args:
            facility: Facility ID
            ssh_host: SSH host for remote access
            signals: Signals to validate
            config: Data source config from facility YAML
            reference_shot: Shot to validate against

        Returns:
            List of check results per signal: {signal_id, valid, shape, dtype, error}
        """
        ...


# =============================================================================
# Scanner Registry
# =============================================================================

_registry: dict[str, DataSourceScanner] = {}


def register_scanner(scanner: DataSourceScanner) -> None:
    """Register a scanner instance in the global registry.

    Args:
        scanner: Scanner instance to register. Must have scanner_type attribute.
    """
    _registry[scanner.scanner_type] = scanner
    logger.debug("Registered scanner: %s", scanner.scanner_type)


def get_scanner(scanner_type: str) -> DataSourceScanner:
    """Get a registered scanner by type.

    Lazily imports and registers scanners on first access.

    Args:
        scanner_type: Data source type (e.g., "tdi", "ppf")

    Returns:
        Scanner instance

    Raises:
        KeyError: If no scanner registered for this type
    """
    if not _registry:
        _auto_register()

    if scanner_type not in _registry:
        msg = (
            f"No scanner registered for '{scanner_type}'. "
            f"Available: {list(_registry.keys())}"
        )
        raise KeyError(msg)

    return _registry[scanner_type]


def list_scanners() -> list[str]:
    """List all registered scanner types."""
    if not _registry:
        _auto_register()
    return list(_registry.keys())


def get_scanners_for_facility(facility: str) -> list[DataSourceScanner]:
    """Get scanners matching a facility's data_sources config.

    Returns scanners for each configured data source type, plus the wiki
    scanner if the facility has wiki_sites configured.

    Args:
        facility: Facility ID (e.g., "tcv", "jet")

    Returns:
        List of scanner instances, ordered by config key order.
        Wiki scanner is appended last as a complementary source.
    """
    from imas_codex.discovery.base.facility import get_facility

    if not _registry:
        _auto_register()

    config = get_facility(facility)
    data_sources = config.get("data_sources", {})

    scanners = []
    for source_type in data_sources:
        if source_type in _registry:
            scanners.append(_registry[source_type])
        else:
            logger.warning(
                "No scanner for data source '%s' (facility: %s)",
                source_type,
                facility,
            )

    # Always include wiki scanner if facility has wiki_sites
    wiki_sites = config.get("wiki_sites", [])
    if wiki_sites and "wiki" in _registry and _registry["wiki"] not in scanners:
        scanners.append(_registry["wiki"])

    return scanners


def get_facility_reference_shot(facility: str) -> int | None:
    """Get the reference shot/pulse for a facility from its config.

    Checks each data source config for a reference_shot or reference_pulse
    field. Returns the first one found.

    Args:
        facility: Facility ID

    Returns:
        Reference shot number, or None if not configured
    """
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    data_sources = config.get("data_sources", {})

    for source_config in data_sources.values():
        if isinstance(source_config, dict):
            ref = source_config.get("reference_shot") or source_config.get(
                "reference_pulse"
            )
            if ref:
                return int(ref)
    return None


def _auto_register() -> None:
    """Auto-register all built-in scanners."""
    # Import triggers module-level registration
    from imas_codex.discovery.signals.scanners import (  # noqa: F401
        edas,
        imas,
        mdsplus,
        ppf,
        tdi,
        wiki,
    )
