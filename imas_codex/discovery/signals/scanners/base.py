"""Base scanner protocol and registry for data source plugins.

Defines the interface all scanners must implement and provides a registry
for dispatching scanners by data source type. Scanner types correspond to
keys in the facility config data_systems section.

Architecture:
    The scanner system is facility-agnostic. Each facility declares its
    data_systems in facility YAML, and the registry dispatches to the
    matching scanner plugin. Scanners handle facility-specific enumeration
    while shared infrastructure handles enrichment and validation.

    Scanner types:
    - tdi: MDSplus TDI function files (.fun) — TCV
    - ppf: JET Processed Pulse Files — JET
    - edas: Experiment Data Access System — JT-60SA
    - mdsplus: Direct MDSplus tree traversal — any MDSplus facility
    - imas: IMAS IDS signal enumeration — ITER, JET, JT-60SA
    - device_xml: EFIT device XML geometry files in git — JET
    - wiki: Internal wiki metadata loader used for enrichment context

    Wiki content is not exposed as a first-class scanner on the CLI. It is
    loaded separately when wiki_sites are configured so enrichment can reuse
    curated descriptions, units, and path hints.
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
    data_systems section.

    Lifecycle:
        1. scan() — discover signals from data source
        2. (shared) enrichment via LLM classification, with wiki context injection
        3. check() — validate signals return data

    Scanners are stateless — all state flows through arguments.
    """

    scanner_type: str
    """Data source type key matching data_systems config (e.g., 'tdi', 'ppf')."""

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
            config: Data source config from facility YAML (e.g., data_systems.tdi)
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
_auto_registered: bool = False


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
    if not _auto_registered:
        _auto_register()

    if scanner_type not in _registry:
        msg = (
            f"No scanner registered for '{scanner_type}'. "
            f"Available: {list(_registry.keys())}"
        )
        raise KeyError(msg)

    return _registry[scanner_type]


def list_scanners() -> list[str]:
    """List user-selectable scanner types.

    Internal metadata helpers such as the wiki context loader are not exposed as
    normal scanner choices on the CLI.
    """
    if not _auto_registered:
        _auto_register()
    return [scanner_type for scanner_type in _registry if scanner_type != "wiki"]


def get_scanners_for_facility(facility: str) -> list[DataSourceScanner]:
    """Get scanners matching a facility's data_systems config.

    Returns scanners for each configured data source type.

    Wiki-derived metadata is loaded separately for enrichment context and is no
    longer treated as a first-class signal scanner.

    Args:
        facility: Facility ID (e.g., "tcv", "jet")

    Returns:
        List of scanner instances in config key order.
    """
    from imas_codex.discovery.base.facility import get_facility

    if not _auto_registered:
        _auto_register()

    config = get_facility(facility)
    data_systems = config.get("data_systems", {})

    scanners = []

    for source_type, source_config in data_systems.items():
        if isinstance(source_config, dict) and source_config.get("available") is False:
            logger.info(
                "Skipping disabled data source '%s' for facility %s",
                source_type,
                facility,
            )
            continue
        if source_type in _registry:
            scanners.append(_registry[source_type])
        else:
            logger.warning(
                "No scanner for data source '%s' (facility: %s)",
                source_type,
                facility,
            )

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
    data_systems = config.get("data_systems", {})

    for source_config in data_systems.values():
        if isinstance(source_config, dict):
            if source_config.get("available") is False:
                continue
            ref = source_config.get("reference_shot") or source_config.get(
                "reference_pulse"
            )
            if ref:
                return int(ref)
    return None


def _auto_register() -> None:
    """Auto-register all built-in scanners."""
    global _auto_registered  # noqa: PLW0603
    _auto_registered = True
    # Import triggers module-level registration
    from imas_codex.discovery.signals.scanners import (  # noqa: F401
        device_xml,
        edas,
        imas,
        jpf,
        mdsplus,
        ppf,
        tdi,
        wiki,
    )
