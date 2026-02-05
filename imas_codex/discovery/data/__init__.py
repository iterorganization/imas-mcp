"""Data signal discovery module.

Discovers and documents facility data signals, their paths, access methods,
units, and sign conventions. Supports multiple data formats used by the fusion
community.

Architecture:
- Parallel async workers: Discover, Enrich, Check
- Graph as coordination (claimed_at pattern from wiki/paths)
- Status transitions: discovered → enriched → checked

Supported data sources:
- MDSplus: tree traversal, TDI function introspection
- IMAS: IDS schema enumeration (future)
- HDF5: dataset path enumeration (future)

Usage:
    from imas_codex.discovery.data import (
        run_parallel_data_discovery,
        get_data_discovery_stats,
    )

    # Run parallel discovery
    result = await run_parallel_data_discovery(
        facility="tcv",
        cost_limit=10.0,
    )
"""

from imas_codex.discovery.data.parallel import (
    get_data_discovery_stats,
    reset_transient_signals,
    run_parallel_data_discovery,
)

__all__ = [
    "get_data_discovery_stats",
    "reset_transient_signals",
    "run_parallel_data_discovery",
]
