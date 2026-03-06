"""MDSplus tree discovery and ingestion.

This module provides facility-agnostic tools for:
- Tree structure discovery via remote introspection
- Epoch boundary detection (structural versioning)
- DataNode extraction with cross-links to versions

Works with any facility that uses MDSplus as a data store.

Usage:
    from imas_codex.mdsplus import discover_epochs_optimized
    from imas_codex.graph import GraphClient

    # Discover epochs (SSH to facility) - optimized batch version
    epochs, structures = discover_epochs_optimized("tcv", "results")
"""

from imas_codex.mdsplus.batch_discovery import (
    BatchDiscovery,
    DiscoveryCheckpoint,
    EpochProgress,
    discover_epochs_optimized,
    refine_boundaries,
)
from imas_codex.mdsplus.extraction import (
    async_extract_tree_version,
    discover_tree,
    extract_tree_version,
    get_static_tree_config,
    ingest_static_tree,
)
from imas_codex.mdsplus.ingestion import (
    compute_canonical_path,
    normalize_mdsplus_path,
)

# Backward-compatible aliases
discover_static_tree = discover_tree

__all__ = [
    "BatchDiscovery",
    "DiscoveryCheckpoint",
    "EpochProgress",
    "async_extract_tree_version",
    "compute_canonical_path",
    "discover_epochs_optimized",
    "discover_static_tree",
    "discover_tree",
    "extract_tree_version",
    "get_static_tree_config",
    "ingest_static_tree",
    "normalize_mdsplus_path",
    "refine_boundaries",
]
