"""MDSplus tree discovery and ingestion.

This module provides facility-agnostic tools for:
- Tree structure discovery via remote introspection
- Epoch boundary detection (structural versioning)
- Super tree construction with applicability ranges
- TreeNode ingestion with cross-links to versions

Works with any facility that uses MDSplus as a data store.

Usage:
    from imas_codex.mdsplus import discover_epochs_optimized, ingest_epochs, ingest_super_tree
    from imas_codex.graph import GraphClient

    # Discover epochs (SSH to facility) - optimized batch version
    epochs, structures = discover_epochs_optimized("epfl", "results")

    # Or use original sequential version
    epochs, structures = discover_epochs("epfl", "results", step=500)

    # Ingest to Neo4j
    with GraphClient() as client:
        ingest_epochs(client, epochs)
        ingest_super_tree(client, "epfl", "results", epochs, structures)
"""

from imas_codex.mdsplus.batch_discovery import (
    BatchDiscovery,
    DiscoveryCheckpoint,
    discover_epochs_optimized,
)
from imas_codex.mdsplus.discovery import (
    TreeDiscovery,
    discover_epochs,
    get_tree_structure,
)
from imas_codex.mdsplus.ingestion import (
    enrich_node_metadata,
    ingest_epochs,
    ingest_super_tree,
)

__all__ = [
    "BatchDiscovery",
    "DiscoveryCheckpoint",
    "TreeDiscovery",
    "discover_epochs",
    "discover_epochs_optimized",
    "enrich_node_metadata",
    "get_tree_structure",
    "ingest_epochs",
    "ingest_super_tree",
]
