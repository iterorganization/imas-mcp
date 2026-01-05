"""MDSplus tree discovery and ingestion.

This module provides facility-agnostic tools for:
- Tree structure discovery via remote introspection
- Epoch boundary detection (structural versioning)
- Super tree construction with applicability ranges
- TreeNode ingestion with cross-links to versions
- Metadata extraction (units, descriptions from COMMENT nodes)

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

        # Enrich with real metadata (units, descriptions)
        enrich_graph_metadata(client, "epfl", "results", shot=89000)

        # Refine rough boundaries (for legacy sequential data)
        refine_boundaries(client, "epfl", "results")
"""

from imas_codex.mdsplus.batch_discovery import (
    BatchDiscovery,
    DiscoveryCheckpoint,
    discover_epochs_optimized,
    refine_boundaries,
)
from imas_codex.mdsplus.discovery import (
    TreeDiscovery,
    discover_epochs,
    get_tree_structure,
)
from imas_codex.mdsplus.ingestion import (
    cleanup_legacy_nodes,
    enrich_node_metadata,
    ingest_epochs,
    ingest_super_tree,
    merge_legacy_metadata,
    normalize_mdsplus_path,
)
from imas_codex.mdsplus.metadata import (
    enrich_graph_metadata,
    extract_metadata,
)

__all__ = [
    "BatchDiscovery",
    "DiscoveryCheckpoint",
    "TreeDiscovery",
    "cleanup_legacy_nodes",
    "discover_epochs",
    "discover_epochs_optimized",
    "enrich_graph_metadata",
    "enrich_node_metadata",
    "extract_metadata",
    "get_tree_structure",
    "ingest_epochs",
    "ingest_super_tree",
    "merge_legacy_metadata",
    "normalize_mdsplus_path",
    "refine_boundaries",
]
