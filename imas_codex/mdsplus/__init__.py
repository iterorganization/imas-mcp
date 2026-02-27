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
    epochs, structures = discover_epochs_optimized("tcv", "results")

    # Or use original sequential version
    epochs, structures = discover_epochs("tcv", "results", step=500)

    # Ingest to Neo4j
    with GraphClient() as client:
        ingest_epochs(client, epochs)
        ingest_super_tree(client, "tcv", "results", epochs, structures)

        # Enrich with real metadata (units, descriptions)
        enrich_graph_metadata(client, "tcv", "results", shot=89000)

        # Refine rough boundaries (for legacy sequential data)
        refine_boundaries(client, "tcv", "results")
"""

from imas_codex.mdsplus.batch_discovery import (
    BatchDiscovery,
    DiscoveryCheckpoint,
    EpochProgress,
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
    compute_canonical_path,
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
from imas_codex.mdsplus.static import (
    discover_static_tree,
    get_static_tree_config,
    ingest_static_tree,
)

__all__ = [
    "BatchDiscovery",
    "DiscoveryCheckpoint",
    "EpochProgress",
    "TreeDiscovery",
    "cleanup_legacy_nodes",
    "compute_canonical_path",
    "discover_epochs",
    "discover_epochs_optimized",
    "discover_static_tree",
    "enrich_graph_metadata",
    "enrich_node_metadata",
    "extract_metadata",
    "get_static_tree_config",
    "get_tree_structure",
    "ingest_epochs",
    "ingest_static_tree",
    "ingest_super_tree",
    "merge_legacy_metadata",
    "normalize_mdsplus_path",
    "refine_boundaries",
]
