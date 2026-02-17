"""Knowledge graph module for facility data.

This module provides:
    - client: Neo4j client for graph operations
    - models: Pydantic models generated from schemas/facility.yaml
    - schema: Schema-driven graph ontology

For query-only usage, import just GraphClient:
    from imas_codex.graph import GraphClient

For schema operations (admin, build scripts), import schema utilities:
    from imas_codex.graph import get_schema, GraphSchema

Auto-regenerates models if schemas are newer than models.py (dev installs only).
"""

# Must run before importing models — regenerates if stale
from imas_codex.graph._ensure_models import ensure_models_fresh as _ensure_fresh

_ensure_fresh()
del _ensure_fresh

from imas_codex.graph.client import GraphClient  # noqa: E402
from imas_codex.graph.dirs import (  # noqa: E402
    GraphDirInfo,
    create_graph_dir,
    find_graph,
    get_active_graph,
    list_local_graphs,
    switch_active_graph,
)
from imas_codex.graph.meta import (  # noqa: E402
    add_facility_to_meta,
    gate_ingestion,
    get_graph_meta,
    init_graph_meta,
    remove_facility_from_meta,
)
from imas_codex.graph.models import (  # noqa: E402
    AnalysisCode,
    AnalysisCodeType,
    Diagnostic,
    DiagnosticCategory,
    Facility,
    IMASMapping,
    IMASPath,
    MDSplusServer,
    MDSplusTree,
    ServerRole,
    ShotRange,
    TDIFunction,
    TreeNode,
    TreeNodeType,
)
from imas_codex.graph.profiles import (  # noqa: E402
    Neo4jProfile,
    list_profiles,
    resolve_neo4j,
)
from imas_codex.graph.schema import (  # noqa: E402
    GraphSchema,
    Relationship,
    get_schema,
    merge_node_query,
    merge_relationship_query,
    to_cypher_props,
)

__all__ = [
    # Client (fast import)
    "GraphClient",
    # Graph directory store
    "GraphDirInfo",
    "create_graph_dir",
    "find_graph",
    "get_active_graph",
    "list_local_graphs",
    "switch_active_graph",
    # Graph identity (meta)
    "add_facility_to_meta",
    "gate_ingestion",
    "get_graph_meta",
    "init_graph_meta",
    "remove_facility_from_meta",
    # Neo4j profiles
    "Neo4jProfile",
    "resolve_neo4j",
    "list_profiles",
    # Schema-driven utilities
    "GraphSchema",
    "Relationship",
    "get_schema",
    "to_cypher_props",
    "merge_node_query",
    "merge_relationship_query",
    # Models (generated from schemas/facility.yaml)
    "AnalysisCode",
    "AnalysisCodeType",
    "Diagnostic",
    "DiagnosticCategory",
    "Facility",
    "IMASMapping",
    "IMASPath",
    "MDSplusServer",
    "MDSplusTree",
    "ServerRole",
    "ShotRange",
    "TDIFunction",
    "TreeNode",
    "TreeNodeType",
]
