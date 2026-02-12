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
    GraphProfile,
    list_profiles,
    resolve_graph,
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
    # Graph profiles
    "GraphProfile",
    "resolve_graph",
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
