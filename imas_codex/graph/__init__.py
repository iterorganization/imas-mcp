"""Knowledge graph module for facility data.

This module provides:
    - client: Neo4j client for graph operations
    - models: Pydantic models generated from schemas/facility.yaml
    - schema: Schema-driven graph ontology

For query-only usage, import just GraphClient:
    from imas_codex.graph import GraphClient

For schema operations (admin, build scripts), import schema utilities:
    from imas_codex.graph import get_schema, GraphSchema
"""

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
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
from imas_codex.graph.schema import (
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
