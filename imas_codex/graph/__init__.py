"""Knowledge graph module for facility data.

This module provides:
    - models: Pydantic models generated from schemas/facility.yaml
    - client: Neo4j client for graph operations
    - schema: Schema-driven graph ontology (node labels, relationships)
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
    # Client
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
