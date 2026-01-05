"""Knowledge graph module for facility data.

This module provides:
    - client: Neo4j client for graph operations (fast import, ~2s)
    - models: Pydantic models generated from schemas/facility.yaml
    - schema: Schema-driven graph ontology (lazy import to avoid linkml overhead)

For query-only usage, import just GraphClient:
    from imas_codex.graph import GraphClient

For schema operations (admin, build scripts), import schema utilities:
    from imas_codex.graph import get_schema, GraphSchema
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

# Schema utilities are lazily imported to avoid linkml_runtime overhead (~10s)
# for query-only usage. They are only needed for admin/build operations.
if TYPE_CHECKING:
    from imas_codex.graph.schema import (
        GraphSchema,
        Relationship,
    )


def __getattr__(name: str):
    """Lazy import for schema utilities to avoid linkml_runtime overhead."""
    schema_exports = {
        "GraphSchema",
        "Relationship",
        "get_schema",
        "merge_node_query",
        "merge_relationship_query",
        "to_cypher_props",
    }
    if name in schema_exports:
        from imas_codex.graph import schema as schema_module

        return getattr(schema_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client (fast import)
    "GraphClient",
    # Schema-driven utilities (lazy import)
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
