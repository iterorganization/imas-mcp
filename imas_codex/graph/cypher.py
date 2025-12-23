"""Neo4j Cypher utilities for the knowledge graph.

This module provides Neo4j-specific constants and utilities that complement
the LinkML-generated models. The models define the schema structure, while
this module handles the Neo4j implementation details.

Node labels and relationship types are defined here to:
1. Provide a single source of truth for Cypher queries
2. Enable type-safe label/relationship references
3. Keep Neo4j concerns separate from the schema definitions
"""

from enum import Enum
from typing import Any


class NodeLabel(str, Enum):
    """Neo4j node labels corresponding to schema classes."""

    # Core
    FACILITY = "Facility"

    # Data Infrastructure
    MDSPLUS_SERVER = "MDSplusServer"
    MDSPLUS_TREE = "MDSplusTree"
    TREE_NODE = "TreeNode"
    TDI_FUNCTION = "TDIFunction"
    DATA_LOCATION = "DataLocation"
    SHOT_RANGE = "ShotRange"

    # Computing Environment
    PYTHON_ENVIRONMENT = "PythonEnvironment"
    OPERATING_SYSTEM = "OperatingSystem"
    COMPILER = "Compiler"
    MODULE_SYSTEM = "ModuleSystem"

    # Tools
    TOOL = "Tool"

    # Data Semantics
    DIAGNOSTIC = "Diagnostic"
    ANALYSIS_CODE = "AnalysisCode"

    # IMAS Mapping
    IMAS_PATH = "IMASPath"
    IMAS_MAPPING = "IMASMapping"


class RelationType(str, Enum):
    """Neo4j relationship types."""

    # Structural relationships (ownership/containment)
    BELONGS_TO = "BELONGS_TO"  # Generic: X belongs to Facility
    HOSTED_BY = "HOSTED_BY"  # Server -> Facility
    STORED_ON = "STORED_ON"  # Tree -> Server
    CHILD_OF = "CHILD_OF"  # TreeNode -> TreeNode (hierarchy)
    CONTAINS = "CONTAINS"  # Tree -> TreeNode

    # Data access relationships
    ACCESSES = "ACCESSES"  # TDIFunction -> TreeNode
    CALLS = "CALLS"  # TDIFunction -> TDIFunction
    READS_FROM = "READS_FROM"  # AnalysisCode -> TreeNode

    # Data production relationships
    PRODUCES = "PRODUCES"  # AnalysisCode -> TreeNode
    WRITES_TO = "WRITES_TO"  # Diagnostic -> TreeNode

    # IMAS mapping relationships
    MAPS_TO = "MAPS_TO"  # TreeNode -> IMASPath
    EQUIVALENT_TO = "EQUIVALENT_TO"  # TDIFunction -> IMASPath

    # Environment relationships
    INSTALLED_ON = "INSTALLED_ON"  # Tool/Compiler/Python -> Facility
    REQUIRES = "REQUIRES"  # Code -> Tool/Compiler


def to_cypher_props(obj: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Convert an object to Neo4j-compatible properties dict.

    Args:
        obj: Object with attributes to convert (dataclass, Pydantic model, etc.)
        exclude: Set of attribute names to exclude

    Returns:
        Dictionary of non-None properties suitable for Cypher queries.
    """
    exclude = exclude or set()

    # Handle different object types
    if hasattr(obj, "model_dump"):
        # Pydantic v2 model
        props = obj.model_dump(exclude_none=True)
    elif hasattr(obj, "dict"):
        # Pydantic v1 model
        props = obj.dict(exclude_none=True)
    elif hasattr(obj, "__dict__"):
        # Regular object/dataclass
        props = {k: v for k, v in obj.__dict__.items() if v is not None}
    else:
        props = dict(obj)

    # Filter excluded and convert enums
    result = {}
    for key, value in props.items():
        if key.startswith("_") or key in exclude:
            continue
        if isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, list):
            # Neo4j can store lists of primitives
            result[key] = [v.value if isinstance(v, Enum) else v for v in value]
        else:
            result[key] = value

    return result


def merge_node_query(label: NodeLabel, id_field: str = "id") -> str:
    """Generate a MERGE query template for a node.

    Args:
        label: Node label
        id_field: Name of the identifier field

    Returns:
        Cypher MERGE query template with $id and $props parameters.
    """
    return f"MERGE (n:{label.value} {{{id_field}: $id}}) SET n += $props"


def merge_relationship_query(
    from_label: NodeLabel,
    to_label: NodeLabel,
    rel_type: RelationType,
    from_id_field: str = "id",
    to_id_field: str = "id",
) -> str:
    """Generate a MERGE query template for a relationship.

    Args:
        from_label: Source node label
        to_label: Target node label
        rel_type: Relationship type
        from_id_field: Source node identifier field
        to_id_field: Target node identifier field

    Returns:
        Cypher MERGE query template with $from_id and $to_id parameters.
    """
    return (
        f"MATCH (a:{from_label.value} {{{from_id_field}: $from_id}}), "
        f"(b:{to_label.value} {{{to_id_field}: $to_id}}) "
        f"MERGE (a)-[r:{rel_type.value}]->(b)"
    )
