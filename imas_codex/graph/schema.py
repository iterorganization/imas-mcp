"""Schema-driven graph ontology derived from LinkML.

This module provides runtime introspection of the LinkML schema
(schemas/facility.yaml) to derive Neo4j node labels, relationship types,
and constraints. This is the single source of truth for the graph structure.

Example:
    >>> from imas_codex.graph.schema import GraphSchema
    >>> schema = GraphSchema()
    >>> print(schema.node_labels)
    ['Facility', 'MDSplusServer', 'MDSplusTree', ...]
    >>> print(schema.get_identifier("Facility"))
    'id'
    >>> print(schema.get_relationships("MDSplusServer"))
    [('MDSplusServer', 'facility_id', 'Facility'), ...]
"""

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

from linkml_runtime.utils.schemaview import SchemaView


def _get_schema_path() -> Path:
    """Get the path to the LinkML schema file."""
    return Path(__file__).parent.parent / "schemas" / "facility.yaml"


@dataclass(frozen=True)
class Relationship:
    """A relationship derived from a LinkML slot with class range.

    Attributes:
        from_class: Source class name (Neo4j label)
        slot_name: LinkML slot name
        to_class: Target class name (Neo4j label)
        multivalued: Whether the slot is multivalued
    """

    from_class: str
    slot_name: str
    to_class: str
    multivalued: bool = False

    @property
    def cypher_type(self) -> str:
        """Convert to Neo4j relationship type (SCREAMING_SNAKE_CASE)."""
        # e.g., "facility_id" -> "FACILITY_ID", "writes_to" -> "WRITES_TO"
        return self.slot_name.upper()


class GraphSchema:
    """Schema-driven graph ontology derived from LinkML.

    This class provides runtime introspection of the LinkML schema to derive:
    - Node labels (class names)
    - Relationship types (slots with class ranges)
    - Identifier fields per class
    - Constraint definitions

    All information is derived from schemas/facility.yaml, making it the
    single source of truth for the graph structure.
    """

    def __init__(self, schema_path: Path | str | None = None) -> None:
        """Initialize from LinkML schema.

        Args:
            schema_path: Path to LinkML schema YAML. Defaults to schemas/facility.yaml.
        """
        self._schema_path = Path(schema_path) if schema_path else _get_schema_path()
        self._view = SchemaView(str(self._schema_path))

    @cached_property
    def node_labels(self) -> list[str]:
        """Get all node labels (non-abstract class names)."""
        return [
            name
            for name, cls in self._view.all_classes().items()
            if not getattr(cls, "abstract", False) and not name.startswith("_")
        ]

    @cached_property
    def abstract_classes(self) -> list[str]:
        """Get abstract class names (not instantiated as nodes)."""
        return [
            name
            for name, cls in self._view.all_classes().items()
            if getattr(cls, "abstract", False)
        ]

    @cached_property
    def relationships(self) -> list[Relationship]:
        """Get all relationships derived from slots with class ranges."""
        rels = []
        for class_name in self.node_labels:
            for slot in self._view.class_induced_slots(class_name):
                slot_range = slot.range
                # Check if range is a class (relationship) vs primitive type
                if slot_range and slot_range in self._view.all_classes():
                    rels.append(
                        Relationship(
                            from_class=class_name,
                            slot_name=slot.name,
                            to_class=slot_range,
                            multivalued=bool(slot.multivalued),
                        )
                    )
        return rels

    @cached_property
    def relationship_types(self) -> list[str]:
        """Get unique relationship type names (SCREAMING_SNAKE_CASE)."""
        return sorted({rel.cypher_type for rel in self.relationships})

    def get_identifier(self, class_name: str) -> str | None:
        """Get the identifier field for a class.

        Args:
            class_name: Name of the class (node label)

        Returns:
            Name of the identifier slot, or None if not found.
        """
        for slot in self._view.class_induced_slots(class_name):
            if getattr(slot, "identifier", False):
                return slot.name
        return None

    def get_model(self, class_name: str) -> type:
        """Get the Pydantic model class for a node label.

        Args:
            class_name: Name of the class (node label)

        Returns:
            Pydantic model class

        Raises:
            ValueError: If class_name is not a valid node label
        """
        if class_name not in self.node_labels:
            msg = f"Unknown node type: {class_name}. Valid: {self.node_labels}"
            raise ValueError(msg)

        # Import models dynamically to avoid circular imports
        from imas_codex.graph import models

        model_class = getattr(models, class_name, None)
        if model_class is None:
            msg = f"Model class not found for {class_name}"
            raise ValueError(msg)
        return model_class

    def get_required_fields(self, class_name: str) -> list[str]:
        """Get required fields for a class.

        Args:
            class_name: Name of the class (node label)

        Returns:
            List of required slot names.
        """
        return [
            slot.name
            for slot in self._view.class_induced_slots(class_name)
            if getattr(slot, "required", False)
        ]

    def get_all_slots(self, class_name: str) -> dict[str, dict[str, Any]]:
        """Get all slots (properties) for a class with their metadata.

        Args:
            class_name: Name of the class (node label)

        Returns:
            Dict mapping slot name to metadata (type, required, identifier).
            Only includes truthy flags for compactness.
        """
        slots = {}
        for slot in self._view.class_induced_slots(class_name):
            # Convert LinkML types (extended_str, TypeDefinitionName) to plain strings
            slot_range = str(slot.range) if slot.range else "string"
            is_relationship = slot_range in self._view.all_classes()

            # Build compact representation (only truthy values)
            info: dict[str, Any] = {"type": slot_range}
            if getattr(slot, "required", False):
                info["required"] = True
            if getattr(slot, "identifier", False):
                info["identifier"] = True
            if getattr(slot, "multivalued", False):
                info["multivalued"] = True
            if is_relationship:
                info["relationship"] = True
            if slot.description:
                info["description"] = str(slot.description)

            slots[slot.name] = info
        return slots

    def get_enums(self) -> dict[str, list[str]]:
        """Get all enums with their permissible values.

        Returns:
            Dict mapping enum name to list of permissible values.
        """
        enums = {}
        for name, enum_def in self._view.all_enums().items():
            if enum_def.permissible_values:
                enums[name] = list(enum_def.permissible_values.keys())
        return enums

    def get_class_description(self, class_name: str) -> str | None:
        """Get the description of a class.

        Args:
            class_name: Name of the class (node label)

        Returns:
            Description string or None.
        """
        cls = self._view.get_class(class_name)
        # Convert LinkML extended_str to plain string for JSON serialization
        return str(cls.description) if cls and cls.description else None

    def get_relationships_from(self, class_name: str) -> list[Relationship]:
        """Get relationships originating from a class.

        Args:
            class_name: Name of the source class

        Returns:
            List of relationships where this class is the source.
        """
        return [rel for rel in self.relationships if rel.from_class == class_name]

    def get_relationships_to(self, class_name: str) -> list[Relationship]:
        """Get relationships targeting a class.

        Args:
            class_name: Name of the target class

        Returns:
            List of relationships where this class is the target.
        """
        return [rel for rel in self.relationships if rel.to_class == class_name]

    def get_slot_info(self, class_name: str, slot_name: str) -> dict[str, Any]:
        """Get detailed information about a slot.

        Args:
            class_name: Name of the class
            slot_name: Name of the slot

        Returns:
            Dictionary with slot metadata.
        """
        for slot in self._view.class_induced_slots(class_name):
            if slot.name == slot_name:
                return {
                    "name": str(slot.name),
                    "range": str(slot.range) if slot.range else None,
                    "required": getattr(slot, "required", False),
                    "identifier": getattr(slot, "identifier", False),
                    "multivalued": getattr(slot, "multivalued", False),
                    "description": str(slot.description) if slot.description else None,
                }
        return {}

    def get_private_slots(self, class_name: str) -> list[str]:
        """Get slot names marked with is_private: true annotation.

        Private slots are stored in *_private.yaml files only,
        never written to the graph or included in OCI artifacts.

        Args:
            class_name: Name of the class (node label)

        Returns:
            List of private slot names.
        """
        private_slots = []
        for slot in self._view.class_induced_slots(class_name):
            if slot.annotations:
                # Use getattr for JsonObj - .get() doesn't work on LinkML JsonObj
                ann = getattr(slot.annotations, "is_private", None)
                if ann and str(ann.value).lower() == "true":
                    private_slots.append(slot.name)
        return private_slots

    def is_private_slot(self, class_name: str, slot_name: str) -> bool:
        """Check if a specific slot is marked is_private: true.

        Args:
            class_name: Name of the class (node label)
            slot_name: Name of the slot to check

        Returns:
            True if the slot is private.
        """
        for slot in self._view.class_induced_slots(class_name):
            if slot.name == slot_name and slot.annotations:
                # Use getattr for JsonObj - .get() doesn't work on LinkML JsonObj
                ann = getattr(slot.annotations, "is_private", None)
                return ann is not None and str(ann.value).lower() == "true"
        return False

    def get_public_slots(self, class_name: str) -> list[str]:
        """Get slot names that are NOT marked private.

        Public slots are safe for the graph and OCI artifacts.

        Args:
            class_name: Name of the class (node label)

        Returns:
            List of public slot names.
        """
        private_set = set(self.get_private_slots(class_name))
        return [
            slot.name
            for slot in self._view.class_induced_slots(class_name)
            if slot.name not in private_set
        ]

    # =========================================================================
    # Cypher Generation Helpers
    # =========================================================================

    def needs_composite_constraint(self, class_name: str) -> bool:
        """Check if a class needs a composite (identifier, facility_id) constraint.

        Classes that have both an identifier field AND a required facility_id
        field need composite uniqueness to support multi-facility graphs.

        Args:
            class_name: Name of the class (node label)

        Returns:
            True if the class needs a composite constraint.
        """
        id_field = self.get_identifier(class_name)
        required = self.get_required_fields(class_name)
        # Needs composite if has identifier, has required facility_id,
        # and identifier is not facility_id itself
        return (
            id_field is not None
            and "facility_id" in required
            and id_field != "facility_id"
        )

    def constraint_statements(self) -> list[str]:
        """Generate Neo4j constraint statements for all node types.

        For facility-owned nodes (those with required facility_id), creates
        composite uniqueness constraints on (identifier, facility_id) to
        support multi-facility graphs where the same logical entity can
        exist at different facilities.

        Returns:
            List of Cypher CREATE CONSTRAINT statements.
        """
        statements = []
        for label in self.node_labels:
            id_field = self.get_identifier(label)
            if not id_field:
                continue

            constraint_name = f"{label.lower()}_{id_field}"

            if self.needs_composite_constraint(label):
                # Composite constraint: (identifier, facility_id)
                # Allows same identifier at different facilities
                statements.append(
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE (n.{id_field}, n.facility_id) IS UNIQUE"
                )
            else:
                # Simple constraint: just the identifier
                # For Facility, IMASPath, and 1:1 facility mappings
                statements.append(
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{id_field} IS UNIQUE"
                )
        return statements

    def index_statements(
        self, indexes: dict[str, list[str]] | None = None
    ) -> list[str]:
        """Generate Neo4j index statements.

        Creates indexes for:
        - facility_id on all facility-owned nodes (for fast facility-scoped queries)
        - Common lookup patterns (category, role, etc.)

        Args:
            indexes: Optional dict mapping label to list of fields to index.
                     If not provided, creates indexes on common lookup patterns.

        Returns:
            List of Cypher CREATE INDEX statements.
        """
        if indexes is None:
            # Default indexes based on common query patterns
            indexes = {
                "Facility": ["ssh_host"],
                "MDSplusServer": ["role"],
                "TreeNode": ["node_type"],
                "Diagnostic": ["category"],
                "AnalysisCode": ["code_type"],
                "Tool": ["category", "available"],
            }

        statements = []

        # Add facility_id index for all facility-owned nodes
        for label in self.node_labels:
            if "facility_id" in self.get_required_fields(label):
                index_name = f"{label.lower()}_facility_id"
                statements.append(
                    f"CREATE INDEX {index_name} IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.facility_id)"
                )

        # Add custom indexes
        for label, fields in indexes.items():
            if label in self.node_labels:
                for field_name in fields:
                    index_name = f"{label.lower()}_{field_name}"
                    statements.append(
                        f"CREATE INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) ON (n.{field_name})"
                    )
        return statements


# =============================================================================
# Utility Functions (migrated from cypher.py)
# =============================================================================


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


def merge_node_query(label: str, id_field: str = "id") -> str:
    """Generate a MERGE query template for a node.

    Args:
        label: Node label (class name)
        id_field: Name of the identifier field

    Returns:
        Cypher MERGE query template with $id and $props parameters.
    """
    return f"MERGE (n:{label} {{{id_field}: $id}}) SET n += $props"


def merge_relationship_query(
    from_label: str,
    to_label: str,
    rel_type: str,
    from_id_field: str = "id",
    to_id_field: str = "id",
) -> str:
    """Generate a MERGE query template for a relationship.

    Args:
        from_label: Source node label
        to_label: Target node label
        rel_type: Relationship type (SCREAMING_SNAKE_CASE)
        from_id_field: Source node identifier field
        to_id_field: Target node identifier field

    Returns:
        Cypher MERGE query template with $from_id and $to_id parameters.
    """
    return (
        f"MATCH (a:{from_label} {{{from_id_field}: $from_id}}), "
        f"(b:{to_label} {{{to_id_field}: $to_id}}) "
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )


# =============================================================================
# Module-level singleton for convenience
# =============================================================================

_schema = GraphSchema()


def get_schema() -> GraphSchema:
    """Get the global GraphSchema instance."""
    return _schema
