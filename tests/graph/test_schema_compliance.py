"""Schema compliance tests.

Verifies that the graph structure matches the LinkML schema definitions:
- All labels and relationship types are schema-defined
- No undeclared properties exist on nodes
- Constraints and indexes exist
- Required fields are populated
- Enum values are valid
- Identifier uniqueness holds
"""

import pytest

from imas_codex.graph.client import EXPECTED_RELATIONSHIP_TYPES
from imas_codex.graph.schema import GraphSchema

pytestmark = pytest.mark.graph


class TestLabelsAndRelationships:
    """Verify graph labels and relationship types match the schema."""

    # Internal labels that are not in the LinkML schema.
    # GraphMeta is an infrastructure singleton for graph identity/metadata.
    INTERNAL_LABELS: set[str] = {"GraphMeta"}

    def test_all_labels_in_schema(self, graph_labels, schema):
        """Every label in the graph must be defined in the schema."""
        dd_schema = GraphSchema(schema_path="imas_codex/schemas/imas_dd.yaml")
        expected = (
            set(schema.node_labels) | set(dd_schema.node_labels) | self.INTERNAL_LABELS
        )
        unexpected = graph_labels - expected
        assert not unexpected, (
            f"Graph contains labels not in schema: {unexpected}. "
            f"Add them to schemas/facility.yaml or schemas/imas_dd.yaml"
        )

    def test_all_relationship_types_in_schema(self, graph_relationship_types, schema):
        """Every relationship type in the graph must be schema-derived."""
        unexpected = graph_relationship_types - EXPECTED_RELATIONSHIP_TYPES
        assert not unexpected, (
            f"Graph contains relationship types not in schema: {unexpected}. "
            f"Add them as slots with relationship_type annotation in LinkML schemas."
        )

    def test_no_undeclared_properties(self, graph_client, graph_labels, schema):
        """Every property on a graph node must be declared in its schema class.

        Properties that exist on nodes but are not in the LinkML schema
        indicate drift between code and schema definitions.
        """
        dd_schema = GraphSchema(schema_path="imas_codex/schemas/imas_dd.yaml")

        # Internal properties added by Neo4j or our infrastructure
        INTERNAL_PROPERTIES = {"embedding"}

        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue

            # Determine which schema owns this label
            if label in schema.node_labels:
                s = schema
            elif label in dd_schema.node_labels:
                s = dd_schema
            else:
                continue  # Covered by test_all_labels_in_schema

            # Get declared slots (properties) for this label
            declared = set(s.get_all_slots(label).keys())

            # Get actual properties from graph (sample for performance)
            result = graph_client.query(
                f"MATCH (n:{label}) "
                f"WITH keys(n) AS props LIMIT 100 "
                f"UNWIND props AS p "
                f"RETURN DISTINCT p AS property"
            )
            actual = {r["property"] for r in result}

            undeclared = actual - declared - INTERNAL_PROPERTIES
            if undeclared:
                violations.append(
                    f"{label}: undeclared properties {sorted(undeclared)}"
                )

        assert not violations, (
            "Graph nodes have properties not in schema:\n  "
            + "\n  ".join(violations)
            + "\n\nAdd these properties to the LinkML schema or remove them from the graph."
        )


class TestConstraints:
    """Verify expected constraints exist in the graph."""

    def test_constraints_created(self, graph_with_schema, schema):
        """All schema-derived constraints should be present (facility + DD).

        Uses ``graph_with_schema`` fixture to ensure constraints are
        initialized before checking (idempotent DDL).
        """
        dd_schema = GraphSchema(schema_path="imas_codex/schemas/imas_dd.yaml")

        expected_names = set()
        for s in [schema, dd_schema]:
            for label in s.node_labels:
                id_field = s.get_identifier(label)
                if id_field:
                    expected_names.add(f"{label.lower()}_{id_field}")

        constraints = graph_with_schema.query(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
            "RETURN name, type, labelsOrTypes, properties"
        )
        existing_names = {c["name"] for c in constraints}
        missing = expected_names - existing_names
        assert not missing, (
            f"Missing constraints: {missing}. "
            f"Run GraphClient().initialize_schema() to create them."
        )

    def test_composite_constraints_correct(
        self, graph_with_schema, schema, graph_labels
    ):
        """Facility-owned nodes have composite (id, facility_id) constraints."""
        constraints = graph_with_schema.query(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
            "RETURN name, type, labelsOrTypes, properties"
        )
        for constraint in constraints:
            label = (
                constraint["labelsOrTypes"][0] if constraint["labelsOrTypes"] else ""
            )
            # Only check labels that have data in the graph
            if label and label in schema.node_labels and label in graph_labels:
                if schema.needs_composite_constraint(label):
                    props = constraint.get("properties", [])
                    assert "facility_id" in props, (
                        f"{label} should have composite constraint including "
                        f"facility_id, got properties: {props}"
                    )


class TestRequiredFields:
    """Verify required fields are populated on all nodes.

    Lifecycle-aware: fields annotated with ``required_after: <status>``
    are only checked on nodes that have reached that status or a later
    terminal state.  This allows nodes in transient states (e.g.
    ``discovered`` signals awaiting enrichment) to exist without the
    field, while still enforcing the constraint on terminal nodes.

    The status field is auto-detected per label — usually ``status``
    but can be ``enrichment_status`` for classes like SignalNode that
    use a separate lifecycle field.
    """

    # Lifecycle ordering per status enum.  Only statuses that represent
    # "the worker ran" are included; terminal bypasses (skipped, failed,
    # excluded, explored) are omitted so nodes that never reached the
    # required stage are not checked.
    _STATUS_ORDER: dict[str, list[str]] = {
        "FacilitySignalStatus": [
            "discovered",
            "enriched",
            "checked",
        ],
        "PathStatus": [
            "discovered",
            "scanned",
            "triaged",
            "scored",
            "enriched",
        ],
        "SourceFileStatus": [
            "discovered",
            "triaged",
            "scored",
            "enriched",
            "ingested",
        ],
        "EnrichmentStatus": [
            "discovered",
            "enriched",
        ],
    }

    def _terminal_statuses(self, enum_name: str, after: str) -> list[str]:
        """Return statuses at or after *after* in the lifecycle."""
        order = self._STATUS_ORDER.get(enum_name)
        if not order or after not in order:
            return []
        idx = order.index(after)
        return order[idx:]

    @staticmethod
    def _find_status_field(
        slots: dict[str, dict],
    ) -> tuple[str | None, str | None]:
        """Find the status field name and its enum type for a node label.

        Checks ``status`` first, then falls back to any slot whose type
        ends with ``Status`` (e.g. ``enrichment_status``).

        Returns:
            (field_name, enum_name) or (None, None) if no status field.
        """
        # Prefer the canonical 'status' slot
        status_slot = slots.get("status")
        if status_slot and status_slot.get("type", "").endswith("Status"):
            return "status", status_slot["type"]
        # Fallback: any slot whose range is a *Status enum
        for name, info in slots.items():
            if name != "status" and info.get("type", "").endswith("Status"):
                return name, info["type"]
        return None, None

    def test_required_fields_present(self, graph_client, schema, graph_labels):
        """Required fields must not be null on any node (lifecycle-aware)."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            required = schema.get_required_fields(label)
            if not required:
                continue

            slots = schema.get_all_slots(label)
            status_field, status_enum = self._find_status_field(slots)

            for field_name in required:
                slot_info = slots.get(field_name, {})
                # Skip relationship fields (stored as edges, not properties)
                if slot_info.get("relationship"):
                    continue

                # Lifecycle scoping: only check nodes past the required_after status
                required_after = slot_info.get("required_after")
                status_filter = ""
                terminal: list[str] = []
                if required_after and status_enum:
                    terminal = self._terminal_statuses(status_enum, required_after)
                    if terminal:
                        quoted = ", ".join(f"'{s}'" for s in terminal)
                        status_filter = f" AND n.{status_field} IN [{quoted}]"

                result = graph_client.query(
                    f"MATCH (n:{label}) WHERE n.{field_name} IS NULL"
                    f"{status_filter} "
                    f"RETURN count(n) AS cnt"
                )
                count = result[0]["cnt"] if result else 0
                if count > 0:
                    scope = f" ({status_field} IN {terminal})" if status_filter else ""
                    violations.append(f"{label}.{field_name}: {count} null{scope}")

        assert not violations, "Nodes with null required fields:\n  " + "\n  ".join(
            violations
        )


class TestIdentifiers:
    """Verify identifier field integrity."""

    def test_identifiers_non_null(self, graph_client, schema, graph_labels):
        """Identifier fields must never be null."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            id_field = schema.get_identifier(label)
            if not id_field:
                continue

            result = graph_client.query(
                f"MATCH (n:{label}) WHERE n.{id_field} IS NULL RETURN count(n) AS cnt"
            )
            count = result[0]["cnt"] if result else 0
            if count > 0:
                violations.append(f"{label}.{id_field}: {count} null")

        assert not violations, "Nodes with null identifiers:\n  " + "\n  ".join(
            violations
        )

    def test_identifier_uniqueness(self, graph_client, schema, graph_labels):
        """Identifiers must be unique per label (or per label+facility_id)."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            id_field = schema.get_identifier(label)
            if not id_field:
                continue

            if schema.needs_composite_constraint(label):
                # Composite: (id, facility_id) must be unique
                result = graph_client.query(
                    f"MATCH (n:{label}) "
                    f"WITH n.{id_field} AS id, n.facility_id AS fid, count(*) AS cnt "
                    f"WHERE cnt > 1 "
                    f"RETURN id, fid, cnt LIMIT 5"
                )
            else:
                result = graph_client.query(
                    f"MATCH (n:{label}) "
                    f"WITH n.{id_field} AS id, count(*) AS cnt "
                    f"WHERE cnt > 1 "
                    f"RETURN id, cnt LIMIT 5"
                )

            if result:
                dupes = ", ".join(f"{r['id']} (x{r['cnt']})" for r in result)
                violations.append(f"{label}: {dupes}")

        assert not violations, "Duplicate identifiers found:\n  " + "\n  ".join(
            violations
        )


class TestEnumValues:
    """Verify enum-typed fields contain only valid values."""

    def test_enum_values_valid(self, graph_client, schema, graph_labels):
        """Fields with enum types must only contain schema-defined values."""
        enums = schema.get_enums()
        enum_names = set(enums.keys())
        violations = []

        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            slots = schema.get_all_slots(label)
            for slot_name, slot_info in slots.items():
                slot_type = slot_info.get("type", "")
                if slot_type not in enum_names:
                    continue

                valid_values = set(enums[slot_type])
                result = graph_client.query(
                    f"MATCH (n:{label}) WHERE n.{slot_name} IS NOT NULL "
                    f"RETURN DISTINCT n.{slot_name} AS val"
                )
                actual_values = {r["val"] for r in result}
                invalid = actual_values - valid_values
                if invalid:
                    violations.append(
                        f"{label}.{slot_name} ({slot_type}): invalid values {invalid}"
                    )

        assert not violations, "Invalid enum values:\n  " + "\n  ".join(violations)
