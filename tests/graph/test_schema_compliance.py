"""Schema compliance tests.

Verifies that the graph structure matches the LinkML schema definitions:
- All labels and relationship types are schema-defined
- Constraints and indexes exist
- Required fields are populated
- Enum values are valid
- Identifier uniqueness holds
"""

import pytest

from imas_codex.graph.client import CODE_CREATED_RELATIONSHIPS

pytestmark = pytest.mark.graph


class TestLabelsAndRelationships:
    """Verify graph labels and relationship types match the schema."""

    # Internal labels that are not in the LinkML schema
    INTERNAL_LABELS = {"_GraphMeta"}

    def test_all_labels_in_schema(self, graph_labels, schema):
        """Every label in the graph must be defined in the schema."""
        from imas_codex.graph.schema import GraphSchema

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
        expected = set(schema.relationship_types)
        # DD schema relationships are SCREAMING_SNAKE too
        from imas_codex.graph.schema import GraphSchema

        dd_schema = GraphSchema(schema_path="imas_codex/schemas/imas_dd.yaml")
        expected |= set(dd_schema.relationship_types)

        # Include code-created relationships documented in client.py
        expected |= set(CODE_CREATED_RELATIONSHIPS.keys())

        unexpected = graph_relationship_types - expected
        assert not unexpected, (
            f"Graph contains relationship types not in schema: {unexpected}"
        )


class TestConstraints:
    """Verify expected constraints exist in the graph."""

    def test_constraints_created(self, graph_constraints, schema):
        """All schema-derived constraints should be present."""
        expected_names = set()
        for label in schema.node_labels:
            id_field = schema.get_identifier(label)
            if id_field:
                expected_names.add(f"{label.lower()}_{id_field}")

        existing_names = {c["name"] for c in graph_constraints}
        missing = expected_names - existing_names
        assert not missing, (
            f"Missing constraints: {missing}. "
            f"Run GraphClient().initialize_schema() to create them."
        )

    def test_composite_constraints_correct(self, graph_constraints, schema):
        """Facility-owned nodes have composite (id, facility_id) constraints."""
        for constraint in graph_constraints:
            label = (
                constraint["labelsOrTypes"][0] if constraint["labelsOrTypes"] else ""
            )
            if label and label in schema.node_labels:
                if schema.needs_composite_constraint(label):
                    props = constraint.get("properties", [])
                    assert "facility_id" in props, (
                        f"{label} should have composite constraint including "
                        f"facility_id, got properties: {props}"
                    )


class TestRequiredFields:
    """Verify required fields are populated on all nodes."""

    def test_required_fields_present(self, graph_client, schema, graph_labels):
        """Required fields must not be null on any node."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            required = schema.get_required_fields(label)
            if not required:
                continue

            for field_name in required:
                # Skip relationship fields (stored as edges, not properties)
                slot_info = schema.get_all_slots(label).get(field_name, {})
                if slot_info.get("relationship"):
                    continue

                result = graph_client.query(
                    f"MATCH (n:{label}) WHERE n.{field_name} IS NULL "
                    f"RETURN count(n) AS cnt"
                )
                count = result[0]["cnt"] if result else 0
                if count > 0:
                    violations.append(f"{label}.{field_name}: {count} null")

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
