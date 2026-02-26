"""Tests for the DD graph schema tool."""

import pytest

from imas_codex.tools.schema_tool import SchemaTool, _build_schema_summary

pytestmark = pytest.mark.graph_mcp


class TestSchemaToolStatic:
    """Test schema tool without requiring Neo4j (static schema introspection)."""

    def test_schema_summary_has_node_types(self):
        summary = _build_schema_summary()
        assert "node_types" in summary
        # Must include core DD node types
        for expected in ["DDVersion", "IDS", "IMASPath", "Unit", "IMASSemanticCluster"]:
            assert expected in summary["node_types"], f"Missing {expected}"

    def test_schema_summary_has_relationships(self):
        summary = _build_schema_summary()
        assert "relationships" in summary
        rel_types = {r["type"] for r in summary["relationships"]}
        assert "PREDECESSOR" in rel_types
        assert "IN_IDS" in rel_types
        assert "INTRODUCED_IN" in rel_types

    def test_schema_summary_has_enums(self):
        summary = _build_schema_summary()
        assert "enums" in summary
        assert "DDDataType" in summary["enums"]
        assert "DDNodeType" in summary["enums"]

    def test_schema_summary_has_vector_indexes(self):
        summary = _build_schema_summary()
        assert "vector_indexes" in summary
        index_names = {idx["name"] for idx in summary["vector_indexes"]}
        assert "imas_path_embedding" in index_names

    def test_schema_summary_has_notes(self):
        summary = _build_schema_summary()
        assert "notes" in summary
        assert "version_lifecycle" in summary["notes"]
        assert "version_evolution" in summary["notes"]
        assert "clusters" in summary["notes"]

    def test_node_type_properties_exclude_relationships(self):
        summary = _build_schema_summary()
        # IMASPath has an 'introduced_in' slot that's a relationship (DDVersion range)
        # It should NOT appear in properties
        imas_path_props = summary["node_types"]["IMASPath"]["properties"]
        for prop_info in imas_path_props.values():
            assert "relationship" not in prop_info

    def test_node_type_has_description(self):
        summary = _build_schema_summary()
        # DDVersion should have a description from the schema
        assert summary["node_types"]["DDVersion"]["description"]

    def test_relationship_directionality(self):
        summary = _build_schema_summary()
        # Find PREDECESSOR relationship
        pred = [r for r in summary["relationships"] if r["type"] == "PREDECESSOR"]
        assert len(pred) >= 1
        assert pred[0]["from"] == "DDVersion"
        assert pred[0]["to"] == "DDVersion"

    def test_enum_values_populated(self):
        summary = _build_schema_summary()
        dd_data_types = summary["enums"]["DDDataType"]
        assert "FLT_1D" in dd_data_types
        assert "INT_0D" in dd_data_types
        assert "STRUCTURE" in dd_data_types


class TestSchemaToolMCP:
    """Test the SchemaTool MCP method."""

    @pytest.mark.anyio
    async def test_get_dd_graph_schema_returns_dict(self):
        tool = SchemaTool()
        result = await tool.get_dd_graph_schema()
        assert isinstance(result, dict)
        assert "node_types" in result
        assert "relationships" in result

    @pytest.mark.anyio
    async def test_schema_is_cached(self):
        """Calling schema tool multiple times should return same object."""
        tool = SchemaTool()
        result1 = await tool.get_dd_graph_schema()
        result2 = await tool.get_dd_graph_schema()
        assert result1 is result2  # Same cached dict
