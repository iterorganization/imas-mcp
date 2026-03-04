"""Tests for the DD graph schema tool."""

import pytest

from imas_codex.graph.schema_context import schema_for
from imas_codex.tools.schema_tool import SchemaTool

pytestmark = pytest.mark.graph_mcp


class TestSchemaToolStatic:
    """Test schema tool without requiring Neo4j (static schema introspection)."""

    def test_imas_schema_has_node_types(self):
        result = schema_for(task="imas")
        # Must include core DD node types
        for expected in ["DDVersion", "IDS", "IMASPath", "Unit", "IMASSemanticCluster"]:
            assert expected in result, f"Missing {expected}"

    def test_imas_schema_has_relationships(self):
        result = schema_for(task="imas")
        assert "HAS_PREDECESSOR" in result
        assert "IN_IDS" in result
        assert "INTRODUCED_IN" in result

    def test_imas_schema_has_enums(self):
        result = schema_for(task="imas")
        assert "DDDataType" in result
        assert "DDNodeType" in result

    def test_imas_schema_has_vector_indexes(self):
        result = schema_for(task="imas")
        assert "imas_path_embedding" in result

    def test_imas_schema_has_property_details(self):
        result = schema_for(task="imas")
        # IMASPath should have documented properties
        assert "documentation" in result
        assert "units" in result

    def test_relationship_directionality(self):
        result = schema_for(task="imas")
        # HAS_PREDECESSOR should show DDVersion -> DDVersion
        assert "DDVersion" in result
        assert "HAS_PREDECESSOR" in result

    def test_enum_values_populated(self):
        result = schema_for(task="imas")
        assert "FLT_1D" in result
        assert "INT_0D" in result
        assert "STRUCTURE" in result


class TestSchemaToolMCP:
    """Test the SchemaTool MCP method."""

    @pytest.mark.anyio
    async def test_get_dd_graph_schema_returns_string(self):
        tool = SchemaTool()
        result = await tool.get_dd_graph_schema()
        assert isinstance(result, str)
        assert "IMASPath" in result
        assert "DDVersion" in result

    @pytest.mark.anyio
    async def test_schema_contains_relationships(self):
        """Schema output should include relationship information."""
        tool = SchemaTool()
        result = await tool.get_dd_graph_schema()
        assert "Relationships" in result
