"""Tests for the DD version metadata tool."""

import pytest

from imas_codex.tools.version_tool import VersionTool

pytestmark = pytest.mark.graph_mcp


class TestVersionToolExecution:
    """Test VersionTool against the fixture graph."""

    @pytest.fixture
    def version_tool(self, graph_client):
        return VersionTool(graph_client)

    @pytest.mark.anyio
    async def test_version_count(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert result["version_count"] == 3

    @pytest.mark.anyio
    async def test_current_version(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert result["current_version"] == "4.1.0"

    @pytest.mark.anyio
    async def test_version_range(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert result["version_range"] == "3.42.0 - 4.1.0"

    @pytest.mark.anyio
    async def test_versions_list(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert "3.42.0" in result["versions"]
        assert "4.0.0" in result["versions"]
        assert "4.1.0" in result["versions"]

    @pytest.mark.anyio
    async def test_versions_ordered(self, version_tool):
        result = await version_tool.get_dd_versions()
        versions = result["versions"]
        # Should be sorted by major, minor, patch
        assert versions.index("3.42.0") < versions.index("4.0.0")
        assert versions.index("4.0.0") < versions.index("4.1.0")
