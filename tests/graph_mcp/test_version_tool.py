"""Tests for the DD version metadata tool."""

import pytest

from imas_codex.tools.version_tool import VersionTool

pytestmark = pytest.mark.graph_mcp


class TestVersionToolExecution:
    """Test VersionTool against the live graph."""

    @pytest.fixture
    def version_tool(self, graph_client):
        return VersionTool(graph_client)

    @pytest.mark.anyio
    async def test_version_count(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert result["version_count"] >= 3

    @pytest.mark.anyio
    async def test_current_version(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert result["current_version"] is not None
        # Current version should be a semver string
        parts = result["current_version"].split(".")
        assert len(parts) == 3

    @pytest.mark.anyio
    async def test_version_range(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert " - " in result["version_range"]
        low, high = result["version_range"].split(" - ")
        assert low.count(".") == 2
        assert high.count(".") == 2

    @pytest.mark.anyio
    async def test_versions_list(self, version_tool):
        result = await version_tool.get_dd_versions()
        assert len(result["versions"]) >= 3
        # Every version should be semver
        for v in result["versions"]:
            assert v.count(".") == 2

    @pytest.mark.anyio
    async def test_versions_ordered(self, version_tool):
        result = await version_tool.get_dd_versions()
        versions = result["versions"]
        # Check ordering: earlier versions come first
        major_versions = [int(v.split(".")[0]) for v in versions]
        assert major_versions == sorted(major_versions)
