"""
Test suite for list_imas_paths tool functionality.

This test suite validates that the list_imas_paths tool works correctly,
covering all output formats and filtering options.
"""

import json

import pytest

from imas_codex.models.result_models import ListPathsResult, ListPathsResultItem
from imas_codex.tools.list_tool import ListTool


class TestListToolInternals:
    """Tests for internal methods of ListTool."""

    @pytest.fixture
    def list_tool(self):
        """Create list tool instance."""
        return ListTool()

    def test_tool_name(self, list_tool):
        """Tool returns correct name."""
        assert list_tool.tool_name == "list_tool"

    def test_build_json_tree_empty(self, list_tool):
        """Empty paths return empty JSON object."""
        result = list_tool._build_json_tree([])
        assert result == "{}"

    def test_build_json_tree_single_path(self, list_tool):
        """Single path creates correct tree structure."""
        paths = ["equilibrium/time_slice/boundary/psi"]
        result = list_tool._build_json_tree(paths)
        parsed = json.loads(result)

        assert "equilibrium" in parsed
        assert "time_slice" in parsed["equilibrium"]
        assert "boundary" in parsed["equilibrium"]["time_slice"]
        assert "psi" in parsed["equilibrium"]["time_slice"]["boundary"]

    def test_build_json_tree_multiple_paths(self, list_tool):
        """Multiple paths create correct tree structure."""
        paths = [
            "equilibrium/time_slice/boundary/psi",
            "equilibrium/time_slice/boundary/psi_norm",
            "equilibrium/time_slice/profiles_1d/pressure",
        ]
        result = list_tool._build_json_tree(paths)
        parsed = json.loads(result)

        # Check shared structure
        boundary = parsed["equilibrium"]["time_slice"]["boundary"]
        assert "psi" in boundary
        assert "psi_norm" in boundary

        # Check separate branch
        profiles = parsed["equilibrium"]["time_slice"]["profiles_1d"]
        assert "pressure" in profiles

    def test_build_dict_tree_empty(self, list_tool):
        """Empty paths return empty dict."""
        result = list_tool._build_dict_tree([])
        assert result == {}

    def test_build_dict_tree_single_path(self, list_tool):
        """Single path creates correct dict structure."""
        paths = ["core_profiles/profiles_1d/electrons/temperature"]
        result = list_tool._build_dict_tree(paths)

        assert "core_profiles" in result
        assert "profiles_1d" in result["core_profiles"]
        assert "electrons" in result["core_profiles"]["profiles_1d"]
        assert "temperature" in result["core_profiles"]["profiles_1d"]["electrons"]

    def test_build_dict_tree_overlapping_paths(self, list_tool):
        """Overlapping paths share common structure."""
        paths = [
            "a/b/c",
            "a/b/d",
            "a/e/f",
        ]
        result = list_tool._build_dict_tree(paths)

        assert "a" in result
        assert "b" in result["a"]
        assert "e" in result["a"]
        assert "c" in result["a"]["b"]
        assert "d" in result["a"]["b"]
        assert "f" in result["a"]["e"]

    def test_build_yaml_tree_empty(self, list_tool):
        """Empty paths return empty string."""
        result = list_tool._build_yaml_tree([])
        assert result == ""

    def test_build_yaml_tree_single_path(self, list_tool):
        """Single path creates correct YAML indentation."""
        paths = ["a/b/c"]
        result = list_tool._build_yaml_tree(paths)

        lines = result.split("\n")
        assert lines[0] == "a"
        assert lines[1] == "  b"
        assert lines[2] == "    c"

    def test_build_yaml_tree_sorted(self, list_tool):
        """YAML tree items are sorted alphabetically."""
        paths = ["z/a", "a/b", "m/c"]
        result = list_tool._build_yaml_tree(paths)

        lines = result.split("\n")
        top_level = [line for line in lines if not line.startswith("  ")]
        assert top_level == ["a", "m", "z"]

    def test_build_yaml_tree_leaf_only(self, list_tool):
        """Leaf only mode affects structure."""
        paths = ["a/b/c", "a/b/d"]
        result_full = list_tool._build_yaml_tree(paths, show_leaf_only=False)
        result_leaf = list_tool._build_yaml_tree(paths, show_leaf_only=True)

        # Both should produce output
        assert len(result_full) > 0
        assert len(result_leaf) > 0

    def test_build_flat_list_empty(self, list_tool):
        """Empty paths return empty list."""
        result = list_tool._build_flat_list([])
        assert result == []

    def test_build_flat_list_sorted(self, list_tool):
        """Flat list is sorted."""
        paths = ["z/path", "a/path", "m/path"]
        result = list_tool._build_flat_list(paths)
        assert result == ["a/path", "m/path", "z/path"]

    def test_filter_paths_by_prefix_none(self, list_tool):
        """None prefix returns all paths."""
        paths = ["a/b", "c/d"]
        result = list_tool._filter_paths_by_prefix(paths, None)
        assert result == paths

    def test_filter_paths_by_prefix_exact_match(self, list_tool):
        """Exact prefix match filters correctly."""
        paths = ["a/b/c", "a/b/d", "x/y/z"]
        result = list_tool._filter_paths_by_prefix(paths, "a/b")

        assert "a/b/c" in result
        assert "a/b/d" in result
        assert "x/y/z" not in result

    def test_filter_paths_by_prefix_strips_slashes(self, list_tool):
        """Prefix with trailing/leading slashes is handled."""
        paths = ["a/b/c", "a/b/d"]
        result = list_tool._filter_paths_by_prefix(paths, "/a/b/")

        assert len(result) == 2

    def test_filter_paths_by_prefix_exact_path(self, list_tool):
        """Exact path match is included."""
        paths = ["a/b", "a/b/c"]
        result = list_tool._filter_paths_by_prefix(paths, "a/b")

        assert "a/b" in result
        assert "a/b/c" in result


class TestListToolExtractPaths:
    """Tests for path extraction from documents."""

    @pytest.fixture
    def list_tool(self):
        """Create list tool instance."""
        return ListTool()

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        from imas_codex.search.document_store import Document, DocumentMetadata

        docs = []
        paths = [
            "equilibrium/time_slice/boundary/psi",
            "equilibrium/time_slice/boundary/psi_norm",
            "equilibrium/time_slice/profiles_1d/pressure",
        ]
        for path in paths:
            metadata = DocumentMetadata(
                path_id=path,
                ids_name="equilibrium",
                path_name=path,
                units="m",
                data_type="float",
                coordinates=(),
                physics_domain="mhd",
                physics_phenomena=(),
            )
            docs.append(
                Document(
                    metadata=metadata,
                    documentation=f"Doc for {path}",
                    relationships={},
                    raw_data={},
                )
            )
        return docs

    def test_extract_paths_basic(self, list_tool, mock_documents):
        """Extract paths from documents."""
        paths = list_tool._extract_paths_from_documents(
            mock_documents,
            ids_name="equilibrium",
            include_ids_prefix=True,
            leaf_only=False,
        )

        assert len(paths) == 3
        assert all(p.startswith("equilibrium/") for p in paths)

    def test_extract_paths_without_ids_prefix(self, list_tool, mock_documents):
        """Extract paths without IDS prefix."""
        paths = list_tool._extract_paths_from_documents(
            mock_documents,
            ids_name="equilibrium",
            include_ids_prefix=False,
            leaf_only=False,
        )

        assert len(paths) == 3
        assert all(not p.startswith("equilibrium/") for p in paths)
        assert "time_slice/boundary/psi" in paths

    def test_extract_paths_leaf_only(self, list_tool, mock_documents):
        """Extract only leaf paths."""
        paths = list_tool._extract_paths_from_documents(
            mock_documents,
            ids_name="equilibrium",
            include_ids_prefix=True,
            leaf_only=True,
        )

        # All paths in mock are leaves (no path is prefix of another)
        assert len(paths) == 3

    def test_extract_paths_sorted(self, list_tool, mock_documents):
        """Extracted paths are sorted."""
        paths = list_tool._extract_paths_from_documents(
            mock_documents,
            ids_name="equilibrium",
            include_ids_prefix=True,
            leaf_only=False,
        )

        assert paths == sorted(paths)


class TestListToolMCPIntegration:
    """Test list tool MCP integration."""

    @pytest.fixture
    async def list_tool(self):
        """Create list tool instance."""
        return ListTool()

    @pytest.mark.asyncio
    async def test_list_imas_paths_yaml_format(self, list_tool):
        """List paths in YAML format."""
        result = await list_tool.list_imas_paths("equilibrium", format="yaml")

        assert isinstance(result, ListPathsResult)
        assert result.format == "yaml"
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_list_imas_paths_list_format(self, list_tool):
        """List paths in list format."""
        result = await list_tool.list_imas_paths("equilibrium", format="list")

        assert isinstance(result, ListPathsResult)
        assert result.format == "list"
        assert len(result.results) > 0

        # First result should have list type paths
        first_result = result.results[0]
        if first_result.paths is not None:
            assert isinstance(first_result.paths, list)

    @pytest.mark.asyncio
    async def test_list_imas_paths_json_format(self, list_tool):
        """List paths in JSON format."""
        result = await list_tool.list_imas_paths("equilibrium", format="json")

        assert isinstance(result, ListPathsResult)
        assert result.format == "json"

    @pytest.mark.asyncio
    async def test_list_imas_paths_dict_format(self, list_tool):
        """List paths in dict format."""
        result = await list_tool.list_imas_paths("equilibrium", format="dict")

        assert isinstance(result, ListPathsResult)
        assert result.format == "dict"

        # First result should have dict type paths
        first_result = result.results[0]
        if first_result.paths is not None:
            assert isinstance(first_result.paths, dict)

    @pytest.mark.asyncio
    async def test_list_imas_paths_multiple_ids(self, list_tool):
        """List paths for multiple IDS."""
        result = await list_tool.list_imas_paths("equilibrium core_profiles")

        assert isinstance(result, ListPathsResult)
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_list_imas_paths_with_path_prefix(self, list_tool):
        """List paths with path prefix filter."""
        result = await list_tool.list_imas_paths(
            "equilibrium/time_slice", format="list"
        )

        assert isinstance(result, ListPathsResult)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_list_imas_paths_leaf_only(self, list_tool):
        """List only leaf paths."""
        result = await list_tool.list_imas_paths(
            "equilibrium", format="list", leaf_only=True
        )

        assert isinstance(result, ListPathsResult)

    @pytest.mark.asyncio
    async def test_list_imas_paths_without_ids_prefix(self, list_tool):
        """List paths without IDS prefix."""
        result = await list_tool.list_imas_paths(
            "equilibrium", format="list", include_ids_prefix=False
        )

        assert isinstance(result, ListPathsResult)

    @pytest.mark.asyncio
    async def test_list_imas_paths_with_max_paths(self, list_tool):
        """List paths with max_paths limit."""
        result = await list_tool.list_imas_paths(
            "equilibrium", format="list", max_paths=10
        )

        assert isinstance(result, ListPathsResult)
        first_result = result.results[0]
        if first_result.paths is not None and isinstance(first_result.paths, list):
            assert len(first_result.paths) <= 10

    @pytest.mark.asyncio
    async def test_list_imas_paths_invalid_ids(self, list_tool):
        """Invalid IDS returns error in result."""
        result = await list_tool.list_imas_paths("nonexistent_ids_xyz")

        assert isinstance(result, ListPathsResult)
        assert len(result.results) == 1
        assert result.results[0].error is not None

    @pytest.mark.asyncio
    async def test_list_imas_paths_summary(self, list_tool):
        """Summary contains expected fields."""
        result = await list_tool.list_imas_paths("equilibrium")

        assert isinstance(result, ListPathsResult)
        assert "total_queries" in result.summary
        assert "successful_queries" in result.summary
        assert "total_paths" in result.summary
        assert "format" in result.summary

    @pytest.mark.asyncio
    async def test_list_imas_paths_empty_query(self, list_tool):
        """Empty query handled gracefully."""
        from imas_codex.models.error_models import ToolError

        result = await list_tool.list_imas_paths("   ")

        # Should return error for empty query
        assert isinstance(result, ToolError | ListPathsResult)


class TestListPathsResultItem:
    """Test ListPathsResultItem model."""

    def test_result_item_with_paths(self):
        """Result item with paths."""
        item = ListPathsResultItem(
            query="equilibrium",
            path_count=100,
            paths=["a/b", "c/d"],
        )

        assert item.query == "equilibrium"
        assert item.path_count == 100
        assert item.paths == ["a/b", "c/d"]
        assert item.error is None
        assert item.truncated_to is None

    def test_result_item_with_error(self):
        """Result item with error."""
        item = ListPathsResultItem(
            query="invalid",
            path_count=0,
            error="IDS not found",
        )

        assert item.error == "IDS not found"
        assert item.path_count == 0

    def test_result_item_truncated(self):
        """Result item that was truncated."""
        item = ListPathsResultItem(
            query="equilibrium",
            path_count=1000,
            truncated_to=100,
            paths=["path"] * 100,
        )

        assert item.truncated_to == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
