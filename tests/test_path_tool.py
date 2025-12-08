"""Tests for the path tool (check_imas_paths and fetch_imas_paths)."""

import pytest

from imas_mcp.core.exclusions import EXCLUSION_REASONS, ExclusionChecker
from imas_mcp.mappings import PathMap, PathMapping, RenameHistoryEntry
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.tools import PathTool

# ============================================================================
# Mock PathMap for testing
# ============================================================================


class MockPathMap(PathMap):
    """Mock PathMap with predefined test data."""

    def __init__(self):
        # Initialize with test data, bypassing file loading
        mapping_data = {
            "metadata": {
                "target_version": "4.0.1",
                "total_mappings": 3,
            },
            "exclusion_reasons": EXCLUSION_REASONS,
            "excluded_paths": {
                "equilibrium/time_slice/profiles_1d/psi_error_lower": "error_field",
                "edge_profiles/ggd/grid/space/objects_per_dimension": "ggd",
            },
            "old_to_new": {
                "equilibrium/time_slice/constraints/bpol_probe": {
                    "new_path": "equilibrium/time_slice/constraints/b_field_pol_probe",
                    "deprecated_in": "4.0.0",
                    "last_valid_version": "3.42.0",
                },
                "equilibrium/time_slice/global_quantities/li": {
                    "new_path": "equilibrium/time_slice/global_quantities/li_3",
                    "deprecated_in": "4.0.0",
                    "last_valid_version": "3.41.0",
                },
                # Deprecated path whose new_path is excluded (error field)
                "core_profiles/profiles_1d/j_tor_error_upper_old": {
                    "new_path": "core_profiles/profiles_1d/j_tor_error_upper",
                    "deprecated_in": "4.0.0",
                    "last_valid_version": "3.42.0",
                },
            },
            "new_to_old": {
                "equilibrium/time_slice/constraints/b_field_pol_probe": [
                    {
                        "old_path": "equilibrium/time_slice/constraints/bpol_probe",
                        "deprecated_in": "4.0.0",
                    }
                ],
                "equilibrium/time_slice/global_quantities/li_3": [
                    {
                        "old_path": "equilibrium/time_slice/global_quantities/li",
                        "deprecated_in": "4.0.0",
                    }
                ],
                "core_profiles/profiles_1d/j_tor_error_upper": [
                    {
                        "old_path": "core_profiles/profiles_1d/j_tor_error_upper_old",
                        "deprecated_in": "4.0.0",
                    }
                ],
            },
        }
        super().__init__(dd_version="4.0.1", mapping_data=mapping_data)


@pytest.fixture
def mock_path_map() -> MockPathMap:
    """Create a mock PathMap for testing."""
    return MockPathMap()


@pytest.fixture
def path_tool(mock_path_map: MockPathMap) -> PathTool:
    """Create a PathTool instance for testing with mocked path map."""
    doc_store = DocumentStore()
    return PathTool(doc_store, path_map=mock_path_map)


# ============================================================================
# Tests for PathMap
# ============================================================================


def test_path_map_get_mapping(mock_path_map: MockPathMap):
    """Test getting mapping info for an old path."""
    mapping = mock_path_map.get_mapping("equilibrium/time_slice/constraints/bpol_probe")

    assert mapping is not None
    assert mapping.new_path == "equilibrium/time_slice/constraints/b_field_pol_probe"
    assert mapping.deprecated_in == "4.0.0"
    assert mapping.last_valid_version == "3.42.0"


def test_path_map_get_mapping_not_found(
    mock_path_map: MockPathMap,
):
    """Test getting mapping info for a path with no mapping."""
    mapping = mock_path_map.get_mapping("fake/path/here")
    assert mapping is None


def test_path_map_get_rename_history(mock_path_map: MockPathMap):
    """Test getting rename history for a current path."""
    history = mock_path_map.get_rename_history(
        "equilibrium/time_slice/constraints/b_field_pol_probe"
    )

    assert len(history) == 1
    assert history[0].old_path == "equilibrium/time_slice/constraints/bpol_probe"
    assert history[0].deprecated_in == "4.0.0"


def test_path_map_get_rename_history_not_found(
    mock_path_map: MockPathMap,
):
    """Test getting rename history for a path with no history."""
    history = mock_path_map.get_rename_history("fake/path/here")
    assert history == []


def test_path_map_metadata(mock_path_map: MockPathMap):
    """Test path map metadata access."""
    assert mock_path_map.target_version == "4.0.1"
    assert mock_path_map.total_mappings == 3


# ============================================================================
# Tests for check_imas_paths - Basic validation
# ============================================================================


@pytest.mark.asyncio
async def test_single_valid_path(path_tool):
    """Test validation of a single existing path."""
    result = await path_tool.check_imas_paths(
        "core_profiles/profiles_1d/electrons/temperature"
    )

    # Now returns CheckPathsResult Pydantic model
    assert result.summary["total"] == 1
    assert result.summary["found"] == 1
    assert result.summary["not_found"] == 0
    assert len(result.results) == 1
    assert result.results[0].exists is True
    assert result.results[0].path == "core_profiles/profiles_1d/electrons/temperature"
    assert result.results[0].ids_name == "core_profiles"


@pytest.mark.asyncio
async def test_single_invalid_path(path_tool):
    """Test validation of a non-existent path."""
    result = await path_tool.check_imas_paths("fake/nonexistent/path")

    assert result.summary["total"] == 1
    assert result.summary["found"] == 0
    assert result.summary["not_found"] == 1
    assert len(result.results) == 1
    assert result.results[0].exists is False
    assert result.results[0].path == "fake/nonexistent/path"


@pytest.mark.asyncio
async def test_multiple_paths_space_delimited(path_tool):
    """Test validation of multiple paths as space-delimited string."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature fake/path"
    )

    assert result.summary["total"] == 3
    assert result.summary["found"] == 2
    assert result.summary["not_found"] == 1
    assert len(result.results) == 3

    # Check first path
    assert result.results[0].exists is True
    assert result.results[0].ids_name == "equilibrium"

    # Check second path
    assert result.results[1].exists is True
    assert result.results[1].ids_name == "core_profiles"

    # Check third path
    assert result.results[2].exists is False
    assert result.results[2].path == "fake/path"


@pytest.mark.asyncio
async def test_multiple_paths_list(path_tool):
    """Test validation of multiple paths as a list."""
    result = await path_tool.check_imas_paths(
        [
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        ]
    )

    assert result.summary["total"] == 2
    assert result.summary["found"] == 2
    assert result.summary["not_found"] == 0
    assert len(result.results) == 2


@pytest.mark.asyncio
async def test_malformed_path_no_slash(path_tool):
    """Test validation with malformed path (no slash)."""
    result = await path_tool.check_imas_paths("notapath")

    assert result.summary["total"] == 1
    assert result.summary["invalid"] == 1
    assert result.results[0].exists is False
    assert result.results[0].error is not None
    assert "Invalid format" in result.results[0].error


@pytest.mark.asyncio
async def test_mixed_valid_invalid_paths(path_tool):
    """Test validation with mix of valid, invalid, and malformed paths."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi invalid/path notapath core_profiles/profiles_1d/electrons/temperature"
    )

    assert result.summary["total"] == 4
    assert result.summary["found"] == 2
    assert result.summary["not_found"] == 1
    assert result.summary["invalid"] == 1


@pytest.mark.asyncio
async def test_returns_structured_response(path_tool):
    """Test that response has proper structure."""
    result = await path_tool.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    # Should be a CheckPathsResult Pydantic model
    from imas_mcp.models.result_models import CheckPathsResult

    assert isinstance(result, CheckPathsResult)

    # Summary should have counts
    assert "total" in result.summary
    assert "found" in result.summary
    assert "not_found" in result.summary
    assert "invalid" in result.summary

    # Results should be a list
    assert isinstance(result.results, list)

    # Each result should have path and exists
    for item in result.results:
        assert item.path is not None
        assert item.exists is not None


@pytest.mark.asyncio
async def test_token_efficient_response(path_tool):
    """Test that response is token-efficient (no verbose fields unless needed)."""
    result = await path_tool.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    # Should not have search-specific fields (check model doesn't have these)
    assert not hasattr(result, "hits")
    assert not hasattr(result, "query_hints")
    assert not hasattr(result, "tool_hints")
    assert not hasattr(result, "physics_context")

    # Results should be minimal
    res = result.results[0]
    assert res.path is not None
    assert res.exists is not None
    assert res.ids_name is not None

    # Should not have documentation (token-heavy) - model doesn't include it
    assert not hasattr(res, "documentation")


# ============================================================================
# Tests for check_imas_paths - IDS prefix handling
# ============================================================================


@pytest.mark.asyncio
async def test_ids_prefix_single_path(path_tool):
    """Test ids parameter with single path."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi", ids="equilibrium"
    )

    assert result.summary["total"] == 1
    assert result.summary["found"] == 1
    assert result.results[0].exists is True
    assert result.results[0].path == "equilibrium/time_slice/boundary/psi"
    assert result.results[0].ids_name == "equilibrium"


@pytest.mark.asyncio
async def test_ids_prefix_multiple_paths(path_tool):
    """Test ids parameter with multiple paths (ensemble checking)."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi time_slice/boundary/psi_norm time_slice/boundary/type",
        ids="equilibrium",
    )

    assert result.summary["total"] == 3
    assert result.summary["found"] == 3
    assert result.summary["not_found"] == 0

    # All paths should be prefixed with equilibrium
    for res in result.results:
        assert res.path.startswith("equilibrium/")
        assert res.exists is True
        assert res.ids_name == "equilibrium"


@pytest.mark.asyncio
async def test_ids_prefix_with_list(path_tool):
    """Test ids parameter with list of paths."""
    result = await path_tool.check_imas_paths(
        ["time_slice/boundary/psi", "time_slice/boundary/psi_norm"], ids="equilibrium"
    )

    assert result.summary["total"] == 2
    assert result.summary["found"] == 2
    assert result.results[0].path == "equilibrium/time_slice/boundary/psi"
    assert result.results[1].path == "equilibrium/time_slice/boundary/psi_norm"


@pytest.mark.asyncio
async def test_ids_prefix_already_present(path_tool):
    """Test that ids prefix is not duplicated if already present."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/boundary/psi", ids="equilibrium"
    )

    assert result.summary["total"] == 1
    assert result.summary["found"] == 1
    # Should not be double-prefixed
    assert result.results[0].path == "equilibrium/time_slice/boundary/psi"
    assert result.results[0].path.count("equilibrium/") == 1


@pytest.mark.asyncio
async def test_ids_prefix_mixed_paths(path_tool):
    """Test ids prefix with some paths having full IDS and some not."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi equilibrium/time_slice/boundary/psi_norm",
        ids="equilibrium",
    )

    assert result.summary["total"] == 2
    assert result.summary["found"] == 2

    # Both should have correct paths without duplication
    assert result.results[0].path == "equilibrium/time_slice/boundary/psi"
    assert result.results[1].path == "equilibrium/time_slice/boundary/psi_norm"


# ============================================================================
# Tests for check_imas_paths - Migration suggestions
# ============================================================================


@pytest.mark.asyncio
async def test_deprecated_path_returns_migration(path_tool):
    """Test that deprecated paths return migration suggestions."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/constraints/bpol_probe"
    )

    assert result.summary["total"] == 1
    assert result.summary["not_found"] == 1

    res = result.results[0]
    assert res.exists is False
    assert res.path == "equilibrium/time_slice/constraints/bpol_probe"

    # Should have migration info
    assert res.migration is not None
    assert (
        res.migration["new_path"]
        == "equilibrium/time_slice/constraints/b_field_pol_probe"
    )
    assert res.migration["deprecated_in"] == "4.0.0"
    assert res.migration["last_valid_version"] == "3.42.0"


@pytest.mark.asyncio
async def test_nonexistent_path_no_migration(path_tool):
    """Test that truly invalid paths don't have migration info."""
    result = await path_tool.check_imas_paths("fake/nonexistent/path")

    res = result.results[0]
    assert res.exists is False
    assert res.migration is None


# ============================================================================
# Tests for fetch_imas_paths - Rich data retrieval
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_single_path(path_tool):
    """Test fetching a single path with full documentation."""
    result = await path_tool.fetch_imas_paths(
        "core_profiles/profiles_1d/electrons/temperature"
    )

    # Check result type
    assert hasattr(result, "nodes")
    assert hasattr(result, "summary")
    assert result.node_count == 1

    # Check summary
    assert result.summary["total_requested"] == 1
    assert result.summary["retrieved"] == 1
    assert result.summary["not_found"] == 0

    # Check node content
    node = result.nodes[0]
    assert node.path == "core_profiles/profiles_1d/electrons/temperature"
    assert node.documentation  # Should have documentation
    assert node.data_type  # Should have data_type


@pytest.mark.asyncio
async def test_fetch_multiple_paths_space_delimited(path_tool):
    """Test fetching multiple paths as space-delimited string."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature"
    )

    assert result.node_count == 2
    assert result.summary["total_requested"] == 2
    assert result.summary["retrieved"] == 2

    # Check that both nodes have full data
    for node in result.nodes:
        assert node.path
        assert node.documentation
        assert node.data_type


@pytest.mark.asyncio
async def test_fetch_multiple_paths_list(path_tool):
    """Test fetching multiple paths as a list."""
    result = await path_tool.fetch_imas_paths(
        [
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        ]
    )

    assert result.node_count == 2
    assert result.summary["retrieved"] == 2


@pytest.mark.asyncio
async def test_fetch_with_ids_prefix(path_tool):
    """Test fetching paths with ids parameter."""
    result = await path_tool.fetch_imas_paths(
        "time_slice/boundary/psi time_slice/boundary/psi_norm", ids="equilibrium"
    )

    assert result.node_count == 2
    assert result.summary["retrieved"] == 2

    # Check that paths are correctly prefixed
    assert result.nodes[0].path == "equilibrium/time_slice/boundary/psi"
    assert result.nodes[1].path == "equilibrium/time_slice/boundary/psi_norm"

    # Both should have full documentation
    for node in result.nodes:
        assert node.documentation


@pytest.mark.asyncio
async def test_fetch_nonexistent_path(path_tool):
    """Test fetching a non-existent path."""
    result = await path_tool.fetch_imas_paths("fake/nonexistent/path")

    assert result.node_count == 0
    assert result.summary["total_requested"] == 1
    assert result.summary["retrieved"] == 0
    assert result.summary["not_found"] == 1


@pytest.mark.asyncio
async def test_fetch_mixed_valid_invalid(path_tool):
    """Test fetching mix of valid and invalid paths."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi invalid/path/here notapath"
    )

    # Only valid path should be retrieved
    assert result.node_count == 1
    assert result.summary["total_requested"] == 3
    assert result.summary["retrieved"] == 1
    assert result.summary["not_found"] == 1
    assert result.summary["invalid"] == 1


@pytest.mark.asyncio
async def test_fetch_returns_cluster_labels(path_tool):
    """Test that fetch_imas_paths includes cluster labels."""
    result = await path_tool.fetch_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    assert result.node_count == 1
    node = result.nodes[0]

    # Check for cluster_labels field
    assert hasattr(node, "cluster_labels")
    # cluster_labels may be None or a list of dicts
    if node.cluster_labels:
        assert isinstance(node.cluster_labels, list)

    # Check that physics_domains are tracked in result
    assert hasattr(result, "physics_domains")


@pytest.mark.asyncio
async def test_fetch_aggregates_physics_domains(path_tool):
    """Test that fetch_imas_paths aggregates physics domains across multiple paths."""
    result = await path_tool.fetch_imas_paths(
        [
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        ]
    )

    assert result.node_count == 2
    # Should aggregate physics domains from all retrieved paths
    assert "physics_domains" in result.summary
    assert isinstance(result.summary["physics_domains"], list)


@pytest.mark.asyncio
async def test_fetch_vs_check_difference(path_tool):
    """Test that fetch returns more data than check."""
    path = "equilibrium/time_slice/profiles_1d/psi"

    # Check (minimal) - returns CheckPathsResult
    check_result = await path_tool.check_imas_paths(path)

    # Fetch (rich)
    fetch_result = await path_tool.fetch_imas_paths(path)

    # Check returns CheckPathsResult with minimal info
    from imas_mcp.models.result_models import CheckPathsResult

    assert isinstance(check_result, CheckPathsResult)
    assert check_result.results[0].exists is True
    # CheckPathsResultItem doesn't have documentation field
    assert not hasattr(check_result.results[0], "documentation")

    # Fetch returns FetchPathsResult with full IdsNode objects
    assert hasattr(fetch_result, "nodes")
    assert fetch_result.node_count == 1
    assert fetch_result.nodes[0].documentation  # Has full documentation


@pytest.mark.asyncio
async def test_fetch_has_tool_result_context(path_tool):
    """Test that fetch_imas_paths returns proper ToolResult with context."""
    result = await path_tool.fetch_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    # Should have ToolResult fields
    assert hasattr(result, "tool_name")
    assert result.tool_name == "fetch_imas_paths"
    assert hasattr(result, "processing_timestamp")
    assert hasattr(result, "version")
    assert hasattr(result, "query")


@pytest.mark.asyncio
async def test_fetch_malformed_path(path_tool):
    """Test fetch with malformed path (no slash)."""
    result = await path_tool.fetch_imas_paths("notapath")

    assert result.node_count == 0
    assert result.summary["invalid"] == 1


# ============================================================================
# Tests for fetch_imas_paths - Deprecated path migration info
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_deprecated_path_returns_migration(path_tool):
    """Test that fetching a deprecated path returns migration info."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/constraints/bpol_probe"
    )

    # Should not be in nodes (path doesn't exist)
    assert result.node_count == 0
    assert result.summary["retrieved"] == 0

    # Should be in deprecated_paths
    assert result.summary["deprecated"] == 1
    assert result.summary["not_found"] == 0
    assert len(result.deprecated_paths) == 1

    deprecated = result.deprecated_paths[0]
    assert deprecated.path == "equilibrium/time_slice/constraints/bpol_probe"
    assert deprecated.new_path == "equilibrium/time_slice/constraints/b_field_pol_probe"
    assert deprecated.deprecated_in == "4.0.0"
    assert deprecated.last_valid_version == "3.42.0"


@pytest.mark.asyncio
async def test_fetch_nonexistent_path_no_deprecated_info(path_tool):
    """Test that truly non-existent paths don't appear in deprecated_paths."""
    result = await path_tool.fetch_imas_paths("fake/nonexistent/path")

    assert result.node_count == 0
    assert result.summary["not_found"] == 1
    assert result.summary["deprecated"] == 0
    assert len(result.deprecated_paths) == 0


@pytest.mark.asyncio
async def test_fetch_mixed_valid_deprecated_invalid(path_tool):
    """Test fetching mix of valid, deprecated, and invalid paths."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi equilibrium/time_slice/constraints/bpol_probe fake/path notapath"
    )

    # One valid path retrieved
    assert result.node_count == 1
    assert result.summary["retrieved"] == 1
    assert result.nodes[0].path == "equilibrium/time_slice/profiles_1d/psi"

    # One deprecated path
    assert result.summary["deprecated"] == 1
    assert len(result.deprecated_paths) == 1
    assert (
        result.deprecated_paths[0].path
        == "equilibrium/time_slice/constraints/bpol_probe"
    )

    # One truly not found, one invalid
    assert result.summary["not_found"] == 1
    assert result.summary["invalid"] == 1


@pytest.mark.asyncio
async def test_fetch_multiple_deprecated_paths(path_tool):
    """Test fetching multiple deprecated paths."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/constraints/bpol_probe equilibrium/time_slice/global_quantities/li"
    )

    assert result.node_count == 0
    assert result.summary["deprecated"] == 2
    assert len(result.deprecated_paths) == 2

    # Check both deprecated paths have migration info
    paths = {dp.path for dp in result.deprecated_paths}
    assert "equilibrium/time_slice/constraints/bpol_probe" in paths
    assert "equilibrium/time_slice/global_quantities/li" in paths

    for dp in result.deprecated_paths:
        assert dp.new_path is not None
        assert dp.deprecated_in
        assert dp.last_valid_version


# ============================================================================
# Tests for ExclusionChecker
# ============================================================================


class TestExclusionChecker:
    """Test the ExclusionChecker class."""

    def test_error_field_exclusion(self):
        """Test that error fields are correctly identified."""
        checker = ExclusionChecker()

        # Various error field patterns
        assert (
            checker.get_exclusion_reason("equilibrium/time_slice/psi_error_upper")
            == "error_field"
        )
        assert (
            checker.get_exclusion_reason("equilibrium/time_slice/psi_error_lower")
            == "error_field"
        )
        assert (
            checker.get_exclusion_reason("equilibrium/time_slice/psi_error_index")
            == "error_field"
        )
        assert (
            checker.get_exclusion_reason("core_profiles/profiles_1d/j_tor_error_upper")
            == "error_field"
        )

    def test_ggd_exclusion(self):
        """Test that GGD paths are correctly identified when GGD is excluded."""
        # Explicitly disable GGD inclusion to test exclusion logic
        checker = ExclusionChecker(include_ggd=False)

        # Various GGD patterns
        assert checker.get_exclusion_reason("edge_profiles/ggd/grid") == "ggd"
        assert checker.get_exclusion_reason("edge_profiles/grids_ggd/grid") == "ggd"
        assert checker.get_exclusion_reason("something/path/ggd/subpath") == "ggd"

    def test_metadata_exclusion(self):
        """Test that metadata paths are correctly identified."""
        checker = ExclusionChecker()

        # Metadata patterns
        assert (
            checker.get_exclusion_reason("equilibrium/ids_properties/homogeneous_time")
            == "metadata"
        )
        assert checker.get_exclusion_reason("core_profiles/code/name") == "metadata"

    def test_non_excluded_paths(self):
        """Test that normal paths are not excluded."""
        checker = ExclusionChecker()

        # Normal paths should not be excluded
        assert (
            checker.get_exclusion_reason("equilibrium/time_slice/profiles_1d/psi")
            is None
        )
        assert (
            checker.get_exclusion_reason(
                "core_profiles/profiles_1d/electrons/temperature"
            )
            is None
        )
        assert checker.get_exclusion_reason("magnetics/flux_loop/flux") is None

    def test_is_excluded_convenience_method(self):
        """Test the is_excluded convenience method."""
        checker = ExclusionChecker()

        assert checker.is_excluded("equilibrium/time_slice/psi_error_upper") is True
        assert checker.is_excluded("equilibrium/time_slice/profiles_1d/psi") is False

    def test_configurable_exclusions(self):
        """Test that exclusions can be configured."""
        # Enable error field inclusion (don't exclude)
        checker = ExclusionChecker(include_error_fields=True)
        assert (
            checker.get_exclusion_reason("equilibrium/time_slice/psi_error_upper")
            is None
        )

        # Enable GGD inclusion (don't exclude)
        checker = ExclusionChecker(include_ggd=True)
        assert checker.get_exclusion_reason("edge_profiles/ggd/grid") is None


# ============================================================================
# Tests for PathMap exclusion methods
# ============================================================================


def test_path_map_get_exclusion_reason(mock_path_map: MockPathMap):
    """Test getting exclusion reason from path map."""
    # Pre-computed excluded path
    reason = mock_path_map.get_exclusion_reason(
        "equilibrium/time_slice/profiles_1d/psi_error_lower"
    )
    assert reason == "error_field"

    # GGD path
    reason = mock_path_map.get_exclusion_reason(
        "edge_profiles/ggd/grid/space/objects_per_dimension"
    )
    assert reason == "ggd"


def test_path_map_get_exclusion_reason_fallback(mock_path_map: MockPathMap):
    """Test that exclusion check falls back to live checker for unmapped paths."""
    # Not in pre-computed excluded_paths but should be detected by live checker
    reason = mock_path_map.get_exclusion_reason(
        "core_profiles/profiles_1d/some_field_error_upper"
    )
    assert reason == "error_field"


def test_path_map_get_exclusion_description(mock_path_map: MockPathMap):
    """Test getting exclusion description from path map."""
    description = mock_path_map.get_exclusion_description("error_field")
    assert "error" in description.lower() or "uncertainty" in description.lower()

    description = mock_path_map.get_exclusion_description("ggd")
    assert "ggd" in description.lower() or "grid" in description.lower()


def test_path_map_is_excluded(mock_path_map: MockPathMap):
    """Test is_excluded convenience method on path map."""
    assert (
        mock_path_map.is_excluded("equilibrium/time_slice/profiles_1d/psi_error_lower")
        is True
    )
    assert mock_path_map.is_excluded("equilibrium/time_slice/profiles_1d/psi") is False


# ============================================================================
# Tests for check_imas_paths - Excluded paths
# ============================================================================


@pytest.mark.asyncio
async def test_check_excluded_path_returns_exclusion_info(path_tool):
    """Test that checking an excluded path returns exclusion info."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi_error_lower"
    )

    assert result.summary["total"] == 1
    assert result.summary["not_found"] == 1

    res = result.results[0]
    assert res.exists is False
    assert res.excluded is not None
    assert res.excluded["reason_key"] == "error_field"
    assert "reason" in res.excluded


@pytest.mark.asyncio
async def test_check_deprecated_path_with_excluded_new_path(path_tool):
    """Test that deprecated paths whose new_path is excluded include that info."""
    result = await path_tool.check_imas_paths(
        "core_profiles/profiles_1d/j_tor_error_upper_old"
    )

    assert result.summary["total"] == 1
    assert result.summary["not_found"] == 1

    res = result.results[0]
    assert res.exists is False
    assert res.migration is not None
    assert res.migration["new_path"] == "core_profiles/profiles_1d/j_tor_error_upper"
    assert res.migration["new_path_excluded"] is True
    assert "exclusion_reason" in res.migration


# ============================================================================
# Tests for fetch_imas_paths - Excluded paths
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_excluded_path_returns_exclusion_info(path_tool):
    """Test that fetching an excluded path returns exclusion info."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi_error_lower"
    )

    # Should not be in nodes (path excluded from index)
    assert result.node_count == 0
    assert result.summary["retrieved"] == 0
    assert result.summary["deprecated"] == 0

    # Should be in excluded_paths
    assert result.summary["excluded"] == 1
    assert len(result.excluded_paths) == 1

    excluded = result.excluded_paths[0]
    assert excluded.path == "equilibrium/time_slice/profiles_1d/psi_error_lower"
    assert excluded.reason_key == "error_field"
    assert excluded.reason_description


@pytest.mark.asyncio
async def test_fetch_deprecated_path_with_excluded_new_path(path_tool):
    """Test that deprecated paths whose new_path is excluded include that info."""
    result = await path_tool.fetch_imas_paths(
        "core_profiles/profiles_1d/j_tor_error_upper_old"
    )

    # Should be in deprecated_paths with exclusion info
    assert result.summary["deprecated"] == 1
    assert len(result.deprecated_paths) == 1

    deprecated = result.deprecated_paths[0]
    assert deprecated.path == "core_profiles/profiles_1d/j_tor_error_upper_old"
    assert deprecated.new_path == "core_profiles/profiles_1d/j_tor_error_upper"
    assert deprecated.new_path_excluded is True
    assert deprecated.exclusion_reason is not None


@pytest.mark.asyncio
async def test_fetch_mixed_valid_deprecated_excluded_invalid(path_tool):
    """Test fetching mix of valid, deprecated, excluded, and invalid paths."""
    result = await path_tool.fetch_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi "  # valid
        "equilibrium/time_slice/constraints/bpol_probe "  # deprecated
        "equilibrium/time_slice/profiles_1d/psi_error_lower "  # excluded
        "fake/path "  # not found
        "notapath"  # invalid
    )

    assert result.summary["retrieved"] == 1
    assert result.summary["deprecated"] == 1
    assert result.summary["excluded"] == 1
    assert result.summary["not_found"] == 1
    assert result.summary["invalid"] == 1

    assert result.node_count == 1
    assert len(result.deprecated_paths) == 1
    assert len(result.excluded_paths) == 1
