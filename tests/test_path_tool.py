"""Tests for the path tool (check_imas_paths and fetch_imas_paths)."""

import pytest

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.tools import PathTool


def _full_path(ids: str, path: str) -> str:
    return f"{ids}/{path}"


@pytest.fixture
def path_tool() -> PathTool:
    """Create a PathTool instance for testing."""
    doc_store = DocumentStore()
    return PathTool(doc_store)


@pytest.fixture
def outdated_paths_with_ids() -> tuple[str, list[str]]:
    return "equilibrium", [
        "time_slice/constraints/bpol_probe",
        "time_slice/constraints/bpol_probe",
    ]


@pytest.fixture
def new_paths_with_ids() -> tuple[str, list[str]]:
    return "equilibrium", [
        "time_slice/constraints/b_field_pol_probe",
        "time_slice/constraints/b_field_pol_probe",
    ]


@pytest.mark.asyncio
async def test_update_imas_path_tool_single(
    path_tool: PathTool,
    outdated_paths_with_ids: tuple[str, list[str]],
    new_paths_with_ids: tuple[str, list[str]],
):
    """Basic test to ensure PathTool is instantiated correctly."""
    _, outdated = outdated_paths_with_ids
    ids, new = new_paths_with_ids

    for o, n in zip(outdated, new, strict=False):
        result = path_tool._convert_path(o, ids=ids, version="3.39.0")
        assert result == n


@pytest.mark.asyncio
async def test_update_imas_path_tool_list(
    path_tool: PathTool,
    outdated_paths_with_ids: tuple[str, list[str]],
    new_paths_with_ids: tuple[str, list[str]],
):
    _, outdated = outdated_paths_with_ids
    ids, new = new_paths_with_ids

    result = await path_tool.update_imas_path(
        outdated, ids_name=ids, source_dd_version="3.39.0"
    )
    assert result == new


@pytest.mark.asyncio
async def test_update_imas_path_tool_spaced_list(
    path_tool: PathTool,
    outdated_paths_with_ids: tuple[str, list[str]],
    new_paths_with_ids: tuple[str, list[str]],
):
    """Test updating IMAS paths provided as space-separated string."""
    _, outdated = outdated_paths_with_ids
    ids, new = new_paths_with_ids

    result = await path_tool.update_imas_path(
        " ".join(outdated), ids_name=ids, source_dd_version="3.39.0"
    )
    assert result == new


@pytest.mark.asyncio
async def test_single_valid_path(path_tool):
    """Test validation of a single existing path."""
    result = await path_tool.check_imas_paths(
        "core_profiles/profiles_1d/electrons/temperature"
    )

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 1
    assert result["summary"]["not_found"] == 0
    assert len(result["results"]) == 1
    assert result["results"][0]["exists"] is True
    assert (
        result["results"][0]["path"]
        == "core_profiles/profiles_1d/electrons/temperature"
    )
    assert result["results"][0]["ids_name"] == "core_profiles"


@pytest.mark.asyncio
async def test_single_invalid_path(path_tool):
    """Test validation of a non-existent path."""
    result = await path_tool.check_imas_paths("fake/nonexistent/path")

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 0
    assert result["summary"]["not_found"] == 1
    assert len(result["results"]) == 1
    assert result["results"][0]["exists"] is False
    assert result["results"][0]["path"] == "fake/nonexistent/path"


@pytest.mark.asyncio
async def test_multiple_paths_space_delimited(path_tool):
    """Test validation of multiple paths as space-delimited string."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature fake/path"
    )

    assert result["summary"]["total"] == 3
    assert result["summary"]["found"] == 2
    assert result["summary"]["not_found"] == 1
    assert len(result["results"]) == 3

    # Check first path
    assert result["results"][0]["exists"] is True
    assert result["results"][0]["ids_name"] == "equilibrium"

    # Check second path
    assert result["results"][1]["exists"] is True
    assert result["results"][1]["ids_name"] == "core_profiles"

    # Check third path
    assert result["results"][2]["exists"] is False
    assert result["results"][2]["path"] == "fake/path"


@pytest.mark.asyncio
async def test_multiple_paths_list(path_tool):
    """Test validation of multiple paths as a list."""
    result = await path_tool.check_imas_paths(
        [
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        ]
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["found"] == 2
    assert result["summary"]["not_found"] == 0
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_malformed_path_no_slash(path_tool):
    """Test validation with malformed path (no slash)."""
    result = await path_tool.check_imas_paths("notapath")

    assert result["summary"]["total"] == 1
    assert result["summary"]["invalid"] == 1
    assert result["results"][0]["exists"] is False
    assert "error" in result["results"][0]
    assert "Invalid format" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_mixed_valid_invalid_paths(path_tool):
    """Test validation with mix of valid, invalid, and malformed paths."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/profiles_1d/psi invalid/path notapath core_profiles/profiles_1d/electrons/temperature"
    )

    assert result["summary"]["total"] == 4
    assert result["summary"]["found"] == 2
    assert result["summary"]["not_found"] == 1
    assert result["summary"]["invalid"] == 1


@pytest.mark.asyncio
async def test_returns_structured_response(path_tool):
    """Test that response has proper structure."""
    result = await path_tool.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    # Should have summary and results
    assert isinstance(result, dict)
    assert "summary" in result
    assert "results" in result

    # Summary should have counts
    assert "total" in result["summary"]
    assert "found" in result["summary"]
    assert "not_found" in result["summary"]
    assert "invalid" in result["summary"]

    # Results should be a list
    assert isinstance(result["results"], list)

    # Each result should have path and exists
    for item in result["results"]:
        assert "path" in item
        assert "exists" in item


@pytest.mark.asyncio
async def test_token_efficient_response(path_tool):
    """Test that response is token-efficient (no verbose fields unless needed)."""
    result = await path_tool.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    # Should not have search-specific fields
    assert "hits" not in result
    assert "query_hints" not in result
    assert "tool_hints" not in result
    assert "physics_context" not in result

    # Results should be minimal
    res = result["results"][0]
    assert "path" in res
    assert "exists" in res
    assert "ids_name" in res

    # Should not have documentation (token-heavy)
    assert "documentation" not in res


@pytest.mark.asyncio
async def test_ids_prefix_single_path(path_tool):
    """Test ids parameter with single path."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi", ids="equilibrium"
    )

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 1
    assert result["results"][0]["exists"] is True
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][0]["ids_name"] == "equilibrium"


@pytest.mark.asyncio
async def test_ids_prefix_multiple_paths(path_tool):
    """Test ids parameter with multiple paths (ensemble checking)."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi time_slice/boundary/psi_norm time_slice/boundary/type",
        ids="equilibrium",
    )

    assert result["summary"]["total"] == 3
    assert result["summary"]["found"] == 3
    assert result["summary"]["not_found"] == 0

    # All paths should be prefixed with equilibrium
    for res in result["results"]:
        assert res["path"].startswith("equilibrium/")
        assert res["exists"] is True
        assert res["ids_name"] == "equilibrium"


@pytest.mark.asyncio
async def test_ids_prefix_with_list(path_tool):
    """Test ids parameter with list of paths."""
    result = await path_tool.check_imas_paths(
        ["time_slice/boundary/psi", "time_slice/boundary/psi_norm"], ids="equilibrium"
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["found"] == 2
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][1]["path"] == "equilibrium/time_slice/boundary/psi_norm"


@pytest.mark.asyncio
async def test_ids_prefix_already_present(path_tool):
    """Test that ids prefix is not duplicated if already present."""
    result = await path_tool.check_imas_paths(
        "equilibrium/time_slice/boundary/psi", ids="equilibrium"
    )

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 1
    # Should not be double-prefixed
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][0]["path"].count("equilibrium/") == 1


@pytest.mark.asyncio
async def test_ids_prefix_mixed_paths(path_tool):
    """Test ids prefix with some paths having full IDS and some not."""
    result = await path_tool.check_imas_paths(
        "time_slice/boundary/psi equilibrium/time_slice/boundary/psi_norm",
        ids="equilibrium",
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["found"] == 2

    # Both should have correct paths without duplication
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][1]["path"] == "equilibrium/time_slice/boundary/psi_norm"


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
    # Units might or might not be present depending on the path


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
async def test_fetch_returns_physics_context(path_tool):
    """Test that fetch_imas_paths includes physics context."""
    result = await path_tool.fetch_imas_paths("equilibrium/time_slice/profiles_1d/psi")

    assert result.node_count == 1
    node = result.nodes[0]

    # Check for physics context if available
    if node.physics_context:
        assert node.physics_context.domain

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

    # Check (minimal)
    check_result = await path_tool.check_imas_paths(path)

    # Fetch (rich)
    fetch_result = await path_tool.fetch_imas_paths(path)

    # Check returns dict with minimal info
    assert isinstance(check_result, dict)
    assert "exists" in check_result["results"][0]
    assert "documentation" not in check_result["results"][0]

    # Fetch returns IdsPathResult with full IdsNode objects
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
