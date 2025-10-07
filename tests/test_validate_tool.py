"""Tests for the check_ids_paths validation tool."""

import pytest

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.tools import ValidateTool


@pytest.fixture
def validate_tool():
    """Create a ValidateTool instance for testing."""
    doc_store = DocumentStore()
    return ValidateTool(doc_store)


@pytest.mark.asyncio
async def test_single_valid_path(validate_tool):
    """Test validation of a single existing path."""
    result = await validate_tool.check_ids_paths(
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
async def test_single_invalid_path(validate_tool):
    """Test validation of a non-existent path."""
    result = await validate_tool.check_ids_paths("fake/nonexistent/path")

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 0
    assert result["summary"]["not_found"] == 1
    assert len(result["results"]) == 1
    assert result["results"][0]["exists"] is False
    assert result["results"][0]["path"] == "fake/nonexistent/path"


@pytest.mark.asyncio
async def test_multiple_paths_space_delimited(validate_tool):
    """Test validation of multiple paths as space-delimited string."""
    result = await validate_tool.check_ids_paths(
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
async def test_multiple_paths_list(validate_tool):
    """Test validation of multiple paths as a list."""
    result = await validate_tool.check_ids_paths(
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
async def test_malformed_path_no_slash(validate_tool):
    """Test validation with malformed path (no slash)."""
    result = await validate_tool.check_ids_paths("notapath")

    assert result["summary"]["total"] == 1
    assert result["summary"]["invalid"] == 1
    assert result["results"][0]["exists"] is False
    assert "error" in result["results"][0]
    assert "Invalid format" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_mixed_valid_invalid_paths(validate_tool):
    """Test validation with mix of valid, invalid, and malformed paths."""
    result = await validate_tool.check_ids_paths(
        "equilibrium/time_slice/profiles_1d/psi invalid/path notapath core_profiles/profiles_1d/electrons/temperature"
    )

    assert result["summary"]["total"] == 4
    assert result["summary"]["found"] == 2
    assert result["summary"]["not_found"] == 1
    assert result["summary"]["invalid"] == 1


@pytest.mark.asyncio
async def test_returns_structured_response(validate_tool):
    """Test that response has proper structure."""
    result = await validate_tool.check_ids_paths(
        "equilibrium/time_slice/profiles_1d/psi"
    )

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
async def test_token_efficient_response(validate_tool):
    """Test that response is token-efficient (no verbose fields unless needed)."""
    result = await validate_tool.check_ids_paths(
        "equilibrium/time_slice/profiles_1d/psi"
    )

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
async def test_ids_prefix_single_path(validate_tool):
    """Test ids parameter with single path."""
    result = await validate_tool.check_ids_paths(
        "time_slice/boundary/psi", ids="equilibrium"
    )

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 1
    assert result["results"][0]["exists"] is True
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][0]["ids_name"] == "equilibrium"


@pytest.mark.asyncio
async def test_ids_prefix_multiple_paths(validate_tool):
    """Test ids parameter with multiple paths (ensemble checking)."""
    result = await validate_tool.check_ids_paths(
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
async def test_ids_prefix_with_list(validate_tool):
    """Test ids parameter with list of paths."""
    result = await validate_tool.check_ids_paths(
        ["time_slice/boundary/psi", "time_slice/boundary/psi_norm"], ids="equilibrium"
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["found"] == 2
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][1]["path"] == "equilibrium/time_slice/boundary/psi_norm"


@pytest.mark.asyncio
async def test_ids_prefix_already_present(validate_tool):
    """Test that ids prefix is not duplicated if already present."""
    result = await validate_tool.check_ids_paths(
        "equilibrium/time_slice/boundary/psi", ids="equilibrium"
    )

    assert result["summary"]["total"] == 1
    assert result["summary"]["found"] == 1
    # Should not be double-prefixed
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][0]["path"].count("equilibrium/") == 1


@pytest.mark.asyncio
async def test_ids_prefix_mixed_paths(validate_tool):
    """Test ids prefix with some paths having full IDS and some not."""
    result = await validate_tool.check_ids_paths(
        "time_slice/boundary/psi equilibrium/time_slice/boundary/psi_norm",
        ids="equilibrium",
    )

    assert result["summary"]["total"] == 2
    assert result["summary"]["found"] == 2

    # Both should have correct paths without duplication
    assert result["results"][0]["path"] == "equilibrium/time_slice/boundary/psi"
    assert result["results"][1]["path"] == "equilibrium/time_slice/boundary/psi_norm"
