"""Direct tests for DD version diagnostics and shared error-field tooling."""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools import Tools
from imas_codex.tools.graph_search import GraphPathTool
from imas_codex.tools.version_tool import VersionTool


@pytest.mark.asyncio
async def test_shared_fetch_error_fields_returns_structured_results():
    gc = MagicMock()
    gc.query.return_value = [
        {
            "path": "equilibrium/time_slice/profiles_1d/psi",
            "error_fields": [
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi_error_upper",
                    "name": "psi_error_upper",
                    "error_type": "upper",
                    "documentation": "Upper error bound",
                    "data_type": "FLT_1D",
                }
            ],
        }
    ]
    tool = GraphPathTool(gc)

    result = await tool.fetch_error_fields("equilibrium/time_slice/profiles_1d/psi")

    assert result["path"] == "equilibrium/time_slice/profiles_1d/psi"
    assert result["count"] == 1
    assert result["not_found"] is False
    assert result["error_fields"][0]["error_type"] == "upper"


@pytest.mark.asyncio
async def test_shared_fetch_error_fields_returns_not_found():
    gc = MagicMock()
    gc.query.return_value = []
    tool = GraphPathTool(gc)

    result = await tool.fetch_error_fields("fake/path")

    assert result["path"] == "fake/path"
    assert result["count"] == 0
    assert result["error_fields"] == []
    assert result["not_found"] is True


@pytest.mark.asyncio
async def test_tools_delegate_get_dd_versions():
    gc = MagicMock()
    gc.query.return_value = [
        {"id": "3.42.0", "major": 3, "minor": 42, "patch": 0, "is_current": False},
        {"id": "4.1.0", "major": 4, "minor": 1, "patch": 0, "is_current": True},
    ]
    tools = Tools(graph_client=gc)

    result = await tools.get_dd_versions()

    assert result["current_version"] == "4.1.0"
    assert result["version_count"] == 2


@pytest.mark.asyncio
async def test_version_context_includes_diagnostics():
    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "core_profiles/profiles_1d/electrons/pressure",
            "introduced_in": "3.22.0",
            "deprecated_in": None,
            "change_count": 1,
            "changes": [
                {"version": "4.0.0", "change_type": "units", "semantic_type": "units",
                 "old_value": "Pa", "new_value": "Pa"},
            ],
        },
        {
            "id": "equilibrium/time_slice/profiles_1d/psi",
            "introduced_in": "3.22.0",
            "deprecated_in": None,
            "change_count": 0,
            "changes": [],
        },
    ]
    tool = VersionTool(gc)

    result = await tool.get_dd_version_context(
        [
            "core_profiles/profiles_1d/electrons/pressure",
            "equilibrium/time_slice/profiles_1d/psi",
            "fake/path",
        ]
    )

    assert result["paths_found"] == [
        "core_profiles/profiles_1d/electrons/pressure",
        "equilibrium/time_slice/profiles_1d/psi",
    ]
    assert result["paths_without_changes"] == [
        "equilibrium/time_slice/profiles_1d/psi"
    ]
    assert result["graph_change_nodes_seen"] == 1
    assert result["not_found"] == ["fake/path"]
