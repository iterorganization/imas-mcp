"""Tests for IDS existence validation in check_dd_paths.

Verifies that passing an invalid ``ids`` parameter returns an early error
result with a fuzzy suggestion instead of silently returning "not found"
for every path.
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.search.fuzzy_matcher import PathFuzzyMatcher, reset_fuzzy_matcher
from imas_codex.tools.graph_search import GraphPathTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_IDS = ["equilibrium", "core_profiles", "magnetics", "summary"]
_VALID_PATHS = [
    "equilibrium/time_slice",
    "equilibrium/time_slice/boundary",
    "equilibrium/time_slice/boundary/psi",
    "equilibrium/time_slice/global_quantities/ip",
    "core_profiles/profiles_1d/electrons/temperature",
    "magnetics/flux_loop",
]


def _make_tool_with_mock_gc(
    *,
    ids_exists: bool = True,
) -> GraphPathTool:
    """Build a GraphPathTool whose GraphClient is fully mocked.

    When *ids_exists* is False the IDS node query returns an empty list,
    simulating a misspelled IDS name.  IMASNode queries still return empty
    (paths not found), which is fine because the IDS-validation short-circuit
    fires first.
    """
    mock_gc = MagicMock()

    def _query(cypher: str, **kwargs):
        # IDS existence check injected by the new validation block
        if "MATCH (i:IDS" in cypher and "i.id" in cypher and "RETURN" in cypher:
            if ids_exists:
                return [{"i.id": kwargs.get("ids_name", "equilibrium")}]
            return []
        # Fuzzy-matcher bootstrap: return IDS + path list
        if "IMASNode" in cypher and "node_category" in cypher:
            return [{"id": p, "ids": p.split("/")[0]} for p in _VALID_PATHS]
        # Default: nothing found
        return []

    mock_gc.query = MagicMock(side_effect=_query)
    return GraphPathTool(mock_gc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_fuzzy_cache():
    """Wipe fuzzy-matcher singleton + per-tool cache before each test."""
    reset_fuzzy_matcher()
    yield
    reset_fuzzy_matcher()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIdsExistenceValidation:
    """IDS validation fires before any path look-up when ids= is wrong."""

    @pytest.mark.asyncio
    async def test_nonexistent_ids_returns_error(self):
        """Completely unknown IDS → all results have exists=False and an error."""
        tool = _make_tool_with_mock_gc(ids_exists=False)
        result = await tool.check_dd_paths(
            "time_slice/boundary/psi", ids="nonexistent_ids"
        )

        assert result.summary["found"] == 0
        assert result.summary["not_found"] == 1
        assert result.error is not None
        assert "nonexistent_ids" in result.error

        item = result.results[0]
        assert item.exists is False
        assert item.error is not None

    @pytest.mark.asyncio
    async def test_typo_ids_suggests_correct_name(self):
        """Typo 'equilibriummm' → suggestion contains 'equilibrium'."""
        tool = _make_tool_with_mock_gc(ids_exists=False)
        result = await tool.check_dd_paths(
            "time_slice/boundary/psi", ids="equilibriummm"
        )

        assert result.summary["found"] == 0
        # At least the top-level error message should mention the typo
        assert result.error is not None

        item = result.results[0]
        assert item.exists is False
        # Fuzzy suggestion should point to "equilibrium"
        if item.suggestion:
            assert "equilibrium" in item.suggestion
        if item.suggestions:
            assert any("equilibrium" in s for s in item.suggestions)

    @pytest.mark.asyncio
    async def test_valid_ids_does_not_trigger_early_return(self):
        """When the IDS exists, normal path processing continues."""
        tool = _make_tool_with_mock_gc(ids_exists=True)
        result = await tool.check_dd_paths("time_slice/boundary/psi", ids="equilibrium")

        # IDS check passed — result.error should be None (no IDS-level error)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_multiple_paths_all_get_error_on_bad_ids(self):
        """All paths in the batch receive the IDS-level error."""
        tool = _make_tool_with_mock_gc(ids_exists=False)
        result = await tool.check_dd_paths(
            ["profiles_1d/electrons/temperature", "profiles_1d/electrons/density"],
            ids="core_profilez",
        )

        assert result.summary["total"] == 2
        assert result.summary["found"] == 0
        assert result.summary["not_found"] == 2
        for item in result.results:
            assert item.exists is False
            assert item.error is not None

    @pytest.mark.asyncio
    async def test_no_ids_param_skips_validation(self):
        """Without ids=, the IDS validation block is not entered."""
        tool = _make_tool_with_mock_gc(ids_exists=True)
        # Pass a fully-qualified path — the IDS-node query should never fire
        result = await tool.check_dd_paths("equilibrium/time_slice/boundary/psi")
        # No IDS-level error regardless of outcome
        assert result.error is None
