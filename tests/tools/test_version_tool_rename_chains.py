"""Tests for VersionTool rename-chain traversal.

Verifies that get_dd_version_context with follow_rename_chains=True
correctly traverses RENAMED_TO graph edges and returns multi-hop lineages.
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools.version_tool import VersionTool


def _make_gc(query_return=None, side_effect=None):
    gc = MagicMock()
    if side_effect is not None:
        gc.query = MagicMock(side_effect=side_effect)
    else:
        gc.query = MagicMock(return_value=query_return or [])
    return gc


class TestRenameChains:
    @pytest.mark.asyncio
    async def test_basic(self):
        """follow_rename_chains=True returns chain data."""
        gc = _make_gc(
            query_return=[
                {
                    "rename_chain": [
                        "equilibrium/profiles_1d/elongation",
                        "equilibrium/time_slice/profiles_1d/elongation",
                    ],
                    "hops": 1,
                },
                {
                    "rename_chain": [
                        "core_profiles/te",
                        "core_profiles/profiles_1d/te",
                        "core_profiles/profiles_1d/electrons/temperature",
                    ],
                    "hops": 2,
                },
            ]
        )
        tool = VersionTool(gc)

        # No paths, no change_type_filter, just follow_rename_chains
        result = await tool.get_dd_version_context(follow_rename_chains=True)

        assert "rename_chains" in result
        chains = result["rename_chains"]
        assert len(chains) == 2
        # Chains should have 'chain' and 'hops' keys
        assert chains[0]["hops"] == 1
        assert len(chains[0]["chain"]) == 2
        assert chains[1]["hops"] == 2
        assert len(chains[1]["chain"]) == 3

    @pytest.mark.asyncio
    async def test_ids_filter(self):
        """Rename chains can be filtered by IDS."""
        gc = _make_gc(
            query_return=[
                {
                    "rename_chain": [
                        "equilibrium/profiles_1d/elongation",
                        "equilibrium/time_slice/profiles_1d/elongation",
                    ],
                    "hops": 1,
                },
            ]
        )
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(
            follow_rename_chains=True,
            ids_filter="equilibrium",
        )

        assert "rename_chains" in result
        # Verify ids_filter was passed to the query
        call_kwargs = gc.query.call_args
        assert call_kwargs[1].get("ids_filter") == "equilibrium"

    @pytest.mark.asyncio
    async def test_combined_with_change_type(self):
        """follow_rename_chains + change_type_filter returns both sections."""
        call_count = 0

        def mock_query(cypher, **kwargs):
            nonlocal call_count
            call_count += 1
            if "IMASNodeChange" in cypher:
                # Bulk change query
                return [
                    {
                        "path": "equilibrium/time_slice/profiles_1d/psi",
                        "ids": "equilibrium",
                        "old_value": "V",
                        "new_value": "Wb",
                        "version": "4.0.0",
                        "change_type": "units",
                        "severity": "breaking",
                        "summary": "Changed units",
                    },
                ]
            elif "RENAMED_TO" in cypher:
                # Rename chain query
                return [
                    {
                        "rename_chain": [
                            "equilibrium/profiles_1d/psi",
                            "equilibrium/time_slice/profiles_1d/psi",
                        ],
                        "hops": 1,
                    },
                ]
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(
            change_type_filter="units",
            follow_rename_chains=True,
        )

        # Should have bulk query results
        assert result["mode"] == "bulk_query"
        assert result["change_count"] == 1

        # Plus rename chains appended
        assert "rename_chains" in result
        assert len(result["rename_chains"]) == 1

    @pytest.mark.asyncio
    async def test_empty_chains(self):
        """Empty rename chains handled gracefully."""
        gc = _make_gc(query_return=[])
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(follow_rename_chains=True)

        assert result["rename_chains"] == []

    @pytest.mark.asyncio
    async def test_query_error_handled(self):
        """Graph errors return error dict instead of raising."""
        gc = _make_gc(side_effect=Exception("Neo4j connection failed"))
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(follow_rename_chains=True)

        assert "error" in result
        assert "rename_chains" in result
        assert result["rename_chains"] == []
