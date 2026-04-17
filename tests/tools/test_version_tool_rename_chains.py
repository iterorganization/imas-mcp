"""Tests for VersionTool rename-chain traversal and version ordering.

Verifies that get_dd_version_context correctly traverses RENAMED_TO graph
edges, includes rename info in per-path mode, and that get_dd_versions
returns properly semver-sorted results.
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools.version_tool import VersionTool, _version_sort_key


def _make_gc(query_return=None, side_effect=None):
    gc = MagicMock()
    if side_effect is not None:
        gc.query = MagicMock(side_effect=side_effect)
    else:
        gc.query = MagicMock(return_value=query_return or [])
    return gc


class TestVersionSortKey:
    """Tests for semver sort key parsing."""

    def test_standard_versions(self):
        versions = ["4.1.0", "3.22.0", "3.9.0", "3.42.2", "4.0.0"]
        result = sorted(versions, key=_version_sort_key)
        assert result == ["3.9.0", "3.22.0", "3.42.2", "4.0.0", "4.1.0"]

    def test_patch_ordering(self):
        versions = ["3.23.3", "3.23.0", "3.23.1", "3.23.2"]
        result = sorted(versions, key=_version_sort_key)
        assert result == ["3.23.0", "3.23.1", "3.23.2", "3.23.3"]

    def test_unparseable_fallback(self):
        assert _version_sort_key("not.a.version") == (0,)
        assert _version_sort_key("") == (0,)


class TestGetDdVersions:
    """Tests for version ordering in get_dd_versions."""

    @pytest.mark.asyncio
    async def test_semver_ordering(self):
        """Versions are sorted by semver, not lexicographically."""
        gc = _make_gc(
            query_return=[
                {"id": "4.1.0", "is_current": True},
                {"id": "3.22.0", "is_current": False},
                {"id": "3.42.2", "is_current": False},
                {"id": "4.0.0", "is_current": False},
            ]
        )
        tool = VersionTool(gc)
        result = await tool.get_dd_versions()

        assert result["versions"] == ["3.22.0", "3.42.2", "4.0.0", "4.1.0"]
        assert result["version_range"] == "3.22.0 - 4.1.0"
        assert result["current_version"] == "4.1.0"


class TestPerPathRenameInfo:
    """Tests for rename info in per-path mode (Mode 1)."""

    @pytest.mark.asyncio
    async def test_per_path_includes_rename_links(self):
        """Per-path mode includes renamed_to and renamed_from."""

        def mock_query(cypher, **kwargs):
            if "UNWIND $path_ids" in cypher and "RENAMED_TO" not in cypher:
                # Main per-path query
                return [
                    {
                        "id": "magnetics/b_field_tor_probe/field",
                        "lifecycle_status": "obsolescent",
                        "introduced_in": "3.22.0",
                        "deprecated_in": "3.39.0",
                        "change_count": 0,
                        "changes": [],
                    }
                ]
            elif "RENAMED_TO" in cypher and "UNWIND" in cypher:
                # Rename links query
                return [
                    {
                        "id": "magnetics/b_field_tor_probe/field",
                        "renamed_to": ["magnetics/b_field_phi_probe/field"],
                        "renamed_from": [],
                    }
                ]
            elif "RENAMED_TO*1..10" in cypher:
                # Full chain query
                return [
                    {
                        "rename_chain": [
                            "magnetics/b_field_tor_probe/field",
                            "magnetics/b_field_phi_probe/field",
                        ],
                        "hops": 1,
                    }
                ]
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(
            paths=["magnetics/b_field_tor_probe/field"],
            follow_rename_chains=True,
        )

        assert "paths" in result
        path_info = result["paths"]["magnetics/b_field_tor_probe/field"]
        assert path_info["renamed_to"] == ["magnetics/b_field_phi_probe/field"]
        assert path_info["renamed_from"] == []
        # Full chains included
        assert "rename_chains" in result
        assert len(result["rename_chains"]) == 1

    @pytest.mark.asyncio
    async def test_per_path_default_follow_rename_chains(self):
        """follow_rename_chains defaults to True."""

        def mock_query(cypher, **kwargs):
            if "RENAMED_TO*1..10" in cypher:
                return []
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(
            paths=["some/path"],
        )

        # With default follow_rename_chains=True, rename_chains key exists
        assert "rename_chains" in result

    @pytest.mark.asyncio
    async def test_per_path_no_chains_when_disabled(self):
        """follow_rename_chains=False excludes chain data."""

        def mock_query(cypher, **kwargs):
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = VersionTool(gc)
        result = await tool.get_dd_version_context(
            paths=["some/path"],
            follow_rename_chains=False,
        )

        assert "rename_chains" not in result


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

        def mock_query(cypher, **kwargs):
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

    @pytest.mark.asyncio
    async def test_path_filter(self):
        """Path filter narrows chains to those containing requested paths."""

        def mock_query(cypher, **kwargs):
            if "RENAMED_TO*1..10" in cypher:
                return [
                    {
                        "rename_chain": ["a/old", "a/new"],
                        "hops": 1,
                    },
                    {
                        "rename_chain": ["b/old", "b/new"],
                        "hops": 1,
                    },
                ]
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = VersionTool(gc)

        # Filter to chains containing "a/old"
        result = await tool._rename_chain_query(path_filter=["a/old"])
        chains = result["rename_chains"]
        assert len(chains) == 1
        assert "a/old" in chains[0]["chain"]
