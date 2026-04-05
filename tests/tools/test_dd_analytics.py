"""Tests for DDAnalyticsTool methods — coverage, units, changes, changelog.

Uses the same mock GraphClient pattern as other tests in tests/tools/.
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools.dd_analytics_tool import DDAnalyticsTool
from imas_codex.tools.version_tool import VersionTool

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_gc(query_return=None, side_effect=None):
    """Create a mock GraphClient with a configurable query return."""
    gc = MagicMock()
    if side_effect is not None:
        gc.query = MagicMock(side_effect=side_effect)
    else:
        gc.query = MagicMock(return_value=query_return or [])
    return gc


# ── get_dd_changelog ─────────────────────────────────────────────────────


class TestGetDDChangelog:
    @pytest.mark.asyncio
    async def test_basic(self):
        """get_dd_changelog returns results with volatility_score."""
        gc = _make_gc(
            query_return=[
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi",
                    "ids": "equilibrium",
                    "change_count": 3,
                    "type_variety": 2,
                    "change_types": ["units", "data_type"],
                    "was_renamed": 0,
                    "volatility_score": 7,
                },
                {
                    "path": "core_profiles/profiles_1d/electrons/temperature",
                    "ids": "core_profiles",
                    "change_count": 1,
                    "type_variety": 1,
                    "change_types": ["definition_clarification"],
                    "was_renamed": 0,
                    "volatility_score": 3,
                },
            ]
        )
        tool = VersionTool(gc)
        result = await tool.get_dd_changelog()

        assert "results" in result
        assert result["total"] == 2
        # Highest volatility first
        assert (
            result["results"][0]["volatility_score"]
            >= result["results"][1]["volatility_score"]
        )

    @pytest.mark.asyncio
    async def test_ids_filter(self):
        """get_dd_changelog respects ids_filter parameter."""
        gc = _make_gc(
            query_return=[
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi",
                    "ids": "equilibrium",
                    "change_count": 2,
                    "type_variety": 1,
                    "change_types": ["units"],
                    "was_renamed": 0,
                    "volatility_score": 4,
                },
            ]
        )
        tool = VersionTool(gc)
        result = await tool.get_dd_changelog(ids_filter="equilibrium")

        assert result["ids_filter"] == "equilibrium"
        assert result["total"] == 1

        # Verify the query was called with the ids_filter param
        call_kwargs = gc.query.call_args
        assert call_kwargs[1]["ids_filter"] == "equilibrium"

    @pytest.mark.asyncio
    async def test_version_range(self):
        """get_dd_changelog passes from_version/to_version to query."""
        gc = _make_gc(query_return=[])
        tool = VersionTool(gc)
        result = await tool.get_dd_changelog(from_version="3.39.0", to_version="4.0.0")

        assert result["total"] == 0
        assert result["version_range"] == {"from": "3.39.0", "to": "4.0.0"}


# ── analyze_dd_coverage ──────────────────────────────────────────────────


class TestAnalyzeDDCoverage:
    @pytest.mark.asyncio
    async def test_basic(self):
        """analyze_dd_coverage returns ranked clusters with IDS counts."""
        gc = _make_gc(
            query_return=[
                {
                    "cluster_id": "cluster_42",
                    "label": "Electron Temperature",
                    "description": "Temperature of thermal electrons",
                    "ids_count": 5,
                    "ids_list": [
                        "core_profiles",
                        "core_transport",
                        "summary",
                        "ece",
                        "edge_profiles",
                    ],
                    "path_count": 12,
                    "domains": ["core_transport", "edge_transport"],
                    "representative_path": "core_profiles/profiles_1d/electrons/temperature",
                },
                {
                    "cluster_id": "cluster_7",
                    "label": "Plasma Current",
                    "description": "Total plasma current",
                    "ids_count": 4,
                    "ids_list": ["equilibrium", "magnetics", "summary", "mhd_linear"],
                    "path_count": 8,
                    "domains": ["magnetics", "equilibrium"],
                    "representative_path": "equilibrium/time_slice/global_quantities/ip",
                },
            ]
        )
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_coverage()

        assert "results" in result
        assert result["total"] == 2
        first = result["results"][0]
        assert first["ids_count"] >= result["results"][1]["ids_count"]
        assert "ids_list" in first
        assert "representative_path" in first

    @pytest.mark.asyncio
    async def test_min_ids_count(self):
        """analyze_dd_coverage respects min_ids_count parameter in Cypher."""
        gc = _make_gc(query_return=[])
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_coverage(min_ids_count=10)

        assert result["min_ids_count"] == 10
        assert result["total"] == 0

        # Verify parameter was passed to the query
        call_kwargs = gc.query.call_args
        assert call_kwargs[1]["min_ids_count"] == 10

    @pytest.mark.asyncio
    async def test_physics_domain_filter(self):
        """analyze_dd_coverage filters by physics_domain post-query."""
        gc = _make_gc(
            query_return=[
                {
                    "cluster_id": "c1",
                    "label": "Magnetic Field",
                    "description": None,
                    "ids_count": 3,
                    "ids_list": ["equilibrium", "magnetics", "mhd_linear"],
                    "path_count": 5,
                    "domains": ["magnetics"],
                    "representative_path": None,
                },
                {
                    "cluster_id": "c2",
                    "label": "Electron Density",
                    "description": None,
                    "ids_count": 4,
                    "ids_list": [
                        "core_profiles",
                        "core_transport",
                        "summary",
                        "edge_profiles",
                    ],
                    "path_count": 8,
                    "domains": ["core_transport"],
                    "representative_path": None,
                },
            ]
        )
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_coverage(physics_domain="magnetics")

        assert result["physics_domain_filter"] == "magnetics"
        assert result["total"] == 1
        assert result["results"][0]["label"] == "Magnetic Field"


# ── check_dd_units ───────────────────────────────────────────────────────


class TestCheckDDUnits:
    @pytest.mark.asyncio
    async def test_basic(self):
        """check_dd_units returns unit inconsistencies grouped by cluster."""
        gc = _make_gc(
            query_return=[
                {
                    "cluster": "Poloidal Flux",
                    "cluster_id": "flux_42",
                    "path1": "equilibrium/time_slice/profiles_1d/psi",
                    "ids1": "equilibrium",
                    "unit1": "Wb",
                    "dim1": "magnetic_flux",
                    "path2": "core_transport/model/profiles_1d/psi",
                    "ids2": "core_transport",
                    "unit2": "V.s",
                    "dim2": "magnetic_flux",
                    "severity": "advisory",
                },
            ]
        )
        tool = DDAnalyticsTool(gc)
        result = await tool.check_dd_units()

        assert "clusters" in result
        assert result["total_inconsistencies"] == 1
        assert result["clusters_affected"] == 1
        cluster = result["clusters"][0]
        assert cluster["cluster"] == "Poloidal Flux"
        assert len(cluster["inconsistencies"]) == 1

    @pytest.mark.asyncio
    async def test_severity_filter(self):
        """check_dd_units filters by severity parameter."""
        gc = _make_gc(
            query_return=[
                {
                    "cluster": "Poloidal Flux",
                    "cluster_id": "flux_42",
                    "path1": "equilibrium/time_slice/profiles_1d/psi",
                    "ids1": "equilibrium",
                    "unit1": "Wb",
                    "dim1": "magnetic_flux",
                    "path2": "core_transport/model/profiles_1d/psi",
                    "ids2": "core_transport",
                    "unit2": "V.s",
                    "dim2": "magnetic_flux",
                    "severity": "advisory",
                },
                {
                    "cluster": "Temperature",
                    "cluster_id": "temp_1",
                    "path1": "core_profiles/profiles_1d/electrons/temperature",
                    "ids1": "core_profiles",
                    "unit1": "eV",
                    "dim1": "energy",
                    "path2": "edge_profiles/ggd/electrons/temperature",
                    "ids2": "edge_profiles",
                    "unit2": "K",
                    "dim2": "temperature",
                    "severity": "incompatible",
                },
            ]
        )
        tool = DDAnalyticsTool(gc)

        # Filter to incompatible only
        result = await tool.check_dd_units(severity="incompatible")
        assert result["severity_filter"] == "incompatible"
        assert result["total_inconsistencies"] == 1
        cluster = result["clusters"][0]
        assert cluster["inconsistencies"][0]["severity"] == "incompatible"

    @pytest.mark.asyncio
    async def test_empty_result(self):
        """check_dd_units handles no inconsistencies gracefully."""
        gc = _make_gc(query_return=[])
        tool = DDAnalyticsTool(gc)
        result = await tool.check_dd_units()

        assert result["total_inconsistencies"] == 0
        assert result["clusters_affected"] == 0
        assert result["clusters"] == []


# ── analyze_dd_changes ───────────────────────────────────────────────────


class TestAnalyzeDDChanges:
    @pytest.mark.asyncio
    async def test_basic(self):
        """analyze_dd_changes returns own changes + siblings + related paths."""
        call_count = 0

        def mock_query(cypher, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Own changes query
                return [
                    {
                        "version": "4.0.0",
                        "change_type": "units",
                        "summary": "Changed units from V to Wb",
                        "breaking_level": None,
                    },
                ]
            elif call_count == 2:
                # Co-changing siblings query
                return [
                    {
                        "path": "core_transport/model/profiles_1d/psi",
                        "ids": "core_transport",
                        "cluster_label": "Poloidal Flux",
                        "sibling_changes": 1,
                        "sibling_change_types": ["units"],
                    },
                ]
            elif call_count == 3:
                # Coordinate-related query
                return [
                    {
                        "path": "core_profiles/profiles_1d/psi",
                        "ids": "core_profiles",
                        "shared_coordinates": ["Normalized toroidal flux"],
                    },
                ]
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_changes(
            path="equilibrium/time_slice/profiles_1d/psi"
        )

        assert result["path"] == "equilibrium/time_slice/profiles_1d/psi"
        assert len(result["own_changes"]) == 1
        assert result["own_changes"][0]["version"] == "4.0.0"
        assert len(result["co_changing_siblings"]) == 1
        assert result["co_changing_siblings"][0]["risk_score"] > 0
        assert len(result["related_paths"]) == 1
        assert result["has_breaking_changes"] is False
        assert result["summary"]["own_change_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_no_changes(self):
        """analyze_dd_changes handles path with no changes gracefully."""
        gc = _make_gc(query_return=[])
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_changes(path="fake/nonexistent/path")

        assert result["own_changes"] == []
        assert result["co_changing_siblings"] == []
        assert result["related_paths"] == []
        assert result["has_breaking_changes"] is False
        assert result["summary"]["own_change_count"] == 0

    @pytest.mark.asyncio
    async def test_breaking_changes_multiply_risk(self):
        """Breaking changes apply 1.5x risk multiplier to siblings."""
        call_count = 0

        def mock_query(cypher, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "version": "4.0.0",
                        "change_type": "sign_convention",
                        "summary": "COCOS sign flip",
                        "breaking_level": "major",
                    },
                ]
            elif call_count == 2:
                return [
                    {
                        "path": "core_transport/model/profiles_1d/psi",
                        "ids": "core_transport",
                        "cluster_label": "Poloidal Flux",
                        "sibling_changes": 2,
                        "sibling_change_types": ["sign_convention", "units"],
                    },
                ]
            return []

        gc = _make_gc(side_effect=mock_query)
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_changes(
            path="equilibrium/time_slice/profiles_1d/psi"
        )

        assert result["has_breaking_changes"] is True
        sibling = result["co_changing_siblings"][0]
        # risk = co_change_count(2) * 3 * 1.5(multiplier) = 9.0
        assert sibling["risk_score"] == 9.0

    @pytest.mark.asyncio
    async def test_version_range_parameters(self):
        """analyze_dd_changes passes version range parameters to query."""
        gc = _make_gc(query_return=[])
        tool = DDAnalyticsTool(gc)
        result = await tool.analyze_dd_changes(
            path="equilibrium/time_slice/profiles_1d/psi",
            from_version="3.39.0",
            to_version="4.0.0",
        )

        assert result["from_version"] == "3.39.0"
        assert result["to_version"] == "4.0.0"


# ── Tools delegate integration ───────────────────────────────────────────


class TestToolsDelegation:
    """Verify Tools delegates to DDAnalyticsTool and VersionTool."""

    @pytest.mark.asyncio
    async def test_tools_delegates_analyze_dd_coverage(self):
        from imas_codex.tools import Tools

        gc = _make_gc(query_return=[])
        tools = Tools(graph_client=gc)
        result = await tools.analyze_dd_coverage(min_ids_count=5)
        assert result["min_ids_count"] == 5

    @pytest.mark.asyncio
    async def test_tools_delegates_check_dd_units(self):
        from imas_codex.tools import Tools

        gc = _make_gc(query_return=[])
        tools = Tools(graph_client=gc)
        result = await tools.check_dd_units(severity="incompatible")
        assert result["severity_filter"] == "incompatible"

    @pytest.mark.asyncio
    async def test_tools_delegates_analyze_dd_changes(self):
        from imas_codex.tools import Tools

        gc = _make_gc(query_return=[])
        tools = Tools(graph_client=gc)
        result = await tools.analyze_dd_changes(path="eq/ts/p1d/psi")
        assert result["path"] == "eq/ts/p1d/psi"

    @pytest.mark.asyncio
    async def test_tools_delegates_get_dd_changelog(self):
        from imas_codex.tools import Tools

        gc = _make_gc(query_return=[])
        tools = Tools(graph_client=gc)
        result = await tools.get_dd_changelog(limit=10)
        assert result["limit"] == 10
