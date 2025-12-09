"""Tests for overview_tool.py internal methods."""

from unittest.mock import patch

import pytest

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.tools.overview_tool import OverviewTool


class TestOverviewToolInternals:
    """Tests for internal methods of OverviewTool."""

    @pytest.fixture
    def overview_tool(self):
        """Create overview tool instance."""
        return OverviewTool()

    def test_get_mcp_tools_returns_list(self, overview_tool):
        """_get_mcp_tools returns a list of tool names."""
        tools = overview_tool._get_mcp_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_physics_domains_empty_catalog(self, overview_tool):
        """Empty catalog returns empty domains."""
        overview_tool._ids_catalog = {}

        domains = overview_tool._get_physics_domains()

        assert domains == {}

    def test_get_physics_domains_groups_by_domain(self, overview_tool):
        """Domains are correctly grouped via PhysicsDomainCategorizer."""
        overview_tool._ids_catalog = {
            "ids_catalog": {
                "equilibrium": {},
                "core_profiles": {},
                "edge_profiles": {},
            }
        }

        # Mock the physics_categorizer to return specific domains
        def mock_get_domain(ids_name):
            domain_map = {
                "equilibrium": PhysicsDomain.MHD,
                "core_profiles": PhysicsDomain.TRANSPORT,
                "edge_profiles": PhysicsDomain.TRANSPORT,
            }
            return domain_map.get(ids_name, PhysicsDomain.GENERAL)

        with patch(
            "imas_mcp.tools.overview_tool.physics_categorizer.get_domain_for_ids",
            side_effect=mock_get_domain,
        ):
            domains = overview_tool._get_physics_domains()

        assert "mhd" in domains
        assert "transport" in domains
        assert len(domains["transport"]) == 2

    def test_filter_ids_by_query_empty_catalog(self, overview_tool):
        """Empty catalog returns empty result."""
        overview_tool._ids_catalog = {}

        result = overview_tool._filter_ids_by_query("test")

        assert result == []

    def test_filter_ids_by_query_matches_name(self, overview_tool):
        """Query matches IDS name."""
        overview_tool._ids_catalog = {
            "ids_catalog": {
                "equilibrium": {
                    "description": "MHD equilibrium",
                },
                "core_profiles": {
                    "description": "Core plasma profiles",
                },
            }
        }

        result = overview_tool._filter_ids_by_query("equilib")

        assert "equilibrium" in result

    def test_filter_ids_by_query_matches_description(self, overview_tool):
        """Query matches description."""
        overview_tool._ids_catalog = {
            "ids_catalog": {
                "equilibrium": {
                    "description": "MHD equilibrium",
                },
            }
        }

        result = overview_tool._filter_ids_by_query("mhd")

        assert "equilibrium" in result

    def test_filter_ids_by_query_matches_domain(self, overview_tool):
        """Query matches physics domain via PhysicsDomainCategorizer."""
        overview_tool._ids_catalog = {
            "ids_catalog": {
                "core_profiles": {
                    "description": "Profiles",
                },
            }
        }

        # Mock the physics_categorizer to return transport domain
        def mock_get_domain(ids_name):
            if ids_name == "core_profiles":
                return PhysicsDomain.TRANSPORT
            return PhysicsDomain.GENERAL

        with patch(
            "imas_mcp.tools.overview_tool.physics_categorizer.get_domain_for_ids",
            side_effect=mock_get_domain,
        ):
            result = overview_tool._filter_ids_by_query("transport")

        assert "core_profiles" in result

    def test_get_complexity_rankings_empty(self, overview_tool):
        """Empty catalog returns empty rankings."""
        overview_tool._ids_catalog = {}

        rankings = overview_tool._get_complexity_rankings()

        assert rankings == []

    def test_get_complexity_rankings_sorted_descending(self, overview_tool):
        """Rankings are sorted by path count descending."""
        overview_tool._ids_catalog = {
            "ids_catalog": {
                "small_ids": {"path_count": 10},
                "large_ids": {"path_count": 1000},
                "medium_ids": {"path_count": 100},
            }
        }

        rankings = overview_tool._get_complexity_rankings()

        assert rankings[0][0] == "large_ids"
        assert rankings[-1][0] == "small_ids"

    def test_generate_recommendations_magnetic_query(self, overview_tool):
        """Magnetic field query generates specific recommendations."""
        recs = overview_tool._generate_recommendations(
            "magnetic field", ["equilibrium"]
        )

        assert len(recs) > 0

    def test_generate_recommendations_temperature_query(self, overview_tool):
        """Temperature query generates specific recommendations."""
        recs = overview_tool._generate_recommendations("temperature", [])

        assert len(recs) > 0

    def test_generate_recommendations_diagnostic_query(self, overview_tool):
        """Diagnostic query generates specific recommendations."""
        recs = overview_tool._generate_recommendations("diagnostic", [])

        assert len(recs) > 0

    def test_generate_recommendations_identifier_query(self, overview_tool):
        """Identifier query generates specific recommendations."""
        recs = overview_tool._generate_recommendations("identifier", [])

        assert len(recs) > 0

    def test_generate_recommendations_single_ids(self, overview_tool):
        """Single IDS generates list_imas_paths recommendation."""
        recs = overview_tool._generate_recommendations(None, ["equilibrium"])

        assert any("list_imas_paths" in str(r) for r in recs)

    def test_generate_recommendations_limited_to_six(self, overview_tool):
        """Recommendations are limited to 6 items."""
        recs = overview_tool._generate_recommendations(None, ["a", "b", "c", "d"])

        assert len(recs) <= 6
