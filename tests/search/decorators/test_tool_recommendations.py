"""Tests for tool_recommendations.py - tool recommendation decorator."""

import pytest

from imas_mcp.search.decorators.tool_recommendations import (
    analyze_search_results,
    generate_concept_suggestions,
    generate_search_suggestions,
    generate_tool_recommendations,
    recommend_tools,
)


class TestAnalyzeSearchResults:
    """Tests for analyze_search_results function."""

    def test_empty_results(self):
        """Test analyzing empty results."""
        result = analyze_search_results([])
        assert result["result_count"] == 0
        assert result["domains"] == []
        assert result["ids_names"] == []

    def test_extracts_ids_names_from_paths(self):
        """Test extracting IDS names from paths."""
        results = [
            {"path": "equilibrium/boundary/psi"},
            {"path": "core_profiles/temperature"},
            {"path": "equilibrium/time_slice"},
        ]
        result = analyze_search_results(results)

        assert result["result_count"] == 3
        assert "equilibrium" in result["ids_names"]
        assert "core_profiles" in result["ids_names"]

    def test_extracts_physics_domains(self):
        """Test extracting physics domains."""
        results = [
            {"path": "a/b", "physics_domain": "MHD"},
            {"path": "c/d", "physics_domain": "Transport"},
            {"path": "e/f", "physics_domain": "MHD"},
        ]
        result = analyze_search_results(results)

        assert "MHD" in result["domains"]
        assert "Transport" in result["domains"]
        assert len(result["domains"]) == 2

    def test_collects_paths(self):
        """Test collecting paths."""
        results = [
            {"path": "a/b/c"},
            {"path": "d/e/f"},
        ]
        result = analyze_search_results(results)

        assert "a/b/c" in result["paths"]
        assert "d/e/f" in result["paths"]

    def test_handles_missing_fields(self):
        """Test handling results with missing fields."""
        results = [
            {"path": "a/b"},
            {"other": "data"},  # No path
            {"path": "c/d", "physics_domain": "MHD"},
        ]
        result = analyze_search_results(results)

        assert result["result_count"] == 3
        assert len(result["paths"]) == 2


class TestGenerateSearchSuggestions:
    """Tests for generate_search_suggestions function."""

    def test_suggestions_with_results(self):
        """Test suggestions when results are found."""
        context = {
            "result_count": 5,
            "domains": ["MHD"],
            "ids_names": ["equilibrium"],
            "paths": ["a/b", "c/d", "e/f"],
        }
        suggestions = generate_search_suggestions("temperature", context)

        assert len(suggestions) > 0
        assert all("tool" in s for s in suggestions)
        assert all("reason" in s for s in suggestions)
        assert all("description" in s for s in suggestions)

    def test_suggests_cluster_exploration(self):
        """Test that cluster exploration is suggested."""
        context = {
            "result_count": 3,
            "domains": [],
            "ids_names": [],
            "paths": ["a", "b", "c"],
        }
        suggestions = generate_search_suggestions("query", context)

        tool_names = [s["tool"] for s in suggestions]
        assert "search_imas_clusters" in tool_names

    def test_suggests_ids_analysis(self):
        """Test that IDS analysis is suggested for specific IDS."""
        context = {
            "result_count": 2,
            "domains": [],
            "ids_names": ["equilibrium", "core_profiles"],
            "paths": ["a", "b"],
        }
        suggestions = generate_search_suggestions("query", context)

        tool_names = [s["tool"] for s in suggestions]
        assert "list_imas_paths" in tool_names

    def test_no_results_suggestions(self):
        """Test suggestions when no results found."""
        context = {
            "result_count": 0,
            "domains": [],
            "ids_names": [],
            "paths": [],
        }
        suggestions = generate_search_suggestions("temperature", context)

        assert len(suggestions) > 0
        tool_names = [s["tool"] for s in suggestions]
        assert "get_imas_overview" in tool_names
        assert "list_imas_identifiers" in tool_names


class TestGenerateConceptSuggestions:
    """Tests for generate_concept_suggestions function."""

    def test_basic_suggestions(self):
        """Test basic concept suggestions."""
        suggestions = generate_concept_suggestions("plasma physics")

        assert len(suggestions) >= 2
        tool_names = [s["tool"] for s in suggestions]
        assert "search_imas_paths" in tool_names
        assert "list_imas_identifiers" in tool_names

    def test_temperature_density_pressure_suggestions(self):
        """Test suggestions for temperature/density/pressure concepts."""
        suggestions = generate_concept_suggestions("electron temperature")

        reasons = [s["reason"].lower() for s in suggestions]
        # Should have core plasma profiles suggestion
        assert any("core" in r or "profile" in r for r in reasons)

    def test_magnetic_field_suggestions(self):
        """Test suggestions for magnetic field concepts."""
        suggestions = generate_concept_suggestions("magnetic equilibrium")

        tool_names = [s["tool"] for s in suggestions]
        assert "list_imas_paths" in tool_names

    def test_transport_flux_suggestions(self):
        """Test suggestions for transport/flux concepts."""
        suggestions = generate_concept_suggestions("transport flux")

        tool_names = [s["tool"] for s in suggestions]
        assert "search_imas_clusters" in tool_names


class TestGenerateToolRecommendations:
    """Tests for generate_tool_recommendations function."""

    def test_error_result_suggestions(self):
        """Test suggestions for error results."""
        result = {"error": "Something went wrong"}
        recommendations = generate_tool_recommendations(result)

        assert len(recommendations) > 0
        tool_names = [r["tool"] for r in recommendations]
        assert "get_imas_overview" in tool_names

    def test_search_result_suggestions(self):
        """Test suggestions for search results."""
        result = {
            "hits": [
                {"path": "a/b", "physics_domain": "MHD"},
                {"path": "c/d"},
            ],
            "query": "temperature",
        }
        recommendations = generate_tool_recommendations(result)

        assert len(recommendations) > 0

    def test_results_key_variant(self):
        """Test handling 'results' key instead of 'hits'."""
        result = {
            "results": [{"path": "a/b"}],
            "query": "test",
        }
        recommendations = generate_tool_recommendations(result)

        assert len(recommendations) > 0

    def test_concept_result_suggestions(self):
        """Test suggestions for concept explanation results."""
        result = {
            "concept": "plasma temperature",
            "explanation": "...",
        }
        recommendations = generate_tool_recommendations(result)

        assert len(recommendations) > 0

    def test_generic_result_suggestions(self):
        """Test suggestions for generic results."""
        result = {"data": "some data", "other": "stuff"}
        recommendations = generate_tool_recommendations(result)

        assert len(recommendations) > 0
        tool_names = [r["tool"] for r in recommendations]
        assert "search_imas_paths" in tool_names
        assert "get_imas_overview" in tool_names


class TestRecommendToolsDecorator:
    """Tests for recommend_tools decorator."""

    @pytest.mark.asyncio
    async def test_adds_suggestions_to_result(self):
        """Test that suggestions are added to result."""

        @recommend_tools()
        async def test_func():
            return {"data": "value"}

        result = await test_func()

        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)

    @pytest.mark.asyncio
    async def test_respects_max_tools(self):
        """Test that max_tools limit is respected."""

        @recommend_tools(max_tools=2)
        async def test_func():
            return {"hits": [{"path": "a/b"} for _ in range(10)]}

        result = await test_func()

        assert len(result["suggestions"]) <= 2

    @pytest.mark.asyncio
    async def test_preserves_original_result(self):
        """Test that original result is preserved."""

        @recommend_tools()
        async def test_func():
            return {"data": "value", "count": 42}

        result = await test_func()

        assert result["data"] == "value"
        assert result["count"] == 42

    @pytest.mark.asyncio
    async def test_skips_error_results(self):
        """Test that error results don't get suggestions added."""

        @recommend_tools()
        async def test_func():
            return {"error": "Something went wrong"}

        result = await test_func()

        # Error results don't get suggestions field added
        assert "suggestions" not in result
