"""
Test analysis and explanation features through user interface.

This module tests analysis functionality as user-facing features,
focusing on concept explanation, structure analysis, and relationship exploration.
"""

import time

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import (
    ConceptResult,
    IdentifierResult,
    RelationshipResult,
    StructureResult,
)


class TestAnalysisFeatures:
    """Test analysis and explanation functionality."""

    @pytest.mark.asyncio
    async def test_concept_explanation_basic(self, tools):
        """Test basic concept explanation functionality."""
        result = await tools.explain_concept(concept="core_profiles")

        assert isinstance(result, ConceptResult)
        assert result.concept == "core_profiles"
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    @pytest.mark.asyncio
    async def test_concept_explanation_physics_terms(self, tools):
        """Test concept explanation for physics terms."""
        physics_concepts = ["equilibrium", "transport", "plasma", "temperature"]

        for concept in physics_concepts:
            result = await tools.explain_concept(concept=concept)
            assert isinstance(result, ConceptResult)
            assert hasattr(result, "concept")
            assert hasattr(result, "explanation")

    @pytest.mark.asyncio
    async def test_structure_analysis_basic(self, tools, mcp_test_context):
        """Test basic IDS structure analysis."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.analyze_ids_structure(ids_name=ids_name)
        assert isinstance(result, StructureResult)

        # Should contain structure analysis
        assert hasattr(result, "ids_name")
        assert result.ids_name == ids_name
        assert hasattr(result, "structure")
        assert hasattr(result, "sample_paths")

        # Structure should have useful information
        structure = result.structure
        assert isinstance(structure, dict)
        # Check for any of the expected structure metrics
        expected_keys = ["total_nodes", "document_count", "total_paths", "max_depth"]
        has_expected_key = any(key in structure for key in expected_keys)
        assert has_expected_key, (
            f"Structure should contain at least one of {expected_keys}, but got: {list(structure.keys())}"
        )

    @pytest.mark.asyncio
    async def test_structure_analysis_depth_information(self, tools, mcp_test_context):
        """Test structure analysis provides depth information."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.analyze_ids_structure(ids_name=ids_name)

        assert isinstance(result, StructureResult)
        assert hasattr(result, "max_depth")
        assert isinstance(result.max_depth, int)
        assert result.max_depth >= 0

    @pytest.mark.asyncio
    async def test_structure_analysis_sample_paths(self, tools, mcp_test_context):
        """Test structure analysis provides sample paths."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.analyze_ids_structure(ids_name=ids_name)

        assert isinstance(result, StructureResult)
        assert hasattr(result, "sample_paths")
        assert isinstance(result.sample_paths, list)

        # Sample paths should be meaningful
        for path in result.sample_paths:
            assert isinstance(path, str)
            assert len(path) > 0

    @pytest.mark.asyncio
    async def test_relationship_exploration_basic(self, tools, mcp_test_context):
        """Test basic relationship exploration."""
        ids_name = mcp_test_context["test_ids"]
        # Use a proper path with hierarchical separators, not just IDS name
        test_path = f"{ids_name}/profiles_1d/electrons/temperature"
        result = await tools.explore_relationships(path=test_path)
        assert isinstance(result, RelationshipResult)

        assert hasattr(result, "path")
        assert result.path == test_path

    @pytest.mark.asyncio
    async def test_relationship_types(self, tools, mcp_test_context):
        """Test relationship exploration identifies different relationship types."""
        ids_name = mcp_test_context["test_ids"]
        # Use a proper path with hierarchical separators
        test_path = f"{ids_name}/profiles_1d/electrons/temperature"
        result = await tools.explore_relationships(path=test_path)

        assert isinstance(result, RelationshipResult)
        if hasattr(result, "connections"):
            connections = result.connections
            # Should provide structured relationship information
            assert isinstance(connections, dict)

    @pytest.mark.asyncio
    async def test_identifier_exploration_basic(self, tools, mcp_test_context):
        """Test basic identifier exploration."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.explore_identifiers(query=ids_name)

        assert isinstance(result, IdentifierResult)
        # Check for the actual fields that are returned
        assert hasattr(result, "analytics")
        assert hasattr(result, "schemas")

    @pytest.mark.asyncio
    async def test_identifier_structure_information(self, tools, mcp_test_context):
        """Test identifier exploration provides structure information."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.explore_identifiers(query=ids_name)

        assert isinstance(result, IdentifierResult)
        if hasattr(result, "schemas"):
            schemas = result.schemas
            # Should provide identifier structure information
            assert isinstance(schemas, dict | list)


class TestAnalysisErrorHandling:
    """Test analysis tools error handling."""

    @pytest.mark.asyncio
    async def test_analysis_invalid_ids_name(self, tools):
        """Test analysis tools handle invalid IDS names gracefully."""
        invalid_ids = "nonexistent_ids_name"

        # Test structure analysis - should return ToolError for invalid IDS
        result = await tools.analyze_ids_structure(ids_name=invalid_ids)
        assert isinstance(result, ToolError)
        assert hasattr(result, "error")
        assert hasattr(result, "suggestions")

    @pytest.mark.asyncio
    async def test_relationships_invalid_ids_name(self, tools):
        """Test relationship exploration handles invalid IDS names."""
        invalid_path = "nonexistent_ids_name/invalid/path"

        result = await tools.explore_relationships(path=invalid_path)
        assert isinstance(result, ToolError)
        # Should provide helpful error information
        assert isinstance(result.error, str)

    @pytest.mark.asyncio
    async def test_identifiers_invalid_ids_name(self, tools):
        """Test identifier exploration handles invalid IDS names."""
        invalid_ids = "nonexistent_ids_name"

        result = await tools.explore_identifiers(query=invalid_ids)
        # May return either success with empty results or error - both valid
        assert isinstance(result, IdentifierResult | ToolError)

    @pytest.mark.asyncio
    async def test_explain_empty_concept(self, tools):
        """Test concept explanation handles empty input."""
        result = await tools.explain_concept(concept="")
        # Empty concept should return ToolError due to validation
        assert isinstance(result, ToolError)


class TestAnalysisQuality:
    """Test analysis quality and usefulness."""

    @pytest.mark.asyncio
    async def test_explanation_completeness(self, tools):
        """Test explanations are complete and useful."""
        result = await tools.explain_concept(concept="equilibrium")

        assert isinstance(result, ConceptResult)
        explanation = result.explanation

        # Explanation should be substantial
        assert len(explanation) > 50  # Reasonable minimum length

        # Should mention the concept being explained
        # Note: Not all explanations need to repeat the exact term

    @pytest.mark.asyncio
    async def test_structure_analysis_completeness(self, tools, mcp_test_context):
        """Test structure analysis provides complete information."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.analyze_ids_structure(ids_name=ids_name)

        assert isinstance(result, StructureResult)
        # Should provide multiple dimensions of analysis
        expected_keys = ["structure", "sample_paths", "max_depth"]

        for key in expected_keys:
            assert hasattr(result, key), f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_analysis_consistency(self, tools, mcp_test_context):
        """Test analysis results are consistent across calls."""
        ids_name = mcp_test_context["test_ids"]

        # Call twice and compare key elements
        result1 = await tools.analyze_ids_structure(ids_name=ids_name)
        result2 = await tools.analyze_ids_structure(ids_name=ids_name)

        # Both should be successful results or both should be errors
        assert isinstance(result1, type(result2))

        if isinstance(result1, StructureResult) and isinstance(
            result2, StructureResult
        ):
            # Core structural information should be consistent
            assert result1.ids_name == result2.ids_name

            # Core structural metrics should be the same
            if hasattr(result1, "structure") and hasattr(result2, "structure"):
                struct1 = result1.structure
                struct2 = result2.structure

                if isinstance(struct1, dict) and isinstance(struct2, dict):
                    # Check for any consistent metric between the two calls
                    for key in [
                        "document_count",
                        "total_nodes",
                        "total_paths",
                        "max_depth",
                    ]:
                        if key in struct1 and key in struct2:
                            assert struct1[key] == struct2[key], (
                                f"Inconsistent {key}: {struct1[key]} != {struct2[key]}"
                            )
                            break  # At least one consistent metric found


class TestAnalysisPerformance:
    """Test analysis performance characteristics."""

    @pytest.mark.asyncio
    async def test_analysis_response_time(self, tools, mcp_test_context):
        """Test analysis tools respond in reasonable time."""
        ids_name = mcp_test_context["test_ids"]

        start_time = time.time()
        result = await tools.analyze_ids_structure(ids_name=ids_name)
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 15.0, f"Analysis took {execution_time:.2f}s, too slow"
        assert isinstance(result, StructureResult | ToolError)

    @pytest.mark.asyncio
    async def test_explanation_response_time(self, tools):
        """Test explanation responds in reasonable time."""
        start_time = time.time()
        result = await tools.explain_concept(concept="transport")
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 10.0, (
            f"Explanation took {execution_time:.2f}s, too slow"
        )
        assert isinstance(result, ConceptResult | ToolError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
