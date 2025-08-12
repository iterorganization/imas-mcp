"""
Tests for abstract metadata properties in result models.

This module tests that all result models properly implement the abstract
metadata properties and that they return correct values.
"""

import time
from datetime import UTC, datetime

import pytest

from imas_mcp.models.constants import (
    DetailLevel,
    IdentifierScope,
    RelationshipType,
    SearchMode,
)
from imas_mcp.models.result_models import (
    ConceptResult,
    DomainExport,
    IdentifierResult,
    IDSExport,
    OverviewResult,
    RelationshipResult,
    SearchResult,
    StructureResult,
    ToolResult,
)


class TestAbstractToolResult:
    """Test the abstract ToolResult base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that ToolResult cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ToolResult(query="test")

    def test_abstract_tool_name_property(self):
        """Test that tool_name is an abstract property."""
        # This is verified by the instantiation test above
        assert hasattr(ToolResult, "tool_name")
        assert getattr(ToolResult.tool_name, "__isabstractmethod__", False)


class TestSearchResultMetadata:
    """Test SearchResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = SearchResult(query="test")
        assert result.tool_name == "search_imas"

    def test_processing_timestamp_format(self):
        """Test that processing_timestamp returns valid ISO format."""
        result = SearchResult(query="test")
        timestamp = result.processing_timestamp

        # Should be valid ISO format
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)

        # Should be recent (within last minute)
        now = datetime.now(UTC)
        time_diff = abs((now - parsed).total_seconds())
        assert time_diff < 60  # Within 60 seconds

    def test_version_property(self):
        """Test that version property returns string."""
        result = SearchResult(query="test")
        version = result.version
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be either "development" or a proper version
        assert version == "development" or "." in version

    def test_properties_are_dynamic(self):
        """Test that timestamp is generated fresh each time."""
        result = SearchResult(query="test")
        timestamp1 = result.processing_timestamp

        # Small delay
        time.sleep(0.001)

        timestamp2 = result.processing_timestamp
        # Should be different timestamps
        assert timestamp1 != timestamp2


class TestOverviewResultMetadata:
    """Test OverviewResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = OverviewResult(content="test", query="test")
        assert result.tool_name == "get_overview"


class TestConceptResultMetadata:
    """Test ConceptResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = ConceptResult(
            concept="test", explanation="test explanation", query="test"
        )
        assert result.tool_name == "explain_concept"


class TestStructureResultMetadata:
    """Test StructureResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = StructureResult(
            ids_name="core_profiles", description="test", query="test"
        )
        assert result.tool_name == "analyze_ids_structure"


class TestIdentifierResultMetadata:
    """Test IdentifierResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = IdentifierResult(query="test")
        assert result.tool_name == "explore_identifiers"


class TestRelationshipResultMetadata:
    """Test RelationshipResult metadata properties."""

    def test_tool_name_property(self):
        """Test that tool_name returns correct value."""
        result = RelationshipResult(path="core_profiles/temperature", query="test")
        assert result.tool_name == "explore_relationships"


class TestExportResultsMetadata:
    """Test export result metadata properties."""

    def test_ids_export_tool_name(self):
        """Test that IDSExport tool_name returns correct value."""
        result = IDSExport(ids_names=["core_profiles"], query="test")
        assert result.tool_name == "export_ids"

    def test_domain_export_tool_name(self):
        """Test that DomainExport tool_name returns correct value."""
        result = DomainExport(domain="equilibrium", query="test")
        assert result.tool_name == "export_physics_domain"


class TestMetadataConsistency:
    """Test metadata consistency across all result types."""

    @pytest.mark.parametrize(
        "result_class,init_kwargs,expected_tool_name",
        [
            (SearchResult, {"query": "test"}, "search_imas"),
            (OverviewResult, {"content": "test", "query": "test"}, "get_overview"),
            (
                ConceptResult,
                {"concept": "test", "explanation": "test", "query": "test"},
                "explain_concept",
            ),
            (
                StructureResult,
                {"ids_name": "test", "description": "test", "query": "test"},
                "analyze_ids_structure",
            ),
            (IdentifierResult, {"query": "test"}, "explore_identifiers"),
            (
                RelationshipResult,
                {"path": "test/path", "query": "test"},
                "explore_relationships",
            ),
            (IDSExport, {"ids_names": ["test"], "query": "test"}, "export_ids"),
            (
                DomainExport,
                {"domain": "test", "query": "test"},
                "export_physics_domain",
            ),
        ],
    )
    def test_all_results_have_consistent_metadata(
        self, result_class, init_kwargs, expected_tool_name
    ):
        """Test that all result classes have consistent metadata properties."""
        result = result_class(**init_kwargs)

        # Test tool_name
        assert result.tool_name == expected_tool_name

        # Test processing_timestamp format
        timestamp = result.processing_timestamp
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)

        # Test version
        version = result.version
        assert isinstance(version, str)
        assert len(version) > 0

    def test_multiple_instances_have_unique_timestamps(self):
        """Test that multiple instances get unique timestamps."""
        results = []
        for _ in range(3):
            result = SearchResult(query="test")
            results.append(result.processing_timestamp)
            time.sleep(0.001)  # Small delay

        # All timestamps should be unique
        assert len(set(results)) == len(results)


class TestResultModelSerialization:
    """Test that result models serialize correctly with metadata."""

    def test_search_result_serialization(self):
        """Test SearchResult serializes metadata properties."""
        result = SearchResult(
            query="temperature", search_mode=SearchMode.SEMANTIC, hits=[]
        )

        # Should serialize to dict including computed properties
        data = result.model_dump()

        assert data["tool_name"] == "search_imas"
        assert "processing_timestamp" in data
        assert "version" in data
        assert data["query"] == "temperature"
        assert data["search_mode"] == "semantic"

    def test_concept_result_serialization(self):
        """Test ConceptResult serializes metadata properties."""
        result = ConceptResult(
            concept="poloidal flux",
            explanation="Magnetic flux through poloidal plane",
            detail_level=DetailLevel.INTERMEDIATE,
            query="psi",
        )

        data = result.model_dump()

        assert data["tool_name"] == "explain_concept"
        assert "processing_timestamp" in data
        assert "version" in data
        assert data["concept"] == "poloidal flux"
        assert data["detail_level"] == "intermediate"
