"""
Tests for abstract metadata properties in result models.

This module tests that all result models properly implement the abstract
metadata properties and that they return correct values.
"""

import time
from datetime import UTC, datetime

import pytest

from imas_mcp import __version__
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

    def test_cannot_instantiate_abstract_class(self):
        """Test that ToolResult cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ToolResult()

    def test_abstract_tool_name_property(self):
        """Test that tool_name is properly abstract."""
        # This is tested implicitly by the fact that concrete classes
        # must implement it to be instantiable
        pass


class TestConcreteResultMetadata:
    """Test metadata properties in concrete result classes."""

    def test_search_result_metadata(self):
        """Test SearchResult metadata properties."""
        result = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="test query",
        )

        # Test abstract property implementation
        assert result.tool_name == "search_imas"

        # Test base class properties
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

        # Test timestamp format (should be ISO format)
        timestamp = result.processing_timestamp
        # Should be parseable as ISO datetime
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed_time, datetime)

    def test_overview_result_metadata(self):
        """Test OverviewResult metadata properties."""
        result = OverviewResult(
            content="Test overview content",
            available_ids=["core_profiles", "equilibrium"],
            hits=[],
            query="test",
        )

        assert result.tool_name == "get_overview"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_concept_result_metadata(self):
        """Test ConceptResult metadata properties."""
        result = ConceptResult(
            concept="test concept",
            explanation="Test explanation",
            detail_level=DetailLevel.BASIC,
            related_topics=[],
            nodes=[],
            query="test",
        )

        assert result.tool_name == "explain_concept"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_structure_result_metadata(self):
        """Test StructureResult metadata properties."""
        result = StructureResult(
            ids_name="core_profiles",
            description="Test IDS description",
            structure={"paths": 100},
            sample_paths=["core_profiles/profiles_1d/temperature"],
            max_depth=3,
            query="test",
        )

        assert result.tool_name == "analyze_ids_structure"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_identifier_result_metadata(self):
        """Test IdentifierResult metadata properties."""
        result = IdentifierResult(
            scope=IdentifierScope.ALL,
            schemas=[],
            paths=[],
            analytics={},
            query="test",
        )

        assert result.tool_name == "explore_identifiers"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_relationship_result_metadata(self):
        """Test RelationshipResult metadata properties."""
        result = RelationshipResult(
            path="core_profiles/profiles_1d/temperature",
            relationship_type=RelationshipType.ALL,
            max_depth=2,
            connections={"related": ["equilibrium/time_slice"]},
            nodes=[],
            query="test",
        )

        assert result.tool_name == "explore_relationships"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_ids_export_metadata(self):
        """Test IDSExport metadata properties."""
        result = IDSExport(
            ids_names=["core_profiles", "equilibrium"],
            include_physics=True,
            include_relationships=True,
        )

        assert result.tool_name == "export_ids"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

    def test_domain_export_metadata(self):
        """Test DomainExport metadata properties."""
        result = DomainExport(
            domain="core_transport",
            domain_info={"description": "Core transport domain"},
            include_cross_domain=False,
            max_paths=10,
        )

        assert result.tool_name == "export_physics_domain"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__


class TestMetadataConsistency:
    """Test consistency of metadata across result types."""

    def test_version_consistency(self):
        """Test that all result types return the same version."""
        results = [
            SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test"),
            OverviewResult(content="test", available_ids=[], hits=[], query="test"),
            ConceptResult(
                concept="test",
                explanation="test",
                detail_level=DetailLevel.BASIC,
                related_topics=[],
                nodes=[],
                query="test",
            ),
            StructureResult(
                ids_name="test",
                description="test",
                structure={},
                sample_paths=[],
                max_depth=0,
                query="test",
            ),
        ]

        versions = [result.version for result in results]
        assert all(v == __version__ for v in versions)
        assert all(v == versions[0] for v in versions)

    def test_timestamp_timing(self):
        """Test that timestamps are cached consistently."""
        # Create result and immediately access timestamp
        result1 = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")
        first_access = result1.processing_timestamp

        # Small delay
        time.sleep(0.01)

        # Second access to same object should return same cached value
        second_access = result1.processing_timestamp
        assert first_access == second_access

        # New object should have different timestamp
        result2 = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")
        new_timestamp = result2.processing_timestamp

        # Different objects should have different timestamps
        assert first_access != new_timestamp

    def test_tool_name_uniqueness(self):
        """Test that each result type has a unique tool_name."""
        results = [
            SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test"),
            OverviewResult(content="test", available_ids=[], hits=[], query="test"),
            ConceptResult(
                concept="test",
                explanation="test",
                detail_level=DetailLevel.BASIC,
                related_topics=[],
                nodes=[],
                query="test",
            ),
            StructureResult(
                ids_name="test",
                description="test",
                structure={},
                sample_paths=[],
                max_depth=0,
                query="test",
            ),
            IdentifierResult(
                scope=IdentifierScope.ALL,
                schemas=[],
                paths=[],
                analytics={},
                query="test",
            ),
            RelationshipResult(
                path="test/path",
                relationship_type=RelationshipType.ALL,
                max_depth=2,
                connections={},
                nodes=[],
                query="test",
            ),
            IDSExport(
                ids_names=["test"], include_physics=True, include_relationships=True
            ),
            DomainExport(
                domain="test", domain_info={}, include_cross_domain=False, max_paths=10
            ),
        ]

        tool_names = [result.tool_name for result in results]

        # All tool names should be unique
        assert len(tool_names) == len(set(tool_names))

        # Verify expected tool names
        expected_names = {
            "search_imas",
            "get_overview",
            "explain_concept",
            "analyze_ids_structure",
            "explore_identifiers",
            "explore_relationships",
            "export_ids",
            "export_physics_domain",
        }
        assert set(tool_names) == expected_names


class TestMetadataInheritance:
    """Test that metadata properties work correctly with inheritance."""

    def test_base_class_properties_accessible(self):
        """Test that base class properties are accessible from concrete classes."""
        result = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")

        # Should be able to access as properties
        assert hasattr(result, "tool_name")
        assert hasattr(result, "processing_timestamp")
        assert hasattr(result, "version")

        # Should return correct types
        assert isinstance(result.tool_name, str)
        assert isinstance(result.processing_timestamp, str)
        assert isinstance(result.version, str)

    def test_property_vs_field_behavior(self):
        """Test that metadata properties behave correctly as properties."""
        result = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")

        # Get initial values
        initial_timestamp = result.processing_timestamp
        initial_version = result.version

        # Small delay
        time.sleep(0.01)

        # Properties should return cached values when accessed again
        new_timestamp = result.processing_timestamp
        same_version = result.version

        # Timestamp should be cached (same value on repeated access)
        assert new_timestamp == initial_timestamp

        # Version should be the same (it's a constant)
        assert same_version == initial_version

    def test_multiple_inheritance_compatibility(self):
        """Test that metadata works with multiple inheritance in result classes."""
        # SearchResult inherits from multiple classes
        result = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")

        # Should have properties from ToolResult
        assert result.tool_name == "search_imas"
        assert isinstance(result.processing_timestamp, str)
        assert result.version == __version__

        # Should also have properties from other base classes
        assert hasattr(result, "hit_count")  # From SearchHits
        assert hasattr(result, "node_count")  # From IdsResult
        assert hasattr(result, "query")  # From QueryContext


class TestMetadataValidation:
    """Test validation of metadata values."""

    def test_tool_name_format(self):
        """Test that tool names follow expected format."""
        results = [
            SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test"),
            OverviewResult(content="test", available_ids=[], hits=[], query="test"),
            ConceptResult(
                concept="test",
                explanation="test",
                detail_level=DetailLevel.BASIC,
                related_topics=[],
                nodes=[],
                query="test",
            ),
        ]

        for result in results:
            tool_name = result.tool_name

            # Should be non-empty string
            assert tool_name
            assert isinstance(tool_name, str)

            # Should be lowercase with underscores (snake_case)
            assert tool_name.islower()
            assert " " not in tool_name  # No spaces

            # Should not contain meaningless suffixes
            meaningless_terms = [
                "improvement",
                "enhancement",
                "advanced",
                "smart",
                "intelligent",
            ]
            for term in meaningless_terms:
                assert term not in tool_name.lower()

    def test_timestamp_format_validation(self):
        """Test that timestamps are in correct ISO format."""
        result = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")
        timestamp = result.processing_timestamp

        # Should be valid ISO format
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            assert isinstance(parsed, datetime)
        except ValueError:
            pytest.fail(f"Timestamp {timestamp} is not valid ISO format")

        # Should include timezone info
        assert (
            timestamp.endswith("Z") or "+" in timestamp or timestamp.endswith("+00:00")
        )

    def test_version_format_validation(self):
        """Test that version follows semantic versioning."""
        result = SearchResult(hits=[], search_mode=SearchMode.AUTO, query="test")
        version = result.version

        # Should be non-empty string
        assert version
        assert isinstance(version, str)

        # Should look like a version (at minimum have dots or be 'development')
        assert "." in version or version == "development"
