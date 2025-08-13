"""
Tests for relationship analysis functionality.
"""

from unittest.mock import Mock, patch

import pytest

from imas_mcp.physics_extraction.relationship_analysis import (
    RelationshipEngine,
    RelationshipStrength,
    RelationshipType,
    SemanticAnalyzer,
)


class TestRelationshipStrength:
    """Test relationship strength classification."""

    def test_strength_values(self):
        """Test strength value constants."""
        assert RelationshipStrength.VERY_STRONG == 0.9
        assert RelationshipStrength.STRONG == 0.7
        assert RelationshipStrength.MODERATE == 0.5
        assert RelationshipStrength.WEAK == 0.3
        assert RelationshipStrength.VERY_WEAK == 0.1

    def test_get_category(self):
        """Test strength categorization."""
        assert RelationshipStrength.get_category(0.95) == "very_strong"
        assert RelationshipStrength.get_category(0.75) == "strong"
        assert RelationshipStrength.get_category(0.5) == "moderate"
        assert RelationshipStrength.get_category(0.35) == "weak"
        assert RelationshipStrength.get_category(0.05) == "very_weak"

    def test_get_category_boundary_cases(self):
        """Test boundary cases for categorization."""
        assert RelationshipStrength.get_category(0.85) == "very_strong"
        assert RelationshipStrength.get_category(0.75) == "strong"
        assert RelationshipStrength.get_category(0.6) == "strong"
        assert RelationshipStrength.get_category(0.4) == "moderate"


class TestSemanticAnalyzer:
    """Test semantic relationship analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SemanticAnalyzer()

    def test_analyze_concept(self, analyzer):
        """Test concept analysis from paths."""
        result = analyzer.analyze_concept("core_profiles/profiles_1d/electrons/density")
        assert "concepts" in result
        assert "density" in result["concepts"]
        assert result["primary_domain"] is not None

    def test_calculate_semantic_similarity(self, analyzer):
        """Test semantic similarity calculation."""
        similarity, details = analyzer.calculate_semantic_similarity(
            "core_profiles/profiles_1d/electrons/density",
            "core_profiles/profiles_1d/ion/density",
        )
        assert 0 <= similarity <= 1
        assert "shared_concepts" in details
        assert "density" in details["shared_concepts"]

    def test_physics_concepts_available(self, analyzer):
        """Test that physics concepts are defined."""
        assert hasattr(analyzer, "physics_concepts")
        assert len(analyzer.physics_concepts) > 0
        # Should have key physics concepts
        assert "density" in analyzer.physics_concepts
        assert "temperature" in analyzer.physics_concepts


class TestRelationshipEngine:
    """Test relationship discovery engine."""

    @pytest.fixture
    def mock_catalog(self):
        """Create mock catalog for testing."""
        return {
            "cross_references": {
                "core_profiles/profiles_1d/electrons/density": {
                    "relationships": [
                        {
                            "path": "core_profiles/profiles_1d/ion/density",
                            "type": "cross_reference",
                        }
                    ]
                }
            },
            "physics_concepts": {
                "core_profiles/profiles_1d/electrons/density": {
                    "relevant_paths": ["edge_profiles/profiles_1d/electrons/density"]
                }
            },
            "unit_families": {
                "m^-3": {
                    "paths_using": [
                        "core_profiles/profiles_1d/electrons/density",
                        "core_profiles/profiles_1d/ion/density",
                    ]
                }
            },
        }

    @pytest.fixture
    def engine(self, mock_catalog):
        """Create engine with mock catalog."""
        return RelationshipEngine(mock_catalog)

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.semantic_analyzer is not None
        assert engine.relationships_catalog is not None

    def test_discover_relationships(self, engine):
        """Test relationship discovery."""
        results = engine.discover_relationships(
            "core_profiles/profiles_1d/electrons/density",
            RelationshipType.ALL,
            max_depth=2,
        )
        assert isinstance(results, dict)
        assert "semantic" in results
        assert "structural" in results
        assert "physics" in results
        assert "measurement" in results

    def test_semantic_analysis(self, engine):
        """Test semantic relationship analysis."""
        results = engine._analyze_semantic_relationships(
            "core_profiles/profiles_1d/electrons/density",
            ["core_profiles/profiles_1d/ion/density"],
            max_depth=2,
        )
        assert isinstance(results, list)

    def test_physics_domain_analysis(self, engine):
        """Test physics domain relationship analysis."""
        results = engine._analyze_physics_domain_relationships(
            "core_profiles/profiles_1d/electrons/density",
            ["core_profiles/profiles_1d/ion/density"],
            max_depth=2,
        )
        assert isinstance(results, list)


class TestSuccessMetrics:
    """Test Priority 2 success metrics."""

    def test_semantic_analysis_available(self):
        """Test that semantic analysis is available."""
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze_concept("core_profiles/profiles_1d/electrons/density")
        assert "concepts" in result
        assert len(result["concepts"]) > 0

    def test_relationship_strength_scoring(self):
        """Test relationship strength scoring system."""
        # Test all strength levels are available
        assert hasattr(RelationshipStrength, "VERY_STRONG")
        assert hasattr(RelationshipStrength, "STRONG")
        assert hasattr(RelationshipStrength, "MODERATE")
        assert hasattr(RelationshipStrength, "WEAK")
        assert hasattr(RelationshipStrength, "VERY_WEAK")

    def test_physics_domain_mapping(self):
        """Test physics domain mapping capability."""
        analyzer = SemanticAnalyzer()
        # Should have physics concepts defined
        assert hasattr(analyzer, "physics_concepts")
        assert len(analyzer.physics_concepts) > 0

    def test_relationship_engine_integration(self):
        """Test relationship engine integrates with existing structure."""
        mock_catalog = {
            "cross_references": {},
            "physics_concepts": {},
            "unit_families": {},
        }
        engine = RelationshipEngine(mock_catalog)

        # Should be able to discover relationships
        results = engine.discover_relationships(
            "test/path", RelationshipType.ALL, max_depth=1
        )
        assert isinstance(results, dict)
        # Should have all relationship type categories
        expected_keys = ["semantic", "structural", "physics", "measurement"]
        for key in expected_keys:
            assert key in results


if __name__ == "__main__":
    pytest.main([__file__])
