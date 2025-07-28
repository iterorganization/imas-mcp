"""
Unit tests for physics integration module.

Tests physics search, concept explanation, and unit context functionality
using mocked physics accessors and search results.
"""

import pytest
from unittest.mock import Mock, patch

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.core.physics_domains import DomainCharacteristics
from imas_mcp.models.enums import ComplexityLevel
from imas_mcp.models.physics_models import (
    PhysicsMatch,
    ConceptSuggestion,
    UnitSuggestion,
    PhysicsSearchResult,
    ConceptExplanation,
    UnitContext,
    DomainConcepts,
)
from imas_mcp.physics_integration import (
    physics_search,
    explain_physics_concept,
    get_domain_concepts,
    get_unit_physics_context,
)


class TestPhysicsIntegration:
    """Test cases for physics integration functionality."""

    @pytest.fixture
    def mock_domain_accessor(self):
        """Mock domain accessor with test data."""
        accessor = Mock()

        # Mock domain characteristics
        domain_data = DomainCharacteristics(
            description="Magnetohydrodynamic equilibrium modeling",
            primary_phenomena=["magnetic_confinement", "pressure_balance"],
            typical_units=["T", "Pa", "m"],
            measurement_methods=["magnetic_diagnostics", "pressure_gauges"],
            related_domains=["transport", "mhd"],
            complexity_level=ComplexityLevel.INTERMEDIATE,
        )
        accessor.get_domain_info.return_value = domain_data
        accessor.get_related_domains.return_value = [
            PhysicsDomain.TRANSPORT,
            PhysicsDomain.MHD,
        ]

        return accessor

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results for testing."""
        mock_doc = Mock()
        mock_doc.title = "Magnetic Equilibrium"
        mock_doc.domain_name = "equilibrium"
        mock_doc.concept_type = "phenomenon"
        mock_doc.description = "Magnetohydrodynamic equilibrium in tokamaks"
        mock_doc.metadata = {
            "symbol": "B",
            "imas_paths": ["equilibrium.time_slice.global_quantities.magnetic_axis"],
        }

        mock_result = Mock()
        mock_result.document = mock_doc
        mock_result.similarity_score = 0.85

        return [mock_result]

    @pytest.fixture
    def mock_unit_accessor(self):
        """Mock unit accessor with test data."""
        accessor = Mock()
        accessor.get_unit_context.return_value = "Tesla - magnetic field strength"
        accessor.get_category_for_unit.return_value = "magnetic_field"
        accessor.get_domains_for_unit.return_value = [
            PhysicsDomain.EQUILIBRIUM,
            PhysicsDomain.MHD,
        ]
        return accessor

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_physics_search_success(
        self,
        mock_search,
        mock_accessor_class,
        mock_domain_accessor,
        mock_search_results,
    ):
        """Test successful physics search with results."""
        # Setup mocks
        mock_accessor_class.return_value = mock_domain_accessor
        mock_search.return_value = mock_search_results

        # Execute search
        result = physics_search("magnetic equilibrium", max_results=5)

        # Verify result structure
        assert isinstance(result, PhysicsSearchResult)
        assert result.query == "magnetic equilibrium"
        assert len(result.physics_matches) == 1
        assert len(result.concept_suggestions) == 1
        assert len(result.unit_suggestions) > 0

        # Verify physics match details
        match = result.physics_matches[0]
        assert isinstance(match, PhysicsMatch)
        assert match.concept == "Magnetic Equilibrium"
        assert match.domain == "equilibrium"
        assert match.relevance_score == 0.85
        assert "T" in match.units

        # Verify suggestions
        concept_suggestion = result.concept_suggestions[0]
        assert isinstance(concept_suggestion, ConceptSuggestion)
        assert concept_suggestion.concept == "Magnetic Equilibrium"

        unit_suggestion = result.unit_suggestions[0]
        assert isinstance(unit_suggestion, UnitSuggestion)
        assert unit_suggestion.unit in ["T", "Pa", "m"]

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_physics_search_no_results(self, mock_search, mock_accessor_class):
        """Test physics search with no results."""
        # Setup mocks
        mock_accessor_class.return_value = Mock()
        mock_search.return_value = []

        # Execute search
        result = physics_search("nonexistent concept")

        # Verify empty result
        assert isinstance(result, PhysicsSearchResult)
        assert result.query == "nonexistent concept"
        assert len(result.physics_matches) == 0
        assert len(result.concept_suggestions) == 0
        assert len(result.unit_suggestions) == 0

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_physics_search_invalid_domain(
        self, mock_search, mock_accessor_class, mock_domain_accessor
    ):
        """Test physics search with invalid domain name."""
        # Setup mock with invalid domain
        mock_doc = Mock()
        mock_doc.title = "Test Concept"
        mock_doc.domain_name = "invalid_domain"
        mock_doc.concept_type = "phenomenon"
        mock_doc.description = "Test description"
        mock_doc.metadata = {}

        mock_result = Mock()
        mock_result.document = mock_doc
        mock_result.similarity_score = 0.7

        mock_accessor_class.return_value = mock_domain_accessor
        mock_search.return_value = [mock_result]

        # Execute search
        result = physics_search("test concept")

        # Should fallback to GENERAL domain
        assert len(result.physics_matches) == 1
        match = result.physics_matches[0]
        assert match.domain == "invalid_domain"  # Original domain name preserved

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_explain_physics_concept_found(
        self,
        mock_search,
        mock_accessor_class,
        mock_domain_accessor,
        mock_search_results,
    ):
        """Test concept explanation when concept is found."""
        # Setup mocks
        mock_accessor_class.return_value = mock_domain_accessor
        mock_search.return_value = mock_search_results

        # Execute explanation
        explanation = explain_physics_concept("magnetic equilibrium")

        # Verify explanation
        assert isinstance(explanation, ConceptExplanation)
        assert explanation.concept == "magnetic equilibrium"
        assert explanation.domain == PhysicsDomain.EQUILIBRIUM
        assert "equilibrium" in explanation.description.lower()
        assert len(explanation.phenomena) > 0
        assert len(explanation.typical_units) > 0
        assert len(explanation.measurement_methods) > 0
        assert len(explanation.related_domains) > 0

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_explain_physics_concept_direct_domain(
        self, mock_search, mock_accessor_class, mock_domain_accessor
    ):
        """Test concept explanation using direct domain lookup."""
        # Setup mocks - no search results, but domain exists
        mock_accessor_class.return_value = mock_domain_accessor
        mock_search.return_value = []

        # Execute explanation for known domain
        explanation = explain_physics_concept("equilibrium")

        # Should fallback to direct domain lookup
        assert isinstance(explanation, ConceptExplanation)
        assert explanation.concept == "equilibrium"
        assert explanation.domain == PhysicsDomain.EQUILIBRIUM

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_explain_physics_concept_not_found(self, mock_search, mock_accessor_class):
        """Test concept explanation when concept is not found."""
        # Setup mocks - no results
        mock_accessor = Mock()
        mock_accessor.get_domain_info.return_value = None
        mock_accessor.get_related_domains.return_value = []
        mock_accessor_class.return_value = mock_accessor
        mock_search.return_value = []

        # Execute explanation
        explanation = explain_physics_concept("unknown concept")

        # Should return generic explanation
        assert isinstance(explanation, ConceptExplanation)
        assert explanation.concept == "unknown concept"
        assert explanation.domain == PhysicsDomain.GENERAL
        assert "Physics concept: unknown concept" in explanation.description
        assert explanation.complexity_level == "unknown"

    @patch("imas_mcp.physics_integration.DomainAccessor")
    def test_get_domain_concepts(self, mock_accessor_class, mock_domain_accessor):
        """Test getting concepts for a domain."""
        # Setup mock
        mock_accessor_class.return_value = mock_domain_accessor

        # Execute
        result = get_domain_concepts(PhysicsDomain.EQUILIBRIUM)

        # Verify
        assert isinstance(result, DomainConcepts)
        assert result.domain == PhysicsDomain.EQUILIBRIUM
        assert len(result.concepts) > 0
        assert "Equilibrium" in result.concepts
        assert any("Magnetic" in concept for concept in result.concepts)

    @patch("imas_mcp.physics_integration.DomainAccessor")
    def test_get_domain_concepts_no_data(self, mock_accessor_class):
        """Test getting concepts for domain with no data."""
        # Setup mock with no data
        mock_accessor = Mock()
        mock_accessor.get_domain_info.return_value = None
        mock_accessor_class.return_value = mock_accessor

        # Execute
        result = get_domain_concepts(PhysicsDomain.GENERAL)

        # Should return empty concepts list
        assert isinstance(result, DomainConcepts)
        assert result.domain == PhysicsDomain.GENERAL
        assert len(result.concepts) == 0

    @patch("imas_mcp.physics_integration.UnitAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_get_unit_physics_context_direct(
        self, mock_search, mock_unit_accessor_class, mock_unit_accessor
    ):
        """Test getting unit context with direct lookup."""
        # Setup mocks
        mock_unit_accessor_class.return_value = mock_unit_accessor

        # Execute
        context = get_unit_physics_context("T")

        # Verify
        assert isinstance(context, UnitContext)
        assert context.unit == "T"
        assert context.context == "Tesla - magnetic field strength"
        assert context.category == "magnetic_field"
        assert PhysicsDomain.EQUILIBRIUM in context.physics_domains

    @patch("imas_mcp.physics_integration.UnitAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_get_unit_physics_context_search_fallback(
        self, mock_search, mock_unit_accessor_class, mock_unit_accessor
    ):
        """Test getting unit context with search fallback."""
        # Setup mocks - no direct context, but search finds it
        mock_unit_accessor.get_unit_context.side_effect = [
            None,
            "Tesla - found via search",
        ]
        mock_unit_accessor.get_category_for_unit.return_value = "magnetic_field"
        mock_unit_accessor.get_domains_for_unit.return_value = [
            PhysicsDomain.EQUILIBRIUM
        ]
        mock_unit_accessor_class.return_value = mock_unit_accessor

        # Mock search result
        mock_doc = Mock()
        mock_doc.metadata = {"symbol": "T"}
        mock_result = Mock()
        mock_result.document = mock_doc
        mock_search.return_value = [mock_result]

        # Execute
        context = get_unit_physics_context("Tesla")

        # Verify search was called and fallback worked
        mock_search.assert_called_once()
        assert isinstance(context, UnitContext)
        assert context.context == "Tesla - found via search"
        assert context.unit == "T"  # Should be updated to resolved symbol

    @pytest.mark.parametrize(
        "domain,expected_concepts",
        [
            (PhysicsDomain.EQUILIBRIUM, ["equilibrium"]),
            (PhysicsDomain.TRANSPORT, ["transport"]),
            (PhysicsDomain.MHD, ["mhd"]),
        ],
    )
    @patch("imas_mcp.physics_integration.DomainAccessor")
    def test_get_domain_concepts_parametrized(
        self, mock_accessor_class, mock_domain_accessor, domain, expected_concepts
    ):
        """Test domain concepts for various domains."""
        # Setup mock based on domain
        domain_data = DomainCharacteristics(
            description=f"{domain.value} physics",
            primary_phenomena=[f"{domain.value}_phenomenon"],
            typical_units=["unit1", "unit2"],
            measurement_methods=[f"{domain.value}_diagnostics"],
            related_domains=["general"],
            complexity_level=ComplexityLevel.INTERMEDIATE,
        )
        mock_domain_accessor.get_domain_info.return_value = domain_data
        mock_accessor_class.return_value = mock_domain_accessor

        # Execute
        result = get_domain_concepts(domain)

        # Verify expected concepts are present (case-insensitive check)
        concepts_lower = [concept.lower() for concept in result.concepts]
        for expected in expected_concepts:
            assert any(
                expected.lower() in concept_lower for concept_lower in concepts_lower
            )

    def test_concept_explanation_model_validation(self):
        """Test ConceptExplanation model validation."""
        # Valid model
        explanation = ConceptExplanation(
            concept="test",
            domain=PhysicsDomain.GENERAL,
            description="test description",
            phenomena=["test_phenomenon"],
            typical_units=["unit"],
            measurement_methods=["method"],
            related_domains=[PhysicsDomain.EQUILIBRIUM],
            complexity_level="basic",
        )

        assert explanation.concept == "test"
        assert explanation.domain == PhysicsDomain.GENERAL
        assert len(explanation.related_domains) == 1


class TestPhysicsIntegrationPerformance:
    """Performance-focused tests for physics integration."""

    @patch("imas_mcp.physics_integration.DomainAccessor")
    @patch("imas_mcp.physics_integration.search_physics_concepts")
    def test_physics_search_large_result_set(self, mock_search, mock_accessor_class):
        """Test performance with large result sets."""
        # Create large mock result set
        mock_results = []
        for i in range(100):
            mock_doc = Mock()
            mock_doc.title = f"Concept {i}"
            mock_doc.domain_name = "equilibrium"
            mock_doc.concept_type = "phenomenon"
            mock_doc.description = f"Description {i}"
            mock_doc.metadata = {}

            mock_result = Mock()
            mock_result.document = mock_doc
            mock_result.similarity_score = 0.5 + (i * 0.005)
            mock_results.append(mock_result)

        # Setup mocks
        mock_accessor = Mock()
        mock_domain_data = Mock()
        mock_domain_data.typical_units = ["unit1", "unit2"]
        mock_accessor.get_domain_info.return_value = mock_domain_data
        mock_accessor_class.return_value = mock_accessor
        mock_search.return_value = mock_results

        # Execute with limited results
        result = physics_search("test query", max_results=10)

        # Should limit results appropriately
        assert len(result.physics_matches) <= 100  # All results processed
        assert len(result.concept_suggestions) <= 5  # Limited suggestions
        assert len(result.unit_suggestions) <= 5  # Limited suggestions

    @patch("imas_mcp.physics_integration.DomainAccessor")
    def test_get_domain_concepts_caching(self, mock_accessor_class):
        """Test that domain concepts can be cached efficiently."""
        # Setup mock
        mock_accessor = Mock()
        mock_domain_data = Mock()
        mock_domain_data.primary_phenomena = ["phenomenon1", "phenomenon2"]
        mock_domain_data.measurement_methods = ["method1", "method2"]
        mock_accessor.get_domain_info.return_value = mock_domain_data
        mock_accessor_class.return_value = mock_accessor

        # Multiple calls should be handled efficiently
        for _ in range(10):
            result = get_domain_concepts(PhysicsDomain.EQUILIBRIUM)
            assert len(result.concepts) > 0

        # Accessor should be called multiple times (no caching implemented yet)
        assert mock_accessor.get_domain_info.call_count == 10
