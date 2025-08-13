"""
Tests for PhysicsDomainAnalyzer YAML-based refactoring.

This module tests the functionality of the domain analyzer after refactoring
to use YAML definitions instead of hardcoded dictionaries.
"""

from unittest.mock import MagicMock

import pytest

from imas_mcp.physics.domain_analyzer import PhysicsDomainAnalyzer


class TestPhysicsDomainAnalyzerYAML:
    """Test suite for YAML-based domain analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create domain analyzer instance for testing."""
        return PhysicsDomainAnalyzer()

    @pytest.fixture
    def mock_result(self):
        """Create a mock search result for testing."""

        def _create_mock(path, data_type="mock", units=None, documentation=""):
            result = MagicMock()
            result.path = path
            result.data_type = data_type
            result.units = units
            result.documentation = documentation
            return result

        return _create_mock

    def test_yaml_definitions_loaded(self, analyzer):
        """Test that all YAML definition files are loaded correctly."""
        # Test measurement types loading
        assert analyzer._measurement_types is not None
        assert "measurement_types" in analyzer._measurement_types
        assert "density_measurement" in analyzer._measurement_types["measurement_types"]

        # Test diagnostic methods loading
        assert analyzer._diagnostic_methods is not None
        assert "diagnostic_methods" in analyzer._diagnostic_methods
        assert (
            "Thomson scattering" in analyzer._diagnostic_methods["diagnostic_methods"]
        )

        # Test physics contexts loading
        assert analyzer._physics_contexts is not None
        assert "theoretical_contexts" in analyzer._physics_contexts
        assert "equilibrium" in analyzer._physics_contexts["theoretical_contexts"]

        # Test research workflows loading
        assert analyzer._research_workflows is not None
        assert "research_applications" in analyzer._research_workflows

    def test_measurement_type_identification(self, analyzer, mock_result):
        """Test measurement type identification using YAML keyword matching."""
        # Test density identification
        result = mock_result("some/path/density/data")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "density_measurement"

        # Test temperature identification
        result = mock_result("some/path/temperature/profile")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "temperature_measurement"

        # Test magnetic field identification
        result = mock_result("some/path/magnetic/field")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "magnetic_field_measurement"

        # Test current identification (use path without density keyword)
        result = mock_result("plasma/current/profile")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "current_measurement"

        # Test pressure identification
        result = mock_result("equilibrium/pressure/data")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "pressure_measurement"

        # Test fallback to data type
        result = mock_result("unknown/path", data_type="custom_type")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "custom_type"

    def test_measurement_description_templating(self, analyzer):
        """Test measurement description using YAML templates."""
        # Test known measurement type
        description = analyzer._describe_measurement("density_measurement", "transport")
        assert "transport" in description
        assert "density" in description.lower()

        # Test another known type
        description = analyzer._describe_measurement(
            "temperature_measurement", "equilibrium"
        )
        assert "equilibrium" in description
        assert "temperature" in description.lower()

        # Test unknown measurement type falls back to default
        description = analyzer._describe_measurement("unknown_measurement", "heating")
        assert "heating" in description

    def test_diagnostic_method_descriptions(self, analyzer):
        """Test diagnostic method descriptions from YAML."""
        # Test Thomson scattering
        description = analyzer._describe_measurement_method(
            "Thomson scattering", "transport"
        )
        assert len(description) > 0
        # Should contain laser or scattering based on YAML definition
        assert any(word in description.lower() for word in ["laser", "scattering"])

        # Test ECE
        description = analyzer._describe_measurement_method("ECE", "transport")
        assert len(description) > 0

        # Test unknown method
        description = analyzer._describe_measurement_method("Unknown Method", "domain")
        assert "domain" in description

    def test_diagnostic_method_outputs(self, analyzer):
        """Test diagnostic method outputs from YAML."""
        # Test Thomson scattering outputs
        outputs = analyzer._get_method_outputs("Thomson scattering")
        assert len(outputs) > 0
        assert any("profile" in output for output in outputs)

        # Test ECE outputs
        outputs = analyzer._get_method_outputs("ECE")
        assert len(outputs) > 0
        assert any("temperature" in output for output in outputs)

        # Test unknown method
        outputs = analyzer._get_method_outputs("Unknown Method")
        assert outputs == ["measurement_data"]

    def test_diagnostic_method_applicability(self, analyzer):
        """Test diagnostic method applicability assessment."""
        # Test high applicability method
        applicability = analyzer._assess_method_applicability(
            "Thomson scattering", "transport"
        )
        assert applicability in ["essential", "high", "moderate", "low"]

        # Test different domain
        applicability = analyzer._assess_method_applicability("MSE", "equilibrium")
        assert applicability in ["essential", "high", "moderate", "low"]

        # Test unknown method/domain combination
        applicability = analyzer._assess_method_applicability("Unknown", "unknown")
        assert applicability in ["essential", "high", "moderate", "low"]

    def test_physics_context_methods(self, analyzer):
        """Test physics context information retrieval."""
        # Test fundamental equations
        equations = analyzer._get_fundamental_equations("equilibrium")
        assert len(equations) > 0
        assert isinstance(equations, list)

        # Test physics scales
        scales = analyzer._get_physics_scales("transport")
        assert isinstance(scales, dict)
        assert "spatial" in scales
        assert "temporal" in scales

        # Test governing parameters
        params = analyzer._get_governing_parameters("mhd")
        assert len(params) > 0
        assert isinstance(params, list)

        # Test typical regimes
        regimes = analyzer._get_typical_regimes("turbulence")
        assert len(regimes) > 0
        assert isinstance(regimes, list)

        # Test unknown domain falls back to defaults
        equations = analyzer._get_fundamental_equations("unknown_domain")
        assert equations == ["domain-specific equations"]

    def test_theoretical_context_building(self, analyzer):
        """Test theoretical context building with YAML data."""
        # Test with known domain
        context = analyzer._build_theoretical_context(
            "equilibrium", ["phenomenon1", "phenomenon2"]
        )
        assert len(context) > 0
        assert "phenomenon1" in context

        # Test without phenomena
        context = analyzer._build_theoretical_context("transport", [])
        assert len(context) > 0

        # Test unknown domain
        context = analyzer._build_theoretical_context("unknown_domain", ["test"])
        assert "unknown_domain" in context

    def test_domain_relationship_classification(self, analyzer):
        """Test domain relationship classification."""
        # Test known relationship
        relationship = analyzer._classify_domain_relationship(
            "equilibrium", "transport"
        )
        assert relationship in [
            "causal",
            "stability",
            "spatial",
            "mechanism",
            "correlative",
        ]

        # Test unknown relationship
        relationship = analyzer._classify_domain_relationship("unknown1", "unknown2")
        assert relationship == "correlative"

    def test_physics_connection_description(self, analyzer):
        """Test physics connection descriptions."""
        # Test known connection
        connection = analyzer._describe_physics_connection("equilibrium", "transport")
        assert len(connection) > 0

        # Test unknown connection
        connection = analyzer._describe_physics_connection("unknown1", "unknown2")
        assert "unknown1" in connection and "unknown2" in connection

    def test_research_applications_identification(self, analyzer):
        """Test research applications identification."""
        # Test known domain
        applications = analyzer._identify_research_applications("equilibrium")
        assert len(applications) > 0
        assert isinstance(applications, list)
        assert all("application" in app for app in applications)

        # Test unknown domain
        applications = analyzer._identify_research_applications("unknown_domain")
        assert len(applications) > 0
        assert "unknown_domain" in applications[0]["description"]

    def test_quality_recommendations(self, analyzer):
        """Test data quality recommendations."""
        # Test high quality
        recommendations = analyzer._generate_quality_recommendations("high_quality")
        assert len(recommendations) > 0

        # Test moderate quality
        recommendations = analyzer._generate_quality_recommendations("moderate_quality")
        assert len(recommendations) > 0

        # Test limited quality
        recommendations = analyzer._generate_quality_recommendations("limited_quality")
        assert len(recommendations) > 0

        # Test unknown quality
        recommendations = analyzer._generate_quality_recommendations("unknown_quality")
        assert len(recommendations) > 0

    def test_analysis_approach_recommendation(self, analyzer):
        """Test analysis approach recommendations."""
        # Test various complexity combinations
        approach = analyzer._recommend_analysis_approach("advanced", "advanced")
        assert approach in [
            "systematic_expert_analysis",
            "guided_analysis_workflow",
            "standard_analysis_tools",
        ]

        approach = analyzer._recommend_analysis_approach("basic", "basic")
        assert approach in [
            "systematic_expert_analysis",
            "guided_analysis_workflow",
            "standard_analysis_tools",
        ]

        approach = analyzer._recommend_analysis_approach("unknown", "unknown")
        assert approach == "guided_analysis_workflow"  # Default

    def test_extract_measurements_integration(self, analyzer, mock_result):
        """Test the integration of measurement extraction with YAML data."""
        # Create mock search results
        results = [
            mock_result("path/density/data", units="m^-3"),
            mock_result("path/temperature/profile", units="eV"),
            mock_result("path/magnetic/field", units="T"),
        ]

        measurements = analyzer._extract_measurements("transport", results)
        assert len(measurements) > 0

        for measurement in measurements:
            assert "measurement_type" in measurement
            assert "paths" in measurement
            assert "description" in measurement
            assert "data_count" in measurement

    def test_workflow_extraction(self, analyzer, mock_result):
        """Test workflow extraction using YAML definitions."""
        # Create mock search results
        results = [
            mock_result("path/density/data"),
            mock_result("path/temperature/data"),
        ]

        workflows = analyzer._extract_workflows("equilibrium", results)
        assert len(workflows) > 0

        for workflow in workflows:
            assert "workflow_name" in workflow
            assert "description" in workflow
            assert "typical_steps" in workflow

    def test_detailed_physics_context(self, analyzer):
        """Test detailed physics context building."""
        context = analyzer._build_detailed_physics_context("equilibrium")

        assert "fundamental_equations" in context
        assert "key_physics_scales" in context
        assert "governing_parameters" in context
        assert "typical_regimes" in context

        # Test unknown domain
        context = analyzer._build_detailed_physics_context("unknown_domain")
        assert "fundamental_equations" in context

    def test_error_handling_for_missing_yaml(self):
        """Test error handling when YAML files are missing or malformed."""
        from pathlib import Path
        from unittest.mock import patch

        # Test with invalid definitions path by mocking the load_definition_file to fail
        with patch(
            "imas_mcp.physics.domain_analyzer.load_definition_file"
        ) as mock_load:
            mock_load.side_effect = Exception("File not found")
            analyzer = PhysicsDomainAnalyzer(definitions_path=Path("/nonexistent/path"))

            # Should not crash and should have empty dictionaries
            assert analyzer._measurement_types == {}
            assert analyzer._diagnostic_methods == {}
            assert analyzer._physics_contexts == {}
            assert analyzer._research_workflows == {}

            # Methods should still work with fallbacks
            description = analyzer._describe_measurement(
                "density_measurement", "transport"
            )
            assert "transport" in description
