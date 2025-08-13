#!/usr/bin/env python3
"""
Tests for YAML-based domain analyzer functionality.
"""

from unittest.mock import Mock

import pytest

from imas_mcp.physics.domain_analyzer import PhysicsDomainAnalyzer


class TestDomainAnalyzerYAMLIntegration:
    """Test that YAML definitions are loaded and used correctly."""

    def test_yaml_loading(self):
        """Test that YAML definitions are loaded correctly."""
        analyzer = PhysicsDomainAnalyzer()

        # Test measurement types loading
        measurement_types = analyzer._measurement_types
        assert "measurement_types" in measurement_types
        assert "density_measurement" in measurement_types["measurement_types"]

        # Test diagnostic methods loading
        diagnostic_methods = analyzer._diagnostic_methods
        assert "diagnostic_methods" in diagnostic_methods
        assert "Thomson scattering" in diagnostic_methods["diagnostic_methods"]

        # Test physics contexts loading
        physics_contexts = analyzer._physics_contexts
        assert "theoretical_contexts" in physics_contexts
        assert "equilibrium" in physics_contexts["theoretical_contexts"]

        # Test research workflows loading
        research_workflows = analyzer._research_workflows
        assert "research_applications" in research_workflows

    def test_measurement_identification(self):
        """Test measurement type identification using YAML definitions."""
        analyzer = PhysicsDomainAnalyzer()

        # Mock result object
        class MockResult:
            def __init__(self, path, data_type="mock"):
                self.path = path
                self.data_type = data_type

        # Test density identification
        result = MockResult("some/path/density/data")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "density_measurement"

        # Test temperature identification
        result = MockResult("some/path/temperature/profile")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "temperature_measurement"

        # Test magnetic field identification
        result = MockResult("some/path/magnetic/field")
        mtype = analyzer._identify_measurement_type(result)
        assert mtype == "magnetic_field_measurement"

    def test_measurement_description(self):
        """Test measurement description using YAML templates."""
        analyzer = PhysicsDomainAnalyzer()

        description = analyzer._describe_measurement("density_measurement", "transport")
        assert "transport" in description
        assert "density" in description.lower()

        description = analyzer._describe_measurement(
            "temperature_measurement", "equilibrium"
        )
        assert "equilibrium" in description
        assert "temperature" in description.lower()

    def test_diagnostic_methods(self):
        """Test diagnostic method information from YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test Thomson scattering description
        description = analyzer._describe_measurement_method(
            "Thomson scattering", "transport"
        )
        assert "laser" in description.lower() or "scattering" in description.lower()

        # Test method outputs
        outputs = analyzer._get_method_outputs("Thomson scattering")
        assert "density_profile" in outputs or "temperature_profile" in outputs

        # Test applicability assessment
        applicability = analyzer._assess_method_applicability(
            "Thomson scattering", "transport"
        )
        assert applicability in ["essential", "high", "moderate", "low"]

    def test_physics_contexts(self):
        """Test physics context information from YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test fundamental equations
        equations = analyzer._get_fundamental_equations("equilibrium")
        assert len(equations) > 0

        # Test physics scales
        scales = analyzer._get_physics_scales("transport")
        assert "spatial" in scales
        assert "temporal" in scales

        # Test governing parameters
        params = analyzer._get_governing_parameters("mhd")
        assert len(params) > 0

    def test_research_applications(self):
        """Test research applications loading from YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test loading research applications
        applications = analyzer._identify_research_applications("equilibrium")
        assert isinstance(applications, list)
        assert len(applications) > 0

        # Each application should have required fields
        for app in applications:
            assert "application" in app
            assert "description" in app

    def test_workflow_extraction(self):
        """Test workflow extraction using YAML definitions."""
        analyzer = PhysicsDomainAnalyzer()

        # Mock search results
        mock_results = [
            Mock(path="density/profile", units="m^-3"),
            Mock(path="temperature/profile", units="eV"),
        ]

        workflows = analyzer._extract_workflows("equilibrium", mock_results)
        assert isinstance(workflows, list)
        assert len(workflows) > 0

        # Check workflow structure
        workflow = workflows[0]
        assert "workflow_name" in workflow
        assert "description" in workflow
        assert "typical_steps" in workflow

    def test_domain_relationships(self):
        """Test domain relationship classification using YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test relationship classification
        relationship = analyzer._classify_domain_relationship(
            "equilibrium", "transport"
        )
        assert isinstance(relationship, str)

        # Test physics connection description
        connection = analyzer._describe_physics_connection("equilibrium", "transport")
        assert isinstance(connection, str)
        assert len(connection) > 0

    def test_quality_recommendations(self):
        """Test data quality recommendations from YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test quality recommendations
        recommendations = analyzer._generate_quality_recommendations("high")
        assert isinstance(recommendations, list)

        recommendations = analyzer._generate_quality_recommendations("moderate")
        assert isinstance(recommendations, list)

        recommendations = analyzer._generate_quality_recommendations("limited")
        assert isinstance(recommendations, list)

    def test_analysis_approach_recommendation(self):
        """Test analysis approach recommendations from YAML."""
        analyzer = PhysicsDomainAnalyzer()

        # Test different complexity combinations
        approach = analyzer._recommend_analysis_approach("advanced", "advanced")
        assert isinstance(approach, str)

        approach = analyzer._recommend_analysis_approach("basic", "basic")
        assert isinstance(approach, str)

        approach = analyzer._recommend_analysis_approach("intermediate", "advanced")
        assert isinstance(approach, str)

    def test_fallback_behavior(self):
        """Test fallback behavior when YAML definitions are missing."""
        analyzer = PhysicsDomainAnalyzer()

        # Test with unknown measurement type
        description = analyzer._describe_measurement(
            "unknown_measurement", "test_domain"
        )
        assert "test_domain" in description

        # Test with unknown diagnostic method
        description = analyzer._describe_measurement_method(
            "unknown_method", "test_domain"
        )
        assert isinstance(description, str)

        # Test with unknown domain
        equations = analyzer._get_fundamental_equations("unknown_domain")
        assert isinstance(equations, list)
        assert len(equations) > 0
