#!/usr/bin/env python3
"""
Quick test to verify the YAML-based domain analyzer refactoring.
"""

from imas_mcp.physics_extraction.domain_analyzer import PhysicsDomainAnalyzer


def test_yaml_loading():
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

    print("✓ All YAML definitions loaded successfully")


def test_measurement_identification():
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

    print("✓ Measurement type identification working correctly")


def test_measurement_description():
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

    print("✓ Measurement description templating working correctly")


def test_diagnostic_methods():
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

    print("✓ Diagnostic method information working correctly")


def test_physics_contexts():
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

    print("✓ Physics context information working correctly")


if __name__ == "__main__":
    test_yaml_loading()
    test_measurement_identification()
    test_measurement_description()
    test_diagnostic_methods()
    test_physics_contexts()
    print(
        "\n🎉 All tests passed! YAML-based domain analyzer refactoring is working correctly."
    )
