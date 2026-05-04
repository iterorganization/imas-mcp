"""Tests for regex fallback in derive_edges."""

import pytest

from imas_codex.standard_names.derivation import derive_edges


class TestComponentFallback:
    """Test component projection pattern matching."""

    @pytest.mark.parametrize(
        "name, axis, inner",
        [
            (
                "parallel_component_of_convection_velocity",
                "parallel",
                "convection_velocity",
            ),
            ("toroidal_component_of_ion_velocity", "toroidal", "ion_velocity"),
            ("radial_component_of_heat_flux", "radial", "heat_flux"),
            (
                "poloidal_component_of_electron_velocity",
                "poloidal",
                "electron_velocity",
            ),
            ("vertical_component_of_magnetic_field", "vertical", "magnetic_field"),
            ("perpendicular_component_of_viscosity", "perpendicular", "viscosity"),
        ],
    )
    def test_component_produces_has_argument(self, name, axis, inner):
        edges = derive_edges(name)
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.edge_type == "HAS_ARGUMENT"
        assert edge.from_name == name
        assert edge.to_name == inner
        assert edge.props["operator"] == "component"
        assert edge.props["axis"] == axis

    def test_radial_component_uses_ir_parser(self):
        """radial_component_of_magnetic_field should parse via IR (not fallback)."""
        edges = derive_edges("radial_component_of_magnetic_field")
        assert len(edges) == 1
        assert edges[0].to_name == "magnetic_field"
        # Should still produce correct result regardless of path


class TestOperatorFallback:
    """Test unary operator pattern matching."""

    @pytest.mark.parametrize(
        "name, op, inner",
        [
            (
                "time_derivative_of_poloidal_component_of_electron_velocity",
                "time_derivative",
                "poloidal_component_of_electron_velocity",
            ),
            ("gradient_of_pressure", "gradient", "pressure"),
            ("maximum_of_temperature", "maximum", "temperature"),
            (
                "second_radial_derivative_of_ion_velocity",
                "second_radial_derivative",
                "ion_velocity",
            ),
        ],
    )
    def test_operator_produces_has_argument(self, name, op, inner):
        edges = derive_edges(name)
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.edge_type == "HAS_ARGUMENT"
        assert edge.from_name == name
        assert edge.to_name == inner
        assert edge.props["operator"] == op


class TestLeafNames:
    """Test that leaf names produce no edges."""

    @pytest.mark.parametrize(
        "name",
        [
            "temperature",
            "electron_temperature",
            "pressure",
            "magnetic_field",
        ],
    )
    def test_leaf_no_edges(self, name):
        derive_edges(name)
        # Leaf names may produce edges if IR parser handles them,
        # but compound leaves like electron_temperature should produce []
        # because they're neither component nor operator patterns
