"""Tests for standard name family detection and axis ordering."""

import pytest

from imas_codex.standard_names.families import (
    AXIS_ORDER,
    DD_DERIVATIVE_MAP,
    detect_families,
    sort_by_axis_convention,
)


class TestAxisOrdering:
    """Phase 6: Component axis ordering convention."""

    def test_cylindrical_right_handed_ordering(self):
        """T17: Children ordered R, φ, Z (right-handed cylindrical)."""
        children = [
            {"name": "vertical_component_of_B", "axis": "vertical"},
            {"name": "toroidal_component_of_B", "axis": "toroidal"},
            {"name": "radial_component_of_B", "axis": "radial"},
        ]
        ordered = sort_by_axis_convention(children)
        assert [c["axis"] for c in ordered] == ["radial", "toroidal", "vertical"]

    def test_geometric_coordinate_ordering(self):
        """T18: Geometric coordinates ordered R, φ, Z."""
        children = [
            {"name": "vertical_position", "axis": "z"},
            {"name": "toroidal_angle", "axis": "phi"},
            {"name": "radial_position", "axis": "r"},
        ]
        ordered = sort_by_axis_convention(children)
        assert [c["axis"] for c in ordered] == ["r", "phi", "z"]

    def test_field_aligned_ordering(self):
        """T19: Field-aligned ordering: parallel before perpendicular."""
        children = [
            {"name": "perpendicular_component_of_v", "axis": "perpendicular"},
            {"name": "parallel_component_of_v", "axis": "parallel"},
        ]
        ordered = sort_by_axis_convention(children)
        assert [c["axis"] for c in ordered] == ["parallel", "perpendicular"]

    def test_normalized_variants_inherit_ordering(self):
        """Normalized axes follow their parent ordering."""
        children = [
            {"axis": "normalized_perpendicular"},
            {"axis": "normalized_radial"},
            {"axis": "normalized_parallel"},
        ]
        ordered = sort_by_axis_convention(children)
        axes = [c["axis"] for c in ordered]
        assert axes == [
            "normalized_radial",
            "normalized_parallel",
            "normalized_perpendicular",
        ]

    def test_unknown_axes_sort_last(self):
        """Axes not in AXIS_ORDER sort after known axes."""
        children = [
            {"axis": "unknown_axis"},
            {"axis": "radial"},
        ]
        ordered = sort_by_axis_convention(children)
        assert [c["axis"] for c in ordered] == ["radial", "unknown_axis"]

    def test_original_list_unchanged(self):
        """sort_by_axis_convention returns a new list."""
        original = [{"axis": "vertical"}, {"axis": "radial"}]
        ordered = sort_by_axis_convention(original)
        assert original[0]["axis"] == "vertical"  # unchanged
        assert ordered[0]["axis"] == "radial"

    def test_cartesian_ordering(self):
        """Cartesian axes: x, y, z."""
        children = [{"axis": "y"}, {"axis": "x"}, {"axis": "z"}]
        ordered = sort_by_axis_convention(children, axis_key="axis")
        assert [c["axis"] for c in ordered] == ["x", "y", "z"]

    def test_right_handedness_r_phi_z(self):
        """Critical: R × φ = Z, confirming right-handed system."""
        assert AXIS_ORDER["radial"] < AXIS_ORDER["toroidal"] < AXIS_ORDER["vertical"]
        assert AXIS_ORDER["r"] < AXIS_ORDER["phi"] < AXIS_ORDER["z"]


class TestFamilyDetection:
    """Phase 1: Family detection from DD paths."""

    def test_detect_physical_vector_family(self):
        """T1: Physical vector detected from j_tor/j_parallel/j_phi."""
        items = [
            {"path": "equilibrium/time_slice/profiles_1d/j_tor", "unit": "A/m^2"},
            {"path": "equilibrium/time_slice/profiles_1d/j_parallel", "unit": "A/m^2"},
            {"path": "equilibrium/time_slice/profiles_1d/j_phi", "unit": "A/m^2"},
        ]
        families = detect_families(items)
        assert len(families) >= 1
        f = [f for f in families if f.family_type == "physical_vector"][0]
        assert f.unit_uniform is True
        assert len(f.members) >= 2

    def test_detect_geometric_coordinate_family(self):
        """T2: Geometric coordinate detected from position/r, z, phi."""
        items = [
            {"path": "barometry/gauge/position/r", "unit": "m"},
            {"path": "barometry/gauge/position/z", "unit": "m"},
            {"path": "barometry/gauge/position/phi", "unit": "rad"},
        ]
        families = detect_families(items)
        assert len(families) >= 1
        f = [f for f in families if f.family_type == "geometric_coordinate"][0]
        assert f.unit_uniform is False
        assert f.parent_name == "position"

    def test_detect_derivative_family(self):
        """T3: Derivative family detected from shared denominator."""
        items = [
            {
                "path": "equilibrium/time_slice/profiles_1d/darea_dpsi",
                "unit": "m^2/Wb",
            },
            {
                "path": "equilibrium/time_slice/profiles_1d/dpressure_dpsi",
                "unit": "Pa/Wb",
            },
            {
                "path": "equilibrium/time_slice/profiles_1d/dvolume_dpsi",
                "unit": "m^3/Wb",
            },
        ]
        families = detect_families(items)
        assert len(families) >= 1
        f = [f for f in families if f.family_type == "derivative"][0]
        assert len(f.members) == 3

    def test_no_false_family_for_unrelated_paths(self):
        """T4: No false family from unrelated paths."""
        items = [
            {"path": "equilibrium/time_slice/profiles_1d/pressure", "unit": "Pa"},
            {"path": "equilibrium/time_slice/profiles_1d/q", "unit": ""},
            {"path": "equilibrium/time_slice/profiles_1d/phi", "unit": "Wb"},
        ]
        families = detect_families(items)
        # phi alone doesn't form a 2+ member family
        vector_fams = [
            f
            for f in families
            if f.family_type in ("physical_vector", "geometric_coordinate")
        ]
        assert len(vector_fams) == 0

    def test_derivative_map_resolves_compound_denominators(self):
        """DD_DERIVATIVE_MAP handles compound denominator patterns."""
        assert DD_DERIVATIVE_MAP["dpsi_drho_tor"] == (
            "poloidal_magnetic_flux",
            "normalised_toroidal_flux_coordinate",
        )

    def test_detect_families_minimum_two_members(self):
        """Families require at least 2 members."""
        items = [
            {"path": "some/parent/r", "unit": "m"},
            {"path": "some/parent/pressure", "unit": "Pa"},
        ]
        families = detect_families(items)
        # Only 1 axis member — no family
        assert len(families) == 0
