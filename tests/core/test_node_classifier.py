"""Tests for node_classifier — two-pass classification logic.

Covers geometry detection, Pass 1 rules (1–15), and Pass 2 rules (R1–R3).
"""

import pytest

from imas_codex.core.node_classifier import (
    COORDINATE_SEGMENTS,
    GEOMETRY_EXCLUSION_PATTERNS,
    GEOMETRY_PATH_PATTERNS,
    SPATIAL_UNITS,
    _is_geometry_path,
    classify_node_pass1,
    classify_node_pass2,
)

# ──────────────────────────────────────────────────────────────────
# Geometry detection helper
# ──────────────────────────────────────────────────────────────────


class TestIsGeometryPath:
    """Tests for _is_geometry_path()."""

    @pytest.mark.parametrize(
        "path,unit",
        [
            ("pf_active/coil/element/geometry/outline/r", "m"),
            ("pf_active/coil/element/geometry/outline/z", "m"),
            ("interferometer/channel/line_of_sight/first_point/r", "m"),
            ("interferometer/channel/line_of_sight/first_point/z", "m"),
            ("bolometer/channel/detector/centre/r", "m"),
            ("wall/description_2d/limiter/unit/outline/r", "m"),
            ("wall/description_2d/vessel/unit/outline/r", "m"),
            ("magnetics/bpol_probe/position/r", "m"),
            ("magnetics/bpol_probe/position/z", "m"),
            ("magnetics/bpol_probe/position/phi", "rad"),
            ("ece/channel/position/r", "m"),
            ("ic_antennas/antenna/surface_current/r", "m"),
            ("pf_passive/loop/geometry/outline/r", "m"),
            ("first_wall/module/element/outline/r", "m"),
            ("divertor/divertor_module/element/outline/r", "m"),
        ],
    )
    def test_geometry_paths(self, path, unit):
        assert _is_geometry_path(path, unit) is True

    @pytest.mark.parametrize(
        "path,unit",
        [
            # Exclusion: equilibrium outputs
            ("equilibrium/time_slice/boundary/outline/r", "m"),
            ("equilibrium/time_slice/boundary/outline/z", "m"),
            ("equilibrium/time_slice/boundary_separatrix/outline/r", "m"),
            ("equilibrium/time_slice/profiles_1d/grid/r", "m"),
            ("core_profiles/profiles_1d/grid/r", "m"),
            # Exclusion: magnetic axis, pedestal, etc.
            ("equilibrium/time_slice/global_quantities/magnetic_axis/r", "m"),
            ("summary/pedestal/position/r", "m"),
            ("equilibrium/time_slice/profiles_1d/x_point/r", "m"),
            ("equilibrium/time_slice/strike_point/outer_inner/r", "m"),
        ],
    )
    def test_exclusion_paths(self, path, unit):
        assert _is_geometry_path(path, unit) is False

    @pytest.mark.parametrize(
        "path,unit",
        [
            # Non-spatial units
            ("pf_active/coil/element/geometry/outline/r", "eV"),
            ("pf_active/coil/element/geometry/outline/r", "T"),
            ("pf_active/coil/element/geometry/outline/r", "Pa"),
            # Inverse-spatial (densities, not geometry)
            ("pf_active/coil/element/geometry/outline/r", "m^-1"),
            ("pf_active/coil/element/geometry/outline/r", "m^-3"),
        ],
    )
    def test_non_spatial_units_rejected(self, path, unit):
        assert _is_geometry_path(path, unit) is False

    def test_no_geometry_ancestor(self):
        """Physics path with spatial unit but no geometry ancestor."""
        assert _is_geometry_path("equilibrium/time_slice/profiles_1d/r", "m") is False

    def test_no_unit(self):
        assert (
            _is_geometry_path("pf_active/coil/element/geometry/outline/r", None)
            is False
        )

    def test_empty_unit(self):
        assert (
            _is_geometry_path("pf_active/coil/element/geometry/outline/r", "") is False
        )

    def test_unit_dash(self):
        """Dash means dimensionless — not spatial."""
        assert (
            _is_geometry_path("pf_active/coil/element/geometry/outline/r", "-") is False
        )


# ──────────────────────────────────────────────────────────────────
# Pass 1 — build-time classification
# ──────────────────────────────────────────────────────────────────


class TestClassifyNodePass1:
    """Tests for classify_node_pass1()."""

    # --- Rule 1: Error fields ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/time_slice/boundary/psi_error_upper", "psi_error_upper"),
            ("core_profiles/profiles_1d/electrons/te_error_lower", "te_error_lower"),
            ("magnetics/ip_error_index", "ip_error_index"),
        ],
    )
    def test_error_fields(self, path, name):
        assert classify_node_pass1(path, name, data_type="FLT_0D") == "error"

    # --- Rule 2: Metadata subtrees ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/ids_properties/homogeneous_time", "homogeneous_time"),
            ("equilibrium/code/name", "name"),
            ("equilibrium/code/version", "version"),
            ("core_profiles/ids_properties/creation_date", "creation_date"),
        ],
    )
    def test_metadata_subtrees(self, path, name):
        assert classify_node_pass1(path, name, data_type="STR_0D") == "metadata"

    # --- Rule 3: IDS root ---

    def test_ids_root_structural(self):
        assert (
            classify_node_pass1("equilibrium", "equilibrium", data_type="STRUCTURE")
            == "structural"
        )

    # --- Rule 4: 'time' leaf ---

    def test_time_leaf_coordinate(self):
        assert (
            classify_node_pass1(
                "equilibrium/time", "time", data_type="FLT_1D", unit="s"
            )
            == "coordinate"
        )

    def test_time_not_leaf(self):
        """time_slice is not the 'time' leaf."""
        result = classify_node_pass1(
            "equilibrium/time_slice", "time_slice", data_type="STRUCT_ARRAY"
        )
        assert result != "coordinate"

    # --- Rule 5: String types ---

    def test_string_metadata(self):
        assert (
            classify_node_pass1(
                "equilibrium/time_slice/profiles_1d/source",
                "source",
                data_type="STR_0D",
            )
            == "metadata"
        )

    def test_str_1d_metadata(self):
        assert (
            classify_node_pass1("something/labels", "labels", data_type="STR_1D")
            == "metadata"
        )

    # --- Rule 6: Metadata leaves ---

    @pytest.mark.parametrize(
        "name", ["description", "name", "comment", "source", "provider"]
    )
    def test_metadata_leaves(self, name):
        path = f"equilibrium/time_slice/boundary/{name}"
        assert classify_node_pass1(path, name, data_type="STR_0D") == "metadata"

    # --- Rule 7: Known coordinate segments ---

    @pytest.mark.parametrize("seg", ["rho_tor_norm", "psi", "phi", "theta", "r", "z"])
    def test_coordinate_segments(self, seg):
        path = f"core_profiles/profiles_1d/{seg}"
        result = classify_node_pass1(path, seg, data_type="FLT_1D")
        assert result == "coordinate"

    # --- Rule 8: Structural keywords ---

    @pytest.mark.parametrize("name", ["grid_index", "type", "flag", "count", "shape"])
    def test_structural_keywords(self, name):
        path = f"something/{name}"
        result = classify_node_pass1(path, name, data_type="INT_0D")
        assert result == "structural"

    # --- Rule 9a/9b: Physics leaf + geometry ---

    def test_physics_leaf_geometry(self):
        """Spatial unit + geometry ancestor → geometry."""
        path = "pf_active/coil/element/geometry/outline/r"
        result = classify_node_pass1(path, "r", data_type="FLT_1D", unit="m")
        assert result == "geometry"

    def test_physics_leaf_quantity(self):
        """Physics unit but NOT geometry → quantity."""
        path = "core_profiles/profiles_1d/electrons/temperature"
        result = classify_node_pass1(path, "temperature", data_type="FLT_1D", unit="eV")
        assert result == "quantity"

    def test_physics_leaf_no_unit_coordinate_name(self):
        """FLT without unit but known coordinate name → coordinate (Rule 10)."""
        path = "equilibrium/time_slice/profiles_1d/psi"
        result = classify_node_pass1(path, "psi", data_type="FLT_1D")
        assert result == "coordinate"

    def test_physics_leaf_no_unit_quantity(self):
        """FLT without unit, not coordinate name → quantity (Rule 10)."""
        path = "equilibrium/time_slice/global_quantities/beta_pol"
        result = classify_node_pass1(path, "beta_pol", data_type="FLT_0D")
        assert result == "quantity"

    # --- Rule 11a/11b: STRUCTURE + unit ---

    def test_structure_geometry(self):
        """STRUCTURE + spatial unit + geometry ancestor → geometry."""
        path = "pf_active/coil/element/geometry/outline"
        result = classify_node_pass1(path, "outline", data_type="STRUCTURE", unit="m")
        assert result == "geometry"

    def test_structure_quantity(self):
        """STRUCTURE + non-spatial unit → quantity."""
        path = "core_profiles/profiles_1d/electrons/density_fit"
        result = classify_node_pass1(
            path, "density_fit", data_type="STRUCTURE", unit="m^-3"
        )
        assert result == "quantity"

    # --- Rule 12: STRUCTURE no unit → structural ---

    def test_structure_no_unit_structural(self):
        path = "equilibrium/time_slice/profiles_1d"
        result = classify_node_pass1(path, "profiles_1d", data_type="STRUCTURE")
        assert result == "structural"

    # --- Rule 13: INT_0D no unit → quantity ---

    def test_int0d_no_unit_quantity(self):
        path = "equilibrium/time_slice/profiles_1d/n_r"
        result = classify_node_pass1(path, "n_r", data_type="INT_0D")
        # Structural keywords would catch most INT_0D — n_r is not a keyword
        assert result == "quantity"

    # --- Rule 14a/14b: INT + unit ---

    def test_int_geometry(self):
        """INT + spatial unit + geometry ancestor → geometry."""
        path = "pf_active/coil/element/geometry/something"
        result = classify_node_pass1(path, "something", data_type="INT_0D", unit="m")
        assert result == "geometry"

    def test_int_quantity_with_unit(self):
        """INT + non-spatial unit → quantity."""
        path = "something/coils_n"
        result = classify_node_pass1(path, "coils_n", data_type="INT_0D", unit="A")
        assert result == "quantity"

    # --- Rule 15: Remaining INT → structural ---

    def test_int_no_unit_structural(self):
        """INT_1D without unit, no structural keyword → structural."""
        path = "something/indices"
        result = classify_node_pass1(path, "indices", data_type="INT_1D")
        assert result == "structural"


# ──────────────────────────────────────────────────────────────────
# Pass 2 — relational refinement
# ──────────────────────────────────────────────────────────────────


class TestClassifyNodePass2:
    """Tests for classify_node_pass2()."""

    # --- R1: Identifier schema ---

    def test_identifier_schema(self):
        result = classify_node_pass2(
            current_category="structural",
            has_identifier_schema=True,
            is_coordinate_target=False,
        )
        assert result == "identifier"

    def test_identifier_overrides_quantity(self):
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=True,
            is_coordinate_target=False,
        )
        assert result == "identifier"

    # --- R2: Coordinate target with name+unit guard ---

    def test_coordinate_target_canonical_name(self):
        """Known coordinate name → always reclassify to coordinate."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=True,
            name="psi",
            unit="Wb",
        )
        assert result == "coordinate"

    def test_coordinate_target_physics_data_guarded(self):
        """Physics data with meaningful unit — keep current category (Bug 2 fix)."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=True,
            name="temperature",
            unit="eV",
        )
        assert result is None  # None = keep current

    def test_coordinate_target_no_name_fallback(self):
        """Without name arg, old behavior: always reclassify."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=True,
        )
        assert result == "coordinate"

    def test_coordinate_target_no_unit(self):
        """Coordinate target with name but no unit → reclassify to coordinate."""
        result = classify_node_pass2(
            current_category="structural",
            has_identifier_schema=False,
            is_coordinate_target=True,
            name="theta",
        )
        assert result == "coordinate"

    def test_coordinate_target_dimensionless(self):
        """Coordinate target with name and dimensionless unit → reclassify."""
        result = classify_node_pass2(
            current_category="structural",
            has_identifier_schema=False,
            is_coordinate_target=True,
            name="some_index",
            unit="-",
        )
        assert result == "coordinate"

    def test_coordinate_target_not_error(self):
        """Error nodes should NOT be reclassified to coordinate."""
        result = classify_node_pass2(
            current_category="error",
            has_identifier_schema=False,
            is_coordinate_target=True,
        )
        assert result is None

    # --- R3: Validate STRUCTURE+unit ---

    def test_r3_structure_with_data_children_keeps_quantity(self):
        """STRUCTURE+unit with data children → keep quantity (return None)."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=False,
            children_categories=["quantity", "structural"],
            data_type="STRUCTURE",
            unit="eV",
        )
        assert result is None

    def test_r3_no_children_demotes_to_structural(self):
        """STRUCTURE+unit with NO data children → demote to structural."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=False,
            children_categories=[],
            data_type="STRUCTURE",
            unit="eV",
        )
        assert result == "structural"

    def test_r3_struct_array_no_children_demotes(self):
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=False,
            children_categories=[],
            data_type="STRUCT_ARRAY",
            unit="m",
        )
        assert result == "structural"

    def test_no_reclassification(self):
        """No signals → return None (keep current)."""
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=False,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────
# Constants consistency
# ──────────────────────────────────────────────────────────────────


class TestConstants:
    """Verify geometry constants are well-formed."""

    def test_geometry_patterns_non_empty(self):
        assert len(GEOMETRY_PATH_PATTERNS) > 0

    def test_exclusion_patterns_non_empty(self):
        assert len(GEOMETRY_EXCLUSION_PATTERNS) > 0

    def test_spatial_units_subset(self):
        assert SPATIAL_UNITS == {"m", "rad", "m^2", "m^3", "deg"}

    def test_no_overlap_geometry_exclusion(self):
        """Geometry inclusion and exclusion should not overlap."""
        assert GEOMETRY_PATH_PATTERNS.isdisjoint(GEOMETRY_EXCLUSION_PATTERNS)

    def test_coordinate_segments_frozen(self):
        assert isinstance(COORDINATE_SEGMENTS, frozenset)
