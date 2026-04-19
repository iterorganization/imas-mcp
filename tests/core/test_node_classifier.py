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

    def test_no_identifier_schema_keeps_quantity(self):
        """Quantity node WITHOUT HAS_IDENTIFIER_SCHEMA must NOT be reclassified.

        Regression test for the OPTIONAL MATCH + count(*) > 0 bug where
        every enrichable node was incorrectly reclassified to identifier.
        """
        result = classify_node_pass2(
            current_category="quantity",
            has_identifier_schema=False,
            is_coordinate_target=False,
            name="pressure",
            unit="Pa",
        )
        assert result is None  # None = keep current category

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


# ──────────────────────────────────────────────────────────────────
# Fit-artifact classification (Pass 1 rules F1, F2)
# ──────────────────────────────────────────────────────────────────


class TestFitArtifactPass1:
    """Tests for fit_artifact classification in Pass 1."""

    # --- Positive cases: fit diagnostics (Rule F1) ---

    @pytest.mark.parametrize(
        "path,name",
        [
            (
                "core_profiles/profiles_1d/electrons/density_fit/chi_squared",
                "chi_squared",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/residual",
                "residual",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/covariance",
                "covariance",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/fitting_weight",
                "fitting_weight",
            ),
            (
                "equilibrium/time_slice/profiles_1d/q_fit/fit_type",
                "fit_type",
            ),
            (
                "core_profiles/profiles_1d/ion/temperature_fit/chi_squared",
                "chi_squared",
            ),
            (
                "core_profiles/profiles_1d/electrons/pressure_fit/residual",
                "residual",
            ),
            (
                "edge_profiles/ggd/electrons/density_fit/fitting_weight",
                "fitting_weight",
            ),
        ],
    )
    def test_fit_diagnostics(self, path, name):
        assert classify_node_pass1(path, name, data_type="FLT_0D") == "fit_artifact"

    # --- Positive cases: fit children under *_fit parent (Rule F2) ---

    @pytest.mark.parametrize(
        "path,name",
        [
            (
                "core_profiles/profiles_1d/electrons/density_fit/measured",
                "measured",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/reconstructed",
                "reconstructed",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/weight",
                "weight",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/time_measurement",
                "time_measurement",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_fit/rho_tor_norm",
                "rho_tor_norm",
            ),
            (
                "equilibrium/time_slice/profiles_1d/q_fit/measured",
                "measured",
            ),
            (
                "core_profiles/profiles_1d/ion/temperature_fit/reconstructed",
                "reconstructed",
            ),
        ],
    )
    def test_fit_children(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit="m^-3")
        assert result == "fit_artifact"

    # --- Negative cases: /measured, /reconstructed NOT under *_fit ---

    @pytest.mark.parametrize(
        "path,name,expected",
        [
            # magnetics/flux_loop/flux/data is structural only when parent_data_type
            # is STRUCTURE — here without parent info, Rule 7 doesn't fire
            (
                "magnetics/flux_loop/flux/data",
                "data",
                "quantity",
            ),
            # 'measured' under a non-fit parent → quantity
            (
                "interferometer/channel/n_e_line/measured",
                "measured",
                "quantity",
            ),
            # 'reconstructed' under a non-fit parent → quantity
            (
                "mse/channel/faraday_angle/reconstructed",
                "reconstructed",
                "quantity",
            ),
            # 'weight' under a non-fit parent → quantity
            (
                "core_transport/model/profiles_1d/weight",
                "weight",
                "quantity",
            ),
            # 'rho_tor_norm' is coordinate in non-fit context
            (
                "core_profiles/profiles_1d/grid/rho_tor_norm",
                "rho_tor_norm",
                "coordinate",
            ),
            # Plain physics quantity should not be affected
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "temperature",
                "quantity",
            ),
        ],
    )
    def test_non_fit_paths_unaffected(self, path, name, expected):
        result = classify_node_pass1(
            path,
            name,
            data_type="FLT_1D",
            unit="m^-3" if expected == "quantity" else None,
        )
        assert result == expected


# ──────────────────────────────────────────────────────────────────
# Representation classification (Pass 1 rules R1, R2)
# ──────────────────────────────────────────────────────────────────


class TestRepresentationPass1:
    """Tests for representation classification in Pass 1."""

    @pytest.mark.parametrize(
        "path,name",
        [
            # GGD subtree markers
            (
                "equilibrium/grids_ggd/grid/space/objects_per_dimension/object/geometry",
                "geometry",
            ),
            (
                "equilibrium/grids_ggd/grid/space/objects_per_dimension/object/measure",
                "measure",
            ),
            (
                "edge_profiles/ggd_fast/electrons/density/coefficients",
                "coefficients",
            ),
            # grid_subset subtree
            (
                "edge_sources/source/ggd/grid_subset/element/object/space",
                "space",
            ),
            (
                "edge_profiles/grids_ggd/grid_subset/element/object/geometry",
                "geometry",
            ),
            # Representation segment matches
            (
                "core_profiles/profiles_1d/electrons/density_fit/coefficients",
                "coefficients",
            ),
            (
                "equilibrium/time_slice/profiles_2d/grid/jacobian",
                "jacobian",
            ),
            (
                "equilibrium/time_slice/profiles_2d/grid/metric",
                "metric",
            ),
            (
                "waves/coherent_wave/profiles_1d/grid_object",
                "grid_object",
            ),
            (
                "edge_profiles/ggd/electrons/density/coefficient",
                "coefficient",
            ),
            # Parent segment match
            (
                "edge_profiles/ggd/grid_subset/dimension",
                "dimension",
            ),
        ],
    )
    def test_representation_paths(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit="m")
        assert result == "representation"

    # --- Negative cases: physics quantity paths that should NOT be representation ---

    @pytest.mark.parametrize(
        "path,name,unit,expected",
        [
            # Boundary outline — exclusion in geometry path, but boundary
            # is in GEOMETRY_EXCLUSION_PATTERNS → quantity (not geometry)
            (
                "equilibrium/time_slice/boundary/outline/r",
                "r",
                "m",
                "quantity",
            ),
            # Plain physics quantity
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "temperature",
                "eV",
                "quantity",
            ),
            # Regular grid quantity
            (
                "equilibrium/time_slice/profiles_1d/pressure",
                "pressure",
                "Pa",
                "quantity",
            ),
        ],
    )
    def test_non_representation_paths(self, path, name, unit, expected):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit=unit)
        assert result == expected


# ──────────────────────────────────────────────────────────────────
# Transport-solver boundary-conditions / coefficients (Rules F3, F4)
# ──────────────────────────────────────────────────────────────────


class TestTransportSolverFitArtifact:
    """Plan 31 WS-A — solver-internal nodes → fit_artifact."""

    @pytest.mark.parametrize(
        "path,name",
        [
            # Rule F3 — boundary_conditions_* subtree → fit_artifact
            (
                "transport_solver_numerics/boundary_conditions_ion/value",
                "value",
            ),
            (
                "transport_solver_numerics/boundary_conditions_ion/rho_tor_norm",
                "rho_tor_norm",
            ),
            (
                "transport_solver_numerics/boundary_conditions_electrons/"
                "particles/value",
                "value",
            ),
            (
                "transport_solver_numerics/boundary_conditions_current/identifier",
                "identifier",
            ),
            # Rule F4 — solver_1d coefficient* leaves → fit_artifact
            (
                "transport_solver_numerics/solver_1d/equation/coefficient/profile",
                "profile",
            ),
            (
                "transport_solver_numerics/solver_1d/equation/"
                "boundary_condition/coefficient",
                "coefficient",
            ),
            (
                "transport_solver_numerics/solver_1d/equation/coefficients",
                "coefficients",
            ),
        ],
    )
    def test_transport_solver_fit_artifact(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit="-")
        assert result == "fit_artifact"

    def test_unrelated_transport_ids_unaffected(self):
        # core_transport is NOT the solver-numerics IDS — should stay quantity.
        result = classify_node_pass1(
            "core_transport/model/profiles_1d/ion/particles/d",
            "d",
            data_type="FLT_1D",
            unit="m^2.s^-1",
        )
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# pulse_schedule reference exclusion (Rule R3)
# ──────────────────────────────────────────────────────────────────


class TestPulseScheduleReference:
    """Plan 31 WS-A — pulse_schedule reference subtrees → representation."""

    @pytest.mark.parametrize(
        "path,name",
        [
            (
                "pulse_schedule/flux_control/i_plasma/reference",
                "reference",
            ),
            (
                "pulse_schedule/flux_control/i_plasma/reference/data",
                "data",
            ),
            (
                "pulse_schedule/density_control/n_e_line/reference_waveform",
                "reference_waveform",
            ),
            (
                "pulse_schedule/density_control/n_e_line/reference_waveform/data",
                "data",
            ),
            (
                "pulse_schedule/ec/launcher/power/reference",
                "reference",
            ),
        ],
    )
    def test_pulse_schedule_reference_representation(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit="A")
        assert result == "representation"

    def test_pulse_schedule_non_reference_unaffected(self):
        # A pulse_schedule leaf that is NOT under /reference should fall
        # through to quantity.
        result = classify_node_pass1(
            "pulse_schedule/time",
            "time",
            data_type="FLT_1D",
            unit="s",
        )
        # /time leaf is coordinate per rule 5
        assert result == "coordinate"

    def test_non_pulse_schedule_reference_unaffected(self):
        # A /reference leaf outside pulse_schedule is a plain quantity.
        result = classify_node_pass1(
            "equilibrium/time_slice/reference",
            "reference",
            data_type="FLT_0D",
            unit="Wb",
        )
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# Diamagnetic-axis exclusion (Rule R4)
# ──────────────────────────────────────────────────────────────────


class TestDiamagneticAxis:
    """Plan 31 WS-A — vector-container /diamagnetic axis → representation."""

    @pytest.mark.parametrize(
        "path,name",
        [
            (
                "core_profiles/profiles_1d/velocity/diamagnetic",
                "diamagnetic",
            ),
            (
                "core_profiles/profiles_1d/ion/velocity/diamagnetic",
                "diamagnetic",
            ),
            (
                "equilibrium/time_slice/profiles_1d/e_field/diamagnetic",
                "diamagnetic",
            ),
            (
                "waves/coherent_wave/profiles_1d/a_field/diamagnetic",
                "diamagnetic",
            ),
            (
                "core_sources/source/profiles_1d/j_tot/diamagnetic",
                "diamagnetic",
            ),
            (
                "equilibrium/time_slice/profiles_2d/b_field/diamagnetic",
                "diamagnetic",
            ),
            # With child leaf under /diamagnetic
            (
                "core_profiles/profiles_1d/velocity/diamagnetic/data",
                "data",
            ),
        ],
    )
    def test_diamagnetic_axis_representation(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D", unit="m.s^-1")
        assert result == "representation"

    def test_diamagnetic_name_outside_vector_container_unaffected(self):
        # /diamagnetic on a non-vector container (e.g. /magnetics) stays
        # quantity — the rule is scoped to velocity/e_field/a_field/j_tot/b_field.
        result = classify_node_pass1(
            "magnetics/diamagnetic_flux/data",
            "data",
            data_type="FLT_1D",
            unit="Wb",
        )
        assert result == "quantity"

    def test_diamagnetic_drift_velocity_unaffected(self):
        # A leaf literally named diamagnetic_drift but not under
        # one of the vector containers should not be flagged.
        result = classify_node_pass1(
            "core_profiles/profiles_1d/diamagnetic_drift_velocity",
            "diamagnetic_drift_velocity",
            data_type="FLT_1D",
            unit="m.s^-1",
        )
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# Pass 2 — fit-child promotion (R4)
# ──────────────────────────────────────────────────────────────────


class TestFitChildPromotionPass2:
    """Tests for R4 fit-child promotion in Pass 2."""

    def test_quantity_under_fit_parent_promoted(self):
        """quantity leaf under *_fit parent → fit_artifact via R4."""
        result = classify_node_pass2(
            "quantity",
            parent_name="density_fit",
            name="some_leaf",
            data_type="FLT_1D",
            unit="m^-3",
        )
        assert result == "fit_artifact"

    def test_quantity_under_non_fit_parent_unchanged(self):
        """quantity leaf under non-fit parent → None (keep quantity)."""
        result = classify_node_pass2(
            "quantity",
            parent_name="electrons",
            name="temperature",
            data_type="FLT_1D",
            unit="eV",
        )
        assert result is None

    def test_structural_under_fit_parent_unchanged(self):
        """structural node under *_fit parent → None (R4 only applies to quantity)."""
        result = classify_node_pass2(
            "structural",
            parent_name="density_fit",
            name="data",
        )
        assert result is None

    def test_fit_parent_various_names(self):
        """Various *_fit parent names trigger R4."""
        for parent in [
            "density_fit",
            "q_profile_fit",
            "temperature_fit",
            "pressure_fit",
        ]:
            result = classify_node_pass2(
                "quantity",
                parent_name=parent,
                name="measured",
                data_type="FLT_1D",
                unit="m^-3",
            )
            assert result == "fit_artifact", f"Failed for parent_name={parent}"

    def test_r4_does_not_override_identifier(self):
        """R1 identifier overrides R4 — identifier_schema takes priority."""
        result = classify_node_pass2(
            "quantity",
            has_identifier_schema=True,
            parent_name="density_fit",
            name="some_field",
        )
        assert result == "identifier"
