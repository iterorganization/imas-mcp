"""Tests for node_classifier — two-pass classification logic.

Covers geometry detection, Pass 1 rules (1–15), and Pass 2 rules (R1–R3).
"""

import pytest

from imas_codex.core.node_classifier import (
    _BOOLEAN_FLAG_LEAVES,
    _DATA_ENTRY_ID_LEAVES,
    _PHYSICS_INT_LEAVES,
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

    # --- Rule 13 (revised): INT_0D no unit → structural (default) or quantity (allowlist) ---

    def test_int0d_no_unit_not_on_allowlist_structural(self):
        path = "equilibrium/time_slice/profiles_1d/n_r"
        result = classify_node_pass1(path, "n_r", data_type="INT_0D")
        # n_r is not in _PHYSICS_INT_LEAVES — revised Rule 13 defaults to structural
        assert result == "structural"

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


# ──────────────────────────────────────────────────────────────────
# Hardware subtree classification (Rule M1)
# ──────────────────────────────────────────────────────────────────


class TestHardwareSubtreePass1:
    """Tests for hardware-configuration subtree → metadata (Rule M1)."""

    @pytest.mark.parametrize(
        "path,name,data_type,unit",
        [
            # ADC subtree — instrument configuration
            ("neutron_diagnostic/detector/adc/bias", "bias", "FLT_0D", None),
            (
                "neutron_diagnostic/detector/adc/input_range",
                "input_range",
                "FLT_0D",
                None,
            ),
            ("neutron_diagnostic/detector/adc/impedance", "impedance", "FLT_0D", None),
            (
                "neutron_diagnostic/detector/adc/sampling_rate",
                "sampling_rate",
                "INT_0D",
                None,
            ),
            (
                "neutron_diagnostic/detectors/adc/discriminator_level_lower",
                "discriminator_level_lower",
                "INT_0D",
                None,
            ),
            # detector_layout subtree — pixel configuration
            (
                "spectrometer_uv/channel/detector_layout/detector_dimensions",
                "detector_dimensions",
                "FLT_1D",
                "m",
            ),
            (
                "spectrometer_uv/channel/detector_layout/pixel_dimensions",
                "pixel_dimensions",
                "FLT_1D",
                "m",
            ),
            # detector_image subtree — image shape configuration
            (
                "spectrometer_visible/channel/detector_image/circular/radius",
                "radius",
                "FLT_0D",
                "m",
            ),
            (
                "spectrometer_visible/channel/detector_image/circular/ellipticity",
                "ellipticity",
                "FLT_0D",
                "1",
            ),
        ],
    )
    def test_hardware_subtree_metadata(self, path, name, data_type, unit):
        result = classify_node_pass1(path, name, data_type=data_type, unit=unit)
        assert result == "metadata"

    @pytest.mark.parametrize(
        "path,name,data_type,unit,expected",
        [
            # Physics quantities that happen to be near hardware paths
            # — must NOT be captured by the hardware rule
            (
                "neutron_diagnostic/detector/mode/count_rate",
                "count_rate",
                "FLT_0D",
                "s^-1",
                "quantity",
            ),
            # Detector temperature as a direct leaf (not under hardware subtree)
            (
                "spectrometer_mass/detector_voltage",
                "detector_voltage",
                "FLT_0D",
                "V",
                "quantity",
            ),
            # Physics quantity with 'detector' in ancestor (but not a hardware
            # subtree segment)
            (
                "bolometer/channel/detector/centre/r",
                "r",
                "FLT_0D",
                "m",
                "geometry",
            ),
        ],
    )
    def test_hardware_subtree_no_false_positives(
        self, path, name, data_type, unit, expected
    ):
        result = classify_node_pass1(path, name, data_type=data_type, unit=unit)
        assert result == expected


# ──────────────────────────────────────────────────────────────────
# Engineering limits classification (Rule M2)
# ──────────────────────────────────────────────────────────────────


class TestEngineeringLimitsPass1:
    """Tests for engineering limits → metadata (Rule M2)."""

    @pytest.mark.parametrize(
        "path,name,data_type,unit",
        [
            # Temperature limits
            (
                "divertors/divertor/target/temperature_limit_max",
                "temperature_limit_max",
                "FLT_0D",
                "K",
            ),
            (
                "breeding_blanket/steel_temperature_limit_max",
                "steel_temperature_limit_max",
                "FLT_0D",
                "K",
            ),
            # Current limits
            (
                "pf_active/supply/current_limit_max",
                "current_limit_max",
                "FLT_0D",
                "A",
            ),
            (
                "pf_active/supply/current_limit_min",
                "current_limit_min",
                "FLT_0D",
                "A",
            ),
            # Voltage limits
            (
                "pf_active/supply/voltage_limit_max",
                "voltage_limit_max",
                "FLT_0D",
                "V",
            ),
            # Count-rate limits
            (
                "neutron_diagnostic/characteristics/reaction/mode/count_limit_max",
                "count_limit_max",
                "FLT_0D",
                "cps",
            ),
            (
                "neutron_diagnostic/detector/mode/count_limit_min",
                "count_limit_min",
                "FLT_0D",
                "s^-1",
            ),
            # Heat flux limits
            (
                "divertors/divertor/target/heat_flux_steady_limit_max",
                "heat_flux_steady_limit_max",
                "FLT_0D",
                "W.m^-2",
            ),
            # Spatial extent limits (position envelopes)
            (
                "ec_launchers/beam/launching_position/r_limit_max",
                "r_limit_max",
                "FLT_0D",
                "m",
            ),
            (
                "ec_launchers/beam/launching_position/r_limit_min",
                "r_limit_min",
                "FLT_0D",
                "m",
            ),
            # Energy limits
            (
                "pf_active/coil/energy_limit_max",
                "energy_limit_max",
                "FLT_0D",
                "J",
            ),
        ],
    )
    def test_engineering_limit_metadata(self, path, name, data_type, unit):
        result = classify_node_pass1(path, name, data_type=data_type, unit=unit)
        assert result == "metadata"

    @pytest.mark.parametrize(
        "path,name,data_type,unit",
        [
            # Physics quantities with 'limit' in name but NOT _limit_max/_limit_min
            (
                "summary/global_quantities/beta_limit",
                "beta_limit",
                "FLT_0D",
                "-",
            ),
            (
                "summary/global_quantities/density_limit",
                "density_limit",
                "FLT_0D",
                "m^-3",
            ),
            # Physics boundary quantities
            (
                "core_profiles/profiles_1d/grid/psi_boundary",
                "psi_boundary",
                "FLT_0D",
                "Wb",
            ),
        ],
    )
    def test_engineering_limit_no_false_positives(self, path, name, data_type, unit):
        """Physics quantities with 'limit'/'boundary' must stay quantity."""
        result = classify_node_pass1(path, name, data_type=data_type, unit=unit)
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# Instrument-specification suffix (Rule M3)
# ──────────────────────────────────────────────────────────────────


class TestInstrumentSpecPass1:
    """Tests for instrument-specification suffixes → metadata (Rule M3)."""

    @pytest.mark.parametrize(
        "path,name",
        [
            ("magnetics/b_field_pol_probe/bandwidth_3db", "bandwidth_3db"),
            ("magnetics/b_field_phi_probe/bandwidth_3db", "bandwidth_3db"),
            ("magnetics/b_field_tor_probe/bandwidth_3db", "bandwidth_3db"),
            ("magnetics/bpol_probe/bandwidth_3db", "bandwidth_3db"),
        ],
    )
    def test_3db_metadata(self, path, name):
        result = classify_node_pass1(path, name, data_type="FLT_1D")
        assert result == "metadata"

    def test_non_3db_bandwidth_stays_quantity(self):
        """IF bandwidth (without _3db) is a physics quantity, keep it."""
        result = classify_node_pass1(
            "ece/channel/if_bandwidth",
            "if_bandwidth",
            data_type="FLT_0D",
        )
        # if_bandwidth doesn't end with _3db → not caught by M3
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# Extended STRUCTURAL_KEYWORDS (W37 additions)
# ──────────────────────────────────────────────────────────────────


class TestExtendedStructuralKeywords:
    """Tests for W37 structural keyword additions: closed, spacing, transformation."""

    @pytest.mark.parametrize(
        "path,name",
        [
            # Outline closure flags
            (
                "wall/description_2d/vessel/unit/annular/outline_outer/closed",
                "closed",
            ),
            (
                "wall/description_2d/vessel/unit/annular/outline_inner/closed",
                "closed",
            ),
            (
                "cryostat/vacuum_vessel/annular/outline_inner/closed",
                "closed",
            ),
            # Grid spacing mode
            (
                "amns_data/coordinate_system/coordinate/spacing",
                "spacing",
            ),
            # Coordinate transformation mode
            (
                "amns_data/coordinate_system/coordinate/transformation",
                "transformation",
            ),
            # result_transformation — also caught by substring match
            (
                "amns_data/process/result_transformation",
                "result_transformation",
            ),
        ],
    )
    def test_structural_config_flags(self, path, name):
        result = classify_node_pass1(path, name, data_type="INT_0D")
        assert result == "structural"

    def test_safety_factor_stays_quantity(self):
        """safety_factor (dimensionless, unit='1') must remain quantity."""
        result = classify_node_pass1(
            "equilibrium/time_slice/global_quantities/q_95",
            "q_95",
            data_type="FLT_0D",
        )
        assert result == "quantity"

    def test_elongation_stays_quantity(self):
        """Dimensionless geometry ratio must remain quantity."""
        result = classify_node_pass1(
            "equilibrium/time_slice/boundary/elongation",
            "elongation",
            data_type="FLT_0D",
        )
        assert result == "quantity"

    def test_beta_pol_stays_quantity(self):
        """Dimensionless plasma beta must remain quantity."""
        result = classify_node_pass1(
            "equilibrium/time_slice/global_quantities/beta_pol",
            "beta_pol",
            data_type="FLT_0D",
        )
        assert result == "quantity"


# ──────────────────────────────────────────────────────────────────
# data_entry_identifier integer slots (Rule M4)
# ──────────────────────────────────────────────────────────────────


class TestDataEntryIdentifierPass1:
    """Tests for Rule M4: data_entry integer identifier slots → metadata.

    INT children of ``data_entry`` structs (run, shot, pulse, …) are
    dataset-provenance coordinates, not physics quantities.  They must not
    enter the StandardName generation pipeline.

    Regression: the ``description`` constant and string siblings must
    remain metadata via earlier rules and must NOT be affected.
    """

    @pytest.mark.parametrize(
        "path,name",
        [
            # amns_data — run and shot under release/data_entry
            ("amns_data/release/data_entry/run", "run"),
            ("amns_data/release/data_entry/shot", "shot"),
            # dataset_description — run and pulse
            ("dataset_description/data_entry/run", "run"),
            ("dataset_description/data_entry/pulse", "pulse"),
            # langmuir_probes — pulse under equilibrium_id/data_entry
            ("langmuir_probes/equilibrium_id/data_entry/pulse", "pulse"),
            ("langmuir_probes/equilibrium_id/data_entry/run", "run"),
        ],
    )
    def test_data_entry_int_leaves_metadata(self, path, name):
        """INT_0D data_entry identifier slots must be classified metadata (Rule M4)."""
        result = classify_node_pass1(path, name, data_type="INT_0D")
        assert result == "metadata"

    def test_data_entry_id_leaves_constant_completeness(self):
        """_DATA_ENTRY_ID_LEAVES must contain the known integer identifier names."""
        assert "run" in _DATA_ENTRY_ID_LEAVES
        assert "shot" in _DATA_ENTRY_ID_LEAVES
        assert "pulse" in _DATA_ENTRY_ID_LEAVES
        assert "occurrence" in _DATA_ENTRY_ID_LEAVES

    # ── Regression: string siblings must still be metadata (Rule 4) ──

    @pytest.mark.parametrize(
        "path,name",
        [
            # STR_0D siblings — caught by Rule 4 before M4 fires
            ("dataset_description/data_entry/user", "user"),
            ("dataset_description/data_entry/machine", "machine"),
            ("dataset_description/data_entry/pulse_type", "pulse_type"),
        ],
    )
    def test_data_entry_string_siblings_still_metadata(self, path, name):
        """STR_0D siblings of INT identifiers must remain metadata (Rule 4 unchanged)."""
        result = classify_node_pass1(path, name, data_type="STR_0D")
        assert result == "metadata"

    # ── Regression: genuine quantity paths must NOT be affected ──

    def test_genuine_quantity_unaffected(self):
        """A physics quantity path far from data_entry must remain quantity."""
        result = classify_node_pass1(
            "core_profiles/profiles_1d/electrons/temperature",
            "temperature",
            data_type="FLT_1D",
            unit="eV",
        )
        assert result == "quantity"

    def test_int_quantity_outside_data_entry_unaffected(self):
        """INT_0D whose parent is NOT data_entry must not be caught by Rule M4.

        Note: n_r is not in _PHYSICS_INT_LEAVES so revised Rule 13 defaults
        it to structural.  This is expected — n_r is a grid resolution count.
        """
        result = classify_node_pass1(
            "equilibrium/time_slice/profiles_1d/n_r",
            "n_r",
            data_type="INT_0D",
        )
        # n_r is not under data_entry AND not in physics allowlist → structural
        assert result == "structural"


# ──────────────────────────────────────────────────────────────────
# INT_0D classifier tightening (systemic fix)
# ──────────────────────────────────────────────────────────────────


class TestINT0DFalsePositiveRegression:
    """INT_0D nodes that MUST NOT be classified as quantity.

    These were false positives under the old Rule 13 (INT_0D no-unit → quantity).
    After the systemic fix each must classify as ``structural`` or ``metadata``.
    """

    @pytest.mark.parametrize(
        "path,name,expected",
        [
            # ── R13a: _validity suffix → structural ──
            (
                "core_profiles/profiles_1d/electrons/density_validity",
                "density_validity",
                "structural",
            ),
            (
                "core_profiles/profiles_1d/electrons/temperature_validity",
                "temperature_validity",
                "structural",
            ),
            (
                "core_profiles/profiles_1d/ion/density_validity",
                "density_validity",
                "structural",
            ),
            (
                "edge_profiles/profiles_1d/electrons/density_validity",
                "density_validity",
                "structural",
            ),
            # ── M5: Epoch timestamps → metadata ──
            (
                "dataset_description/pulse_time_begin_epoch/seconds",
                "seconds",
                "metadata",
            ),
            (
                "dataset_description/pulse_time_begin_epoch/nanoseconds",
                "nanoseconds",
                "metadata",
            ),
            (
                "summary/pulse_time_end_epoch/seconds",
                "seconds",
                "metadata",
            ),
            (
                "summary/pulse_time_end_epoch/nanoseconds",
                "nanoseconds",
                "metadata",
            ),
            # ── M6: IDS-reference occurrence/pulse/shot → metadata ──
            (
                "langmuir_probes/equilibrium_id/occurrence",
                "occurrence",
                "metadata",
            ),
            (
                "thomson_scattering/equilibrium_id/occurrence",
                "occurrence",
                "metadata",
            ),
            ("dataset_description/pulse", "pulse", "metadata"),
            (
                "dataset_description/parent_entry/pulse",
                "pulse",
                "metadata",
            ),
            ("summary/pulse", "pulse", "metadata"),
            # ── R13d: Boolean flags → structural ──
            (
                "coils_non_axisymmetric/is_periodic",
                "is_periodic",
                "structural",
            ),
            (
                "langmuir_probes/probe/reciprocating",
                "reciprocating",
                "structural",
            ),
            (
                "gyrokinetics/model/adiabatic_electrons",
                "adiabatic_electrons",
                "structural",
            ),
            (
                "waves/coherent_wave/beam_tracing/beam/wave_vector/varying_n_tor",
                "varying_n_tor",
                "structural",
            ),
            # ── R13e: Generic /value containers → structural ──
            (
                "summary/disruption/mitigation_valve/value",
                "value",
                "structural",
            ),
            ("summary/pellets/occurrence/value", "value", "structural"),
            (
                "temporary/constant_integer0d/value",
                "value",
                "metadata",
            ),
        ],
    )
    def test_int0d_false_positive_not_quantity(self, path, name, expected):
        """INT_0D false positives must be reclassified (not quantity)."""
        cat = classify_node_pass1(path, name, data_type="INT_0D")
        assert cat == expected, f"{path} got {cat!r}, expected {expected!r}"


class TestINT0DPhysicsTruePositive:
    """INT_0D nodes that MUST remain classified as ``quantity``.

    These are genuine physics integers on the ``_PHYSICS_INT_LEAVES`` allowlist.
    """

    @pytest.mark.parametrize(
        "path,name",
        [
            # Nuclear charge
            ("core_profiles/profiles_1d/ion/element/z_n", "z_n"),
            ("nbi/unit/species/z_n", "z_n"),
            # Atom count
            ("core_profiles/profiles_1d/ion/element/atoms_n", "atoms_n"),
            (
                "gas_injection/valve/species/element/atoms_n",
                "atoms_n",
            ),
            # Mode numbers
            (
                "mhd_linear/time_slice/toroidal_mode/n_tor",
                "n_tor",
            ),
            ("ntms/time_slice/mode/n_tor", "n_tor"),
            (
                "ntms/time_slice/mode/detailed_evolution/m_pol",
                "m_pol",
            ),
            # Coil turns
            ("magnetics/bpol_probe/turns", "turns"),
            ("magnetics/b_field_tor_probe/turns", "turns"),
            # Module/coil counts
            ("breeding_blanket/modules_n", "modules_n"),
            ("coils_non_axisymmetric/coils_n", "coils_n"),
            # Convergence iterations
            (
                "transport_solver_numerics/solver_1d/equation/convergence/iterations_n",
                "iterations_n",
            ),
            # Species fraction
            (
                "core_profiles/profiles_1d/ion/fraction",
                "fraction",
            ),
        ],
    )
    def test_int0d_physics_remains_quantity(self, path, name):
        """Genuine physics integers must stay classified as quantity."""
        cat = classify_node_pass1(path, name, data_type="INT_0D")
        assert cat == "quantity", f"{path} should be quantity but got {cat!r}"


class TestValidityClassification:
    """Paths with ``_validity`` suffix → structural (Rule 6 + R13a)."""

    @pytest.mark.parametrize(
        "path,name",
        [
            # Exact match (existing Rule 6 behaviour)
            ("core_profiles/profiles_1d/validity", "validity"),
            ("core_profiles/profiles_1d/validity_timed", "validity_timed"),
            # Suffix match (R13a extension)
            (
                "core_profiles/profiles_1d/electrons/density_validity",
                "density_validity",
            ),
            (
                "core_profiles/profiles_1d/electrons/temperature_validity",
                "temperature_validity",
            ),
            (
                "edge_profiles/profiles_1d/electrons/density_validity",
                "density_validity",
            ),
            (
                "core_profiles/profiles_1d/ion/state/density_validity",
                "density_validity",
            ),
        ],
    )
    def test_validity_structural(self, path, name):
        """Validity indicators must be classified as structural."""
        cat = classify_node_pass1(path, name, data_type="INT_0D")
        assert cat == "structural", f"{path} got {cat!r}, expected 'structural'"


class TestEpochTimestampClassification:
    """Epoch timestamp paths → metadata (Rule M5)."""

    @pytest.mark.parametrize(
        "path,name",
        [
            (
                "dataset_description/pulse_time_begin_epoch/seconds",
                "seconds",
            ),
            (
                "dataset_description/pulse_time_begin_epoch/nanoseconds",
                "nanoseconds",
            ),
            ("summary/pulse_time_end_epoch/seconds", "seconds"),
            (
                "summary/pulse_time_end_epoch/nanoseconds",
                "nanoseconds",
            ),
        ],
    )
    def test_epoch_metadata(self, path, name):
        """Epoch timestamp leaves must be classified as metadata."""
        cat = classify_node_pass1(path, name, data_type="INT_0D")
        assert cat == "metadata", f"{path} got {cat!r}, expected 'metadata'"


class TestPhysicsQuantityNonRegression:
    """Core FLT physics paths must always classify as quantity.

    Guards that the INT_0D tightening did NOT break FLT/CPX rules.
    """

    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "temperature",
                "FLT_1D",
                "eV",
            ),
            (
                "equilibrium/time_slice/global_quantities/ip",
                "ip",
                "FLT_0D",
                "A",
            ),
            (
                "equilibrium/time_slice/profiles_1d/psi",
                "psi",
                "FLT_1D",
                "Wb",
            ),
            (
                "core_profiles/profiles_1d/electrons/density",
                "density",
                "FLT_1D",
                "m^-3",
            ),
            (
                "equilibrium/time_slice/profiles_1d/elongation",
                "elongation",
                "FLT_1D",
                None,
            ),
            (
                "equilibrium/time_slice/global_quantities/beta_tor",
                "beta_tor",
                "FLT_0D",
                None,
            ),
        ],
    )
    def test_flt_physics_quantity(self, path, name, dt, unit):
        """FLT physics quantities must remain unaffected by INT_0D changes."""
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "quantity", f"{path} should be quantity but got {cat!r}"


# ──────────────────────────────────────────────────────────────────
# Leak-by-construction guarantees (Standard-Name extraction safety net)
# ──────────────────────────────────────────────────────────────────


class TestStandardNameLeakClasses:
    """Each parametrized leak class must NOT classify as quantity/geometry/coordinate.

    These categories are the SN-extraction inputs (``SN_SOURCE_CATEGORIES``);
    any leak here is a structural bug.  All checks are deterministic — no LLM
    or graph context required — so the rule set in ``classify_node_pass1`` is
    the single point of enforcement.
    """

    # Leaf names that, with INT_0D + no unit, must NOT become quantity.
    @pytest.mark.parametrize(
        "path,name",
        [
            # *_flag patterns (W12A regression)
            (
                "pf_active/coil/toroidal_field_coil_periodicity_flag",
                "toroidal_field_coil_periodicity_flag",
            ),
            (
                "coils_non_axisymmetric/non_axisymmetric_coil_periodicity_flag",
                "non_axisymmetric_coil_periodicity_flag",
            ),
            ("ec_launchers/beam/some_flag", "some_flag"),
            # *_index storage indexes
            ("equilibrium/time_slice/profiles_2d/grid_index", "grid_index"),
            # *_identifier
            ("ic_antennas/antenna/some_identifier", "some_identifier"),
            # *_status
            ("interferometer/status", "status"),
        ],
    )
    def test_int_storage_keywords_not_quantity(self, path, name):
        cat = classify_node_pass1(path, name, data_type="INT_0D", unit=None)
        assert cat == "structural", f"{path} → {cat!r} (expected structural)"

    # data_entry/* metadata leaks
    @pytest.mark.parametrize(
        "path,name,dt",
        [
            ("dataset_description/data_entry/run", "run", "INT_0D"),
            ("dataset_description/data_entry/pulse", "pulse", "INT_0D"),
            ("dataset_description/data_entry/user", "user", "STR_0D"),
            (
                "thomson_scattering/equilibrium_id/data_entry/run",
                "run",
                "INT_0D",
            ),
            (
                "amns_data/release/data_entry/description",
                "description",
                "STR_0D",
            ),
        ],
    )
    def test_data_entry_subtree_metadata(self, path, name, dt):
        cat = classify_node_pass1(path, name, data_type=dt, unit=None)
        assert cat == "metadata", f"{path} → {cat!r} (expected metadata)"

    # Scratch IDS — every node, regardless of dtype/unit, is metadata.
    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            ("temporary/constant_float0d/value", "value", "FLT_0D", "1"),
            ("temporary/constant_float1d/value", "value", "FLT_1D", "1"),
            ("temporary/constant_float5d/value", "value", "FLT_5D", "mixed"),
            ("temporary/constant_float6d/value", "value", "FLT_6D", "mixed"),
            ("temporary/constant_integer0d/value", "value", "INT_0D", None),
            ("temporary/constant_string0d/value", "value", "STR_0D", None),
        ],
    )
    def test_scratch_ids_temporary_metadata(self, path, name, dt, unit):
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "metadata", f"{path} → {cat!r} (expected metadata)"

    # Legacy temporary_storage_* path-segment leaks (W28A user-flagged).
    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            (
                "summary/temporary_storage_integer_value/value",
                "value",
                "INT_0D",
                None,
            ),
            (
                "summary/temporary_storage_float/data",
                "data",
                "FLT_1D",
                "1",
            ),
            (
                "ec_launchers/temporary_storage_string/foo",
                "foo",
                "STR_0D",
                None,
            ),
        ],
    )
    def test_legacy_temporary_storage_metadata(self, path, name, dt, unit):
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "metadata", f"{path} → {cat!r} (expected metadata)"

    # transport_solver_numerics convergence subtree → fit_artifact.
    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            (
                "transport_solver_numerics/convergence/equations/ion/energy/delta_relative/value",
                "value",
                "FLT_0D",
                "-",
            ),
            (
                "transport_solver_numerics/convergence/equations/current/iterations_n",
                "iterations_n",
                "INT_0D",
                None,
            ),
            (
                "transport_solver_numerics/convergence/time_step",
                "time_step",
                "STRUCTURE",
                "s",
            ),
            (
                "transport_solver_numerics/convergence/equations/electrons/particles",
                "particles",
                "STRUCTURE",
                "m^-3.s^-1",
            ),
        ],
    )
    def test_transport_solver_convergence_fit_artifact(self, path, name, dt, unit):
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "fit_artifact", f"{path} → {cat!r} (expected fit_artifact)"

    # Singular ``boundary_condition`` under solver_1d/equation/ → fit_artifact.
    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            (
                "transport_solver_numerics/solver_1d/equation/boundary_condition/value",
                "value",
                "FLT_1D",
                "mixed",
            ),
            (
                "transport_solver_numerics/solver_1d/equation/boundary_condition/position",
                "position",
                "FLT_0D",
                "mixed",
            ),
        ],
    )
    def test_solver_1d_boundary_condition_singular(self, path, name, dt, unit):
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "fit_artifact", f"{path} → {cat!r} (expected fit_artifact)"

    # summary/<topic>/occurrence flag containers → metadata.
    @pytest.mark.parametrize(
        "topic",
        ["pellets", "rmps", "kicks"],
    )
    def test_summary_occurrence_flag_metadata(self, topic):
        path = f"summary/{topic}/occurrence"
        cat = classify_node_pass1(path, "occurrence", data_type="STRUCTURE", unit="Hz")
        assert cat == "metadata", f"{path} → {cat!r} (expected metadata)"

    # Pre-existing F3 plural form must continue to work.
    def test_transport_solver_boundary_conditions_plural(self):
        cat = classify_node_pass1(
            "transport_solver_numerics/boundary_conditions_ion/value",
            "value",
            data_type="FLT_1D",
            unit="m^-3",
        )
        assert cat == "fit_artifact"

    # Non-leak guards: legitimate physics that share suffixes/segments.
    @pytest.mark.parametrize(
        "path,name,dt,unit",
        [
            # refractive_index is a real physics quantity, despite ending _index.
            (
                "camera_visible/channel/optical_element/material_properties/refractive_index",
                "refractive_index",
                "FLT_1D",
                "1",
            ),
            # 'value' inside a real summary physics container is fine.
            (
                "summary/global_quantities/energy_total/value",
                "value",
                "FLT_1D",
                "J",
            ),
            # equilibrium boundary outline r/z (geometry exclusion handled
            # elsewhere — must remain a quantity in this regime).
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "temperature",
                "FLT_1D",
                "eV",
            ),
        ],
    )
    def test_non_leak_physics_remains_quantity(self, path, name, dt, unit):
        cat = classify_node_pass1(path, name, data_type=dt, unit=unit)
        assert cat == "quantity", f"{path} → {cat!r} (expected quantity)"
