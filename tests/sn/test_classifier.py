"""Tests for naming-scope classifier.

Validates ``classify_path`` against a comprehensive gold set of ~50 edge-case
DD paths, plus focused unit tests for each classification rule.
"""

from __future__ import annotations

import pytest

from imas_codex.sn.classifier import (
    ERROR_SUFFIXES,
    PHYSICS_LEAF_TYPES,
    STRUCTURE_TYPES,
    Scope,
    classify_path,
)

# ============================================================================
# Gold set — ~50 edge-case paths for parametrised testing
# ============================================================================
#
# Format: (path, data_type, unit, parent_type, description, node_category, expected)

GOLD_SET: list[tuple[str, str, str | None, str | None, str, str | None, Scope]] = [
    # -----------------------------------------------------------------------
    # quantity — common physics quantities
    # -----------------------------------------------------------------------
    (
        "core_profiles/profiles_1d/electrons/temperature",
        "FLT_1D",
        "eV",
        "STRUCTURE",
        "Electron temperature",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/profiles_1d/psi",
        "FLT_1D",
        "Wb",
        None,
        "Poloidal magnetic flux",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/global_quantities/ip",
        "FLT_0D",
        "A",
        None,
        "Plasma current",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/electrons/density",
        "FLT_1D",
        "m^-3",
        "STRUCTURE",
        "Electron density",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/ion/pressure",
        "FLT_1D",
        "Pa",
        "STRUCTURE",
        "Ion pressure",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/electrons/velocity/radial",
        "FLT_1D",
        "m/s",
        None,
        "Radial velocity",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor",
        "FLT_0D",
        "T",
        None,
        "Toroidal magnetic field on axis",
        None,
        "quantity",
    ),
    (
        "core_transport/model/profiles_1d/electrons/energy/flux",
        "FLT_1D",
        "W.m^-2",
        None,
        "Electron energy flux",
        None,
        "quantity",
    ),
    (
        "magnetics/flux_loop/flux/data",
        "FLT_1D",
        "Wb",
        "STRUCTURE",
        "Magnetic flux data",
        None,
        # /data under STRUCTURE with unit → metadata (Rule 1 takes precedence)
        "metadata",
    ),
    (
        "equilibrium/time_slice/profiles_1d/phi",
        "FLT_1D",
        "Wb",
        None,
        "Toroidal flux",
        None,
        "quantity",
    ),
    (
        "core_sources/source/profiles_1d/electrons/energy",
        "FLT_1D",
        "W.m^-3",
        None,
        "Electron energy source",
        None,
        "quantity",
    ),
    # quantity — complex data types
    (
        "waves/coherent_wave/profiles_1d/e_field_plus",
        "CPX_1D",
        "V/m",
        None,
        "Left-hand rotating electric field component",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/zeff",
        "FLT_1D",
        None,
        None,
        "Effective charge",
        None,
        "quantity",
    ),
    # quantity — dimensionless quantities (no unit)
    (
        "equilibrium/time_slice/profiles_1d/safety_factor",
        "FLT_1D",
        None,
        None,
        "Safety factor (q)",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/elongation",
        "FLT_1D",
        None,
        None,
        "Plasma elongation",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/global_quantities/beta_pol",
        "FLT_0D",
        None,
        None,
        "Poloidal beta",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/global_quantities/beta_tor",
        "FLT_0D",
        None,
        None,
        "Toroidal beta",
        None,
        "quantity",
    ),
    (
        "equilibrium/time_slice/profiles_1d/triangularity_upper",
        "FLT_1D",
        None,
        None,
        "Upper triangularity",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/electrons/collisionality_norm",
        "FLT_1D",
        None,
        None,
        "Normalised collisionality",
        None,
        "quantity",
    ),
    # quantity — INT_0D dimensionless physics
    (
        "mhd_linear/toroidal_mode_number",
        "INT_0D",
        None,
        None,
        "Toroidal mode number",
        None,
        "quantity",
    ),
    (
        "mhd_linear/time_slice/toroidal_mode/n_tor",
        "INT_0D",
        None,
        None,
        "Toroidal mode number n",
        None,
        "quantity",
    ),
    (
        "core_profiles/profiles_1d/ion/z_ion",
        "INT_0D",
        None,
        None,
        "Ion charge state",
        None,
        "quantity",
    ),
    # quantity — STRUCTURE with unit (Rule 10)
    (
        "barometry/gauge/pressure",
        "STRUCTURE",
        "Pa",
        None,
        "Pressure measurement",
        None,
        "quantity",
    ),
    (
        "magnetics/flux_loop/flux",
        "STRUCTURE",
        "Wb",
        None,
        "Magnetic flux measurement",
        None,
        "quantity",
    ),
    # quantity — INT_1D / INT_2D with unit
    (
        "core_profiles/profiles_1d/ion/charge_state",
        "INT_1D",
        "C",
        None,
        "Charge state array",
        None,
        "quantity",
    ),
    # -----------------------------------------------------------------------
    # metadata — /data under STRUCTURE parent with unit
    # -----------------------------------------------------------------------
    (
        "barometry/gauge/pressure/data",
        "FLT_1D",
        "Pa",
        "STRUCTURE",
        "Pressure data array",
        None,
        "metadata",
    ),
    (
        "magnetics/bpol_probe/field/data",
        "FLT_1D",
        "T",
        "STRUCTURE",
        "Poloidal field data",
        None,
        "metadata",
    ),
    (
        "interferometer/channel/n_e_line/data",
        "FLT_1D",
        "m^-2",
        "STRUCTURE",
        "Line-integrated electron density data",
        None,
        "metadata",
    ),
    (
        "bolometer/channel/power/data",
        "FLT_1D",
        "W",
        "STRUCT_ARRAY",
        "Power data array",
        None,
        "metadata",
    ),
    # metadata — /data under STRUCTURE but NO unit → falls through to quantity (Rule 8/9)
    (
        "some_ids/path/data",
        "FLT_1D",
        None,
        "STRUCTURE",
        "Some dimensionless data",
        None,
        # No unit on parent, Rule 1 does NOT match → falls to Rule 8/9
        "quantity",
    ),
    # metadata — /time leaf
    (
        "core_profiles/profiles_1d/electrons/temperature/time",
        "FLT_1D",
        "s",
        None,
        "Time base",
        None,
        "metadata",
    ),
    (
        "equilibrium/time",
        "FLT_1D",
        "s",
        None,
        "Time array for equilibrium",
        None,
        "metadata",
    ),
    (
        "magnetics/flux_loop/flux/time",
        "FLT_1D",
        "s",
        None,
        "Time base for flux measurement",
        None,
        "metadata",
    ),
    # metadata — /validity and /validity_timed
    (
        "equilibrium/time_slice/profiles_1d/psi/validity",
        "INT_0D",
        None,
        None,
        "Validity flag",
        None,
        "metadata",
    ),
    (
        "core_profiles/profiles_1d/electrons/temperature/validity_timed",
        "INT_1D",
        None,
        None,
        "Time-dependent validity",
        None,
        "metadata",
    ),
    (
        "magnetics/bpol_probe/field/validity",
        "INT_0D",
        None,
        None,
        "Validity indicator for the measurement",
        None,
        "metadata",
    ),
    # -----------------------------------------------------------------------
    # skip — STR_0D string fields
    # -----------------------------------------------------------------------
    (
        "core_profiles/profiles_1d/electrons/label",
        "STR_0D",
        None,
        None,
        "Species label",
        None,
        "skip",
    ),
    (
        "wall/description_2d/limiter/type/name",
        "STR_0D",
        None,
        None,
        "Limiter type name",
        None,
        "skip",
    ),
    (
        "equilibrium/time_slice/boundary/type/description",
        "STR_0D",
        None,
        None,
        "Boundary type description",
        None,
        "skip",
    ),
    # skip — error fields
    (
        "core_profiles/profiles_1d/grid/rho_tor_norm_error_upper",
        "FLT_1D",
        None,
        None,
        "Upper error bound",
        None,
        "skip",
    ),
    (
        "equilibrium/time_slice/profiles_1d/psi_error_lower",
        "FLT_1D",
        None,
        None,
        "Lower error bound",
        None,
        "skip",
    ),
    (
        "equilibrium/time_slice/profiles_1d/psi_error_index",
        "INT_0D",
        None,
        None,
        "Error index",
        None,
        "skip",
    ),
    # skip — INT_0D indices, flags, and counters
    (
        "magnetics/flux_loop/flux/index",
        "INT_0D",
        None,
        None,
        "Index in the flux_loop array",
        None,
        "skip",
    ),
    (
        "pf_active/coil/element/turns_with_sign/count",
        "INT_0D",
        None,
        None,
        "Number of turns",
        None,
        "skip",
    ),
    (
        "core_profiles/profiles_1d/ion/element/z_n",
        "INT_0D",
        None,
        None,
        "Number of nucleons (count of protons and neutrons)",
        None,
        "skip",
    ),
    (
        "equilibrium/time_slice/convergence/flag",
        "INT_0D",
        None,
        None,
        "Convergence flag for equilibrium solve",
        None,
        "skip",
    ),
    (
        "core_profiles/profiles_1d/ion/type/identifier",
        "INT_0D",
        None,
        None,
        "Identifier for the ion species type",
        None,
        "skip",
    ),
    (
        "pf_active/coil/number_of_points",
        "INT_0D",
        None,
        None,
        "Number of points defining the coil geometry",
        None,
        "skip",
    ),
    # skip — STRUCTURE / STRUCT_ARRAY without unit (containers)
    (
        "core_profiles/profiles_1d/electrons",
        "STRUCTURE",
        None,
        None,
        "Electron species profiles",
        None,
        "skip",
    ),
    (
        "equilibrium/time_slice",
        "STRUCT_ARRAY",
        None,
        None,
        "Equilibrium time slice",
        None,
        "skip",
    ),
    (
        "wall/description_2d",
        "STRUCT_ARRAY",
        None,
        None,
        "2D wall description",
        None,
        "skip",
    ),
]


@pytest.mark.parametrize(
    "path,data_type,unit,parent_type,description,node_category,expected",
    GOLD_SET,
    ids=[g[0].rsplit("/", 1)[-1] for g in GOLD_SET],
)
def test_classify_path(
    path: str,
    data_type: str,
    unit: str | None,
    parent_type: str | None,
    description: str,
    node_category: str | None,
    expected: Scope,
) -> None:
    """Gold-set parametrised test — each case maps to one expected scope."""
    node = {
        "path": path,
        "data_type": data_type,
        "unit": unit,
        "parent_type": parent_type,
        "description": description or "",
        "node_category": node_category,
        "parent_path": "/".join(path.split("/")[:-1]) if "/" in path else None,
        "cluster_label": None,
    }
    assert classify_path(node) == expected


# ============================================================================
# Focused unit tests — one per classification rule
# ============================================================================


class TestRule1DataUnderStructureWithUnit:
    """Rule 1: /data under STRUCTURE parent with unit → metadata."""

    def _make(self, *, parent_type: str = "STRUCTURE", unit: str | None = "Pa") -> dict:
        return {
            "path": "barometry/gauge/pressure/data",
            "data_type": "FLT_1D",
            "unit": unit,
            "parent_type": parent_type,
            "description": "Pressure data",
            "node_category": None,
            "parent_path": "barometry/gauge/pressure",
            "cluster_label": None,
        }

    def test_structure_with_unit(self) -> None:
        assert classify_path(self._make()) == "metadata"

    def test_struct_array_with_unit(self) -> None:
        assert classify_path(self._make(parent_type="STRUCT_ARRAY")) == "metadata"

    def test_structure_without_unit_falls_through(self) -> None:
        """Without a unit, Rule 1 doesn't match → falls through to Rule 8/9."""
        assert classify_path(self._make(unit=None)) == "quantity"

    def test_non_structure_parent_falls_through(self) -> None:
        """If parent is not STRUCTURE, Rule 1 doesn't match."""
        result = classify_path(self._make(parent_type="FLT_1D"))
        assert result == "quantity"


class TestRule2TimeLeaf:
    """Rule 2: /time leaf → metadata."""

    def test_time_leaf(self) -> None:
        node = {
            "path": "equilibrium/time",
            "data_type": "FLT_1D",
            "unit": "s",
            "parent_type": None,
            "description": "Time array",
            "node_category": None,
            "parent_path": "equilibrium",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"

    def test_nested_time_leaf(self) -> None:
        node = {
            "path": "core_profiles/profiles_1d/electrons/temperature/time",
            "data_type": "FLT_1D",
            "unit": "s",
            "parent_type": None,
            "description": "Time base",
            "node_category": None,
            "parent_path": "core_profiles/profiles_1d/electrons/temperature",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"

    def test_time_in_middle_of_path_is_not_matched(self) -> None:
        """Only the last segment matters — 'time_slice' should not match."""
        node = {
            "path": "equilibrium/time_slice/profiles_1d/psi",
            "data_type": "FLT_1D",
            "unit": "Wb",
            "parent_type": None,
            "description": "Poloidal flux",
            "node_category": None,
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"


class TestRule3Validity:
    """Rule 3: /validity and /validity_timed → metadata."""

    def test_validity(self) -> None:
        node = {
            "path": "equilibrium/time_slice/profiles_1d/psi/validity",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Validity flag",
            "node_category": None,
            "parent_path": "equilibrium/time_slice/profiles_1d/psi",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"

    def test_validity_timed(self) -> None:
        node = {
            "path": "core_profiles/profiles_1d/electrons/temperature/validity_timed",
            "data_type": "INT_1D",
            "unit": None,
            "parent_type": None,
            "description": "Time-dependent validity",
            "node_category": None,
            "parent_path": "core_profiles/profiles_1d/electrons/temperature",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"


class TestRule4ErrorFields:
    """Rule 4: Error fields → skip."""

    @pytest.mark.parametrize("suffix", ["_error_upper", "_error_lower", "_error_index"])
    def test_error_suffixes(self, suffix: str) -> None:
        node = {
            "path": f"equilibrium/time_slice/profiles_1d/psi{suffix}",
            "data_type": "FLT_1D",
            "unit": None,
            "parent_type": None,
            "description": "Error bound",
            "node_category": None,
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"


class TestRule5StringFields:
    """Rule 5: STR_0D → skip."""

    def test_string_label(self) -> None:
        node = {
            "path": "core_profiles/profiles_1d/electrons/label",
            "data_type": "STR_0D",
            "unit": None,
            "parent_type": None,
            "description": "Species label",
            "node_category": None,
            "parent_path": "core_profiles/profiles_1d/electrons",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_string_with_unit_still_skipped(self) -> None:
        """Even if a STR_0D somehow has a unit, it's still skip."""
        node = {
            "path": "some/string_field",
            "data_type": "STR_0D",
            "unit": "eV",
            "parent_type": None,
            "description": "Unusual string field",
            "node_category": None,
            "parent_path": "some",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"


class TestRule6And7IntWithoutUnit:
    """Rules 6–7: INT_0D without unit — index/flag vs physics."""

    def test_index_keyword_skipped(self) -> None:
        node = {
            "path": "magnetics/flux_loop/index",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Index in the flux_loop array",
            "node_category": None,
            "parent_path": "magnetics/flux_loop",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_flag_keyword_skipped(self) -> None:
        node = {
            "path": "equilibrium/convergence/flag",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Convergence flag for equilibrium solve",
            "node_category": None,
            "parent_path": "equilibrium/convergence",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_number_of_keyword_skipped(self) -> None:
        node = {
            "path": "pf_active/coil/n_points",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Number of points defining the coil",
            "node_category": None,
            "parent_path": "pf_active/coil",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_count_of_keyword_skipped(self) -> None:
        node = {
            "path": "core_profiles/n_elements",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Count of elements in the array",
            "node_category": None,
            "parent_path": "core_profiles",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_identifier_keyword_skipped(self) -> None:
        node = {
            "path": "core_profiles/type/id",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Identifier for the species type",
            "node_category": None,
            "parent_path": "core_profiles/type",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_physics_int_is_quantity(self) -> None:
        """INT_0D with physics description (no index/flag keywords) → quantity."""
        node = {
            "path": "mhd_linear/toroidal_mode_number",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Toroidal mode number",
            "node_category": None,
            "parent_path": "mhd_linear",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"

    def test_int0d_with_unit_is_quantity(self) -> None:
        """INT_0D with a unit bypasses Rule 6/7 and hits Rule 8."""
        node = {
            "path": "some_ids/integer_quantity",
            "data_type": "INT_0D",
            "unit": "C",
            "parent_type": None,
            "description": "Some integer physics quantity",
            "node_category": None,
            "parent_path": "some_ids",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"


class TestRule8And9PhysicsLeaf:
    """Rules 8–9: Physics leaf types → quantity."""

    @pytest.mark.parametrize(
        "data_type",
        sorted(PHYSICS_LEAF_TYPES - {"INT_0D"}),  # INT_0D tested above
    )
    def test_physics_leaf_with_unit(self, data_type: str) -> None:
        node = {
            "path": "some_ids/some_field",
            "data_type": data_type,
            "unit": "eV",
            "parent_type": None,
            "description": "A physics field",
            "node_category": None,
            "parent_path": "some_ids",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"

    @pytest.mark.parametrize(
        "data_type",
        ["FLT_0D", "FLT_1D", "FLT_2D", "CPX_0D"],
    )
    def test_physics_leaf_without_unit_is_dimensionless_quantity(
        self, data_type: str
    ) -> None:
        node = {
            "path": "equilibrium/time_slice/profiles_1d/safety_factor",
            "data_type": data_type,
            "unit": None,
            "parent_type": None,
            "description": "Safety factor",
            "node_category": None,
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"


class TestRule10StructureWithUnit:
    """Rule 10: STRUCTURE/STRUCT_ARRAY with unit → quantity."""

    def test_structure_with_unit(self) -> None:
        node = {
            "path": "barometry/gauge/pressure",
            "data_type": "STRUCTURE",
            "unit": "Pa",
            "parent_type": None,
            "description": "Pressure measurement",
            "node_category": None,
            "parent_path": "barometry/gauge",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"

    def test_struct_array_with_unit(self) -> None:
        node = {
            "path": "some_ids/signal_array",
            "data_type": "STRUCT_ARRAY",
            "unit": "V",
            "parent_type": None,
            "description": "Signal array measurement",
            "node_category": None,
            "parent_path": "some_ids",
            "cluster_label": None,
        }
        assert classify_path(node) == "quantity"

    def test_structure_without_unit_is_skip(self) -> None:
        node = {
            "path": "core_profiles/profiles_1d/electrons",
            "data_type": "STRUCTURE",
            "unit": None,
            "parent_type": None,
            "description": "Electron species profiles",
            "node_category": None,
            "parent_path": "core_profiles/profiles_1d",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"


class TestRule11EverythingElse:
    """Rule 11: Everything else → skip."""

    def test_unknown_data_type_is_skip(self) -> None:
        node = {
            "path": "some/unknown",
            "data_type": "UNKNOWN_TYPE",
            "unit": None,
            "parent_type": None,
            "description": "Unknown thing",
            "node_category": None,
            "parent_path": "some",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_empty_data_type_is_skip(self) -> None:
        node = {
            "path": "some/node",
            "data_type": "",
            "unit": None,
            "parent_type": None,
            "description": "No data type",
            "node_category": None,
            "parent_path": "some",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"


# ============================================================================
# Edge cases and robustness
# ============================================================================


class TestEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_empty_unit_string_treated_as_none(self) -> None:
        """An empty-string unit should be normalised to None."""
        node = {
            "path": "equilibrium/time_slice/profiles_1d/safety_factor",
            "data_type": "FLT_1D",
            "unit": "",
            "parent_type": None,
            "description": "Safety factor",
            "node_category": None,
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "cluster_label": None,
        }
        # Empty unit → None → dimensionless quantity (Rule 9)
        assert classify_path(node) == "quantity"

    def test_missing_keys_dont_crash(self) -> None:
        """Minimal dict should still produce a valid classification."""
        node = {"path": "something", "data_type": "FLT_0D"}
        assert classify_path(node) == "quantity"

    def test_rule_priority_data_before_error(self) -> None:
        """/data check (Rule 1) runs before error check (Rule 4)."""
        node = {
            "path": "barometry/gauge/pressure/data",
            "data_type": "FLT_1D",
            "unit": "Pa",
            "parent_type": "STRUCTURE",
            "description": "Pressure data",
            "node_category": None,
            "parent_path": "barometry/gauge/pressure",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"

    def test_validity_takes_priority_over_int_rules(self) -> None:
        """Rule 3 (/validity) should fire before Rule 6/7 for INT_0D."""
        node = {
            "path": "equilibrium/something/validity",
            "data_type": "INT_0D",
            "unit": None,
            "parent_type": None,
            "description": "Validity flag",
            "node_category": None,
            "parent_path": "equilibrium/something",
            "cluster_label": None,
        }
        assert classify_path(node) == "metadata"

    def test_error_in_middle_of_segment_name(self) -> None:
        """_error_upper in a segment means skip, even if path looks physics-y."""
        node = {
            "path": "core_profiles/profiles_1d/electrons/temperature_error_upper",
            "data_type": "FLT_1D",
            "unit": "eV",
            "parent_type": None,
            "description": "Upper error of electron temperature",
            "node_category": None,
            "parent_path": "core_profiles/profiles_1d/electrons",
            "cluster_label": None,
        }
        assert classify_path(node) == "skip"

    def test_constants_are_exposed(self) -> None:
        """Verify public constants are importable and non-empty."""
        assert len(PHYSICS_LEAF_TYPES) >= 12
        assert len(STRUCTURE_TYPES) == 2
        assert len(ERROR_SUFFIXES) == 3
