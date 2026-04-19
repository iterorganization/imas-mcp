"""Tests for the extended representation_artifact_check audit.

Plan 31 WS-A (A.4) — extended regex to cover ggd-coefficient,
interpolation-coefficient, finite-element coefficient real/imaginary
parts, and a heuristic ``_on_ggd$`` suffix gated by DD source path.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.audits import representation_artifact_check

# =========================================================================
# Existing suffixes still flagged
# =========================================================================


class TestRepresentationArtifactExisting:
    """Baseline suffixes already flagged by the original regex."""

    @pytest.mark.parametrize(
        "name",
        [
            "magnetic_field_coefficients",
            "electron_density_basis",
            "pressure_spline",
            "poloidal_flux_fourier_modes",
            "current_density_harmonics_coefficients",
        ],
    )
    def test_baseline_flagged(self, name):
        issues = representation_artifact_check({"id": name})
        assert issues and "representation_artifact_check" in issues[0]


# =========================================================================
# A.4 — new suffixes flagged
# =========================================================================


class TestRepresentationArtifactExtended:
    """New suffix variants added by plan 31 A.4."""

    @pytest.mark.parametrize(
        "name",
        [
            # _ggd_coefficients
            "electron_density_ggd_coefficients",
            "ion_temperature_ggd_coefficients",
            # _coefficient_on_ggd
            "electron_density_coefficient_on_ggd",
            "magnetic_field_coefficient_on_ggd",
            # _interpolation_coefficient(s)
            "electron_density_interpolation_coefficient",
            "magnetic_field_interpolation_coefficients",
            # _interpolation_coefficient(s)_on_ggd
            "electron_density_interpolation_coefficient_on_ggd",
            "magnetic_field_interpolation_coefficients_on_ggd",
            # _finite_element_coefficients_(real|imaginary)_part
            "poloidal_flux_finite_element_coefficients_real_part",
            "poloidal_flux_finite_element_coefficients_imaginary_part",
            # Base finite-element coefficients
            "electron_density_finite_element_coefficients",
            "pressure_finite_element_interpolation_coefficients",
        ],
    )
    def test_extended_flagged(self, name):
        issues = representation_artifact_check({"id": name})
        assert issues, f"expected '{name}' to be flagged"
        assert "representation_artifact_check" in issues[0]


# =========================================================================
# A.4 — heuristic _on_ggd$ gated by source_path
# =========================================================================


class TestOnGgdHeuristic:
    """Bare ``_on_ggd$`` suffix only fires when source_path carries GGD."""

    def test_on_ggd_with_ggd_source_path_flagged(self):
        issues = representation_artifact_check(
            {"id": "electron_density_on_ggd"},
            source_path="edge_profiles/ggd/electrons/density",
        )
        assert issues and "_on_ggd" in issues[0]

    def test_on_ggd_with_grids_ggd_source_path_flagged(self):
        issues = representation_artifact_check(
            {"id": "magnetic_field_on_ggd"},
            source_path="equilibrium/grids_ggd/grid/space",
        )
        assert issues and "_on_ggd" in issues[0]

    def test_on_ggd_without_source_path_still_flagged_by_general_regex(self):
        # The general ``_ggd$`` alternative in the main regex matches any
        # ``*_on_ggd`` name — so even without a source_path, it is flagged.
        # The heuristic path is additive insurance; the general path fires first.
        issues = representation_artifact_check(
            {"id": "electron_density_on_ggd"},
            source_path=None,
        )
        assert issues and "representation_artifact_check" in issues[0]

    def test_on_ggd_with_non_ggd_source_path_still_flagged(self):
        # Same: the general ``_ggd$`` rule fires regardless of source_path.
        issues = representation_artifact_check(
            {"id": "electron_density_on_ggd"},
            source_path="core_profiles/profiles_1d/electrons/density_on_ggd",
        )
        assert issues and "representation_artifact_check" in issues[0]

    def test_legitimate_on_axis_not_flagged(self):
        # Non-_on_ggd suffix must never be flagged by the heuristic.
        issues = representation_artifact_check(
            {"id": "electron_density_on_axis"},
            source_path="edge_profiles/ggd/electrons/density",
        )
        assert issues == []


# =========================================================================
# Negative cases — physics quantities unaffected
# =========================================================================


class TestRepresentationArtifactNegatives:
    """Plain physics names must not be flagged."""

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "electron_density",
            "plasma_current",
            "poloidal_flux",
            "toroidal_magnetic_field",
            "diamagnetic_flux",
        ],
    )
    def test_plain_quantity_unflagged(self, name):
        assert representation_artifact_check({"id": name}) == []

    def test_empty_name_unflagged(self):
        assert representation_artifact_check({"id": ""}) == []
        assert representation_artifact_check({}) == []
