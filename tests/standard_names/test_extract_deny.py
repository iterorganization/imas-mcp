"""Tests for the extract-phase deny gate (W19A Issue 2).

Validates that paths matched by ``config/extract_deny.yaml`` rules are
excluded from standard-name extraction and recorded as skipped SNS nodes.

Covers five denial classes:
1. Boolean constraint selectors (equilibrium /exact fields)
2. Engineering coil geometry (pf_active/pf_passive element & coil level)
3. Control-system parameters (force_self_per_unit_length)
4. Metadata uncertainty index fields
5. GGD structural grid object geometry

Tests use ACTUAL DD path strings harvested from the graph in W21 recon.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.extract_deny import (
    DenyRule,
    _glob_match,
    _load_rules,
    match_deny_rule,
)


class TestGlobMatch:
    """Verify glob pattern matching behaviour."""

    def test_exact_match(self) -> None:
        assert _glob_match("a/b/c", "a/b/c")

    def test_exact_no_match(self) -> None:
        assert not _glob_match("a/b/c", "a/b/d")

    def test_single_star_matches_one_segment(self) -> None:
        assert _glob_match("a/*/c", "a/b/c")
        assert not _glob_match("a/*/c", "a/b/d/c")

    def test_double_star_matches_zero_segments(self) -> None:
        assert _glob_match("**/c", "c")

    def test_double_star_matches_multiple_segments(self) -> None:
        assert _glob_match("**/c", "a/b/c")

    def test_trailing_wildcard(self) -> None:
        """``equilibrium/time_slice/constraints/**/exact`` matches /exact paths."""
        assert _glob_match(
            "equilibrium/time_slice/constraints/**/exact",
            "equilibrium/time_slice/constraints/flux_loop/exact",
        )
        assert _glob_match(
            "equilibrium/time_slice/constraints/**/exact",
            "equilibrium/time_slice/constraints/iron_core_segment/magnetisation_r/exact",
        )

    def test_pf_active_element_geometry(self) -> None:
        """Element-level geometry: no extra segment between element and geometry."""
        pattern = "pf_active/*/element/geometry/oblique/*"
        # Actual DD path — no index segment between element and geometry
        assert _glob_match(pattern, "pf_active/coil/element/geometry/oblique/alpha")
        assert not _glob_match(
            pattern, "pf_active/coil/element/geometry/rectangle/width"
        )

    def test_pf_active_coil_geometry(self) -> None:
        """Coil-level geometry (no element layer)."""
        pattern = "pf_active/*/geometry/oblique/*"
        assert _glob_match(pattern, "pf_active/coil/geometry/oblique/alpha")

    def test_pf_passive_element_geometry(self) -> None:
        pattern = "pf_passive/*/element/geometry/outline/*"
        assert _glob_match(pattern, "pf_passive/loop/element/geometry/outline/r")

    def test_thick_line_nested(self) -> None:
        """thick_line has nested subfields — must use ** at leaf level."""
        pattern = "pf_active/*/element/geometry/thick_line/**"
        assert _glob_match(
            pattern, "pf_active/coil/element/geometry/thick_line/first_point/r"
        )
        assert _glob_match(
            pattern, "pf_active/coil/element/geometry/thick_line/first_point/z"
        )

    def test_old_broken_pattern_does_not_match_actual_paths(self) -> None:
        """Confirm the W20B-broken pattern (*element/*/geometry/*) no longer fires."""
        broken_pattern = "pf_active/*/element/*/geometry/oblique/*"
        # This pattern requires an extra segment (e.g. '0') between element and
        # geometry which does NOT exist in the actual DD.
        assert not _glob_match(
            broken_pattern, "pf_active/coil/element/geometry/oblique/alpha"
        )


class TestDenyRuleMatching:
    """Verify DenyRule.matches method."""

    def test_matches_constraint_exact(self) -> None:
        rule = DenyRule(
            path_pattern="equilibrium/time_slice/constraints/**/exact",
            skip_reason="boolean_constraint_selector",
            reason="test",
        )
        assert rule.matches("equilibrium/time_slice/constraints/flux_loop/exact")
        assert not rule.matches("equilibrium/time_slice/profiles_1d/psi")


class TestLoadRules:
    """Verify YAML config loads correctly."""

    def test_rules_load_without_error(self) -> None:
        rules = _load_rules()
        assert len(rules) > 0, "extract_deny.yaml must contain at least one rule"

    def test_all_rules_have_required_fields(self) -> None:
        for rule in _load_rules():
            assert rule.path_pattern, f"Rule missing path_pattern: {rule}"
            assert rule.skip_reason, f"Rule missing skip_reason: {rule}"
            assert rule.reason, f"Rule missing reason: {rule}"

    def test_has_five_deny_classes(self) -> None:
        """All five skip_reason classes must be present in the config."""
        reasons = {r.skip_reason for r in _load_rules()}
        assert "boolean_constraint_selector" in reasons
        assert "engineering_coil_geometry" in reasons
        assert "control_system_parameter" in reasons
        assert "metadata_uncertainty_index" in reasons
        assert "ggd_structural_geometry" in reasons


class TestMatchDenyRule:
    """Verify deny-rule matching using actual DD paths from W21 reconnaissance."""

    # --- Class 1: Boolean constraint selectors (actual DD paths) ---

    def test_constraint_flux_loop_exact_denied(self) -> None:
        """Actual DD path: equilibrium/time_slice/constraints/flux_loop/exact."""
        rule = match_deny_rule("equilibrium/time_slice/constraints/flux_loop/exact")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_constraint_safety_factor_exact_denied(self) -> None:
        rule = match_deny_rule("equilibrium/time_slice/constraints/q/exact")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_constraint_diamagnetic_flux_exact_denied(self) -> None:
        rule = match_deny_rule(
            "equilibrium/time_slice/constraints/diamagnetic_flux/exact"
        )
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_constraint_deep_iron_core_exact_denied(self) -> None:
        """Depth-5 path: constraints/iron_core_segment/magnetisation_r/exact."""
        rule = match_deny_rule(
            "equilibrium/time_slice/constraints/iron_core_segment/magnetisation_r/exact"
        )
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_constraint_bpol_probe_exact_denied(self) -> None:
        rule = match_deny_rule("equilibrium/time_slice/constraints/bpol_probe/exact")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_constraint_ip_exact_denied(self) -> None:
        rule = match_deny_rule("equilibrium/time_slice/constraints/ip/exact")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    # --- Class 2: Engineering coil geometry (actual DD paths) ---

    # pf_active element-level
    def test_pf_active_element_oblique_alpha_denied(self) -> None:
        """Actual DD path (no index segment): pf_active/coil/element/geometry/oblique/alpha."""
        rule = match_deny_rule("pf_active/coil/element/geometry/oblique/alpha")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_oblique_beta_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/geometry/oblique/beta")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_arcs_r_denied(self) -> None:
        rule = match_deny_rule(
            "pf_active/coil/element/geometry/arcs_of_circle/curvature_radii"
        )
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_outline_r_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/geometry/outline/r")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_rectangle_height_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/geometry/rectangle/height")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_annulus_radius_inner_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/geometry/annulus/radius_inner")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_element_thick_line_first_point_r_denied(self) -> None:
        """thick_line has nested subfields — pf_active element level."""
        rule = match_deny_rule(
            "pf_active/coil/element/geometry/thick_line/first_point/r"
        )
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    # pf_active coil-level (no element layer)
    def test_pf_active_coil_oblique_alpha_denied(self) -> None:
        """Coil-level geometry path (no element layer)."""
        rule = match_deny_rule("pf_active/coil/geometry/oblique/alpha")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_coil_annulus_r_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/geometry/annulus/r")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_coil_thick_line_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/geometry/thick_line/first_point/r")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    # pf_passive element-level
    def test_pf_passive_element_oblique_denied(self) -> None:
        rule = match_deny_rule("pf_passive/loop/element/geometry/oblique/alpha")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_passive_element_outline_r_denied(self) -> None:
        rule = match_deny_rule("pf_passive/loop/element/geometry/outline/r")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_passive_element_annulus_z_denied(self) -> None:
        rule = match_deny_rule("pf_passive/loop/element/geometry/annulus/z")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_passive_element_thick_line_denied(self) -> None:
        rule = match_deny_rule(
            "pf_passive/loop/element/geometry/thick_line/first_point/r"
        )
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    # Confirm the previously broken pattern (element/*/geometry) WOULD NOT match
    def test_broken_pattern_path_still_denied_via_corrected_rule(self) -> None:
        """The old test path had an extra '0' segment; verify actual path is denied."""
        # Old broken test used: pf_active/coil/element/0/geometry/oblique/alpha
        # Real DD path is:      pf_active/coil/element/geometry/oblique/alpha
        rule = match_deny_rule("pf_active/coil/element/geometry/oblique/alpha")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    # --- Class 3: Control-system parameters ---

    def test_force_self_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/force_self_per_unit_length")
        assert rule is not None
        assert rule.skip_reason == "control_system_parameter"

    def test_force_other_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/force_other_per_unit_length")
        assert rule is not None
        assert rule.skip_reason == "control_system_parameter"

    # --- Class 4: Metadata uncertainty index ---

    def test_uncertainty_index_denied(self) -> None:
        """Forward-looking rule; no current DD paths, but future-proofing."""
        rule = match_deny_rule("some_ids/path/uncertainty_index_of_something")
        assert rule is not None
        assert rule.skip_reason == "metadata_uncertainty_index"

    # --- Class 5: GGD structural geometry ---

    def test_ggd_object_geometry_denied(self) -> None:
        """Actual DD path harvested from W21 recon."""
        rule = match_deny_rule(
            "distribution_sources/source/ggd/grid/space/objects_per_dimension/object/geometry"
        )
        assert rule is not None
        assert rule.skip_reason == "ggd_structural_geometry"

    def test_ggd_object_geometry_2d_denied(self) -> None:
        rule = match_deny_rule(
            "edge_profiles/grid_ggd/space/objects_per_dimension/object/geometry_2d"
        )
        assert rule is not None
        assert rule.skip_reason == "ggd_structural_geometry"

    def test_ggd_geometry_content_denied(self) -> None:
        rule = match_deny_rule(
            "distribution_sources/source/ggd/grid/space/objects_per_dimension/geometry_content"
        )
        assert rule is not None
        assert rule.skip_reason == "ggd_structural_geometry"

    def test_ggd_geometry_content_index_denied(self) -> None:
        rule = match_deny_rule(
            "edge_profiles/grid_ggd/space/objects_per_dimension/geometry_content/index"
        )
        assert rule is not None
        assert rule.skip_reason == "ggd_structural_geometry"

    # --- Paths that should NOT be denied ---

    def test_electron_temperature_not_denied(self) -> None:
        """Physics quantities must pass through."""
        assert (
            match_deny_rule("core_profiles/profiles_1d/electrons/temperature") is None
        )

    def test_equilibrium_psi_not_denied(self) -> None:
        assert match_deny_rule("equilibrium/time_slice/profiles_1d/psi") is None

    def test_pf_active_current_not_denied(self) -> None:
        """Current in PF coil is a physics quantity, not denied."""
        assert match_deny_rule("pf_active/coil/current/data") is None

    def test_magnetics_flux_loop_flux_not_denied(self) -> None:
        """The actual flux loop measurement — must pass through."""
        assert match_deny_rule("magnetics/flux_loop/flux/data") is None

    def test_boundary_r_not_denied(self) -> None:
        """Plasma boundary geometry is physics, not engineering."""
        assert match_deny_rule("equilibrium/time_slice/boundary/outline/r") is None

    def test_equilibrium_constraint_measured_value_not_denied(self) -> None:
        """The 'measured' sub-field of a constraint is physics, not a selector."""
        assert (
            match_deny_rule(
                "equilibrium/time_slice/constraints/flux_loop/measured/data"
            )
            is None
        )

    def test_equilibrium_constraint_source_not_denied(self) -> None:
        """Source field of a constraint is not a boolean selector."""
        assert (
            match_deny_rule("equilibrium/time_slice/constraints/flux_loop/source")
            is None
        )


class TestActualDDPathsDenied:
    """Regression tests: actual DD paths from W21 graph reconnaissance must be denied.

    If these tests fail, the deny config has regressed and real engineering
    paths will pollute the standard-name compose queue.
    """

    EQ_CONSTRAINT_EXACT = [
        "equilibrium/time_slice/constraints/b_field_pol_probe/exact",
        "equilibrium/time_slice/constraints/b_field_tor_vacuum_r/exact",
        "equilibrium/time_slice/constraints/bpol_probe/exact",
        "equilibrium/time_slice/constraints/diamagnetic_flux/exact",
        "equilibrium/time_slice/constraints/faraday_angle/exact",
        "equilibrium/time_slice/constraints/flux_loop/exact",
        "equilibrium/time_slice/constraints/ip/exact",
        "equilibrium/time_slice/constraints/iron_core_segment/magnetisation_r/exact",
        "equilibrium/time_slice/constraints/iron_core_segment/magnetisation_z/exact",
        "equilibrium/time_slice/constraints/iron_core_segment/magnetization_r/exact",
        "equilibrium/time_slice/constraints/iron_core_segment/magnetization_z/exact",
        "equilibrium/time_slice/constraints/j_parallel/exact",
        "equilibrium/time_slice/constraints/j_phi/exact",
        "equilibrium/time_slice/constraints/j_tor/exact",
        "equilibrium/time_slice/constraints/mse_polarisation_angle/exact",
        "equilibrium/time_slice/constraints/n_e/exact",
        "equilibrium/time_slice/constraints/n_e_line/exact",
        "equilibrium/time_slice/constraints/pf_current/exact",
        "equilibrium/time_slice/constraints/pressure/exact",
        "equilibrium/time_slice/constraints/q/exact",
        "equilibrium/time_slice/constraints/strike_point/exact",
        "equilibrium/time_slice/constraints/x_point/exact",
    ]

    PF_ACTIVE_ELEMENT_GEOMETRY = [
        "pf_active/coil/element/geometry/oblique/alpha",
        "pf_active/coil/element/geometry/oblique/beta",
        "pf_active/coil/element/geometry/oblique/r",
        "pf_active/coil/element/geometry/oblique/z",
        "pf_active/coil/element/geometry/oblique/thickness",
        "pf_active/coil/element/geometry/oblique/length_alpha",
        "pf_active/coil/element/geometry/oblique/length_beta",
        "pf_active/coil/element/geometry/arcs_of_circle/r",
        "pf_active/coil/element/geometry/arcs_of_circle/z",
        "pf_active/coil/element/geometry/arcs_of_circle/curvature_radii",
        "pf_active/coil/element/geometry/outline/r",
        "pf_active/coil/element/geometry/outline/z",
        "pf_active/coil/element/geometry/rectangle/height",
        "pf_active/coil/element/geometry/annulus/r",
        "pf_active/coil/element/geometry/annulus/z",
        "pf_active/coil/element/geometry/annulus/radius_inner",
        "pf_active/coil/element/geometry/annulus/radius_outer",
        "pf_active/coil/element/geometry/thick_line/first_point/r",
        "pf_active/coil/element/geometry/thick_line/first_point/z",
    ]

    PF_ACTIVE_COIL_GEOMETRY = [
        "pf_active/coil/geometry/oblique/alpha",
        "pf_active/coil/geometry/oblique/beta",
        "pf_active/coil/geometry/arcs_of_circle/r",
        "pf_active/coil/geometry/arcs_of_circle/curvature_radii",
        "pf_active/coil/geometry/outline/r",
        "pf_active/coil/geometry/rectangle/height",
        "pf_active/coil/geometry/annulus/r",
        "pf_active/coil/geometry/annulus/radius_inner",
        "pf_active/coil/geometry/thick_line/first_point/r",
    ]

    PF_PASSIVE_ELEMENT_GEOMETRY = [
        "pf_passive/loop/element/geometry/oblique/alpha",
        "pf_passive/loop/element/geometry/oblique/beta",
        "pf_passive/loop/element/geometry/arcs_of_circle/r",
        "pf_passive/loop/element/geometry/outline/r",
        "pf_passive/loop/element/geometry/rectangle/height",
        "pf_passive/loop/element/geometry/annulus/r",
        "pf_passive/loop/element/geometry/annulus/radius_inner",
        "pf_passive/loop/element/geometry/thick_line/first_point/r",
    ]

    def test_all_eq_constraint_exact_denied(self) -> None:
        for path in self.EQ_CONSTRAINT_EXACT:
            rule = match_deny_rule(path)
            assert rule is not None, f"Not denied: {path}"
            assert rule.skip_reason == "boolean_constraint_selector", (
                f"Wrong reason for {path}: {rule.skip_reason}"
            )

    def test_all_pf_active_element_geometry_denied(self) -> None:
        for path in self.PF_ACTIVE_ELEMENT_GEOMETRY:
            rule = match_deny_rule(path)
            assert rule is not None, f"Not denied: {path}"
            assert rule.skip_reason == "engineering_coil_geometry", (
                f"Wrong reason for {path}: {rule.skip_reason}"
            )

    def test_all_pf_active_coil_geometry_denied(self) -> None:
        for path in self.PF_ACTIVE_COIL_GEOMETRY:
            rule = match_deny_rule(path)
            assert rule is not None, f"Not denied: {path}"
            assert rule.skip_reason == "engineering_coil_geometry", (
                f"Wrong reason for {path}: {rule.skip_reason}"
            )

    def test_all_pf_passive_element_geometry_denied(self) -> None:
        for path in self.PF_PASSIVE_ELEMENT_GEOMETRY:
            rule = match_deny_rule(path)
            assert rule is not None, f"Not denied: {path}"
            assert rule.skip_reason == "engineering_coil_geometry", (
                f"Wrong reason for {path}: {rule.skip_reason}"
            )


class TestApplyExtractDeny:
    """Verify the pipeline integration function."""

    def test_filters_denied_paths(self) -> None:
        from imas_codex.standard_names.sources.dd import _apply_extract_deny

        rows = [
            {
                "path": "equilibrium/time_slice/constraints/flux_loop/exact",
                "description": "flag",
            },
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Te",
            },
            {
                "path": "pf_active/coil/element/geometry/oblique/alpha",
                "description": "angle",
            },
        ]
        kept = _apply_extract_deny(rows, write_skipped=False)
        paths = [r["path"] for r in kept]
        assert len(kept) == 1
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_empty_input(self) -> None:
        from imas_codex.standard_names.sources.dd import _apply_extract_deny

        assert _apply_extract_deny([], write_skipped=False) == []

    def test_all_kept_when_no_matches(self) -> None:
        from imas_codex.standard_names.sources.dd import _apply_extract_deny

        rows = [
            {"path": "equilibrium/time_slice/profiles_1d/psi", "description": "psi"},
            {
                "path": "core_profiles/profiles_1d/electrons/density",
                "description": "ne",
            },
        ]
        kept = _apply_extract_deny(rows, write_skipped=False)
        assert len(kept) == 2
