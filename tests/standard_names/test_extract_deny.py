"""Tests for the extract-phase deny gate (W19A Issue 2).

Validates that paths matched by ``config/extract_deny.yaml`` rules are
excluded from standard-name extraction and recorded as skipped SNS nodes.

Covers three denial classes:
1. Boolean constraint selectors (``use_exact_*``)
2. Engineering coil geometry (``pf_active/.../geometry/oblique/...``)
3. Control-system parameters (``force_self_per_unit_length``)
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
        """``**/use_exact_*`` matches paths ending with use_exact_X."""
        assert _glob_match(
            "**/use_exact_*", "equilibrium/time_slice/constraints/use_exact_phi"
        )
        assert _glob_match(
            "**/use_exact_*", "equilibrium/time_slice/constraints/use_exact_bpol"
        )

    def test_pf_active_geometry(self) -> None:
        pattern = "pf_active/*/element/*/geometry/oblique/*"
        assert _glob_match(pattern, "pf_active/coil/element/0/geometry/oblique/alpha")
        assert not _glob_match(
            pattern, "pf_active/coil/element/0/geometry/rectangle/width"
        )

    def test_pf_passive_geometry(self) -> None:
        pattern = "pf_passive/*/element/*/geometry/outline/*"
        assert _glob_match(pattern, "pf_passive/loop/element/0/geometry/outline/r")


class TestDenyRuleMatching:
    """Verify DenyRule.matches method."""

    def test_matches(self) -> None:
        rule = DenyRule(
            path_pattern="**/use_exact_*",
            skip_reason="boolean_constraint_selector",
            reason="test",
        )
        assert rule.matches("equilibrium/time_slice/constraints/use_exact_phi")
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


class TestMatchDenyRule:
    """Verify deny-rule matching for representative paths from each class."""

    # --- Class 1: Boolean constraint selectors ---

    def test_use_exact_phi_denied(self) -> None:
        rule = match_deny_rule("equilibrium/time_slice/constraints/use_exact_phi")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_use_exact_bpol_denied(self) -> None:
        rule = match_deny_rule(
            "equilibrium/time_slice/constraints/use_exact_bpol_probe"
        )
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    def test_use_exact_deep_path(self) -> None:
        """use_exact_* works regardless of nesting depth."""
        rule = match_deny_rule("equilibrium/time_slice/constraints/q/use_exact_q")
        assert rule is not None
        assert rule.skip_reason == "boolean_constraint_selector"

    # --- Class 2: Engineering coil geometry ---

    def test_pf_active_oblique_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/0/geometry/oblique/alpha")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_rectangle_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/0/geometry/rectangle/width")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_passive_outline_denied(self) -> None:
        rule = match_deny_rule("pf_passive/loop/element/0/geometry/outline/r")
        assert rule is not None
        assert rule.skip_reason == "engineering_coil_geometry"

    def test_pf_active_arcs_denied(self) -> None:
        rule = match_deny_rule("pf_active/coil/element/0/geometry/arcs_of_circle/r")
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

    def test_boundary_r_not_denied(self) -> None:
        """Plasma boundary geometry is physics, not engineering."""
        assert match_deny_rule("equilibrium/time_slice/boundary/outline/r") is None


class TestApplyExtractDeny:
    """Verify the pipeline integration function."""

    def test_filters_denied_paths(self) -> None:
        from imas_codex.standard_names.sources.dd import _apply_extract_deny

        rows = [
            {
                "path": "equilibrium/time_slice/constraints/use_exact_phi",
                "description": "flag",
            },
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Te",
            },
            {
                "path": "pf_active/coil/element/0/geometry/oblique/alpha",
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
