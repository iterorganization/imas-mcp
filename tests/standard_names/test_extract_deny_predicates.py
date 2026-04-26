"""Tests for attribute-predicate deny rules (W29 — Phase 1).

Validates the extended ``DenyRule`` schema with attribute predicates
(``data_type_in``, ``units_empty``, ``doc_contains_any``) and the new
``configuration_flag`` deny class for DD paths that are configuration
metadata rather than physical quantities.

Covers:
  - Predicate parsing from YAML
  - Predicate evaluation (all combinations)
  - ``data_type_in`` filter
  - ``units_empty`` filter (handles None, empty, '-')
  - ``doc_contains_any`` (case-insensitive)
  - Backward compat: rules without predicates work as before
  - New ``configuration_flag`` class catches known flag paths
  - ``status`` field propagation (``not_physical_quantity``)
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.extract_deny import (
    DenyRule,
    _load_rules,
    _predicates_match,
    match_deny_rule,
)

# ---------------------------------------------------------------------------
# DenyRule construction
# ---------------------------------------------------------------------------


class TestDenyRuleConstruction:
    """Verify DenyRule dataclass defaults and field storage."""

    def test_defaults_backward_compat(self) -> None:
        rule = DenyRule(path_pattern="**/exact", skip_reason="test", reason="test")
        assert rule.data_type_in == ()
        assert rule.units_empty is False
        assert rule.doc_contains_any == ()
        assert rule.status == "skipped"

    def test_full_construction(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="configuration_flag",
            reason="flag",
            data_type_in=("INT_0D", "INT_1D"),
            units_empty=True,
            doc_contains_any=("Flag = 1", "Flag = 0"),
            status="not_physical_quantity",
        )
        assert rule.data_type_in == ("INT_0D", "INT_1D")
        assert rule.units_empty is True
        assert rule.doc_contains_any == ("Flag = 1", "Flag = 0")
        assert rule.status == "not_physical_quantity"


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


class TestPredicatesMatch:
    """Test _predicates_match logic in isolation."""

    def test_no_predicates_always_matches(self) -> None:
        rule = DenyRule(path_pattern="**", skip_reason="x", reason="x")
        assert _predicates_match(rule, None) is True
        assert _predicates_match(rule, {}) is True

    def test_predicates_without_attrs_fails(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            data_type_in=("INT_0D",),
        )
        assert _predicates_match(rule, None) is False

    def test_data_type_in_match(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            data_type_in=("INT_0D", "INT_1D"),
        )
        assert _predicates_match(rule, {"data_type": "INT_0D"}) is True
        assert _predicates_match(rule, {"data_type": "INT_1D"}) is True
        assert _predicates_match(rule, {"data_type": "FLT_0D"}) is False

    def test_units_empty_none(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            units_empty=True,
        )
        assert _predicates_match(rule, {"units": None}) is True

    def test_units_empty_empty_string(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            units_empty=True,
        )
        assert _predicates_match(rule, {"units": ""}) is True

    def test_units_empty_dash(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            units_empty=True,
        )
        assert _predicates_match(rule, {"units": "-"}) is True

    def test_units_empty_fails_real_unit(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            units_empty=True,
        )
        assert _predicates_match(rule, {"units": "m"}) is False
        assert _predicates_match(rule, {"units": "eV"}) is False

    def test_doc_contains_any_case_insensitive(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            doc_contains_any=("Flag = 1",),
        )
        assert (
            _predicates_match(
                rule, {"documentation": "flag = 1 if collision conserves momentum"}
            )
            is True
        )

    def test_doc_contains_any_no_match(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            doc_contains_any=("Flag = 1",),
        )
        assert (
            _predicates_match(rule, {"documentation": "Electron temperature in eV"})
            is False
        )

    def test_doc_contains_any_none_doc(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            doc_contains_any=("Flag = 1",),
        )
        assert _predicates_match(rule, {"documentation": None}) is False

    def test_all_predicates_must_hold(self) -> None:
        """When multiple predicates are specified, ALL must match."""
        rule = DenyRule(
            path_pattern="**",
            skip_reason="x",
            reason="x",
            data_type_in=("INT_0D",),
            units_empty=True,
            doc_contains_any=("Flag = 1",),
        )
        # All match
        attrs = {
            "data_type": "INT_0D",
            "units": None,
            "documentation": "Flag = 1 if conservation is enforced",
        }
        assert _predicates_match(rule, attrs) is True

        # data_type mismatch
        attrs_bad_dt = {**attrs, "data_type": "FLT_0D"}
        assert _predicates_match(rule, attrs_bad_dt) is False

        # units not empty
        attrs_bad_units = {**attrs, "units": "eV"}
        assert _predicates_match(rule, attrs_bad_units) is False

        # doc mismatch
        attrs_bad_doc = {**attrs, "documentation": "Electron density"}
        assert _predicates_match(rule, attrs_bad_doc) is False


# ---------------------------------------------------------------------------
# DenyRule.matches with predicates
# ---------------------------------------------------------------------------


class TestDenyRuleMatchesWithPredicates:
    """Test DenyRule.matches() with path + predicate evaluation."""

    def test_path_only_rule_no_attrs(self) -> None:
        """Path-only rules match without node_attrs (backward compat)."""
        rule = DenyRule(
            path_pattern="equilibrium/**/exact",
            skip_reason="boolean_constraint_selector",
            reason="test",
        )
        assert (
            rule.matches("equilibrium/time_slice/constraints/flux_loop/exact") is True
        )

    def test_predicate_rule_requires_attrs(self) -> None:
        """Rules with predicates need attrs to match."""
        rule = DenyRule(
            path_pattern="**",
            skip_reason="configuration_flag",
            reason="test",
            data_type_in=("INT_0D",),
            units_empty=True,
        )
        # No attrs → fails
        assert (
            rule.matches("gyrokinetics/model/collisions_momentum_conservation") is False
        )

    def test_predicate_rule_matches_with_attrs(self) -> None:
        rule = DenyRule(
            path_pattern="**",
            skip_reason="configuration_flag",
            reason="test",
            data_type_in=("INT_0D",),
            units_empty=True,
            doc_contains_any=("Flag = 1",),
        )
        attrs = {
            "data_type": "INT_0D",
            "units": None,
            "documentation": "Flag = 1 if the collision operator conserves momentum",
        }
        assert (
            rule.matches("gyrokinetics/model/collisions_momentum_conservation", attrs)
            is True
        )

    def test_path_mismatch_short_circuits(self) -> None:
        """If the path doesn't match, predicates are never checked."""
        rule = DenyRule(
            path_pattern="gyrokinetics/**",
            skip_reason="configuration_flag",
            reason="test",
            data_type_in=("INT_0D",),
        )
        attrs = {"data_type": "INT_0D", "units": None, "documentation": "Flag = 1"}
        # Path doesn't match
        assert (
            rule.matches("core_profiles/profiles_1d/electrons/temperature", attrs)
            is False
        )


# ---------------------------------------------------------------------------
# YAML loading with predicates
# ---------------------------------------------------------------------------


class TestLoadRulesPredicates:
    """Verify _load_rules parses optional predicate fields."""

    def test_configuration_flag_class_loaded(self) -> None:
        """The configuration_flag class must be present with predicates."""
        _load_rules.cache_clear()
        try:
            rules = _load_rules()
            flag_rules = [r for r in rules if r.skip_reason == "configuration_flag"]
            assert len(flag_rules) == 1, (
                f"Expected 1 configuration_flag rule, found {len(flag_rules)}"
            )
            rule = flag_rules[0]
            assert rule.status == "not_physical_quantity"
            assert "INT_0D" in rule.data_type_in
            assert "INT_1D" in rule.data_type_in
            assert rule.units_empty is True
            assert len(rule.doc_contains_any) >= 3
        finally:
            _load_rules.cache_clear()

    def test_backward_compat_rules_still_load(self) -> None:
        """Existing path-only rules still load correctly."""
        _load_rules.cache_clear()
        try:
            rules = _load_rules()
            path_only = [
                r for r in rules if r.skip_reason == "boolean_constraint_selector"
            ]
            assert len(path_only) >= 1
            rule = path_only[0]
            assert rule.data_type_in == ()
            assert rule.units_empty is False
            assert rule.doc_contains_any == ()
            assert rule.status == "skipped"
        finally:
            _load_rules.cache_clear()


# ---------------------------------------------------------------------------
# match_deny_rule with node_attrs
# ---------------------------------------------------------------------------


class TestMatchDenyRuleWithAttrs:
    """Test match_deny_rule with the node_attrs parameter."""

    def test_path_only_match_backward_compat(self) -> None:
        """Existing path-only rules still match without node_attrs."""
        _load_rules.cache_clear()
        try:
            rule = match_deny_rule("equilibrium/time_slice/constraints/flux_loop/exact")
            assert rule is not None
            assert rule.skip_reason == "boolean_constraint_selector"
        finally:
            _load_rules.cache_clear()

    def test_configuration_flag_matches(self) -> None:
        """Configuration flag rule matches with correct attrs."""
        _load_rules.cache_clear()
        try:
            attrs = {
                "data_type": "INT_0D",
                "units": None,
                "documentation": (
                    "Flag = 1 if the collision operator conserves momentum, 0 otherwise"
                ),
            }
            rule = match_deny_rule(
                "gyrokinetics/model/collisions_momentum_conservation",
                node_attrs=attrs,
            )
            assert rule is not None
            assert rule.skip_reason == "configuration_flag"
            assert rule.status == "not_physical_quantity"
        finally:
            _load_rules.cache_clear()

    def test_configuration_flag_no_match_wrong_type(self) -> None:
        """Configuration flag rule doesn't match FLT_0D paths."""
        _load_rules.cache_clear()
        try:
            attrs = {
                "data_type": "FLT_0D",
                "units": None,
                "documentation": "Flag = 1 if something",
            }
            rule = match_deny_rule(
                "gyrokinetics/model/collisions_momentum_conservation",
                node_attrs=attrs,
            )
            # Should NOT match — configuration_flag requires INT_0D/INT_1D
            # (may match a path-only rule if one covers this path, but
            # configuration_flag specifically should not)
            if rule is not None:
                assert rule.skip_reason != "configuration_flag"
        finally:
            _load_rules.cache_clear()

    def test_configuration_flag_no_match_has_units(self) -> None:
        """Configuration flag rule doesn't match paths with real units."""
        _load_rules.cache_clear()
        try:
            attrs = {
                "data_type": "INT_0D",
                "units": "eV",
                "documentation": "Flag = 1 if something",
            }
            rule = match_deny_rule(
                "gyrokinetics/model/collisions_momentum_conservation",
                node_attrs=attrs,
            )
            if rule is not None:
                assert rule.skip_reason != "configuration_flag"
        finally:
            _load_rules.cache_clear()

    def test_configuration_flag_no_match_wrong_doc(self) -> None:
        """Configuration flag rule doesn't match non-flag documentation."""
        _load_rules.cache_clear()
        try:
            attrs = {
                "data_type": "INT_0D",
                "units": None,
                "documentation": "Number of modes in the simulation",
            }
            rule = match_deny_rule(
                "gyrokinetics/model/collisions_momentum_conservation",
                node_attrs=attrs,
            )
            if rule is not None:
                assert rule.skip_reason != "configuration_flag"
        finally:
            _load_rules.cache_clear()

    def test_real_quantity_not_caught(self) -> None:
        """A real physics quantity with FLT_0D and real units is not denied."""
        _load_rules.cache_clear()
        try:
            attrs = {
                "data_type": "FLT_0D",
                "units": "eV",
                "documentation": "Electron temperature",
            }
            rule = match_deny_rule(
                "core_profiles/profiles_1d/electrons/temperature",
                node_attrs=attrs,
            )
            assert rule is None
        finally:
            _load_rules.cache_clear()


# ---------------------------------------------------------------------------
# Known configuration flag paths (from W29 evidence)
# ---------------------------------------------------------------------------


class TestKnownConfigurationFlags:
    """Verify the 26 known configuration flag paths match the new rule."""

    # Representative sample from the 26 confirmed paths (87% gyrokinetics)
    KNOWN_FLAGS = [
        (
            "gyrokinetics/model/collisions_momentum_conservation",
            "INT_0D",
            "Flag = 1 if the collision operator conserves momentum, 0 otherwise",
        ),
        (
            "gyrokinetics/model/collisions_energy_conservation",
            "INT_0D",
            "Flag = 1 if the collision operator conserves energy, 0 otherwise",
        ),
        (
            "gyrokinetics/model/collisions_finite_larmor_radius",
            "INT_0D",
            "Flag = 1 if the collision operator includes finite Larmor radius effects",
        ),
        (
            "gyrokinetics/model/non_linear_run",
            "INT_0D",
            "Flag = 1 if the simulation is non-linear, 0 if not",
        ),
        (
            "gyrokinetics/model/initial_value_run",
            "INT_0D",
            "Flag = 1 if the simulation is an initial value run, 0 if not",
        ),
    ]

    @pytest.mark.parametrize(
        "path,data_type,doc",
        KNOWN_FLAGS,
        ids=[p[0].split("/")[-1] for p in KNOWN_FLAGS],
    )
    def test_known_flag_matched(self, path: str, data_type: str, doc: str) -> None:
        _load_rules.cache_clear()
        try:
            attrs = {"data_type": data_type, "units": None, "documentation": doc}
            rule = match_deny_rule(path, node_attrs=attrs)
            assert rule is not None, f"Expected {path} to be denied"
            assert rule.skip_reason == "configuration_flag"
            assert rule.status == "not_physical_quantity"
        finally:
            _load_rules.cache_clear()
