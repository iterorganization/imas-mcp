"""Tests for the _parent_supports_uncertainty_index semantic gate (Phase C).

Verifies that mint_error_siblings() skips uncertainty_index_of_<P> siblings
when the parent name or unit is semantically unsuitable, while still minting
upper/lower uncertainty siblings unconditionally.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: produce a pass-through ISN grammar mock so integration tests don't
# depend on the imas_standard_names package parsing specific name strings.
# ---------------------------------------------------------------------------


def _make_isn_passthrough():
    """Return a pair of (parse_mock, compose_mock) that echo input unchanged."""

    def _parse(name: str):
        mock_result = MagicMock()
        mock_result.ir = name
        return mock_result

    def _compose(ir):
        return ir  # identity

    return _parse, _compose


# ---------------------------------------------------------------------------
# Unit tests for _parent_supports_uncertainty_index
# ---------------------------------------------------------------------------


class TestParentSupportsUncertaintyIndex:
    """Direct unit tests for the gate helper function."""

    def test_allow_temperature(self):
        """W24 policy gate: ALL parents now return False for uncertainty_index.

        Prior to W24 audit, dimensional scalars were allowed.  W20A/W20B/W24
        audits found zero useful uncertainty_index_of_* names; Rule 6 closes
        the gate unconditionally.
        """
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("electron_temperature", "eV") is False

    def test_allow_current(self):
        """W24 policy gate: dimensional scalar (A) now also blocked by Rule 6."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("plasma_current", "A") is False

    def test_allow_ion_density(self):
        """W24 policy gate: dimensional scalar (m^-3) now also blocked by Rule 6."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("ion_density", "m^-3") is False

    def test_deny_process_term(self):
        """Name containing _due_to_ → denied (process attribution)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("power_due_to_thermalization", "W")
            is False
        )

    def test_deny_caused_by_pattern(self):
        """Name containing caused_by_ → denied (process attribution)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("energy_caused_by_radiation", "J")
            is False
        )

    def test_deny_dimensionless_empty(self):
        """Empty unit string → denied (dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("safety_factor", "") is False

    def test_deny_unit_one(self):
        """Unit '1' → denied (explicitly dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("safety_factor", "1") is False

    def test_deny_unit_none(self):
        """Unit None → denied (no unit = dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("some_quantity", None) is False

    def test_deny_unit_dash(self):
        """Unit '-' → denied (dimensionless dash convention)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("some_quantity", "-") is False

    def test_deny_status_suffix(self):
        """Name ending in _status → denied (categorical field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("plasma_status", "") is False

    def test_deny_type_suffix(self):
        """Name ending in _type → denied (categorical field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("ion_type", "") is False

    def test_deny_index_suffix(self):
        """Name ending in _index with dimensionless unit → denied."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("phase_index", "") is False

    def test_deny_id_suffix(self):
        """Name ending in _id → denied (identifier field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("node_id", "") is False

    def test_deny_label_suffix(self):
        """Name ending in _label → denied (categorical label)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("grid_label", "") is False

    def test_deny_constant_prefix(self):
        """Name starting with constant_ → denied (data-type descriptor)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("constant_float_value", "m") is False

    def test_deny_generic_prefix(self):
        """Name starting with generic_ → denied (data-type descriptor)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("generic_quantity", "Pa") is False


# ---------------------------------------------------------------------------
# Integration tests for mint_error_siblings (gate wired in)
# ---------------------------------------------------------------------------


class TestMintErrorSiblingsGate:
    """Integration tests verifying the gate is applied inside mint_error_siblings."""

    def test_mint_skips_denied_parent(self):
        """Process-term parent → no uncertainty_index sibling produced."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "power_due_to_thermalization",
                error_node_ids=[
                    "fast_particles/power_due_to_thermalization_error_index",
                ],
                unit="W",
                physics_domain="heating",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"Expected no uncertainty_index sibling, got: {ids}"
        )

    def test_mint_allows_approved_parent(self):
        """W24 policy gate: uncertainty_index is NO LONGER produced for any parent.

        Prior to W24, dimensional parents (eV) were allowed to produce
        uncertainty_index_of_* siblings.  Rule 6 now blocks all of them.
        upper/lower uncertainty siblings are still produced (not gated).
        """
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "electron_temperature",
                error_node_ids=[
                    "core_profiles/profiles_1d/electrons/temperature_error_index",
                ],
                unit="eV",
                physics_domain="transport",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"W24 policy gate: uncertainty_index siblings must not be produced, got: {ids}"
        )

    def test_upper_lower_not_blocked_for_denied_parent(self):
        """Gate only applies to _error_index; upper/lower always pass through."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "power_due_to_thermalization",
                error_node_ids=[
                    "fast_particles/power_due_to_thermalization_error_upper",
                    "fast_particles/power_due_to_thermalization_error_lower",
                    "fast_particles/power_due_to_thermalization_error_index",
                ],
                unit="W",
                physics_domain="heating",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        # upper and lower pass through, index is blocked
        assert len(siblings) == 2, f"Expected 2 siblings (upper+lower), got: {ids}"
        assert any("upper_uncertainty" in sid for sid in ids)
        assert any("lower_uncertainty" in sid for sid in ids)
        assert not any("uncertainty_index" in sid for sid in ids)

    def test_dimensionless_unit_blocks_index_only(self):
        """Dimensionless unit blocks uncertainty_index but not upper/lower."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "safety_factor",
                error_node_ids=[
                    "x/safety_factor_error_upper",
                    "x/safety_factor_error_lower",
                    "x/safety_factor_error_index",
                ],
                unit="1",  # dimensionless
                physics_domain="equilibrium",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert len(siblings) == 2, f"Expected 2 siblings (upper+lower), got: {ids}"
        assert not any("uncertainty_index" in sid for sid in ids)


class TestW24UncertaintyIndexLeak:
    """Regression tests for W24 audit: uncertainty_index_of_* leak via error_siblings.

    Root cause: error_siblings pipeline mints uncertainty_index_of_<parent>
    deterministically from parent names + error_node_ids, bypassing the
    extract_deny gate (which only applies to DD path extraction).  For
    dimensional parents like current density (A.m^-2), the previous gate
    allowed the sibling.  W24 confirmed 5 leaks:
      - uncertainty_index_of_vertical_component_of_inertial_current_density
      - uncertainty_index_of_radial_component_of_diamagnetic_current_density
      (and 3 similar forms)

    Fix: Rule 6 in _parent_supports_uncertainty_index always returns False,
    blocking ALL uncertainty_index_of_* sibling creation.
    """

    def test_current_density_component_blocked(self):
        """W24 leak pattern: component_of current density must not produce uncertainty_index."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.error_siblings import mint_error_siblings

        def _parse(name):
            m = MagicMock()
            m.ir = name
            return m

        with (
            patch("imas_standard_names.grammar.parser.parse", side_effect=_parse),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=lambda ir: ir,
            ),
        ):
            siblings = mint_error_siblings(
                "vertical_component_of_inertial_current_density",
                error_node_ids=[
                    "edge_profiles/ggd/j_inertial/z_error_index",
                ],
                unit="A.m^-2",
                physics_domain="edge_plasma_physics",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"W24 regression: uncertainty_index_of_vertical_component_of_inertial_"
            f"current_density must not be generated, got: {ids}"
        )

    def test_diamagnetic_current_density_blocked(self):
        """W24 leak pattern: diamagnetic current density component blocked."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.error_siblings import mint_error_siblings

        def _parse(name):
            m = MagicMock()
            m.ir = name
            return m

        with (
            patch("imas_standard_names.grammar.parser.parse", side_effect=_parse),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=lambda ir: ir,
            ),
        ):
            siblings = mint_error_siblings(
                "radial_component_of_diamagnetic_current_density",
                error_node_ids=[
                    "edge_profiles/ggd/j_diamagnetic/radial_error_index",
                ],
                unit="A.m^-2",
                physics_domain="edge_plasma_physics",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"W24 regression: uncertainty_index_of_radial_component_of_diamagnetic_"
            f"current_density must not be generated, got: {ids}"
        )

    def test_geometry_dimension_prefix_blocked(self):
        """Rule 5 regression: length_of_* parents must not produce uncertainty_index."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("length_of_magnetic_field_probe", "m")
            is False
        )
        assert (
            _parent_supports_uncertainty_index(
                "major_radius_of_magnetic_field_probe", "m"
            )
            is False
        )
        assert (
            _parent_supports_uncertainty_index("minor_radius_of_plasma_boundary", "m")
            is False
        )
