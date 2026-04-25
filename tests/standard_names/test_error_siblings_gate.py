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
        """Dimensional scalar (eV) with no deny pattern → allowed."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("electron_temperature", "eV") is True

    def test_allow_current(self):
        """Dimensional scalar (A) with no deny pattern → allowed."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("plasma_current", "A") is True

    def test_allow_ion_density(self):
        """Dimensional scalar (m^-3) with no deny pattern → allowed."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("ion_density", "m^-3") is True

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
        """Physical scalar (eV, plasma_current) → uncertainty_index IS produced."""
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
        assert any("uncertainty_index" in sid for sid in ids), (
            f"Expected uncertainty_index sibling, got: {ids}"
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
