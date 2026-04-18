"""Tests for naming-scope classifier.

After Plan 30, the SN classifier has three defensive rules:
- S0: STR_* data types → skip (string-typed leaves are never standard names)
- S1: core_instant_changes IDS → skip (dedup policy)
- S2: _error_* suffix → skip (defensive)

All other semantic filtering is owned by DD node_category
(the extractor's SN_SOURCE_CATEGORIES pre-filter excludes
fit_artifact, representation, coordinate, structural, etc.).
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.classifier import (
    ERROR_SUFFIXES,
    Scope,
    classify_path,
)

# ============================================================================
# Helpers
# ============================================================================


def _node(path: str, **overrides: object) -> dict:
    """Build a minimal node dict for classify_path."""
    return {
        "path": path,
        "data_type": overrides.pop("data_type", "FLT_1D"),
        **overrides,
    }


# ============================================================================
# Gold set — parametrised testing
# ============================================================================
#
# Format: (path, expected)
# The classifier is now binary — only S1 and S2 filter; everything else
# is "quantity" because DD pre-filtering already excluded non-quantity nodes.

GOLD_SET: list[tuple[str, Scope]] = [
    # -----------------------------------------------------------------------
    # quantity — normal physics paths pass through
    # -----------------------------------------------------------------------
    ("core_profiles/profiles_1d/electrons/temperature", "quantity"),
    ("equilibrium/time_slice/profiles_1d/psi", "quantity"),
    ("equilibrium/time_slice/global_quantities/ip", "quantity"),
    ("core_profiles/profiles_1d/electrons/density", "quantity"),
    ("core_profiles/profiles_1d/ion/pressure", "quantity"),
    ("magnetics/flux_loop/flux/data", "quantity"),
    ("equilibrium/time_slice/profiles_1d/phi", "quantity"),
    ("equilibrium/time_slice/global_quantities/beta_pol", "quantity"),
    ("barometry/gauge/pressure", "quantity"),
    ("mhd_linear/toroidal_mode_number", "quantity"),
    # Paths that used to be "metadata" or "skip" are now "quantity"
    # because DD pre-filters them — the classifier trusts DD
    ("magnetics/bpol_probe/field/data", "quantity"),
    ("equilibrium/time", "quantity"),
    ("core_profiles/profiles_1d/electrons/temperature/validity", "quantity"),
    # -----------------------------------------------------------------------
    # skip — S1: core_instant_changes
    # -----------------------------------------------------------------------
    ("core_instant_changes/change/profiles_1d/electrons/density", "skip"),
    ("core_instant_changes/change/profiles_1d/e_field/parallel", "skip"),
    ("core_instant_changes/vacuum_toroidal_field/b0", "skip"),
    ("core_instant_changes", "skip"),
    # -----------------------------------------------------------------------
    # skip — S2: error fields (defensive)
    # -----------------------------------------------------------------------
    ("core_profiles/profiles_1d/grid/rho_tor_norm_error_upper", "skip"),
    ("equilibrium/time_slice/profiles_1d/psi_error_lower", "skip"),
    ("equilibrium/time_slice/profiles_1d/psi_error_index", "skip"),
    ("core_profiles/profiles_1d/electrons/temperature_error_upper", "skip"),
]

# S0 cases need data_type override, tested separately in TestS0StringTypes


@pytest.mark.parametrize(
    "path,expected",
    GOLD_SET,
    ids=[g[0].rsplit("/", 1)[-1] for g in GOLD_SET],
)
def test_classify_path(path: str, expected: Scope) -> None:
    """Gold-set parametrised test."""
    assert classify_path(_node(path)) == expected


# ============================================================================
# Focused unit tests — S0 (string type defensive)
# ============================================================================


class TestS0StringTypes:
    """S0: STR_* data types → skip (names/descriptions can't be SN)."""

    def test_str_0d_skip(self) -> None:
        node = _node("core_profiles/profiles_1d/electrons/label", data_type="STR_0D")
        assert classify_path(node) == "skip"

    def test_str_1d_skip(self) -> None:
        node = _node("core_profiles/profiles_1d/ion/label", data_type="STR_1D")
        assert classify_path(node) == "skip"

    def test_flt_1d_passes(self) -> None:
        node = _node(
            "core_profiles/profiles_1d/electrons/temperature", data_type="FLT_1D"
        )
        assert classify_path(node) == "quantity"

    def test_empty_data_type_passes(self) -> None:
        node = _node("core_profiles/profiles_1d/electrons/temperature", data_type="")
        assert classify_path(node) == "quantity"


# ============================================================================
# Focused unit tests — S1 (core_instant_changes)
# ============================================================================


class TestS1CoreInstantChanges:
    """S1: core_instant_changes IDS-level dedup → skip."""

    def test_root(self) -> None:
        assert classify_path(_node("core_instant_changes")) == "skip"

    def test_nested_leaf(self) -> None:
        assert (
            classify_path(
                _node("core_instant_changes/change/profiles_1d/electrons/density")
            )
            == "skip"
        )

    def test_similar_prefix_not_matched(self) -> None:
        """core_instant_changes_extra should NOT be matched."""
        assert (
            classify_path(
                _node("core_instant_changes_extra/profiles_1d/electrons/density")
            )
            == "quantity"
        )

    def test_other_ids_not_matched(self) -> None:
        assert (
            classify_path(_node("core_profiles/profiles_1d/electrons/density"))
            == "quantity"
        )


# ============================================================================
# Focused unit tests — S2 (error fields defensive)
# ============================================================================


class TestS2ErrorFields:
    """S2: _error_* suffix defensive check → skip."""

    @pytest.mark.parametrize("suffix", ERROR_SUFFIXES)
    def test_error_suffixes(self, suffix: str) -> None:
        node = _node(f"equilibrium/time_slice/profiles_1d/psi{suffix}")
        assert classify_path(node) == "skip"

    def test_error_in_middle_of_segment_name(self) -> None:
        """_error_upper in a segment means skip."""
        node = _node("core_profiles/profiles_1d/electrons/temperature_error_upper")
        assert classify_path(node) == "skip"

    def test_non_error_suffix_passes(self) -> None:
        """Paths without error suffix → quantity."""
        node = _node("core_profiles/profiles_1d/electrons/temperature")
        assert classify_path(node) == "quantity"


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Verify public constants are importable."""

    def test_error_suffixes(self) -> None:
        assert len(ERROR_SUFFIXES) == 3
        assert "_error_upper" in ERROR_SUFFIXES
