"""Tests for imas_codex.core.paths."""

import pytest

from imas_codex.core.paths import strip_path_annotations


class TestStripPathAnnotations:
    """Tests for strip_path_annotations."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            # Parenthesized DD annotations
            ("flux_loop(i1)/flux/data(:)", "flux_loop/flux/data"),
            ("time_slice(itime)/profiles_1d(i1)/psi", "time_slice/profiles_1d/psi"),
            ("channel(i1)/position/r", "channel/position/r"),
            ("profiles_1d(:)/grid/rho_tor_norm", "profiles_1d/grid/rho_tor_norm"),
            # Bracket annotations
            ("time_slice[1]/profiles_1d[:]/psi", "time_slice/profiles_1d/psi"),
            ("channel[0]/position/r", "channel/position/r"),
            ("profiles_1d[0:3]/grid/rho_tor_norm", "profiles_1d/grid/rho_tor_norm"),
            # No annotations — passthrough
            (
                "equilibrium/time_slice/profiles_1d/psi",
                "equilibrium/time_slice/profiles_1d/psi",
            ),
            ("magnetics/flux_loop/flux/data", "magnetics/flux_loop/flux/data"),
            # Mixed
            ("time_slice(itime)/profiles_1d[0]/psi", "time_slice/profiles_1d/psi"),
            # Empty / simple
            ("", ""),
            ("magnetics", "magnetics"),
        ],
    )
    def test_strip(self, input_path: str, expected: str) -> None:
        assert strip_path_annotations(input_path) == expected
