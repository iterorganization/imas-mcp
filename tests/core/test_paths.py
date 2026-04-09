"""Tests for imas_codex.core.paths."""

import pytest

from imas_codex.core.paths import (
    _looks_like_path,
    normalize_imas_path,
    strip_path_annotations,
)


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


# ---------------------------------------------------------------------------
# _looks_like_path
# ---------------------------------------------------------------------------


class TestLooksLikePath:
    """Guard function that distinguishes IMAS paths from natural language."""

    @pytest.mark.parametrize(
        "text",
        [
            # Dot-separated IMAS paths
            "equilibrium.time_slice.profiles_1d.psi",
            "core_profiles.profiles_1d.electrons.temperature",
            "magnetics.ip.data",
            # Slash-separated IMAS paths
            "equilibrium/time_slice/profiles_1d/psi",
            "magnetics/flux_loop/flux/data",
            # Mixed dot/slash
            "equilibrium.time_slice/profiles_1d",
        ],
    )
    def test_path_detected(self, text: str) -> None:
        assert _looks_like_path(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Natural language with spaces
            "electron temperature",
            "plasma current measurement",
            # Natural language with periods
            "electron temperature e.g. in eV",
            "Find B0.",
            "temperature i.e. Te",
            "plasma current. Also check safety factor.",
            # Single words (no separator)
            "equilibrium",
            "magnetics",
            "ip",
            "",
            # Units with special chars
            "m^-1.s^-2",
            "eV.s",
            # Numeric/version strings
            "3.39.0",
        ],
    )
    def test_non_path_rejected(self, text: str) -> None:
        assert _looks_like_path(text) is False


# ---------------------------------------------------------------------------
# normalize_imas_path — dot-notation conversion
# ---------------------------------------------------------------------------


class TestNormalizeImasPathDotNotation:
    """Dot→slash conversion for IMAS paths, with natural-language safety."""

    # --- Dot-notation paths: dots MUST become slashes ---

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            # Pure dot notation
            (
                "equilibrium.time_slice.profiles_1d.psi",
                "equilibrium/time_slice/profiles_1d/psi",
            ),
            (
                "core_profiles.profiles_1d.electrons.temperature",
                "core_profiles/profiles_1d/electrons/temperature",
            ),
            ("magnetics.ip.data", "magnetics/ip/data"),
            # Mixed dot/slash
            (
                "equilibrium.time_slice/profiles_1d",
                "equilibrium/time_slice/profiles_1d",
            ),
            # Two segments
            ("magnetics.ip", "magnetics/ip"),
        ],
    )
    def test_dots_converted_to_slashes(self, input_path: str, expected: str) -> None:
        assert normalize_imas_path(input_path) == expected

    # --- Slash-notation paths: pass through unchanged ---

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            (
                "equilibrium/time_slice/profiles_1d/psi",
                "equilibrium/time_slice/profiles_1d/psi",
            ),
            ("magnetics/flux_loop/flux/data", "magnetics/flux_loop/flux/data"),
        ],
    )
    def test_slash_paths_unchanged(self, input_path: str, expected: str) -> None:
        assert normalize_imas_path(input_path) == expected

    # --- Natural language: dots MUST NOT become slashes ---

    @pytest.mark.parametrize(
        "query",
        [
            "electron temperature e.g. in eV",
            "Find B0.",
            "temperature i.e. Te",
            "plasma current. Also check safety factor.",
            "What is the toroidal field B0?",
            "Find plasma current measurement",
            "electron density profile for ITER scenario 2.",
        ],
    )
    def test_natural_language_preserved(self, query: str) -> None:
        """Natural language with periods must not be mangled."""
        result = normalize_imas_path(query)
        # Stripping is allowed, but dots must remain dots
        assert "." in query.strip() if "." in query else True
        # No slash should appear where there wasn't one
        original_slashes = query.strip().count("/")
        result_slashes = result.count("/")
        assert result_slashes == original_slashes, (
            f"Dot→slash leaked into natural language: {query!r} → {result!r}"
        )


# ---------------------------------------------------------------------------
# normalize_imas_path — annotation stripping
# ---------------------------------------------------------------------------


class TestNormalizeImasPathAnnotations:
    """Index/array annotation stripping combined with dot-notation."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            # Parenthesized annotations
            ("flux_loop(i1)/flux/data(:)", "flux_loop/flux/data"),
            (
                "time_slice(itime)/profiles_1d(i1)/psi",
                "time_slice/profiles_1d/psi",
            ),
            # Bracket annotations
            ("time_slice[1]/profiles_1d[:]/psi", "time_slice/profiles_1d/psi"),
            ("channel[0]/position/r", "channel/position/r"),
            # Dot-notation WITH annotations
            (
                "equilibrium.time_slice(itime).profiles_1d.psi",
                "equilibrium/time_slice/profiles_1d/psi",
            ),
            # Mixed dots + brackets
            (
                "magnetics.flux_loop[0].flux.data",
                "magnetics/flux_loop/flux/data",
            ),
        ],
    )
    def test_annotations_stripped(self, input_path: str, expected: str) -> None:
        assert normalize_imas_path(input_path) == expected


# ---------------------------------------------------------------------------
# normalize_imas_path — whitespace and edge cases
# ---------------------------------------------------------------------------


class TestNormalizeImasPathEdgeCases:
    """Whitespace handling, empty strings, single-word inputs."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            # Leading/trailing whitespace stripped
            ("  equilibrium/time_slice  ", "equilibrium/time_slice"),
            (" equilibrium.time_slice ", "equilibrium/time_slice"),
            # Leading/trailing slashes stripped
            ("/equilibrium/time_slice/", "equilibrium/time_slice"),
            # Single word (no separator) — passthrough
            ("equilibrium", "equilibrium"),
            ("magnetics", "magnetics"),
            # Empty / whitespace-only
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_edge_cases(self, input_path: str, expected: str) -> None:
        assert normalize_imas_path(input_path) == expected


# ---------------------------------------------------------------------------
# _normalize_paths — multi-path splitting + per-path normalization
# ---------------------------------------------------------------------------


class TestNormalizePaths:
    """The _normalize_paths helper splits then normalizes each path."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from imas_codex.tools.graph_search import _normalize_paths

        self._normalize = _normalize_paths

    # --- Space-separated dot-notation paths ---

    def test_space_separated_dot_paths(self) -> None:
        """Multiple dot-notation paths separated by spaces must each convert."""
        result = self._normalize(
            "equilibrium.time_slice.profiles_1d.psi "
            "core_profiles.profiles_1d.electrons.temperature"
        )
        assert result == [
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        ]

    def test_comma_separated_dot_paths(self) -> None:
        """Comma-separated dot-notation paths must each convert."""
        result = self._normalize(
            "equilibrium.time_slice.profiles_1d.psi,magnetics.ip.data"
        )
        assert result == [
            "equilibrium/time_slice/profiles_1d/psi",
            "magnetics/ip/data",
        ]

    def test_mixed_notation_multi_path(self) -> None:
        """Mix of dot, slash, and annotated paths in one string."""
        result = self._normalize(
            "equilibrium.time_slice.profiles_1d.psi "
            "magnetics/flux_loop/flux/data "
            "core_profiles.profiles_1d(i1).electrons.temperature"
        )
        assert result == [
            "equilibrium/time_slice/profiles_1d/psi",
            "magnetics/flux_loop/flux/data",
            "core_profiles/profiles_1d/electrons/temperature",
        ]

    def test_list_input_dot_paths(self) -> None:
        """List[str] input with dot-notation paths."""
        result = self._normalize(
            [
                "equilibrium.time_slice.profiles_1d.psi",
                "magnetics.ip.data",
            ]
        )
        assert result == [
            "equilibrium/time_slice/profiles_1d/psi",
            "magnetics/ip/data",
        ]

    def test_json_array_dot_paths(self) -> None:
        """JSON array string with dot-notation paths."""
        result = self._normalize(
            '["equilibrium.time_slice.profiles_1d.psi", "magnetics.ip.data"]'
        )
        assert result == [
            "equilibrium/time_slice/profiles_1d/psi",
            "magnetics/ip/data",
        ]

    def test_single_dot_path(self) -> None:
        result = self._normalize("equilibrium.time_slice")
        assert result == ["equilibrium/time_slice"]

    def test_single_slash_path(self) -> None:
        result = self._normalize("equilibrium/time_slice")
        assert result == ["equilibrium/time_slice"]
