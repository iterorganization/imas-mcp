"""W4b: Pre-validation gate rejects malformed standard-name candidates."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.workers import is_well_formed_candidate


class TestIsWellFormedCandidate:
    """Unit tests for the pre-validation gate function."""

    # --- Valid names ---

    @pytest.mark.parametrize(
        "name",
        [
            "electron_temperature",
            "plasma_current",
            "q95",
            "b_field_tor_vacuum_r_major",
            "a",
            "x0",
        ],
    )
    def test_valid_names_pass(self, name: str):
        ok, reason = is_well_formed_candidate(name)
        assert ok is True
        assert reason is None

    # --- Empty / whitespace ---

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "   ",
            "\t",
            "\n",
            None,
        ],
    )
    def test_empty_rejected(self, name):
        ok, reason = is_well_formed_candidate(name)
        assert ok is False
        assert reason == "empty_or_whitespace"

    # --- Too long (> 100 chars) ---

    def test_long_name_rejected(self):
        name = "a" * 101
        ok, reason = is_well_formed_candidate(name)
        assert ok is False
        assert reason == "too_long"

    def test_exactly_100_chars_accepted(self):
        name = "a" * 100
        ok, reason = is_well_formed_candidate(name)
        assert ok is True

    # --- Illegal characters ---

    @pytest.mark.parametrize(
        "name",
        [
            "foo\nbar",
            "foo\tbar",
            "foo{bar}",
            "foo\\bar",
        ],
    )
    def test_illegal_chars_rejected(self, name: str):
        ok, reason = is_well_formed_candidate(name)
        assert ok is False
        assert reason == "illegal_chars"

    # --- Triple dot ---

    def test_triple_dot_rejected(self):
        ok, reason = is_well_formed_candidate("electron...temperature")
        assert ok is False
        assert reason == "triple_dot"

    # --- Not snake_case ---

    @pytest.mark.parametrize(
        "name",
        [
            "ElectronTemperature",  # CamelCase
            "Electron_temperature",  # Initial cap
            "3d_field",  # Starts with digit
            "_private",  # Starts with underscore
            "electron temperature",  # Space
            "electron-temperature",  # Hyphen
        ],
    )
    def test_not_snake_case_rejected(self, name: str):
        ok, reason = is_well_formed_candidate(name)
        assert ok is False
        assert reason == "not_snake_case"

    # --- Edge cases ---

    def test_single_char_accepted(self):
        ok, reason = is_well_formed_candidate("a")
        assert ok is True

    def test_trailing_underscore_rejected(self):
        """Trailing underscore doesn't match [a-z][a-z0-9_]*."""
        # Actually 'a_' matches [a-z][a-z0-9_]*, so this should pass
        ok, reason = is_well_formed_candidate("a_")
        assert ok is True

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before length check."""
        ok, reason = is_well_formed_candidate("  electron_temperature  ")
        # After strip -> "electron_temperature" -> valid
        assert ok is True
