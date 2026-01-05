"""Tests for MDSplus path extraction."""

import pytest

from imas_codex.code_examples.mdsplus_extractor import (
    MDSplusReference,
    extract_mdsplus_paths,
    normalize_mdsplus_path,
)


class TestNormalizeMDSplusPath:
    """Tests for normalize_mdsplus_path function."""

    def test_uppercase(self) -> None:
        """Should uppercase paths."""
        assert normalize_mdsplus_path("results::i_p") == "\\RESULTS::I_P"

    def test_single_backslash(self) -> None:
        """Should normalize to single backslash prefix."""
        assert normalize_mdsplus_path("\\\\RESULTS::I_P") == "\\RESULTS::I_P"
        assert normalize_mdsplus_path("RESULTS::I_P") == "\\RESULTS::I_P"

    def test_complex_path(self) -> None:
        """Should handle complex nested paths."""
        assert (
            normalize_mdsplus_path("results::thomson.profiles.auto:te")
            == "\\RESULTS::THOMSON.PROFILES.AUTO:TE"
        )


class TestExtractMDSplusPaths:
    """Tests for extract_mdsplus_paths function."""

    def test_direct_path_string(self) -> None:
        """Should extract paths from string literals."""
        code = """data = conn.get("\\\\RESULTS::I_P")"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 1
        assert refs[0].path == "\\RESULTS::I_P"
        assert refs[0].ref_type == "mdsplus_path"

    def test_lowercase_path(self) -> None:
        """Should handle lowercase paths."""
        code = """data = conn.get("\\\\results::thomson.profiles.auto:te")"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 1
        assert refs[0].path == "\\RESULTS::THOMSON.PROFILES.AUTO:TE"

    def test_f_string_pattern(self) -> None:
        """Should extract paths from f-strings."""
        code = """
eq_tree = "\\\\results::top.equil_1.results"
psi = _load_data(conn, f"{eq_tree}:PSI")
"""
        refs = extract_mdsplus_paths(code)
        paths = {r.path for r in refs}
        assert "\\RESULTS::PSI" in paths or "\\RESULTS::TOP.EQUIL_1.RESULTS" in paths

    def test_tcv_eq_call(self) -> None:
        """Should extract TDI function quantities."""
        code = """ip = tcv_eq("I_P")"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 1
        assert refs[0].path == "\\RESULTS::I_P"
        assert refs[0].ref_type == "tdi_call"

    def test_tcv_get_call(self) -> None:
        """Should extract tcv_get quantities."""
        code = """ip = tcv_get("IP")"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 1
        assert refs[0].path == "\\RESULTS::IP"
        assert refs[0].ref_type == "tdi_call"

    def test_multiple_paths(self) -> None:
        """Should extract multiple unique paths."""
        code = """
psi = conn.get("\\\\RESULTS::PSI")
ip = conn.get("\\\\RESULTS::I_P")
psi_again = conn.get("\\\\RESULTS::PSI")  # duplicate
"""
        refs = extract_mdsplus_paths(code)
        paths = {r.path for r in refs}
        assert paths == {"\\RESULTS::PSI", "\\RESULTS::I_P"}

    def test_mixed_references(self) -> None:
        """Should handle mixed path types."""
        code = """
# Direct path
psi = conn.get("\\\\RESULTS::PSI")
# TDI call
ip = tcv_eq("I_P")
"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 2
        types = {r.ref_type for r in refs}
        assert types == {"mdsplus_path", "tdi_call"}

    def test_ignores_non_mdsplus_strings(self) -> None:
        """Should not extract regular strings."""
        code = """
name = "equilibrium"
path = "/home/user/file.py"
"""
        refs = extract_mdsplus_paths(code)
        assert len(refs) == 0

    def test_real_world_code_example(self) -> None:
        """Test with realistic code from TCV analysis."""
        code = """
eq_tree = f"\\\\results::top.{equilibrium}.results"

dimensions = (
    (
        "time_equilibrium",
        _load_data(conn, f"{eq_tree}:TIME_PSI"),
        "s",
        "Time array for equilibrium 2D profiles."
    ),
    (
        "psi",
        np.swapaxes(_load_data(conn, f"{eq_tree}:PSI"), 1, 2),
        ("time_equilibrium", "radius_equilibrium", "height_equilibrium"),
        "Wb / 2*pi",
        "Poloidal flux function 2D array against (R, Z)."
    ),
    (
        "plasma_current",
        _load_data(conn, f"{eq_tree}:I_PL"),
        ("time_equilibrium",),
        "A",
        "Plasma current."
    ),
)
"""
        refs = extract_mdsplus_paths(code)
        paths = {r.path for r in refs}
        # Should find the quantities from f-strings
        assert any("TIME_PSI" in p for p in paths) or any("PSI" in p for p in paths)
