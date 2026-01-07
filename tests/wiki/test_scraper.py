"""Tests for wiki scraper module."""

import re

import pytest

from imas_codex.wiki.scraper import (
    COCOS_PATTERN,
    IMAS_PATH_PATTERN,
    MDSPLUS_PATH_PATTERN,
    SIGN_CONVENTION_PATTERN,
    UNIT_PATTERN,
    WikiPage,
    extract_conventions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_units,
)


class TestMDSplusPathPattern:
    """Tests for the MDSplus path regex pattern."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            (r"\RESULTS::THOMSON:NE", True),
            (r"\\RESULTS::LIUQE:PSI", True),
            (r"\ACQUISITION::TCV:IP", True),
            (r"\RESULTS::TOP.SUB:VALUE", True),
            (r"no path here", False),
            (r"\single", False),  # Too short, no ::
        ],
    )
    def test_pattern_matching(self, text: str, expected: bool):
        """Test MDSplus path pattern matches correctly."""
        match = MDSPLUS_PATH_PATTERN.search(text)
        assert (match is not None) == expected


class TestIMASPathPattern:
    """Tests for the IMAS path regex pattern."""

    @pytest.mark.parametrize(
        "text,expected_count",
        [
            ("equilibrium/time_slice/boundary/psi", 1),
            ("core_profiles.profiles_1d.electrons.temperature", 1),
            ("magnetics/flux_loop/flux", 1),
            ("No IMAS paths here", 0),
            ("simple_word", 0),  # Single word, not a path
        ],
    )
    def test_pattern_matching(self, text: str, expected_count: int):
        """Test IMAS path pattern matches correctly."""
        matches = IMAS_PATH_PATTERN.findall(text)
        assert len(matches) == expected_count


class TestMDSplusPathExtraction:
    """Tests for MDSplus path extraction from text."""

    def test_single_backslash_path(self):
        """Extract path with single backslash."""
        text = r"The signal is stored at \RESULTS::THOMSON:NE"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::THOMSON:NE" in paths

    def test_double_backslash_path(self):
        """Extract path with double backslash (escaped)."""
        text = r"Use \\RESULTS::LIUQE:PSI for the poloidal flux"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::LIUQE:PSI" in paths

    def test_multiple_paths(self):
        """Extract multiple paths from text."""
        text = r"""
        Electron temperature: \RESULTS::THOMSON:TE
        Electron density: \RESULTS::THOMSON:NE
        Ion temperature: \RESULTS::TI:TI0
        """
        paths = extract_mdsplus_paths(text)
        assert len(paths) == 3
        assert r"\RESULTS::THOMSON:TE" in paths
        assert r"\RESULTS::THOMSON:NE" in paths
        assert r"\RESULTS::TI:TI0" in paths

    def test_normalize_to_uppercase(self):
        """Paths should be normalized to uppercase."""
        text = r"Use \results::thomson:ne for density"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::THOMSON:NE" in paths

    def test_deduplicate_paths(self):
        """Duplicate paths should be deduplicated."""
        text = r"\RESULTS::PSI and again \RESULTS::PSI"
        paths = extract_mdsplus_paths(text)
        assert len(paths) == 1

    def test_nested_path(self):
        """Extract deeply nested path."""
        text = r"\RESULTS::LIUQE:PSI:ERROR_BAR"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::LIUQE:PSI:ERROR_BAR" in paths

    def test_path_with_underscore(self):
        """Extract path with underscores."""
        text = r"\RESULTS::NE_FIT:NE_PROFILE"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::NE_FIT:NE_PROFILE" in paths


class TestIMASPathExtraction:
    """Tests for IMAS path extraction from text."""

    def test_slash_separated_path(self):
        """Extract path with slash separator."""
        text = "Maps to equilibrium/time_slice/boundary/psi"
        paths = extract_imas_paths(text)
        # The function extracts the IDS name and path separately
        assert any("equilibrium" in p for p in paths)

    def test_dot_separated_path(self):
        """Extract path with dot separator."""
        text = "Maps to core_profiles.profiles_1d.electrons.temperature"
        paths = extract_imas_paths(text)
        assert any("core_profiles" in p for p in paths)

    def test_lowercase_normalization(self):
        """Paths should be normalized to lowercase."""
        text = "Maps to Equilibrium/Time_Slice/Global_Quantities/Ip"
        paths = extract_imas_paths(text)
        # Should have one path, lowercased
        assert any("equilibrium" in p for p in paths)

    def test_various_ids(self):
        """Extract paths from various IDS."""
        text = """
        - magnetics/flux_loop/flux
        - thomson_scattering/channel/n_e
        - charge_exchange/channel/ti
        """
        paths = extract_imas_paths(text)
        assert len(paths) >= 3

    def test_mixed_separators(self):
        """Paths with mixed separators normalize correctly."""
        text = "equilibrium.time_slice/boundary.psi"
        paths = extract_imas_paths(text)
        # Should normalize to slash separator
        assert any("equilibrium" in p for p in paths)


class TestUnitExtraction:
    """Tests for physical unit extraction."""

    def test_energy_units(self):
        """Extract energy units."""
        text = "Temperature is 500 eV or 0.5 keV"
        units = extract_units(text)
        assert "eV" in units or "keV" in units

    def test_magnetic_units(self):
        """Extract magnetic field units."""
        text = "Field strength: 2.5 T and flux: 1.2 Wb"
        units = extract_units(text)
        assert any(u in units for u in ["T", "Wb"])

    def test_current_units(self):
        """Extract current units."""
        text = "Plasma current: 400 kA or 0.4 MA"
        units = extract_units(text)
        assert any(u in units for u in ["kA", "MA", "A"])

    def test_density_units(self):
        """Extract density units."""
        text = "Density: 1e19 m^-3 or 1e13 cm^-3"
        units = extract_units(text)
        # Should find at least one density unit
        assert len(units) > 0

    def test_deduplicate_units(self):
        """Duplicate units should be deduplicated."""
        text = "5 eV plus 10 eV equals 15 eV"
        units = extract_units(text)
        assert units.count("eV") == 1


class TestConventionExtraction:
    """Tests for sign convention extraction."""

    def test_cocos_number(self):
        """Extract COCOS index."""
        text = "TCV uses COCOS 11 for all equilibrium data"
        conventions = extract_conventions(text)
        assert len(conventions) > 0
        cocos_conv = [c for c in conventions if c["type"] == "cocos"]
        assert len(cocos_conv) > 0
        assert cocos_conv[0]["cocos_index"] == 11

    def test_cocos_equals(self):
        """Extract COCOS with equals sign."""
        text = "The convention is cocos=2"
        conventions = extract_conventions(text)
        cocos_conv = [c for c in conventions if c["type"] == "cocos"]
        assert len(cocos_conv) > 0
        assert cocos_conv[0]["cocos_index"] == 2

    def test_sign_convention(self):
        """Extract sign convention."""
        text = "Positive toroidal direction is counter-clockwise from above"
        conventions = extract_conventions(text)
        assert len(conventions) > 0
        sign_conv = [c for c in conventions if c["type"] == "sign"]
        assert len(sign_conv) > 0

    def test_multiple_conventions(self):
        """Extract multiple conventions from text."""
        text = """
        TCV uses COCOS 11. Positive Bt is clockwise from above.
        """
        conventions = extract_conventions(text)
        # Should find at least the COCOS reference
        cocos_conv = [c for c in conventions if c["type"] == "cocos"]
        assert len(cocos_conv) >= 1


class TestWikiPage:
    """Tests for WikiPage dataclass."""

    def test_content_hash(self):
        """Content hash should be deterministic."""
        page1 = WikiPage(
            url="https://example.com/wiki/Test",
            title="Test",
            content_html="<html>Test content</html>",
        )
        page2 = WikiPage(
            url="https://example.com/wiki/Test",
            title="Test",
            content_html="<html>Test content</html>",
        )
        assert page1.content_hash == page2.content_hash

    def test_content_hash_changes(self):
        """Different content should have different hash."""
        page1 = WikiPage(
            url="https://example.com/wiki/Test",
            title="Test",
            content_html="<html>Content A</html>",
        )
        page2 = WikiPage(
            url="https://example.com/wiki/Test",
            title="Test",
            content_html="<html>Content B</html>",
        )
        assert page1.content_hash != page2.content_hash

    def test_page_name_extraction(self):
        """Page name should be extracted from URL."""
        page = WikiPage(
            url="https://spcwiki.epfl.ch/wiki/Thomson_Scattering",
            title="Thomson Scattering",
            content_html="",
        )
        assert page.page_name == "Thomson_Scattering"

    def test_page_name_with_complex_url(self):
        """Page name should handle complex URLs."""
        page = WikiPage(
            url="https://spcwiki.epfl.ch/wiki/Data/LIUQE/PSI?version=2",
            title="LIUQE PSI",
            content_html="",
        )
        # Should get the path after /wiki/
        assert "Data/LIUQE/PSI" in page.page_name

    def test_mdsplus_paths_default(self):
        """MDSplus paths should default to empty list."""
        page = WikiPage(
            url="https://spcwiki.epfl.ch/wiki/Test",
            title="Test",
            content_html="",
        )
        assert page.mdsplus_paths == []


class TestPatternCompilation:
    """Tests that regex patterns compile correctly."""

    def test_mdsplus_pattern_compiles(self):
        """MDSplus pattern should compile."""
        assert MDSPLUS_PATH_PATTERN is not None
        # Pattern is already compiled, verify it works
        assert MDSPLUS_PATH_PATTERN.search(r"\TEST::NODE") is not None

    def test_imas_pattern_compiles(self):
        """IMAS pattern should compile."""
        assert IMAS_PATH_PATTERN is not None
        # Pattern is already compiled, verify it works
        assert IMAS_PATH_PATTERN.search("equilibrium/time") is not None

    def test_cocos_pattern_compiles(self):
        """COCOS pattern should work."""
        # COCOS_PATTERN is already compiled
        assert COCOS_PATTERN is not None
        match = COCOS_PATTERN.search("COCOS 11")
        assert match is not None

    def test_sign_pattern_compiles(self):
        """Sign convention pattern should work."""
        # SIGN_CONVENTION_PATTERN is already compiled
        assert SIGN_CONVENTION_PATTERN is not None
        match = SIGN_CONVENTION_PATTERN.search("positive clockwise")
        assert match is not None

    def test_unit_pattern_compiles(self):
        """Unit pattern should work."""
        # UNIT_PATTERN is already compiled
        assert UNIT_PATTERN is not None
        match = UNIT_PATTERN.search("5 eV")
        assert match is not None
