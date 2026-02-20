"""Tests for facility-aware entity extraction.

Tests the FacilityEntityExtractor class and standalone extraction functions
from entity_extraction.py.
"""

import pytest

from imas_codex.discovery.wiki.entity_extraction import (
    ExtractionResult,
    FacilityEntityExtractor,
    _build_imas_pattern,
    extract_conventions,
    extract_facility_tool_mentions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_ppf_paths,
    extract_units,
    get_all_ids_names,
)

# =============================================================================
# IDS Name Discovery
# =============================================================================


class TestGetAllIdsNames:
    """Tests for IDS name discovery from data dictionary."""

    def test_returns_tuple(self):
        names = get_all_ids_names()
        assert isinstance(names, tuple)

    def test_contains_core_ids(self):
        names = get_all_ids_names()
        for ids in ("equilibrium", "core_profiles", "magnetics", "thomson_scattering"):
            assert ids in names, f"Missing core IDS: {ids}"

    def test_has_many_ids(self):
        """DD has 80+ IDS, should return at least 50."""
        names = get_all_ids_names()
        assert len(names) >= 50

    def test_sorted(self):
        names = get_all_ids_names()
        assert names == tuple(sorted(names))


class TestBuildImasPattern:
    """Tests for IMAS path pattern construction."""

    def test_returns_compiled_pattern(self):
        import re

        pattern = _build_imas_pattern()
        assert isinstance(pattern, re.Pattern)

    def test_matches_all_core_ids(self):
        pattern = _build_imas_pattern()
        for ids in ("equilibrium", "core_profiles", "magnetics", "nbi", "ece"):
            text = f"{ids}/time_slice/profiles_1d"
            match = pattern.search(text)
            assert match is not None, f"Pattern should match {ids} path"

    def test_does_not_match_random_text(self):
        pattern = _build_imas_pattern()
        assert pattern.search("just some random text about plasma") is None

    def test_findall_returns_full_match(self):
        """Critical: findall must return full path, not just IDS name."""
        pattern = _build_imas_pattern()
        text = "The equilibrium/time_slice/boundary/psi was measured."
        matches = pattern.findall(text)
        assert len(matches) == 1
        assert "equilibrium" in matches[0]
        assert "time_slice" in matches[0]


# =============================================================================
# MDSplus Path Extraction
# =============================================================================


class TestExtractMdsplusPaths:
    """Tests for MDSplus path extraction."""

    def test_single_backslash(self):
        text = r"Signal at \RESULTS::THOMSON:NE"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::THOMSON:NE" in paths

    def test_double_backslash(self):
        text = r"Signal at \\RESULTS::LIUQE:PSI"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::LIUQE:PSI" in paths

    def test_normalization_uppercase(self):
        text = r"Signal at \results::thomson:ne"
        paths = extract_mdsplus_paths(text)
        assert r"\RESULTS::THOMSON:NE" in paths

    def test_deduplicated(self):
        text = r"\RESULTS::IP and also \RESULTS::IP again"
        paths = extract_mdsplus_paths(text)
        assert paths.count(r"\RESULTS::IP") == 1

    def test_empty_text(self):
        assert extract_mdsplus_paths("") == []

    def test_no_matches(self):
        assert extract_mdsplus_paths("no MDSplus paths here") == []

    def test_multiple_paths(self):
        text = r"\ATLAS::IP and \RESULTS::THOMSON:TE"
        paths = extract_mdsplus_paths(text)
        assert len(paths) == 2


# =============================================================================
# IMAS Path Extraction (Full DD Coverage)
# =============================================================================


class TestExtractImasPaths:
    """Tests for IMAS path extraction with full DD IDS list."""

    def test_slash_separator(self):
        paths = extract_imas_paths("equilibrium/time_slice/boundary/psi")
        assert "equilibrium/time_slice/boundary/psi" in paths

    def test_dot_separator(self):
        paths = extract_imas_paths("core_profiles.profiles_1d.electrons.temperature")
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_ids_prefix_stripped(self):
        paths = extract_imas_paths("ids.equilibrium.global_quantities.ip")
        assert "equilibrium/global_quantities/ip" in paths

    def test_previously_missing_ids(self):
        """IDS names that were NOT in old hardcoded list of 12."""
        for ids in (
            "thomson_scattering",
            "langmuir_probes",
            "soft_x_rays",
            "wall",
            "summary",
            "pf_active",
            "distributions",
        ):
            text = f"{ids}/some_field/value"
            paths = extract_imas_paths(text)
            assert len(paths) >= 1, f"{ids} path should be extracted"

    def test_empty_text(self):
        assert extract_imas_paths("") == []

    def test_no_matches(self):
        assert extract_imas_paths("no IMAS paths here") == []

    def test_array_index_notation(self):
        """Array indices are accepted in input but stripped during normalization."""
        paths = extract_imas_paths("equilibrium/time_slice[0]/profiles_1d/psi")
        assert len(paths) == 1
        assert paths[0] == "equilibrium/time_slice/profiles_1d/psi"

    def test_multiple_paths(self):
        text = "equilibrium/ip and core_profiles/te and magnetics/flux"
        paths = extract_imas_paths(text)
        assert len(paths) == 3

    def test_deduplicated(self):
        text = "equilibrium/ip and equilibrium/ip again"
        paths = extract_imas_paths(text)
        assert len(paths) == 1


# =============================================================================
# PPF Path Extraction
# =============================================================================


class TestExtractPpfPaths:
    """Tests for JET PPF DDA/Dtype path extraction."""

    def test_known_dda(self):
        paths = extract_ppf_paths("PPF signal EFIT/Q95 for q-profile")
        assert "EFIT/Q95" in paths

    def test_multiple_ddas(self):
        paths = extract_ppf_paths("Compare KK3/TE with HRTS/TE profiles")
        assert "KK3/TE" in paths
        assert "HRTS/TE" in paths

    def test_unknown_dda_rejected(self):
        """Unknown DDAs should be filtered to reduce false positives."""
        paths = extract_ppf_paths("See HTML/PAGE for the document")
        assert len(paths) == 0

    def test_case_normalized(self):
        paths = extract_ppf_paths("efit/q95 value")
        # PPF pattern requires uppercase, so lowercase won't match
        assert len(paths) == 0

    def test_bolo_dda(self):
        paths = extract_ppf_paths("BOLO/KB5H bolometer channel")
        assert "BOLO/KB5H" in paths

    def test_empty_text(self):
        assert extract_ppf_paths("") == []


# =============================================================================
# Unit Extraction
# =============================================================================


class TestExtractUnits:
    """Tests for physical unit extraction."""

    def test_energy_units(self):
        units = extract_units("Temperature is 5 keV")
        assert "keV" in units

    def test_magnetic_units(self):
        units = extract_units("Field of 2.3 Tesla")
        assert "Tesla" in units

    def test_current_units(self):
        units = extract_units("Plasma current of 400 kA")
        assert "kA" in units

    def test_empty_text(self):
        assert extract_units("") == []


# =============================================================================
# Convention Extraction
# =============================================================================


class TestExtractConventions:
    """Tests for COCOS and sign convention extraction."""

    def test_cocos_reference(self):
        conventions = extract_conventions("Uses COCOS 11 convention")
        assert any(c["type"] == "cocos" for c in conventions)
        cocos = [c for c in conventions if c["type"] == "cocos"]
        assert cocos[0]["cocos_index"] == 11

    def test_sign_convention(self):
        conventions = extract_conventions("positive clockwise for Bt")
        assert any(c["type"] == "sign" for c in conventions)

    def test_empty_text(self):
        assert extract_conventions("") == []


# =============================================================================
# Facility Tool Mentions
# =============================================================================


class TestExtractFacilityToolMentions:
    """Tests for facility-specific tool mention extraction."""

    def test_key_tool_match(self):
        mentions = extract_facility_tool_mentions(
            "Use tdiExecute to get the signal",
            key_tools=["tdiExecute", "tcv_get"],
        )
        assert "tdiExecute" in mentions

    def test_code_import_pattern(self):
        mentions = extract_facility_tool_mentions(
            "import ppf\nppf.ppfdata(pulse, dda, dtype)",
            code_import_patterns=["import ppf", "ppf.ppfdata"],
        )
        assert "import ppf" in mentions
        assert "ppf.ppfdata" in mentions

    def test_word_boundary_prevents_partial(self):
        """'sal' should not match 'universal'."""
        mentions = extract_facility_tool_mentions(
            "The universal truth about plasma",
            key_tools=["sal"],
        )
        assert len(mentions) == 0

    def test_word_boundary_matches_standalone(self):
        """'sal' should match 'import sal'."""
        mentions = extract_facility_tool_mentions(
            "import sal\nsal.get(pulse, signal)",
            key_tools=["sal"],
        )
        assert "sal" in mentions

    def test_no_tools(self):
        mentions = extract_facility_tool_mentions("some text", key_tools=None)
        assert mentions == []

    def test_empty_text(self):
        mentions = extract_facility_tool_mentions("", key_tools=["ppfget"])
        assert mentions == []

    def test_case_insensitive(self):
        mentions = extract_facility_tool_mentions(
            "Used PPFGET to fetch data",
            key_tools=["ppfget"],
        )
        assert "ppfget" in mentions


# =============================================================================
# FacilityEntityExtractor
# =============================================================================


class TestFacilityEntityExtractor:
    """Tests for the FacilityEntityExtractor class."""

    def test_init_unknown_facility(self):
        """Should not raise even if facility config doesn't exist."""
        extractor = FacilityEntityExtractor("nonexistent_facility")
        assert extractor.facility_id == "nonexistent_facility"
        assert not extractor.has_mdsplus
        assert not extractor.has_ppf

    def test_extract_returns_result(self):
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("some text")
        assert isinstance(result, ExtractionResult)

    def test_extract_empty_text(self):
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("")
        assert result == ExtractionResult()

    def test_imas_always_extracted(self):
        """IMAS paths should be extracted regardless of facility."""
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("equilibrium/time_slice/boundary/psi")
        assert len(result.imas_paths) >= 1

    def test_units_always_extracted(self):
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("Temperature of 5 keV")
        assert "keV" in result.units

    def test_conventions_always_extracted(self):
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("COCOS 11 convention used")
        assert len(result.conventions) >= 1

    def test_mdsplus_not_extracted_without_config(self):
        """No MDSplus extraction for unknown facility (no data_systems)."""
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract(r"\RESULTS::THOMSON:NE")
        assert result.mdsplus_paths == []

    def test_ppf_not_extracted_without_config(self):
        """No PPF extraction for unknown facility (no data_systems)."""
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("EFIT/Q95 value")
        assert result.ppf_paths == []

    def test_to_chunk_properties(self):
        """to_chunk_properties returns dict with correct keys."""
        extractor = FacilityEntityExtractor("nonexistent_facility")
        result = extractor.extract("equilibrium/ip at 5 keV with COCOS 11")
        props = extractor.to_chunk_properties(result)
        assert "mdsplus_paths" in props
        assert "imas_paths" in props
        assert "ppf_paths" in props
        assert "units" in props
        assert "conventions" in props
        assert "tool_mentions" in props
        assert isinstance(props["imas_paths"], list)


class TestFacilityEntityExtractorWithConfig:
    """Tests that use real facility configs (if available).

    These tests verify integration with actual facility YAML configs.
    They are lenient — if a facility config is missing, the test still
    passes since FacilityEntityExtractor handles missing configs gracefully.
    """

    def test_tcv_has_mdsplus(self):
        """TCV should have MDSplus data system."""
        extractor = FacilityEntityExtractor("tcv")
        if extractor._data_systems:
            assert extractor.has_mdsplus
            result = extractor.extract(r"\RESULTS::THOMSON:NE")
            assert len(result.mdsplus_paths) >= 1

    def test_jet_has_ppf(self):
        """JET should have PPF data system."""
        extractor = FacilityEntityExtractor("jet")
        if extractor._data_systems:
            assert extractor.has_ppf
            result = extractor.extract("EFIT/Q95 profile data")
            assert len(result.ppf_paths) >= 1

    def test_tcv_tool_mentions(self):
        """TCV extractor should find tdiExecute mentions."""
        extractor = FacilityEntityExtractor("tcv")
        if extractor._key_tools:
            result = extractor.extract("Use tdiExecute to read the signal")
            assert len(result.tool_mentions) >= 1

    def test_jet_tool_mentions(self):
        """JET extractor should find ppfget mentions."""
        extractor = FacilityEntityExtractor("jet")
        if extractor._key_tools:
            result = extractor.extract("Call ppfget for the data")
            assert len(result.tool_mentions) >= 1

    def test_iter_imas_extraction(self):
        """ITER should extract IMAS paths (universal)."""
        extractor = FacilityEntityExtractor("iter")
        result = extractor.extract("equilibrium/time_slice[0]/profiles_1d/psi")
        assert len(result.imas_paths) >= 1
