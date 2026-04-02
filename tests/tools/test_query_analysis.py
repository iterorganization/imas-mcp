"""Tests for query analysis module."""

import pytest

from imas_codex.tools.query_analysis import (
    ACCESSOR_TERMINALS,
    PHYSICS_ABBREVIATIONS,
    QueryAnalyzer,
    strip_accessor_suffix,
)


class TestQueryAnalyzer:
    """Test suite for QueryAnalyzer."""

    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    # --- Path queries ---

    def test_exact_path_with_ids(self):
        intent = self.analyzer.analyze("equilibrium/time_slice/profiles_1d/psi")
        assert intent.query_type == "path_exact"
        assert intent.ids_hint == "equilibrium"
        assert intent.path_segments == [
            "equilibrium",
            "time_slice",
            "profiles_1d",
            "psi",
        ]

    def test_partial_path_no_ids(self):
        intent = self.analyzer.analyze("time_slice/profiles_1d/psi")
        assert intent.query_type == "path_exact"
        assert intent.path_segments == ["time_slice", "profiles_1d", "psi"]

    def test_path_with_accessor_terminal(self):
        intent = self.analyzer.analyze("magnetics/flux_loop/flux/data")
        assert intent.query_type == "path_exact"
        assert intent.accessor_hint == "data"
        assert intent.stripped_query == "magnetics/flux_loop/flux"

    def test_path_with_value_terminal(self):
        intent = self.analyzer.analyze("nbi/unit/power_launched/value")
        assert intent.query_type == "path_exact"
        assert intent.accessor_hint == "value"
        assert intent.stripped_query == "nbi/unit/power_launched"

    # --- Abbreviation queries ---

    def test_single_abbreviation_ip(self):
        intent = self.analyzer.analyze("ip")
        assert intent.query_type == "concept"
        assert intent.is_abbreviation is True
        assert "plasma current" in intent.expanded_terms
        assert "ip" in intent.expanded_terms

    def test_single_abbreviation_te(self):
        intent = self.analyzer.analyze("te")
        assert intent.is_abbreviation is True
        assert "electron temperature" in intent.expanded_terms

    def test_single_abbreviation_ne(self):
        intent = self.analyzer.analyze("ne")
        assert intent.is_abbreviation is True
        assert "electron density" in intent.expanded_terms

    def test_single_abbreviation_q(self):
        intent = self.analyzer.analyze("q")
        assert intent.is_abbreviation is True
        assert "safety factor" in intent.expanded_terms

    # --- Concept queries ---

    def test_concept_query(self):
        intent = self.analyzer.analyze("electron temperature")
        assert intent.query_type == "concept"
        assert intent.is_abbreviation is False

    def test_concept_query_expanded(self):
        intent = self.analyzer.analyze("plasma boundary shape")
        assert intent.query_type == "concept"

    # --- Hybrid queries ---

    def test_hybrid_query(self):
        intent = self.analyzer.analyze("te core_profiles")
        assert intent.query_type == "hybrid"
        assert intent.is_abbreviation is True
        assert "electron temperature" in intent.expanded_terms

    # --- Edge cases ---

    def test_empty_query(self):
        intent = self.analyzer.analyze("")
        assert intent.query_type == "concept"

    def test_whitespace_query(self):
        intent = self.analyzer.analyze("   ")
        assert intent.query_type == "concept"

    def test_unknown_single_word(self):
        """Single word not in abbreviations -> concept query."""
        intent = self.analyzer.analyze("elongation")
        assert intent.query_type == "concept"
        assert intent.is_abbreviation is False


class TestStripAccessorSuffix:
    """Tests for strip_accessor_suffix helper."""

    def test_strip_data(self):
        assert (
            strip_accessor_suffix("magnetics/flux_loop/flux/data")
            == "magnetics/flux_loop/flux"
        )

    def test_strip_value(self):
        assert (
            strip_accessor_suffix("nbi/unit/power_launched/value")
            == "nbi/unit/power_launched"
        )

    def test_strip_time(self):
        assert (
            strip_accessor_suffix("equilibrium/time_slice/time")
            == "equilibrium/time_slice"
        )

    def test_no_strip_non_accessor(self):
        assert (
            strip_accessor_suffix("equilibrium/time_slice/profiles_1d/psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )

    def test_no_strip_empty(self):
        assert strip_accessor_suffix("") == ""

    def test_strip_r(self):
        assert (
            strip_accessor_suffix("wall/description_2d/limiter/unit/outline/r")
            == "wall/description_2d/limiter/unit/outline"
        )


class TestPhysicsAbbreviations:
    """Validate that the abbreviation map is well-formed."""

    def test_all_have_full_name(self):
        """Each abbreviation should have at least one expansion with more than 3 chars."""
        for abbrev, expansions in PHYSICS_ABBREVIATIONS.items():
            long_terms = [e for e in expansions if len(e) > 3]
            assert long_terms, f"Abbreviation '{abbrev}' has no long expansion"

    def test_abbreviation_included(self):
        """Each abbreviation should include itself in expansions."""
        for abbrev, expansions in PHYSICS_ABBREVIATIONS.items():
            assert abbrev in expansions, (
                f"Abbreviation '{abbrev}' not in its own expansions"
            )


class TestAccessorTerminals:
    """Validate accessor terminal set."""

    def test_common_terminals(self):
        assert "data" in ACCESSOR_TERMINALS
        assert "value" in ACCESSOR_TERMINALS
        assert "time" in ACCESSOR_TERMINALS
        assert "r" in ACCESSOR_TERMINALS
        assert "z" in ACCESSOR_TERMINALS

    def test_not_physics_terms(self):
        """Physics concepts should not be accessor terminals."""
        assert "psi" not in ACCESSOR_TERMINALS
        assert "temperature" not in ACCESSOR_TERMINALS
        assert "density" not in ACCESSOR_TERMINALS
