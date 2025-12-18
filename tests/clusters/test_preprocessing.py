"""Tests for preprocessing.py - path filtering and unit family building."""

import pytest

from imas_codex.clusters.config import RelationshipExtractionConfig
from imas_codex.clusters.preprocessing import PathFilter, UnitFamilyBuilder
from imas_codex.embeddings.config import EncoderConfig


class TestPathFilter:
    """Tests for PathFilter class.

    Note: The semantic-first approach trusts embeddings to distinguish
    meaningful from generic paths, so mechanical filtering is minimal.
    Only truly empty documentation is filtered.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration with default settings."""
        return RelationshipExtractionConfig(
            encoder_config=EncoderConfig(),
        )

    @pytest.fixture
    def path_filter(self, config):
        """Create PathFilter instance."""
        return PathFilter(config)

    def test_filter_meaningful_paths_empty_input(self, path_filter):
        """Test filtering with empty input."""
        result = path_filter.filter_meaningful_paths({})
        assert result == {}

    def test_filter_meaningful_paths_keeps_all_documented_paths(self, path_filter):
        """Test that all paths with any documentation are kept (semantic-first)."""
        ids_data = {
            "test_ids": {
                "paths": {
                    "test_ids/good_path": {
                        "documentation": "This is good documentation"
                    },
                    "test_ids/short_path": {"documentation": "Hi"},
                }
            }
        }
        result = path_filter.filter_meaningful_paths(ids_data)
        # Semantic-first: all documented paths are kept
        assert "test_ids/good_path" in result
        assert "test_ids/short_path" in result

    def test_filter_meaningful_paths_filters_empty_docs(self, path_filter):
        """Test that paths with empty documentation are filtered."""
        ids_data = {
            "test_ids": {
                "paths": {
                    "test_ids/good_path": {
                        "documentation": "Electron temperature in Kelvin"
                    },
                    "test_ids/empty_path": {"documentation": ""},
                    "test_ids/whitespace_path": {"documentation": "   "},
                }
            }
        }
        result = path_filter.filter_meaningful_paths(ids_data)
        assert "test_ids/good_path" in result
        assert "test_ids/empty_path" not in result
        assert "test_ids/whitespace_path" not in result

    def test_filter_meaningful_paths_includes_all_documented(self, path_filter):
        """Test that all paths with documentation are included (semantic-first)."""
        ids_data = {
            "test_ids": {
                "paths": {
                    "test_ids/good_path": {"documentation": "Valid path documentation"},
                    "test_ids/ids_properties/version": {
                        "documentation": "Version info"
                    },
                }
            }
        }
        result = path_filter.filter_meaningful_paths(ids_data)
        # Semantic-first: no skip patterns, all documented paths kept
        assert "test_ids/good_path" in result
        assert "test_ids/ids_properties/version" in result

    def test_filter_meaningful_paths_includes_description(self, path_filter):
        """Test that filtered paths include semantic description."""
        ids_data = {
            "core_profiles": {
                "paths": {
                    "core_profiles/temperature": {
                        "documentation": "Electron temperature profile in keV"
                    }
                }
            }
        }
        result = path_filter.filter_meaningful_paths(ids_data)
        assert "core_profiles/temperature" in result
        path_info = result["core_profiles/temperature"]
        assert "ids" in path_info
        assert "description" in path_info
        assert "data" in path_info
        assert path_info["ids"] == "core_profiles"

    def test_build_semantic_description_with_doc(self, path_filter):
        """Test building semantic description with documentation."""
        path_data = {
            "documentation": "Plasma electron temperature. High-precision measurement."
        }
        description = path_filter._build_semantic_description(
            "test/temperature", path_data
        )
        assert "temperature" in description.lower() or "plasma" in description.lower()

    def test_build_semantic_description_no_doc(self, path_filter):
        """Test building semantic description without documentation."""
        # Use a path that results in empty context to trigger fallback
        description = path_filter._build_semantic_description("test/time_slice", {})
        assert description == "Generic data field"

    def test_build_semantic_description_with_context_only(self, path_filter):
        """Test building semantic description with only path context."""
        description = path_filter._build_semantic_description("test/path", {})
        assert description == "Context: path"

    def test_clean_documentation_removes_cross_references(self, path_filter):
        """Test that cross-references are cleaned from documentation."""
        doc = "Primary description. Within equilibrium IDS: nested context."
        cleaned = path_filter._clean_documentation(doc)
        # Should remove the "Within X IDS:" part
        assert "Primary description" in cleaned

    def test_extract_path_context(self, path_filter):
        """Test extracting context from path components."""
        context = path_filter._extract_path_context(
            "equilibrium/time_slice/boundary/psi"
        )
        # Should skip time_slice and extract meaningful parts
        assert "boundary" in context or "psi" in context

    def test_extract_path_context_short_path(self, path_filter):
        """Test extracting context from short path."""
        context = path_filter._extract_path_context("equilibrium")
        assert context == ""


class TestUnitFamilyBuilder:
    """Tests for UnitFamilyBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create UnitFamilyBuilder instance."""
        return UnitFamilyBuilder()

    def test_build_unit_families_empty(self, builder):
        """Test building unit families with empty input."""
        result = builder.build_unit_families({})
        assert result == {}

    def test_build_unit_families_single_path_per_unit(self, builder):
        """Test that single-path units are not included."""
        filtered_paths = {
            "a/b": {"data": {"units": "m"}},
            "c/d": {"data": {"units": "kg"}},
        }
        result = builder.build_unit_families(filtered_paths)
        # Each unit only has one path, so no families
        assert result == {}

    def test_build_unit_families_shared_units(self, builder):
        """Test building families for shared units."""
        filtered_paths = {
            "a/temperature": {"data": {"units": "K"}},
            "b/temperature": {"data": {"units": "K"}},
            "c/temperature": {"data": {"units": "K"}},
            "a/pressure": {"data": {"units": "Pa"}},
            "b/pressure": {"data": {"units": "Pa"}},
        }
        result = builder.build_unit_families(filtered_paths)

        assert "K" in result
        assert "Pa" in result
        assert len(result["K"]["paths_using"]) == 3
        assert len(result["Pa"]["paths_using"]) == 2
        assert result["K"]["base_unit"] == "K"

    def test_build_unit_families_ignores_dimensionless(self, builder):
        """Test that dimensionless units are ignored."""
        filtered_paths = {
            "a/ratio": {"data": {"units": "1"}},
            "b/ratio": {"data": {"units": "1"}},
            "c/index": {"data": {"units": "-"}},
            "d/index": {"data": {"units": "-"}},
            "e/empty": {"data": {"units": ""}},
            "f/empty": {"data": {"units": ""}},
        }
        result = builder.build_unit_families(filtered_paths)
        # Dimensionless units should be ignored
        assert "1" not in result
        assert "-" not in result
        assert "" not in result

    def test_build_unit_families_mixed(self, builder):
        """Test with mixed valid and invalid units."""
        filtered_paths = {
            "a/temp": {"data": {"units": "K"}},
            "b/temp": {"data": {"units": "K"}},
            "c/dim": {"data": {"units": "1"}},
            "d/dim": {"data": {"units": "1"}},
            "e/pressure": {"data": {"units": "Pa"}},
        }
        result = builder.build_unit_families(filtered_paths)
        # Only K should form a family (2+ paths, not dimensionless)
        assert "K" in result
        assert "1" not in result
        assert "Pa" not in result  # Only one path
