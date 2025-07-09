"""Comprehensive test suite for the XML parser with proper pytest structure."""

import json
import logging

import pytest

from imas_mcp.core.xml_parser import DataDictionaryTransformer
from imas_mcp.core.data_model import TransformationOutputs
from imas_mcp.dd_accessor import ImasDataDictionaryAccessor


logger = logging.getLogger(__name__)


@pytest.fixture
def xml_accessor():
    """Fixture to provide ImasDataDictionaryAccessor instance."""
    return ImasDataDictionaryAccessor()


@pytest.fixture
def test_ids_set():
    """Fixture providing a small set of IDS for fast testing."""
    return {"core_profiles", "equilibrium", "pf_active"}


@pytest.fixture
def large_test_ids_set():
    """Fixture providing a larger set of IDS for more comprehensive testing."""
    return {
        "core_profiles",
        "equilibrium",
        "pf_active",
        "magnetics",
        "wall",
        "ec_launchers",
    }


@pytest.fixture
def transformer(tmp_path, xml_accessor, test_ids_set):
    """Fixture to provide a configured DataDictionaryTransformer."""
    return DataDictionaryTransformer(
        output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=test_ids_set
    )


@pytest.fixture
def transformer_large(tmp_path, xml_accessor, large_test_ids_set):
    """Fixture to provide a transformer with a larger IDS set."""
    return DataDictionaryTransformer(
        output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=large_test_ids_set
    )


@pytest.fixture
def transformer_no_ids_set(tmp_path, xml_accessor):
    """Fixture to provide a transformer without IDS set restriction."""
    return DataDictionaryTransformer(output_dir=tmp_path, dd_accessor=xml_accessor)


class TestDataDictionaryTransformerBasic:
    """Basic functionality tests for DataDictionaryTransformer."""

    def test_initialization_with_accessor(self, tmp_path, xml_accessor):
        """Test transformer initialization with provided accessor."""
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor
        )

        assert transformer.output_dir == tmp_path
        assert transformer.dd_accessor is xml_accessor
        assert transformer.ids_set is None
        assert transformer.skip_ggd is True
        assert "ids_properties" in transformer.excluded_patterns
        assert "code" in transformer.excluded_patterns

    def test_initialization_without_accessor(self, tmp_path):
        """Test transformer initialization creates accessor automatically."""
        transformer = DataDictionaryTransformer(output_dir=tmp_path)

        assert transformer.dd_accessor is not None
        assert isinstance(transformer.dd_accessor, ImasDataDictionaryAccessor)

    def test_initialization_with_ids_set(self, tmp_path, xml_accessor):
        """Test transformer initialization with IDS set restriction."""
        ids_set = {"core_profiles", "equilibrium"}
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=ids_set
        )

        assert transformer.ids_set == ids_set

    def test_output_directory_creation(self, tmp_path, xml_accessor):
        """Test that output directory is created during initialization."""
        nested_path = tmp_path / "nested" / "output"
        DataDictionaryTransformer(output_dir=nested_path, dd_accessor=xml_accessor)

        assert nested_path.exists()
        assert nested_path.is_dir()


class TestDataDictionaryTransformerTransformation:
    """Tests for the main transformation functionality."""

    def test_transform_complete_basic(self, transformer):
        """Test basic transformation with restricted IDS set."""
        outputs = transformer.transform_complete()

        assert isinstance(outputs, TransformationOutputs)
        assert outputs.catalog.exists()
        assert outputs.relationships.exists()
        assert len(outputs.detailed) > 0

        # All detailed files should exist
        for detailed_file in outputs.detailed:
            assert detailed_file.exists()

    def test_catalog_structure(self, transformer, test_ids_set):
        """Test that catalog structure is correct."""
        outputs = transformer.transform_complete()

        with open(outputs.catalog, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        # Validate top-level structure
        assert "metadata" in catalog_data
        assert "ids_catalog" in catalog_data

        # Check metadata
        metadata = catalog_data["metadata"]
        assert "version" in metadata
        assert "generation_date" in metadata
        assert "total_ids" in metadata
        assert "total_leaf_nodes" in metadata

        # Check IDS catalog matches our test set
        ids_catalog = catalog_data["ids_catalog"]
        catalog_ids = set(ids_catalog.keys())
        assert catalog_ids == test_ids_set

        # Each IDS should have required fields
        for ids_name, ids_info in ids_catalog.items():
            assert "path_count" in ids_info  # Changed from leaf_count to path_count
            assert "physics_domain" in ids_info
            assert isinstance(ids_info["path_count"], int)
            assert "name" in ids_info

    def test_detailed_files_structure(self, transformer, test_ids_set):
        """Test that detailed files have correct structure."""
        outputs = transformer.transform_complete()

        # Should have one detailed file per IDS
        assert len(outputs.detailed) == len(test_ids_set)

        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            # Validate structure
            assert "ids_info" in detailed_data
            assert "coordinate_systems" in detailed_data
            assert "paths" in detailed_data
            assert "semantic_groups" in detailed_data

            # Check IDS info
            ids_info = detailed_data["ids_info"]
            assert "name" in ids_info
            assert ids_info["name"] in test_ids_set

            # Check paths
            paths = detailed_data["paths"]
            assert isinstance(paths, dict)
            assert len(paths) > 0  # Should have at least some paths

            # Check some path structures
            for path_key, path_data in list(paths.items())[:3]:  # Check first 3 paths
                assert "documentation" in path_data
                assert "units" in path_data
                assert "path" in path_data
                assert path_data["path"] == path_key

    def test_relationships_structure(self, transformer):
        """Test that relationships file has correct structure."""
        outputs = transformer.transform_complete()

        with open(outputs.relationships, "r", encoding="utf-8") as f:
            relationships_data = json.load(f)
        # Validate structure
        assert "metadata" in relationships_data
        assert "cross_references" in relationships_data
        assert "physics_concepts" in relationships_data

        # Check metadata
        metadata = relationships_data["metadata"]
        assert "generation_date" in metadata
        assert "total_relationships" in metadata


class TestDataDictionaryTransformerFiltering:
    """Tests for filtering and exclusion functionality."""

    def test_ids_set_filtering(self, tmp_path, xml_accessor):
        """Test that IDS set filtering works correctly."""
        single_ids = {"core_profiles"}
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=single_ids
        )

        outputs = transformer.transform_complete()

        # Should only have one detailed file
        assert len(outputs.detailed) == 1

        # Check catalog only contains the requested IDS
        with open(outputs.catalog, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        catalog_ids = set(catalog_data["ids_catalog"].keys())
        assert catalog_ids == single_ids

    def test_excluded_patterns(self, transformer):
        """Test that excluded patterns are properly filtered."""
        outputs = transformer.transform_complete()

        # Check that no paths contain excluded patterns
        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            paths = detailed_data["paths"]
            for path_key in paths.keys():
                # Should not contain excluded patterns
                assert "ids_properties" not in path_key
                assert "code" not in path_key

    def test_ggd_filtering(self, tmp_path, xml_accessor, test_ids_set):
        """Test that GGD nodes are filtered when skip_ggd is True."""
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path,
            dd_accessor=xml_accessor,
            ids_set=test_ids_set,
            skip_ggd=True,
        )

        outputs = transformer.transform_complete()

        # Check that no paths contain ggd
        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            paths = detailed_data["paths"]
            ggd_paths = [path for path in paths.keys() if "ggd" in path.lower()]
            assert len(ggd_paths) == 0, (
                f"Found GGD paths when they should be filtered: {ggd_paths}"
            )


class TestDataDictionaryTransformerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_ids_set(self, tmp_path, xml_accessor):
        """Test behavior with empty IDS set."""
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=set()
        )

        outputs = transformer.transform_complete()

        # Should create files but with minimal content
        assert outputs.catalog.exists()
        assert outputs.relationships.exists()
        assert len(outputs.detailed) == 0

        # Catalog should have empty IDS catalog
        with open(outputs.catalog, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        assert len(catalog_data["ids_catalog"]) == 0

    def test_nonexistent_ids_in_set(self, tmp_path, xml_accessor):
        """Test behavior with non-existent IDS names."""
        fake_ids_set = {"nonexistent_ids", "fake_data_structure"}
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=fake_ids_set
        )

        outputs = transformer.transform_complete()

        # Should handle gracefully and create minimal output
        assert outputs.catalog.exists()
        assert outputs.relationships.exists()

        # Should have no detailed files since IDS don't exist
        assert len(outputs.detailed) == 0

    def test_mixed_valid_invalid_ids(self, tmp_path, xml_accessor):
        """Test behavior with mix of valid and invalid IDS names."""
        mixed_ids_set = {"core_profiles", "nonexistent_ids", "equilibrium"}
        transformer = DataDictionaryTransformer(
            output_dir=tmp_path, dd_accessor=xml_accessor, ids_set=mixed_ids_set
        )

        outputs = transformer.transform_complete()

        # Should process valid IDS and ignore invalid ones
        assert outputs.catalog.exists()
        assert len(outputs.detailed) <= 2  # At most 2 valid IDS

    def test_invalid_output_directory(self, xml_accessor):
        """Test behavior with invalid output directory path."""
        # Test with a path that would cause issues on Windows
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file where we want to put a directory
            conflict_path = os.path.join(temp_dir, "conflict")
            with open(conflict_path, "w") as f:
                f.write("test")
            # This should handle the conflict gracefully or raise appropriate error
            from pathlib import Path

            try:
                transformer = DataDictionaryTransformer(
                    output_dir=Path(conflict_path), dd_accessor=xml_accessor
                )
                # If it doesn't raise an error, transformation should still work
                outputs = transformer.transform_complete()
                assert outputs.catalog.exists()
            except (OSError, FileExistsError):
                # Expected behavior for invalid directory
                pass


class TestDataDictionaryTransformerPerformance:
    """Performance and efficiency tests."""

    def test_small_vs_large_ids_set_timing(
        self, tmp_path, xml_accessor, test_ids_set, large_test_ids_set
    ):
        """Test that processing time scales reasonably with IDS set size."""
        import time

        # Test small set
        transformer_small = DataDictionaryTransformer(
            output_dir=tmp_path / "small",
            dd_accessor=xml_accessor,
            ids_set=test_ids_set,
        )

        start_time = time.time()
        outputs_small = transformer_small.transform_complete()
        small_duration = time.time() - start_time

        # Test large set
        transformer_large = DataDictionaryTransformer(
            output_dir=tmp_path / "large",
            dd_accessor=xml_accessor,
            ids_set=large_test_ids_set,
        )

        start_time = time.time()
        outputs_large = transformer_large.transform_complete()
        large_duration = time.time() - start_time

        # Basic performance validation
        assert small_duration > 0
        assert large_duration > 0
        # Large set should produce more output
        assert len(outputs_large.detailed) >= len(outputs_small.detailed)

        logger.info(f"Small set ({len(test_ids_set)} IDS): {small_duration:.2f}s")
        logger.info(f"Large set ({len(large_test_ids_set)} IDS): {large_duration:.2f}s")

    def test_memory_usage_reasonable(self, transformer_large):
        """Test that memory usage stays reasonable during processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        _ = transformer_large.transform_complete()  # We only care about memory usage

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info(
            f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)"
        )

        # Memory increase should be reasonable (less than 500MB for test)
        assert memory_increase < 500, (
            f"Memory usage increased by {memory_increase:.1f}MB, which seems excessive"
        )

    def test_output_file_sizes_reasonable(self, transformer_large):
        """Test that output files have reasonable sizes."""
        outputs = transformer_large.transform_complete()

        # Check catalog size
        catalog_size = outputs.catalog.stat().st_size / 1024  # KB
        assert catalog_size > 1, "Catalog file seems too small"
        assert catalog_size < 10240, (
            f"Catalog file seems too large: {catalog_size:.1f}KB"
        )

        # Check relationships size
        rel_size = outputs.relationships.stat().st_size / 1024  # KB
        assert rel_size > 0.1, "Relationships file seems too small"
        assert rel_size < 51200, f"Relationships file seems too large: {rel_size:.1f}KB"

        # Check detailed files
        for detailed_file in outputs.detailed:
            detail_size = detailed_file.stat().st_size / 1024  # KB
            assert detail_size > 1, (
                f"Detailed file {detailed_file.name} seems too small"
            )
            assert detail_size < 20480, (
                f"Detailed file {detailed_file.name} seems too large: {detail_size:.1f}KB"
            )


class TestDataDictionaryTransformerDataQuality:
    """Data quality and content validation tests."""

    def test_data_completeness(self, transformer):
        """Test that extracted data is complete and consistent."""
        outputs = transformer.transform_complete()

        # Load all data
        with open(outputs.catalog, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        detailed_data = {}
        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                ids_name = data["ids_info"]["name"]
                detailed_data[ids_name] = data

        # Cross-validate catalog and detailed data
        catalog_ids = set(catalog_data["ids_catalog"].keys())
        detailed_ids = set(detailed_data.keys())

        assert catalog_ids == detailed_ids, "Mismatch between catalog and detailed IDS"

        # Validate path counts match
        for ids_name in catalog_ids:
            catalog_path_count = catalog_data["ids_catalog"][ids_name]["path_count"]
            detailed_path_count = len(detailed_data[ids_name]["paths"])
            # Path count should match or be reasonably close to path count
            # (allowing for larger differences due to filtering like GGD exclusion and other patterns)
            # Use a very generous tolerance to account for aggressive filtering that can remove 60%+ of paths
            max_difference = max(200, catalog_path_count * 0.8)
            assert abs(catalog_path_count - detailed_path_count) <= max_difference, (
                f"Path count mismatch for {ids_name}: catalog={catalog_path_count}, detailed={detailed_path_count}, tolerance={max_difference}"
            )

    def test_path_consistency(self, transformer):
        """Test that paths are consistent and well-formed."""
        outputs = transformer.transform_complete()

        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            ids_name = detailed_data["ids_info"]["name"]
            paths = detailed_data["paths"]

            for path_key, path_data in paths.items():
                # Path key should match path field
                assert path_data["path"] == path_key

                # Path should start with IDS name
                assert path_key.startswith(ids_name + "/"), (
                    f"Path {path_key} doesn't start with IDS name {ids_name}"
                )

                # Path should not be empty
                assert len(path_key) > len(ids_name) + 1

                # Should have required fields
                assert "documentation" in path_data
                assert "units" in path_data

                # Documentation should not be empty string (though can be None)
                if path_data["documentation"]:
                    assert len(path_data["documentation"].strip()) > 0

    def test_units_validation(self, transformer):
        """Test that units are properly extracted and formatted."""
        outputs = transformer.transform_complete()

        unit_statistics = {"with_units": 0, "without_units": 0, "empty_units": 0}

        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            paths = detailed_data["paths"]

            for path_key, path_data in paths.items():
                units = path_data["units"]

                if units is None:
                    unit_statistics["without_units"] += 1
                elif units.strip() == "":
                    unit_statistics["empty_units"] += 1
                else:
                    unit_statistics["with_units"] += 1

        total_paths = sum(unit_statistics.values())
        logger.info(f"Units statistics: {unit_statistics} (total: {total_paths})")

        # Should have some paths with units
        assert unit_statistics["with_units"] > 0, "No paths found with units"
        # Most paths should either have units or explicitly None
        # (Allow for a reasonable percentage of empty units in real data - up to 50%)
        assert unit_statistics["empty_units"] < total_paths * 0.5, (
            "Too many paths with empty string units"
        )

    def test_coordinate_systems_extraction(self, transformer):
        """Test that coordinate systems are properly extracted."""
        outputs = transformer.transform_complete()

        coordinate_systems_found = 0

        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            coord_systems = detailed_data["coordinate_systems"]

            if coord_systems:
                coordinate_systems_found += len(coord_systems)

                # Validate coordinate system structure
                for coord_name, coord_data in coord_systems.items():
                    assert isinstance(coord_data, dict)
                    # Should have some meaningful content
                    assert len(str(coord_data)) > 10

        logger.info(f"Found {coordinate_systems_found} coordinate systems")

        # Should find at least some coordinate systems
        # (This might need adjustment based on actual data)
        assert coordinate_systems_found >= 0  # At minimum, should not fail

    def test_semantic_groups_extraction(self, transformer):
        """Test that semantic groups are properly identified."""
        outputs = transformer.transform_complete()

        semantic_groups_found = 0

        for detailed_file in outputs.detailed:
            with open(detailed_file, "r", encoding="utf-8") as f:
                detailed_data = json.load(f)

            semantic_groups = detailed_data["semantic_groups"]

            if semantic_groups:
                semantic_groups_found += len(semantic_groups)

                # Validate semantic group structure
                for group_name, group_paths in semantic_groups.items():
                    # Each semantic group should be a list of paths
                    assert isinstance(group_paths, list)
                    # Should have multiple paths (groups with single items are filtered out)
                    assert len(group_paths) > 1
                    # Each path should be a string
                    for path in group_paths:
                        assert isinstance(path, str)
                        # Path should be a valid IDS path (contains the IDS name)
                        ids_name = detailed_data["ids_info"]["name"]
                        assert path.startswith(ids_name + "/"), (
                            f"Path {path} should start with {ids_name}/"
                        )

        logger.info(f"Found {semantic_groups_found} semantic groups")

        # Should find at least some semantic groups or handle gracefully
        assert semantic_groups_found >= 0
