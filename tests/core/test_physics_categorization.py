"""
Tests for physics_categorization module.

Tests physics domain classification and IDS-to-domain mappings.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.core.data_model import PhysicsDomain
from imas_codex.core.physics_categorization import (
    PhysicsDomainCategorizer,
    _load_physics_mappings,
    physics_categorizer,
)


class TestLoadPhysicsMappings:
    """Tests for _load_physics_mappings function."""

    def test_load_mappings_success(self, tmp_path):
        """Test successful loading of physics mappings."""
        # Create a mock mappings file
        mappings_data = {
            "mappings": {
                "core_profiles": "transport",
                "equilibrium": "mhd",
                "wall": "engineering",
            }
        }

        # Create mock accessor
        mock_accessor = MagicMock()
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        mock_accessor.schemas_dir = schemas_dir

        mapping_path = schemas_dir / "physics_domains.json"
        with open(mapping_path, "w") as f:
            json.dump(mappings_data, f)

        # Patch both DOMAINS_FILE (to non-existent) and ResourcePathAccessor
        fake_domains_file = tmp_path / "nonexistent" / "domains.json"
        with (
            patch(
                "imas_codex.core.physics_categorization.DOMAINS_FILE",
                fake_domains_file,
            ),
            patch(
                "imas_codex.core.physics_categorization.ResourcePathAccessor",
                return_value=mock_accessor,
            ),
        ):
            # Clear the cache to force reload
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == mappings_data["mappings"]
        assert "core_profiles" in result
        assert result["core_profiles"] == "transport"

    def test_load_mappings_file_not_found(self, tmp_path):
        """Test handling of missing mappings file."""
        mock_accessor = MagicMock()
        mock_accessor.schemas_dir = tmp_path / "nonexistent"

        # Patch both DOMAINS_FILE and ResourcePathAccessor to non-existent paths
        fake_domains_file = tmp_path / "nonexistent" / "domains.json"
        with (
            patch(
                "imas_codex.core.physics_categorization.DOMAINS_FILE",
                fake_domains_file,
            ),
            patch(
                "imas_codex.core.physics_categorization.ResourcePathAccessor",
                return_value=mock_accessor,
            ),
        ):
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == {}

    def test_load_mappings_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file."""
        mock_accessor = MagicMock()
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        mock_accessor.schemas_dir = schemas_dir

        mapping_path = schemas_dir / "physics_domains.json"
        with open(mapping_path, "w") as f:
            f.write("invalid json {")

        # Patch both DOMAINS_FILE (to non-existent) and ResourcePathAccessor
        fake_domains_file = tmp_path / "nonexistent" / "domains.json"
        with (
            patch(
                "imas_codex.core.physics_categorization.DOMAINS_FILE",
                fake_domains_file,
            ),
            patch(
                "imas_codex.core.physics_categorization.ResourcePathAccessor",
                return_value=mock_accessor,
            ),
        ):
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == {}

    def test_load_mappings_from_definitions_first(self, tmp_path):
        """Test that definitions file is loaded before resources."""
        # Create definitions file with version-keyed structure
        definitions_data = {
            "4.1.0": {
                "mappings": {
                    "equilibrium": "equilibrium",
                    "core_profiles": "transport",
                }
            }
        }
        definitions_file = tmp_path / "definitions" / "domains.json"
        definitions_file.parent.mkdir(parents=True)
        with open(definitions_file, "w") as f:
            json.dump(definitions_data, f)

        # Create resources file with different data (should not be loaded)
        resources_data = {
            "mappings": {
                "equilibrium": "general",
                "wall": "engineering",
            }
        }
        mock_accessor = MagicMock()
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        mock_accessor.schemas_dir = schemas_dir
        mapping_path = schemas_dir / "physics_domains.json"
        with open(mapping_path, "w") as f:
            json.dump(resources_data, f)

        with (
            patch(
                "imas_codex.core.physics_categorization.DOMAINS_FILE",
                definitions_file,
            ),
            patch(
                "imas_codex.core.physics_categorization.ResourcePathAccessor",
                return_value=mock_accessor,
            ),
            patch(
                "imas_codex.core.physics_categorization.dd_version",
                "4.1.0",
            ),
        ):
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        # Should load from definitions, not resources
        assert result == definitions_data["4.1.0"]["mappings"]
        assert result["equilibrium"] == "equilibrium"
        assert "wall" not in result


class TestPhysicsDomainCategorizer:
    """Tests for PhysicsDomainCategorizer class."""

    @pytest.fixture
    def categorizer(self):
        """Create a categorizer with mock mappings."""
        cat = PhysicsDomainCategorizer()
        cat._mappings = {
            "core_profiles": "transport",
            "equilibrium": "mhd",
            "wall": "wall",
            "edge_profiles": "edge_physics",
            "waves": "heating",
        }
        return cat

    def test_mappings_property_lazy_loading(self):
        """Test that mappings are lazily loaded."""
        cat = PhysicsDomainCategorizer()
        cat._mappings = None

        with patch(
            "imas_codex.core.physics_categorization._load_physics_mappings",
            return_value={"test": "transport"},
        ) as mock_load:
            result = cat.mappings
            mock_load.assert_called_once()
            assert result == {"test": "transport"}

    def test_get_domain_for_ids_known(self, categorizer):
        """Test getting domain for known IDS."""
        result = categorizer.get_domain_for_ids("core_profiles")
        assert result == PhysicsDomain.TRANSPORT

        result = categorizer.get_domain_for_ids("equilibrium")
        assert result == PhysicsDomain.MHD

        result = categorizer.get_domain_for_ids("wall")
        assert result == PhysicsDomain.WALL

    def test_get_domain_for_ids_case_insensitive(self, categorizer):
        """Test that IDS name lookup is case-insensitive."""
        result = categorizer.get_domain_for_ids("CORE_PROFILES")
        assert result == PhysicsDomain.TRANSPORT

        result = categorizer.get_domain_for_ids("Core_Profiles")
        assert result == PhysicsDomain.TRANSPORT

    def test_get_domain_for_ids_unknown(self, categorizer):
        """Test getting domain for unknown IDS returns GENERAL."""
        result = categorizer.get_domain_for_ids("unknown_ids")
        assert result == PhysicsDomain.GENERAL

    def test_get_domain_for_ids_invalid_domain_string(self, categorizer):
        """Test handling of invalid domain string in mappings."""
        categorizer._mappings["bad_ids"] = "invalid_domain"

        result = categorizer.get_domain_for_ids("bad_ids")
        assert result == PhysicsDomain.GENERAL

    def test_analyze_domain_distribution(self, categorizer):
        """Test domain distribution analysis."""
        ids_list = [
            "core_profiles",
            "equilibrium",
            "wall",
            "core_profiles",
            "unknown",
        ]

        result = categorizer.analyze_domain_distribution(ids_list)

        assert result[PhysicsDomain.TRANSPORT] == 2
        assert result[PhysicsDomain.MHD] == 1
        assert result[PhysicsDomain.WALL] == 1
        assert result[PhysicsDomain.GENERAL] == 1

    def test_analyze_domain_distribution_empty(self, categorizer):
        """Test distribution analysis with empty list."""
        result = categorizer.analyze_domain_distribution([])
        assert result == {}

    def test_get_all_mappings(self, categorizer):
        """Test getting all mappings as enum values."""
        result = categorizer.get_all_mappings()

        assert isinstance(result, dict)
        assert result["core_profiles"] == PhysicsDomain.TRANSPORT
        assert result["equilibrium"] == PhysicsDomain.MHD
        assert result["wall"] == PhysicsDomain.WALL

    def test_get_all_mappings_with_invalid(self, categorizer):
        """Test get_all_mappings handles invalid domain strings."""
        categorizer._mappings["bad"] = "not_a_domain"

        result = categorizer.get_all_mappings()

        assert result["bad"] == PhysicsDomain.GENERAL

    def test_get_ids_for_domain(self, categorizer):
        """Test getting IDS names for a domain."""
        result = categorizer.get_ids_for_domain(PhysicsDomain.TRANSPORT)
        assert "core_profiles" in result

        result = categorizer.get_ids_for_domain(PhysicsDomain.MHD)
        assert "equilibrium" in result

    def test_get_ids_for_domain_empty(self, categorizer):
        """Test getting IDS for domain with no matches."""
        result = categorizer.get_ids_for_domain(PhysicsDomain.COILS)
        assert result == []


class TestGlobalCategorizer:
    """Tests for the global physics_categorizer instance."""

    def test_global_instance_exists(self):
        """Test that global categorizer instance exists."""
        assert physics_categorizer is not None
        assert isinstance(physics_categorizer, PhysicsDomainCategorizer)

    def test_global_instance_usable(self):
        """Test that global instance can be used."""
        # This will work even without real mappings (returns GENERAL)
        result = physics_categorizer.get_domain_for_ids("test")
        assert isinstance(result, PhysicsDomain)
