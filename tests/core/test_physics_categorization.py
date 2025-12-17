"""
Tests for physics_categorization module.

Tests physics domain classification and IDS-to-domain mappings.
"""

import json
from unittest.mock import patch

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
        """Test successful loading of physics mappings from definitions."""
        # Create a mock mappings file
        mappings_data = {
            "ids_domain_mappings": {
                "core_profiles": "transport",
                "equilibrium": "mhd",
                "wall": "engineering",
            }
        }

        mapping_path = tmp_path / "ids_domains.json"
        with open(mapping_path, "w") as f:
            json.dump(mappings_data, f)

        with patch(
            "imas_codex.core.physics_categorization.IDS_DOMAINS_FILE",
            mapping_path,
        ):
            # Clear the cache to force reload
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == mappings_data["ids_domain_mappings"]
        assert "core_profiles" in result
        assert result["core_profiles"] == "transport"

    def test_load_mappings_file_not_found(self, tmp_path):
        """Test handling of missing mappings file."""
        nonexistent_path = tmp_path / "nonexistent" / "ids_domains.json"

        with patch(
            "imas_codex.core.physics_categorization.IDS_DOMAINS_FILE",
            nonexistent_path,
        ):
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == {}

    def test_load_mappings_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file."""
        mapping_path = tmp_path / "ids_domains.json"
        with open(mapping_path, "w") as f:
            f.write("invalid json {")

        with patch(
            "imas_codex.core.physics_categorization.IDS_DOMAINS_FILE",
            mapping_path,
        ):
            _load_physics_mappings.cache_clear()
            result = _load_physics_mappings()

        assert result == {}


class TestPhysicsDomainCategorizer:
    """Tests for PhysicsDomainCategorizer class."""

    @pytest.fixture
    def categorizer(self):
        """Create a categorizer with mock mappings."""
        cat = PhysicsDomainCategorizer()
        cat._mappings = {
            "core_profiles": "transport",
            "equilibrium": "magnetohydrodynamics",
            "wall": "plasma_wall_interactions",
            "edge_profiles": "edge_plasma_physics",
            "waves": "auxiliary_heating",
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
        assert result == PhysicsDomain.MAGNETOHYDRODYNAMICS

        result = categorizer.get_domain_for_ids("wall")
        assert result == PhysicsDomain.PLASMA_WALL_INTERACTIONS

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
        assert result[PhysicsDomain.MAGNETOHYDRODYNAMICS] == 1
        assert result[PhysicsDomain.PLASMA_WALL_INTERACTIONS] == 1
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
        assert result["equilibrium"] == PhysicsDomain.MAGNETOHYDRODYNAMICS
        assert result["wall"] == PhysicsDomain.PLASMA_WALL_INTERACTIONS

    def test_get_all_mappings_with_invalid(self, categorizer):
        """Test get_all_mappings handles invalid domain strings."""
        categorizer._mappings["bad"] = "not_a_domain"

        result = categorizer.get_all_mappings()

        assert result["bad"] == PhysicsDomain.GENERAL

    def test_get_ids_for_domain(self, categorizer):
        """Test getting IDS names for a domain."""
        result = categorizer.get_ids_for_domain(PhysicsDomain.TRANSPORT)
        assert "core_profiles" in result

        result = categorizer.get_ids_for_domain(PhysicsDomain.MAGNETOHYDRODYNAMICS)
        assert "equilibrium" in result

    def test_get_ids_for_domain_empty(self, categorizer):
        """Test getting IDS for domain with no matches."""
        result = categorizer.get_ids_for_domain(PhysicsDomain.MAGNETIC_FIELD_SYSTEMS)
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
