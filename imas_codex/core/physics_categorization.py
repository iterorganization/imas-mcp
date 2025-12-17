"""
Physics domain categorization for IDS.

This module provides IDS-to-physics-domain mapping loaded from LLM-generated
JSON file. The mappings are loaded from:
1. definitions/physics/domains.json (version-controlled cache)
2. resources/schemas/physics_domains.json (build artifact, fallback)

The mappings are generated at build time by the build-physics script.
"""

import json
import logging
from functools import lru_cache

from imas_codex import dd_version
from imas_codex.core.data_model import PhysicsDomain
from imas_codex.definitions.physics import DOMAINS_FILE
from imas_codex.resource_path_accessor import ResourcePathAccessor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_physics_mappings() -> dict[str, str]:
    """Load physics domain mappings from JSON file.

    Loads from definitions first (version-controlled), then falls back
    to resources (build artifact).

    Returns:
        Dictionary mapping IDS names to physics domain strings.
        Returns empty dict if no file exists.
    """
    # Try definitions first (version-controlled, keyed by DD version)
    if DOMAINS_FILE.exists():
        try:
            with open(DOMAINS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # Get mappings for current DD version
            version_data = data.get(dd_version, {})
            mappings = version_data.get("mappings", {})
            if mappings:
                logger.debug(
                    f"Loaded {len(mappings)} mappings from definitions for version {dd_version}"
                )
                return mappings
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load from definitions: {e}")

    # Fall back to resources (build artifact)
    try:
        accessor = ResourcePathAccessor(dd_version)
        mapping_path = accessor.schemas_dir / "physics_domains.json"

        if not mapping_path.exists():
            logger.warning(
                "Physics domain mappings not found. Run 'build-physics' to generate."
            )
            return {}

        with open(mapping_path, encoding="utf-8") as f:
            data = json.load(f)

        mappings = data.get("mappings", {})
        logger.debug(f"Loaded {len(mappings)} physics domain mappings from resources")
        return mappings

    except Exception as e:
        logger.error(f"Failed to load physics domain mappings: {e}")
        return {}


class PhysicsDomainCategorizer:
    """Categorization system using LLM-generated physics domain mappings."""

    def __init__(self):
        """Initialize the categorizer with cached mappings."""
        self._mappings: dict[str, str] | None = None

    @property
    def mappings(self) -> dict[str, str]:
        """Get the IDS to domain mappings (lazy loaded)."""
        if self._mappings is None:
            self._mappings = _load_physics_mappings()
        return self._mappings

    def get_domain_for_ids(self, ids_name: str) -> PhysicsDomain:
        """Get the physics domain for a given IDS name.

        Args:
            ids_name: Name of the IDS (e.g., 'equilibrium', 'core_profiles')

        Returns:
            PhysicsDomain enum value for the IDS, or GENERAL if not found.
        """
        domain_str = self.mappings.get(ids_name.lower())
        if domain_str:
            try:
                return PhysicsDomain(domain_str)
            except ValueError:
                logger.warning(
                    f"Unknown physics domain '{domain_str}' for IDS '{ids_name}'"
                )
                return PhysicsDomain.GENERAL
        return PhysicsDomain.GENERAL

    def analyze_domain_distribution(
        self, ids_list: list[str]
    ) -> dict[PhysicsDomain, int]:
        """Analyze the distribution of IDS across physics domains.

        Args:
            ids_list: List of IDS names to analyze

        Returns:
            Dictionary mapping domains to count of IDS in each domain.
        """
        distribution: dict[PhysicsDomain, int] = {}
        for ids_name in ids_list:
            domain = self.get_domain_for_ids(ids_name)
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def get_all_mappings(self) -> dict[str, PhysicsDomain]:
        """Get all IDS to domain mappings as enum values.

        Returns:
            Dictionary mapping IDS names to PhysicsDomain enum values.
        """
        result = {}
        for ids_name, domain_str in self.mappings.items():
            try:
                result[ids_name] = PhysicsDomain(domain_str)
            except ValueError:
                result[ids_name] = PhysicsDomain.GENERAL
        return result

    def get_ids_for_domain(self, domain: PhysicsDomain) -> list[str]:
        """Get all IDS names belonging to a specific domain.

        Args:
            domain: Physics domain to filter by

        Returns:
            List of IDS names in the specified domain.
        """
        return [
            ids_name
            for ids_name, domain_str in self.mappings.items()
            if domain_str == domain.value
        ]


# Global instance for easy access
physics_categorizer = PhysicsDomainCategorizer()
