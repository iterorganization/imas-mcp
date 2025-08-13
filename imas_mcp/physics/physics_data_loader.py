"""
Physics data loader for relationship analysis.

This module loads physics definitions from external YAML files.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PhysicsDataLoader:
    """Loads physics definitions from external YAML files."""

    def __init__(self, definitions_root: Path | None = None):
        """Initialize the physics data loader.

        Args:
            definitions_root: Root path to definitions directory.
                            If None, uses package default.
        """
        if definitions_root is None:
            # Default to package definitions directory
            current_file = Path(__file__)
            self.definitions_root = current_file.parent.parent / "definitions"
        else:
            self.definitions_root = Path(definitions_root)

        self.physics_root = self.definitions_root / "physics"
        self._domain_characteristics = None
        self._domain_relationships = None
        self._unit_contexts = None
        self._physics_concepts = None

    def load_domain_characteristics(self) -> dict[str, Any]:
        """Load physics domain characteristics."""
        if self._domain_characteristics is None:
            characteristics_file = (
                self.physics_root / "domains" / "characteristics.yaml"
            )
            try:
                with open(characteristics_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self._domain_characteristics = data.get("domains", {})
            except FileNotFoundError:
                logger.warning(
                    f"Domain characteristics file not found: {characteristics_file}"
                )
                self._domain_characteristics = {}
            except Exception as e:
                logger.error(f"Error loading domain characteristics: {e}")
                self._domain_characteristics = {}

        return self._domain_characteristics

    def load_domain_relationships(self) -> dict[str, list[str]]:
        """Load physics domain relationships."""
        if self._domain_relationships is None:
            relationships_file = self.physics_root / "domains" / "relationships.yaml"
            try:
                with open(relationships_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self._domain_relationships = data.get("relationships", {})
            except FileNotFoundError:
                logger.warning(
                    f"Domain relationships file not found: {relationships_file}"
                )
                self._domain_relationships = {}
            except Exception as e:
                logger.error(f"Error loading domain relationships: {e}")
                self._domain_relationships = {}

        return self._domain_relationships

    def load_unit_contexts(self) -> dict[str, Any]:
        """Load unit context definitions."""
        if self._unit_contexts is None:
            units_file = self.physics_root / "units" / "contexts.yaml"
            try:
                with open(units_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self._unit_contexts = data  # Load full data structure
            except FileNotFoundError:
                logger.warning(f"Unit contexts file not found: {units_file}")
                self._unit_contexts = {}
            except Exception as e:
                logger.error(f"Error loading unit contexts: {e}")
                self._unit_contexts = {}

        return self._unit_contexts

    def generate_physics_concepts(self) -> dict[str, dict[str, Any]]:
        """Generate physics concepts from domain characteristics.

        This converts the domain characteristics into the concept format
        needed by the relationship analyzer.
        """
        if self._physics_concepts is not None:
            return self._physics_concepts

        characteristics = self.load_domain_characteristics()
        concepts = {}

        # Extract physics concepts from domain primary phenomena
        for domain_name, domain_data in characteristics.items():
            phenomena = domain_data.get("primary_phenomena", [])
            units = domain_data.get("typical_units", [])
            methods = domain_data.get("measurement_methods", [])
            related_domains = domain_data.get("related_domains", [])

            # Create concepts for each phenomenon
            for phenomenon in phenomena:
                # Clean up phenomenon name (remove spaces, special chars)
                concept_key = phenomenon.replace(" ", "_").replace("-", "_").lower()

                concepts[concept_key] = {
                    "domain": domain_name,
                    "related_concepts": phenomena,  # Other phenomena in same domain
                    "measurement_types": methods,
                    "typical_units": units,
                    "related_domains": related_domains,
                    "source_domain": domain_name,
                }

        # Add direct domain concepts (for paths that contain domain names)
        for domain_name, domain_data in characteristics.items():
            if domain_name not in concepts:
                concepts[domain_name] = {
                    "domain": domain_name,
                    "related_concepts": domain_data.get("primary_phenomena", []),
                    "measurement_types": domain_data.get("measurement_methods", []),
                    "typical_units": domain_data.get("typical_units", []),
                    "related_domains": domain_data.get("related_domains", []),
                    "source_domain": domain_name,
                }

        self._physics_concepts = concepts
        return concepts

    def get_related_domains(self, domain: str) -> list[str]:
        """Get domains related to the given domain."""
        relationships = self.load_domain_relationships()
        return relationships.get(domain, [])

    def get_domain_characteristics(self, domain: str) -> dict[str, Any]:
        """Get characteristics for a specific domain."""
        characteristics = self.load_domain_characteristics()
        return characteristics.get(domain, {})

    def get_unit_category(self, unit: str) -> str | None:
        """Get the category for a given unit."""
        contexts = self.load_unit_contexts()
        unit_categories = contexts.get("unit_categories", {})

        for category, units in unit_categories.items():
            if unit in units:
                return category
        return None

    def get_physics_domain_for_unit(self, unit: str) -> list[str]:
        """Get likely physics domains for a unit."""
        contexts = self.load_unit_contexts()
        unit_category = self.get_unit_category(unit)

        if unit_category:
            domain_hints = contexts.get("physics_domain_hints", {})
            return domain_hints.get(unit_category, [])
        return []
