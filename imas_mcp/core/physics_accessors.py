"""
Domain accessor methods for physics domains definitions.

This module provides high-level accessor methods for interacting with YAML-based
physics domain definitions. Keeps data access methods close to the data files.
"""

import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .data_model import PhysicsDomain
from .physics_domains import (
    DomainCharacteristics,
    DomainRelationshipsData,
    IDSMappingData,
    PhysicsContext,
    PhysicsDomainsData,
    UnitContext,
    UnitContextsData,
)

logger = logging.getLogger(__name__)


class DomainAccessor:
    """Accessor for physics domains with caching and validation."""

    def __init__(self, definitions_path: Path | None = None):
        """Initialize with optional custom path."""
        if definitions_path is None:
            # Point to definitions structure
            definitions_resource = resources.files("imas_mcp") / "definitions"
            definitions_path = Path(str(definitions_resource))

        self.definitions_path = definitions_path
        self._context: PhysicsContext | None = None

    @property
    def context(self) -> PhysicsContext:
        """Get the physics context, loading if necessary."""
        if self._context is None:
            self._context = self._load_context()
        return self._context

    def _load_context(self) -> PhysicsContext:
        """Load all physics definitions into a physics context."""
        try:
            # Load domains
            domains_data = self._load_domains_data()

            # Load IDS mapping
            ids_mapping_data = self._load_ids_mapping_data()

            # Load relationships
            relationships_data = self._load_relationships_data()

            # Load unit contexts
            unit_contexts_data = self._load_unit_contexts_data()

            # Create physics context
            context = PhysicsContext(
                domains=domains_data,
                ids_mapping=ids_mapping_data,
                relationships=relationships_data,
                unit_contexts=unit_contexts_data,
            )

            return context

        except Exception as e:
            logger.error(f"Failed to load physics context: {e}")
            raise

    def _load_domains_data(self) -> PhysicsDomainsData:
        """Load domain characteristics data."""
        file_path = (
            self.definitions_path / "physics" / "domains" / "characteristics.yaml"
        )
        return self._load_yaml_model(file_path, PhysicsDomainsData)

    def _load_ids_mapping_data(self) -> IDSMappingData:
        """Load IDS mapping data."""
        file_path = self.definitions_path / "physics" / "domains" / "ids_mapping.yaml"
        return self._load_yaml_model(file_path, IDSMappingData)

    def _load_relationships_data(self) -> DomainRelationshipsData:
        """Load domain relationships data."""
        file_path = self.definitions_path / "physics" / "domains" / "relationships.yaml"
        return self._load_yaml_model(file_path, DomainRelationshipsData)

    def _load_unit_contexts_data(self) -> UnitContextsData:
        """Load unit contexts data."""
        file_path = self.definitions_path / "physics" / "units" / "contexts.yaml"
        return self._load_yaml_model(file_path, UnitContextsData)

    def _load_yaml_model(self, file_path: Path, model_class):
        """Load and validate YAML file using Pydantic model."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            return model_class.model_validate(data)

        except FileNotFoundError:
            logger.error(f"Definition file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Validation error in {file_path}: {e}")
            raise

    # High-level accessor methods
    def get_all_domains(self) -> set[PhysicsDomain]:
        """Get all available physics domains."""
        domain_names = self.context.get_domain_names()
        domains = set()
        for domain_name in domain_names:
            try:
                domains.add(PhysicsDomain(domain_name))
            except ValueError:
                domains.add(PhysicsDomain.GENERAL)
        return domains

    def get_domain_info(self, domain: PhysicsDomain) -> DomainCharacteristics | None:
        """Get detailed information about a domain."""
        return self.context.get_domain_characteristics(domain.value)

    def get_domain_ids(self, domain: PhysicsDomain) -> list[str]:
        """Get all IDS for a domain."""
        return self.context.get_ids_for_domain(domain.value)

    def get_ids_domain(self, ids_name: str) -> PhysicsDomain | None:
        """Get domain for an IDS."""
        domain_str = self.context.get_domain_for_ids(ids_name)
        if domain_str:
            try:
                return PhysicsDomain(domain_str)
            except ValueError:
                return PhysicsDomain.GENERAL
        return None

    def get_related_domains(self, domain: PhysicsDomain) -> list[PhysicsDomain]:
        """Get domains related to the specified domain."""
        related_strs = self.context.get_related_domains(domain.value)
        related_domains = []
        for domain_str in related_strs:
            try:
                related_domains.append(PhysicsDomain(domain_str))
            except ValueError:
                related_domains.append(PhysicsDomain.GENERAL)
        return related_domains

    def search_domains_by_phenomenon(self, phenomenon: str) -> list[PhysicsDomain]:
        """Find domains containing a specific phenomenon."""
        domain_strs = self.context.get_domains_by_phenomenon(phenomenon)
        domains = []
        for domain_str in domain_strs:
            try:
                domains.append(PhysicsDomain(domain_str))
            except ValueError:
                domains.append(PhysicsDomain.GENERAL)
        return domains

    def search_domains_by_unit(self, unit: str) -> list[PhysicsDomain]:
        """Find domains that use a specific unit."""
        domain_strs = self.context.get_domains_by_unit(unit)
        domains = []
        for domain_str in domain_strs:
            try:
                domains.append(PhysicsDomain(domain_str))
            except ValueError:
                domains.append(PhysicsDomain.GENERAL)
        return domains

    def get_domain_summary(self) -> dict[PhysicsDomain, dict]:
        """Get summary of all domains with key statistics."""
        summary = {}

        for domain_enum in self.get_all_domains():
            characteristics = self.get_domain_info(domain_enum)
            ids_list = self.get_domain_ids(domain_enum)
            related = self.get_related_domains(domain_enum)

            if characteristics:
                summary[domain_enum] = {
                    "description": characteristics.description,
                    "complexity": characteristics.complexity_level.value,
                    "ids_count": len(ids_list),
                    "phenomena_count": len(characteristics.primary_phenomena),
                    "units_count": len(characteristics.typical_units),
                    "related_domains": related,
                    "sample_phenomena": characteristics.primary_phenomena[:3],
                    "sample_units": characteristics.typical_units[:3],
                    "sample_ids": ids_list[:3],
                }

        return summary

    def get_cross_domain_analysis(self, domain: PhysicsDomain) -> dict[str, Any]:
        """Get comprehensive cross-domain analysis."""
        return self.context.get_cross_domain_analysis(domain.value)

    def validate_definitions(self) -> dict[str, Any]:
        """Validate the consistency of all definitions."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        try:
            # Check domain consistency
            all_domains = self.get_all_domains()
            all_domain_strs = {domain.value for domain in all_domains}

            # Check IDS mappings reference valid domains
            ids_mapping = self.context.ids_mapping.get_domain_mappings()
            mapped_domains = set(ids_mapping.keys())

            missing_domain_chars = mapped_domains - all_domain_strs
            if missing_domain_chars:
                validation_results["errors"].append(
                    f"IDS mappings reference undefined domains: {missing_domain_chars}"
                )
                validation_results["valid"] = False

            # Check relationships reference valid domains
            for domain, related in self.context.relationships.relationships.items():
                if domain not in all_domain_strs:
                    validation_results["warnings"].append(
                        f"Relationship source domain not defined: {domain}"
                    )

                invalid_related = set(related) - all_domain_strs
                if invalid_related:
                    validation_results["warnings"].append(
                        f"Domain {domain} references undefined related domains: {invalid_related}"
                    )

            # Generate statistics
            total_ids = sum(len(ids_list) for ids_list in ids_mapping.values())
            validation_results["statistics"] = {
                "total_domains": len(all_domains),
                "total_ids_mapped": total_ids,
                "domains_with_relationships": len(
                    self.context.relationships.relationships
                ),
                "total_unit_contexts": len(self.context.unit_contexts.unit_contexts),
            }

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation failed: {e}")

        return validation_results

    def get_enhanced_unit_contexts(self) -> list[UnitContext]:
        """Get enhanced unit contexts with physics domain information."""
        return self.context.unit_contexts.get_enhanced_unit_contexts()

    def find_units_for_domain(self, domain: PhysicsDomain) -> list[str]:
        """Find typical units for a specific domain."""
        characteristics = self.get_domain_info(domain)
        return characteristics.typical_units if characteristics else []

    def find_measurement_methods_for_domain(self, domain: PhysicsDomain) -> list[str]:
        """Find measurement methods for a specific domain."""
        characteristics = self.get_domain_info(domain)
        return characteristics.measurement_methods if characteristics else []

    def get_domain_complexity_distribution(self) -> dict[str, list[PhysicsDomain]]:
        """Get domains grouped by complexity level."""
        distribution = {"basic": [], "intermediate": [], "advanced": []}

        for domain_enum in self.get_all_domains():
            characteristics = self.get_domain_info(domain_enum)
            if characteristics:
                complexity = characteristics.complexity_level.value
                distribution[complexity].append(domain_enum)

        return distribution


class UnitAccessor:
    """Accessor for unit contexts and physics domain relationships."""

    def __init__(self):
        """Initialize unit accessor."""
        self._domain_accessor = get_domain_accessor()

    def get_all_unit_contexts(self) -> dict[str, str]:
        """Get all unit contexts."""
        return self._domain_accessor.context.unit_contexts.unit_contexts

    def get_unit_context(self, unit: str) -> str | None:
        """Get context for a specific unit."""
        return self._domain_accessor.context.unit_contexts.unit_contexts.get(unit)

    def get_unit_categories(self) -> dict[str, list[str]]:
        """Get unit categories."""
        return self._domain_accessor.context.unit_contexts.unit_categories

    def get_physics_domain_hints(self) -> dict[str, list[str]]:
        """Get physics domain hints for unit categories."""
        return self._domain_accessor.context.unit_contexts.physics_domain_hints

    def get_units_for_category(self, category: str) -> list[str]:
        """Get units for a specific category."""
        categories = self.get_unit_categories()
        return categories.get(category, [])

    def get_category_for_unit(self, unit: str) -> str | None:
        """Get category for a specific unit."""
        categories = self.get_unit_categories()
        for category, units in categories.items():
            if unit in units:
                return category
        return None

    def get_domains_for_unit(self, unit: str) -> list[PhysicsDomain]:
        """Get physics domains that typically use this unit."""
        category = self.get_category_for_unit(unit)
        if category:
            hints = self.get_physics_domain_hints()
            domain_strs = hints.get(category, [])
            domains = []
            for domain_str in domain_strs:
                try:
                    domains.append(PhysicsDomain(domain_str))
                except ValueError:
                    domains.append(PhysicsDomain.GENERAL)
            return domains
        return []

    def search_units_by_context(self, search_term: str) -> list[str]:
        """Search units by context description."""
        matching_units = []
        search_lower = search_term.lower()

        contexts = self.get_all_unit_contexts()
        for unit, context in contexts.items():
            if search_lower in context.lower():
                matching_units.append(unit)

        return matching_units


# Global accessor instances
_global_accessor: DomainAccessor | None = None
_global_unit_accessor: UnitAccessor | None = None


@lru_cache(maxsize=1)
def get_domain_accessor() -> DomainAccessor:
    """Get the global domain accessor instance with caching."""
    global _global_accessor
    if _global_accessor is None:
        _global_accessor = DomainAccessor()
    return _global_accessor


@lru_cache(maxsize=1)
def get_unit_accessor() -> UnitAccessor:
    """Get the global unit accessor instance with caching."""
    global _global_unit_accessor
    if _global_unit_accessor is None:
        _global_unit_accessor = UnitAccessor()
    return _global_unit_accessor


# Convenience functions for easy access
def get_all_physics_domains() -> set[PhysicsDomain]:
    """Get all available physics domains."""
    return get_domain_accessor().get_all_domains()


def get_domain_characteristics(
    domain: PhysicsDomain,
) -> DomainCharacteristics | None:
    """Get characteristics for a specific domain."""
    return get_domain_accessor().get_domain_info(domain)


def get_ids_for_domain(domain: PhysicsDomain) -> list[str]:
    """Get IDS names for a domain."""
    return get_domain_accessor().get_domain_ids(domain)


def get_domain_for_ids(ids_name: str) -> PhysicsDomain | None:
    """Get domain for an IDS name."""
    return get_domain_accessor().get_ids_domain(ids_name)


def search_domains_by_concept(concept: str) -> list[PhysicsDomain]:
    """Search domains by phenomenon or concept."""
    return get_domain_accessor().search_domains_by_phenomenon(concept)


def get_domain_summary() -> dict[PhysicsDomain, dict]:
    """Get summary of all domains."""
    return get_domain_accessor().get_domain_summary()


def validate_physics_definitions() -> dict[str, Any]:
    """Validate all physics definitions for consistency."""
    return get_domain_accessor().validate_definitions()
