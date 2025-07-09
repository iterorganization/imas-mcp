"""
Physics domain categorization for IMAS Data Dictionary.

This module provides comprehensive categorization of IDS into physics domains
and analysis tools for understanding the structure of plasma physics data.
Data is loaded from YAML definition files for better maintainability.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Any

from .data_model import PhysicsDomain
from .domain_loader import get_domain_loader


@dataclass
class DomainCharacteristics:
    """Characteristics of a physics domain."""

    description: str
    primary_phenomena: List[str]
    typical_units: List[str]
    measurement_methods: List[str]
    related_domains: List[str]
    complexity_level: str  # "basic", "intermediate", "advanced"


class PhysicsDomainCategorizer:
    """Categorization system for IMAS physics domains."""

    def __init__(self):
        self.domain_loader = get_domain_loader()
        self.domain_characteristics = self._load_domain_characteristics()
        self.ids_to_domain_mapping = self._load_ids_mapping()
        self.physics_relationships = self._load_physics_relationships()

    def _load_domain_characteristics(
        self,
    ) -> Dict[PhysicsDomain, DomainCharacteristics]:
        """Load domain characteristics from YAML definitions."""
        yaml_characteristics = self.domain_loader.load_domain_characteristics()

        characteristics = {}
        for domain_name, domain_data in yaml_characteristics.items():
            try:
                # Convert domain name to enum
                domain_enum = PhysicsDomain(domain_name)
                characteristics[domain_enum] = DomainCharacteristics(
                    description=domain_data["description"],
                    primary_phenomena=domain_data["primary_phenomena"],
                    typical_units=domain_data["typical_units"],
                    measurement_methods=domain_data["measurement_methods"],
                    related_domains=domain_data["related_domains"],
                    complexity_level=domain_data["complexity_level"],
                )
            except ValueError:
                # Skip domains not in the enum
                continue

        return characteristics

    def _load_ids_mapping(self) -> Dict[str, PhysicsDomain]:
        """Load IDS to domain mapping from YAML definitions."""
        yaml_mapping = self.domain_loader.load_ids_mapping()

        mapping = {}
        for ids_name, domain_name in yaml_mapping.items():
            try:
                domain_enum = PhysicsDomain(domain_name)
                mapping[ids_name] = domain_enum
            except ValueError:
                # Default to GENERAL for unknown domains
                mapping[ids_name] = PhysicsDomain.GENERAL

        return mapping

    def _load_physics_relationships(self) -> Dict[PhysicsDomain, Set[PhysicsDomain]]:
        """Load physics relationships from YAML definitions."""
        yaml_relationships = self.domain_loader.load_domain_relationships()

        relationships = {}
        for domain_name, related_names in yaml_relationships.items():
            try:
                domain_enum = PhysicsDomain(domain_name)
                related_enums = set()

                for related_name in related_names:
                    try:
                        related_enum = PhysicsDomain(related_name)
                        related_enums.add(related_enum)
                    except ValueError:
                        continue

                relationships[domain_enum] = related_enums
            except ValueError:
                continue

        return relationships

    def validate_yaml_definitions(self) -> Dict[str, Any]:
        """Validate the YAML definitions for consistency."""
        return self.domain_loader.validate_definitions()

    def get_domain_for_ids(self, ids_name: str) -> PhysicsDomain:
        """Get the physics domain for a given IDS name."""
        return self.ids_to_domain_mapping.get(ids_name.lower(), PhysicsDomain.GENERAL)

    def get_domain_characteristics(
        self, domain: PhysicsDomain
    ) -> DomainCharacteristics:
        """Get characteristics for a physics domain."""
        return self.domain_characteristics.get(
            domain, self.domain_characteristics[PhysicsDomain.GENERAL]
        )

    def get_related_domains(self, domain: PhysicsDomain) -> Set[PhysicsDomain]:
        """Get domains related to the given domain."""
        return self.physics_relationships.get(domain, set())

    def analyze_domain_distribution(
        self, ids_list: List[str]
    ) -> Dict[PhysicsDomain, int]:
        """Analyze the distribution of IDS across physics domains."""
        distribution = {}
        for ids_name in ids_list:
            domain = self.get_domain_for_ids(ids_name)
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def get_domain_summary(self) -> Dict[str, Dict]:
        """Get a comprehensive summary of all physics domains."""
        summary = {}

        # Group IDS by domain
        domain_ids = {}
        for ids_name, domain in self.ids_to_domain_mapping.items():
            if domain not in domain_ids:
                domain_ids[domain] = []
            domain_ids[domain].append(ids_name)

        # Create summary for each domain
        for domain, characteristics in self.domain_characteristics.items():
            ids_count = len(domain_ids.get(domain, []))
            summary[domain.value] = {
                "description": characteristics.description,
                "ids_count": ids_count,
                "ids_list": domain_ids.get(domain, []),
                "primary_phenomena": characteristics.primary_phenomena,
                "typical_units": characteristics.typical_units,
                "complexity_level": characteristics.complexity_level,
                "related_domains": [d.value for d in self.get_related_domains(domain)],
            }

        return summary

    def suggest_domain_improvements(
        self, ids_name: str, current_paths: List[str]
    ) -> Dict[str, Any]:
        """Suggest domain improvements based on IDS content analysis."""
        current_domain = self.get_domain_for_ids(ids_name)
        suggestions = {
            "current_domain": current_domain.value,
            "confidence": "high" if ids_name in self.ids_to_domain_mapping else "low",
            "alternative_domains": [],
            "reasoning": [],
        }

        # Analyze path content for domain hints
        transport_indicators = [
            "temperature",
            "density",
            "velocity",
            "flux",
            "diffusion",
            "convection",
        ]
        equilibrium_indicators = [
            "magnetic",
            "field",
            "flux",
            "current",
            "pressure",
            "force",
        ]

        # Count indicators in paths
        for path in current_paths[:10]:  # Analyze first 10 paths
            path_lower = path.lower()
            if any(indicator in path_lower for indicator in transport_indicators):
                if PhysicsDomain.TRANSPORT not in [current_domain]:
                    suggestions["alternative_domains"].append(
                        PhysicsDomain.TRANSPORT.value
                    )
                    suggestions["reasoning"].append(
                        f"Transport-related paths found: {path}"
                    )

            if any(indicator in path_lower for indicator in equilibrium_indicators):
                if PhysicsDomain.EQUILIBRIUM not in [current_domain]:
                    suggestions["alternative_domains"].append(
                        PhysicsDomain.EQUILIBRIUM.value
                    )
                    suggestions["reasoning"].append(
                        f"Equilibrium-related paths found: {path}"
                    )

        return suggestions


# Global instance for easy access
physics_categorizer = PhysicsDomainCategorizer()
