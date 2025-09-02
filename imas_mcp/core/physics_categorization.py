"""
This module provides comprehensive categorization of IDS into physics domains
using the PhysicsDomain enum with domain accessor backend.
"""

from typing import Any

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.core.physics_accessors import DomainAccessor
from imas_mcp.core.physics_domains import DomainCharacteristics


class PhysicsDomainCategorizer:
    """Categorization system using PhysicsDomain enum with domain accessor."""

    def __init__(self):
        self.domains_accessor = DomainAccessor()

    def get_domain_for_ids(self, ids_name: str) -> PhysicsDomain:
        """Get the physics domain for a given IDS name."""
        domain_str = self.domains_accessor.get_ids_domain(ids_name)
        if domain_str:
            try:
                return PhysicsDomain(domain_str)
            except ValueError:
                return PhysicsDomain.GENERAL
        return PhysicsDomain.GENERAL

    def get_domain_characteristics(
        self, domain: PhysicsDomain
    ) -> DomainCharacteristics | None:
        """Get characteristics for a physics domain."""
        return self.domains_accessor.get_domain_info(domain)

    def get_related_domains(self, domain: PhysicsDomain) -> set[PhysicsDomain]:
        """Get domains related to the given domain."""
        related_domains = self.domains_accessor.get_related_domains(domain)
        return set(related_domains)

    def analyze_domain_distribution(
        self, ids_list: list[str]
    ) -> dict[PhysicsDomain, int]:
        """Analyze the distribution of IDS across physics domains."""
        distribution = {}
        for ids_name in ids_list:
            domain = self.get_domain_for_ids(ids_name)
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def get_domain_summary(self) -> dict[PhysicsDomain, dict[str, Any]]:
        """Get domain summary using PhysicsDomain enum."""
        summary_data = self.domains_accessor.get_domain_summary()
        enum_summary = {}

        for domain_str, info in summary_data.items():
            try:
                domain_enum = PhysicsDomain(domain_str)
            except ValueError:
                domain_enum = PhysicsDomain.GENERAL
            enum_summary[domain_enum] = info

        return enum_summary

    def suggest_domain_improvements(
        self, ids_name: str, current_paths: list[str] | None = None
    ) -> dict[str, Any]:
        """Domain suggestions using PhysicsDomain enum."""
        current_domain = self.get_domain_for_ids(ids_name)
        cross_analysis = self.domains_accessor.get_cross_domain_analysis(current_domain)

        suggestions = {
            "current_domain": current_domain,
            "confidence": "high" if cross_analysis else "low",
            "alternative_domains": [],
            "reasoning": [],
            "analysis": cross_analysis,
        }

        if cross_analysis:
            related_domains = cross_analysis.get("shared_phenomena_domains", [])
            for domain_str in related_domains[:3]:
                try:
                    enum_domain = PhysicsDomain(domain_str)
                    if enum_domain != current_domain:
                        suggestions["alternative_domains"].append(enum_domain)
                        suggestions["reasoning"].append(
                            f"Shares phenomena with {domain_str}"
                        )
                except ValueError:
                    continue

        return suggestions

    def validate_definitions(self) -> dict[str, Any]:
        """Validate the domain definitions."""
        return self.domains_accessor.validate_definitions()

    def search_domains_by_concept(self, concept: str) -> list[PhysicsDomain]:
        """Search domains by concept or phenomenon."""
        return self.domains_accessor.search_domains_by_phenomenon(concept)

    def get_domains_by_complexity(self, complexity: str) -> list[PhysicsDomain]:
        """Get domains filtered by complexity level."""
        complexity_distribution = (
            self.domains_accessor.get_domain_complexity_distribution()
        )
        return complexity_distribution.get(complexity, [])

    def get_measurement_methods_for_domain(self, domain: PhysicsDomain) -> list[str]:
        """Get measurement methods for a domain."""
        return self.domains_accessor.find_measurement_methods_for_domain(domain)

    def get_typical_units_for_domain(self, domain: PhysicsDomain) -> list[str]:
        """Get typical units for a domain."""
        return self.domains_accessor.find_units_for_domain(domain)


# Global instance for easy access
physics_categorizer = PhysicsDomainCategorizer()
