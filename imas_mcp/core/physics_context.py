"""
Physics context provider for IMAS MCP tools.

This module provides physics context enrichment using YAML domain definitions
and unit contexts to enhance MCP tool responses with domain-specific information.
"""

from typing import Any

from pydantic import BaseModel, Field

from imas_mcp.core.physics_accessors import get_domain_accessor, get_unit_accessor


class DomainContext(BaseModel):
    """Rich domain context from YAML definitions."""

    domain_name: str
    description: str
    primary_phenomena: list[str] = Field(default_factory=list)
    typical_units: list[str] = Field(default_factory=list)
    measurement_methods: list[str] = Field(default_factory=list)
    related_domains: list[str] = Field(default_factory=list)
    complexity_level: str = "unknown"
    match_type: str = "direct"


class UnitContext(BaseModel):
    """Rich unit context from YAML definitions."""

    unit: str
    context_description: str | None = None
    category: str | None = None
    physics_domains: list[str] = Field(default_factory=list)
    measurement_significance: str = "Physical unit"


class PhysicsContextProvider:
    """Provider for physics context enrichment using YAML definitions."""

    def __init__(self):
        self.domain_accessor = get_domain_accessor()
        self.unit_accessor = get_unit_accessor()

    def get_domain_context(self, concept: str) -> DomainContext | None:
        """Get domain context for a concept."""
        concept_lower = concept.lower()

        domain_info = self.domain_accessor.get_domain_info(concept_lower)
        if domain_info:
            return DomainContext(
                domain_name=concept_lower,
                description=domain_info.description,
                primary_phenomena=domain_info.primary_phenomena,
                typical_units=domain_info.typical_units,
                measurement_methods=domain_info.measurement_methods,
                related_domains=domain_info.related_domains,
                complexity_level=domain_info.complexity_level.value
                if domain_info.complexity_level
                else "unknown",
                match_type="direct",
            )

        all_domains = self.domain_accessor.get_all_domains()
        for domain_name in all_domains:
            domain_data = self.domain_accessor.get_domain_info(domain_name)
            if domain_data:
                searchable_text = (
                    " ".join(domain_data.primary_phenomena)
                    + " "
                    + " ".join(domain_data.measurement_methods)
                    + " "
                    + domain_data.description
                ).lower()

                if concept_lower in searchable_text:
                    return DomainContext(
                        domain_name=domain_name,
                        description=domain_data.description,
                        primary_phenomena=domain_data.primary_phenomena[:3],
                        typical_units=domain_data.typical_units[:3],
                        measurement_methods=domain_data.measurement_methods[:3],
                        related_domains=domain_data.related_domains[:3],
                        complexity_level=domain_data.complexity_level.value
                        if domain_data.complexity_level
                        else "unknown",
                        match_type="partial",
                    )

        return None

    def get_unit_context(self, concept: str) -> UnitContext | None:
        """Get unit context for a concept."""
        unit_context = self.unit_accessor.get_unit_context(concept)
        if unit_context:
            category = self.unit_accessor.get_category_for_unit(concept)
            physics_domains = self.unit_accessor.get_domains_for_unit(concept)
            return UnitContext(
                unit=concept,
                context_description=unit_context,
                category=category,
                physics_domains=physics_domains,
                measurement_significance=f"Used in {category} measurements"
                if category
                else "Physical unit",
            )
        return None

    def enhance_explanation(
        self, concept: str, detail_level: str = "intermediate"
    ) -> dict[str, str]:
        """Generate enhanced explanation content based on physics context."""
        domain_context = self.get_domain_context(concept)
        unit_context = self.get_unit_context(concept)

        explanation = {
            "definition": f"Analysis of '{concept}' within IMAS data dictionary context",
            "physics_context": "No specific physics domain context found",
            "measurement_scope": "General measurement context",
            "data_availability": "Standard data paths available",
        }

        if domain_context:
            complexity_level = domain_context.complexity_level

            if detail_level == "basic" or complexity_level == "basic":
                explanation["definition"] = (
                    f"{concept}: {domain_context.description}. Primary phenomena: {', '.join(domain_context.primary_phenomena[:2])}"
                )
                explanation["measurement_scope"] = (
                    f"Measured using: {', '.join(domain_context.measurement_methods[:2])}"
                )
            elif detail_level == "advanced" or complexity_level == "advanced":
                explanation["definition"] = (
                    f"{concept}: {domain_context.description}. Encompasses {len(domain_context.primary_phenomena)} primary phenomena including {', '.join(domain_context.primary_phenomena[:4])}"
                )
                explanation["measurement_scope"] = (
                    f"Advanced measurements: {', '.join(domain_context.measurement_methods)}"
                )
                explanation["related_physics"] = (
                    f"Coupled to: {', '.join(domain_context.related_domains)}"
                )
            else:
                explanation["definition"] = (
                    f"{concept}: {domain_context.description}. Key phenomena: {', '.join(domain_context.primary_phenomena[:3])}"
                )
                explanation["measurement_scope"] = (
                    f"Typical measurements: {', '.join(domain_context.measurement_methods[:3])}"
                )

            explanation["typical_units"] = (
                f"Units: {', '.join(domain_context.typical_units[:5])}"
            )
            explanation["complexity_level"] = f"Complexity: {complexity_level}"

        if unit_context:
            explanation["unit_context"] = unit_context.context_description or ""
            explanation["unit_category"] = unit_context.category or "Unspecified"
            explanation["unit_domains"] = (
                ", ".join(unit_context.physics_domains)
                if unit_context.physics_domains
                else "Not specified"
            )
            explanation["unit_significance"] = unit_context.measurement_significance

        return explanation

    def get_domain_definition(self, domain: str) -> dict[str, Any] | None:
        """Get domain definition for export enhancement."""
        domain_info = self.domain_accessor.get_domain_info(domain.lower())
        if not domain_info:
            return None

        result = {
            "domain_name": domain,
            "description": domain_info.description,
            "primary_phenomena": domain_info.primary_phenomena,
            "typical_units": domain_info.typical_units,
            "measurement_methods": domain_info.measurement_methods,
            "related_domains": domain_info.related_domains,
            "complexity_level": domain_info.complexity_level.value
            if domain_info.complexity_level
            else "unknown",
            "definition_source": "YAML domain definitions",
        }

        units_context = {}
        for unit in domain_info.typical_units:
            unit_context = self.unit_accessor.get_unit_context(unit)
            category = self.unit_accessor.get_category_for_unit(unit)
            if unit_context or category:
                units_context[unit] = {
                    "context": unit_context,
                    "category": category,
                    "definition_priority": "high",
                }

        if units_context:
            result["units_context"] = units_context

        return result

    def get_cross_domain_analysis(self, domain: str) -> dict[str, Any]:
        """Get cross-domain analysis using definition relationships."""
        domain_info = self.domain_accessor.get_domain_info(domain.lower())
        if not domain_info or not domain_info.related_domains:
            return {}

        cross_analysis = {}
        for related_domain in domain_info.related_domains:
            related_domain_def = self.domain_accessor.get_domain_info(related_domain)
            if related_domain_def:
                cross_analysis[related_domain] = {
                    "has_definition": True,
                    "complexity_level": related_domain_def.complexity_level.value
                    if related_domain_def.complexity_level
                    else "unknown",
                    "common_units": list(
                        set(domain_info.typical_units)
                        & set(related_domain_def.typical_units)
                    ),
                    "common_methods": list(
                        set(domain_info.measurement_methods)
                        & set(related_domain_def.measurement_methods)
                    ),
                    "relationship_strength": "strong",
                }

        return cross_analysis

    def enhance_search_suggestions(self, query: str, result_count: int) -> list[str]:
        """Generate enhanced search suggestions based on physics context."""
        suggestions = []

        domain_context = self.get_domain_context(query)
        unit_context = self.get_unit_context(query)

        if result_count == 0:
            suggestions.extend(
                [
                    "Try a broader search term",
                    "Check spelling of physics terms",
                    "Use search_mode='semantic' for concept-based search",
                ]
            )

            if unit_context and unit_context.category:
                suggestions.append(f"Try related {unit_context.category} units")

            if domain_context:
                suggestions.extend(
                    [
                        f"Try '{method}' for measurement methods"
                        for method in domain_context.measurement_methods[:2]
                    ]
                )

        elif result_count < 3:
            suggestions.extend(
                [
                    "Try related physics concepts",
                    "Use search_mode='hybrid' for comprehensive results",
                ]
            )

            if domain_context and domain_context.related_domains:
                suggestions.append(
                    f"Explore related domains: {', '.join(domain_context.related_domains[:2])}"
                )

        return suggestions


_physics_context_provider: PhysicsContextProvider | None = None


def get_physics_context_provider() -> PhysicsContextProvider:
    """Get the global physics context provider instance."""
    global _physics_context_provider
    if _physics_context_provider is None:
        _physics_context_provider = PhysicsContextProvider()
    return _physics_context_provider
