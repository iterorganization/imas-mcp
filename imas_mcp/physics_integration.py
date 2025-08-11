"""
Physics integration for IMAS MCP Tools using YAML-based domain definitions.

This module provides physics-aware search and concept explanation capabilities
using the YAML domain definitions with semantic search over embeddings.
"""

import logging

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.core.physics_accessors import DomainAccessor, UnitAccessor
from imas_mcp.models.constants import ComplexityLevel, UnitCategory
from imas_mcp.models.physics_models import (
    ConceptExplanation,
    ConceptSuggestion,
    DomainConcepts,
    PhysicsMatch,
    PhysicsSearchResult,
    UnitContext,
    UnitSuggestion,
)
from imas_mcp.search.physics_search import (
    get_physics_search,
    search_physics_concepts,
)

logger = logging.getLogger(__name__)


def physics_search(query: str, max_results: int = 10) -> PhysicsSearchResult:
    """
    Search physics concepts using semantic similarity over YAML domain data.

    Args:
        query: Search query for physics concepts
        max_results: Maximum number of results to return

    Returns:
        PhysicsSearchResult with matched concepts and domain context
    """
    # Use semantic search over physics definitions
    semantic_results = search_physics_concepts(
        query,
        max_results=max_results,
        min_similarity=0.2,  # More inclusive threshold
    )

    # Convert to physics matches with proper domain handling
    physics_matches = []
    concept_suggestions = []
    unit_suggestions = []
    symbol_suggestions = []
    imas_path_suggestions = []

    domain_accessor = DomainAccessor()

    for result in semantic_results:
        doc = result.document

        # Convert domain string to PhysicsDomain enum
        try:
            domain_enum = PhysicsDomain(doc.domain_name)
        except ValueError:
            domain_enum = PhysicsDomain.GENERAL

        # Get domain data for additional context
        domain_data = domain_accessor.get_domain_info(domain_enum)

        physics_match = PhysicsMatch(
            concept=doc.title,
            quantity_name=doc.title,
            symbol=doc.metadata.get("symbol", ""),
            units=", ".join(domain_data.typical_units[:3] if domain_data else []),
            description=doc.description,
            imas_paths=doc.metadata.get("imas_paths", []),
            domain=domain_enum,
            relevance_score=result.similarity_score,
        )

        physics_matches.append(physics_match)

        # Add concept suggestions
        if doc.concept_type == "phenomenon":
            concept_suggestions.append(
                ConceptSuggestion(concept=doc.title, description=doc.description)
            )

        # Add unit suggestions
        if domain_data and domain_data.typical_units:
            for unit in domain_data.typical_units[:2]:
                unit_suggestions.append(
                    UnitSuggestion(
                        unit=unit,
                        description=f"Typical unit for {doc.domain_name}",
                        example_quantities=[doc.title],
                    )
                )

    return PhysicsSearchResult(
        query=query,
        physics_matches=physics_matches,
        concept_suggestions=concept_suggestions[:5],
        unit_suggestions=unit_suggestions[:5],
        symbol_suggestions=symbol_suggestions,
        imas_path_suggestions=imas_path_suggestions,
    )


def explain_physics_concept(
    concept: str, detail_level: str = "intermediate"
) -> ConceptExplanation:
    """
    Explain a physics concept using YAML domain definitions.

    Args:
        concept: Physics concept to explain
        detail_level: Level of detail (basic, intermediate, advanced)

    Returns:
        ConceptExplanation with domain context and related information
    """
    # Search for the concept in domain data
    search_results = search_physics_concepts(concept, max_results=1, min_similarity=0.3)

    domain_accessor = DomainAccessor()

    if not search_results:
        # Fallback: search domain names directly
        concept_lower = concept.lower().replace(" ", "_")

        try:
            domain_enum = PhysicsDomain(concept_lower)
            domain_data = domain_accessor.get_domain_info(domain_enum)
            if domain_data:
                return ConceptExplanation(
                    concept=concept,
                    domain=domain_enum,
                    description=domain_data.description,
                    phenomena=domain_data.primary_phenomena,
                    typical_units=domain_data.typical_units,
                    measurement_methods=domain_data.measurement_methods,
                    related_domains=domain_accessor.get_related_domains(domain_enum),
                    complexity_level=domain_data.complexity_level,
                )
        except ValueError:
            pass

        # If still not found, provide a generic explanation
        return ConceptExplanation(
            concept=concept,
            domain=PhysicsDomain.GENERAL,
            description=f"Physics concept: {concept}",
            phenomena=[],
            typical_units=[],
            measurement_methods=[],
            related_domains=[],
            complexity_level=ComplexityLevel.BASIC,
        )

    # Use the best matching result
    best_result = search_results[0]
    doc = best_result.document

    # Convert domain string to enum
    try:
        domain_enum = PhysicsDomain(doc.domain_name)
    except ValueError:
        domain_enum = PhysicsDomain.GENERAL

    # Get full domain data
    domain_data = domain_accessor.get_domain_info(domain_enum)

    if not domain_data:
        return ConceptExplanation(
            concept=concept,
            domain=domain_enum,
            description=doc.description,
            phenomena=[],
            typical_units=[],
            measurement_methods=[],
            related_domains=[],
            complexity_level=ComplexityLevel.BASIC,
        )

    return ConceptExplanation(
        concept=concept,
        domain=domain_enum,
        description=domain_data.description,
        phenomena=domain_data.primary_phenomena,
        typical_units=domain_data.typical_units,
        measurement_methods=domain_data.measurement_methods,
        related_domains=domain_accessor.get_related_domains(domain_enum),
        complexity_level=domain_data.complexity_level,
    )


def get_domain_concepts(domain: PhysicsDomain) -> DomainConcepts:
    """Get all physics concepts for a specific domain."""
    domain_accessor = DomainAccessor()
    domain_data = domain_accessor.get_domain_info(domain)

    if not domain_data:
        return DomainConcepts(domain=domain, concepts=[])

    concepts = [domain.value.replace("_", " ").title()]
    concepts.extend(
        [p.replace("_", " ").title() for p in domain_data.primary_phenomena]
    )
    concepts.extend(
        [m.replace("_", " ").title() for m in domain_data.measurement_methods]
    )

    return DomainConcepts(domain=domain, concepts=concepts)


def get_unit_physics_context(unit: str) -> UnitContext:
    """Get physics context for a unit from YAML definitions."""
    unit_accessor = UnitAccessor()
    context = unit_accessor.get_unit_context(unit)

    if context is None:
        # Try semantic search to find the unit
        results = search_physics_concepts(unit, max_results=1, concept_types=["unit"])
        if results:
            # Extract the unit symbol from the best match
            best_match = results[0]
            unit_symbol = best_match.document.metadata.get("symbol", unit)
            context = unit_accessor.get_unit_context(unit_symbol)
            unit = unit_symbol  # Use the resolved symbol

    category = unit_accessor.get_category_for_unit(unit)
    # Convert string category to UnitCategory enum if needed
    if category and isinstance(category, str) and category.strip():
        try:
            category = UnitCategory(category)
        except ValueError:
            category = None
    else:
        category = None

    physics_domains = unit_accessor.get_domains_for_unit(unit)

    return UnitContext(
        unit=unit,
        context=context,
        category=category,
        physics_domains=physics_domains,
    )


# Initialize embeddings on import (non-blocking)
def _initialize_physics_embeddings():
    """Initialize physics embeddings in background."""
    try:
        physics_search_engine = get_physics_search()
        # Try to load cache, build if needed
        physics_search_engine.build_embeddings(force_rebuild=False)
    except Exception as e:
        # Don't fail import if embeddings can't be built
        logger.warning(f"Could not initialize physics embeddings: {e}")


# Initialize embeddings when module is imported
_initialize_physics_embeddings()
