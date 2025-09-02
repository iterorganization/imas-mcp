"""Clean, focused Pydantic models for physics search and semantic analysis."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.models.constants import ComplexityLevel, ConceptType, UnitCategory

# ============================================================================
# PHYSICS SEARCH COMPONENTS
# ============================================================================


class PhysicsMatch(BaseModel):
    """A physics concept match from search."""

    concept: str
    quantity_name: str
    symbol: str
    units: str
    description: str
    imas_paths: list[str] = Field(default_factory=list)
    domain: PhysicsDomain
    relevance_score: float


class ConceptSuggestion(BaseModel):
    """A concept suggestion."""

    concept: str
    description: str | None = None


class UnitSuggestion(BaseModel):
    """A unit suggestion."""

    unit: str
    description: str
    example_quantities: list[str] = Field(default_factory=list)


class SymbolSuggestion(BaseModel):
    """A symbol suggestion."""

    symbol: str
    concept: str | None = None
    description: str | None = None


# ============================================================================
# PHYSICS SEARCH RESULTS
# ============================================================================


class PhysicsSearchResult(BaseModel):
    """Complete physics search result."""

    query: str
    physics_matches: list[PhysicsMatch] = Field(default_factory=list)
    concept_suggestions: list[ConceptSuggestion] = Field(default_factory=list)
    unit_suggestions: list[UnitSuggestion] = Field(default_factory=list)
    symbol_suggestions: list[SymbolSuggestion] = Field(default_factory=list)
    imas_path_suggestions: list[str] = Field(default_factory=list)


# ============================================================================
# CONCEPT & DOMAIN MODELS
# ============================================================================


class ConceptExplanation(BaseModel):
    """Explanation of a physics concept with domain context."""

    concept: str
    domain: PhysicsDomain
    description: str
    phenomena: list[str] = Field(default_factory=list)
    typical_units: list[str] = Field(default_factory=list)
    measurement_methods: list[str] = Field(default_factory=list)
    related_domains: list[PhysicsDomain] = Field(default_factory=list)
    complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE


class UnitContext(BaseModel):
    """Physics context for a unit."""

    unit: str
    context: str | None = None
    category: UnitCategory | None = None
    physics_domains: list[PhysicsDomain] = Field(default_factory=list)


class DomainConcepts(BaseModel):
    """All concepts for a physics domain."""

    domain: PhysicsDomain
    concepts: list[str] = Field(default_factory=list)


# ============================================================================
# SEMANTIC SEARCH MODELS
# ============================================================================


class EmbeddingDocument(BaseModel):
    """Document for physics concepts that can be embedded and searched."""

    concept_id: str
    concept_type: ConceptType
    domain_name: str
    title: str
    description: str
    content: str  # Rich content for embedding
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class SemanticResult(BaseModel):
    """Result from physics semantic search."""

    document: EmbeddingDocument
    similarity_score: float
    rank: int

    @property
    def concept_id(self) -> str:
        return self.document.concept_id

    @property
    def domain_name(self) -> str:
        return self.document.domain_name


class SemanticSearchRequest(BaseModel):
    """Request parameters for physics semantic search."""

    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0)
    concept_types: list[str] | None = None
    domains: list[str] | None = None


class SemanticSearchResult(BaseModel):
    """Response from physics semantic search."""

    query: str
    results: list[SemanticResult] = Field(default_factory=list)
    total_results: int
    max_results: int
    min_similarity: float
