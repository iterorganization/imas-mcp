"""Clean, focused Pydantic models for physics search and semantic analysis."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.models.constants import ConceptType, ComplexityLevel, UnitCategory


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
    imas_paths: List[str] = Field(default_factory=list)
    domain: PhysicsDomain
    relevance_score: float


class ConceptSuggestion(BaseModel):
    """A concept suggestion."""

    concept: str
    description: Optional[str] = None


class UnitSuggestion(BaseModel):
    """A unit suggestion."""

    unit: str
    description: str
    example_quantities: List[str] = Field(default_factory=list)


class SymbolSuggestion(BaseModel):
    """A symbol suggestion."""

    symbol: str
    concept: Optional[str] = None
    description: Optional[str] = None


# ============================================================================
# PHYSICS SEARCH RESULTS
# ============================================================================


class PhysicsSearchResult(BaseModel):
    """Complete physics search result."""

    query: str
    physics_matches: List[PhysicsMatch] = Field(default_factory=list)
    concept_suggestions: List[ConceptSuggestion] = Field(default_factory=list)
    unit_suggestions: List[UnitSuggestion] = Field(default_factory=list)
    symbol_suggestions: List[SymbolSuggestion] = Field(default_factory=list)
    imas_path_suggestions: List[str] = Field(default_factory=list)


# ============================================================================
# CONCEPT & DOMAIN MODELS
# ============================================================================


class ConceptExplanation(BaseModel):
    """Explanation of a physics concept with domain context."""

    concept: str
    domain: PhysicsDomain
    description: str
    phenomena: List[str] = Field(default_factory=list)
    typical_units: List[str] = Field(default_factory=list)
    measurement_methods: List[str] = Field(default_factory=list)
    related_domains: List[PhysicsDomain] = Field(default_factory=list)
    complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE


class UnitContext(BaseModel):
    """Physics context for a unit."""

    unit: str
    context: Optional[str] = None
    category: Optional[UnitCategory] = None
    physics_domains: List[PhysicsDomain] = Field(default_factory=list)


class DomainConcepts(BaseModel):
    """All concepts for a physics domain."""

    domain: PhysicsDomain
    concepts: List[str] = Field(default_factory=list)


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
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    concept_types: Optional[List[str]] = None
    domains: Optional[List[str]] = None


class SemanticSearchResult(BaseModel):
    """Response from physics semantic search."""

    query: str
    results: List[SemanticResult] = Field(default_factory=list)
    total_results: int
    max_results: int
    min_similarity: float
