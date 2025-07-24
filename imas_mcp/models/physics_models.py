"""Pydantic models for physics search results and semantic search."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from imas_mcp.core.data_model import PhysicsDomain


class PhysicsMatch(BaseModel):
    """A physics concept match from search."""

    concept: str
    quantity_name: str
    symbol: str
    units: str
    description: str
    imas_paths: List[str] = Field(default_factory=list)
    domain: str
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


class PhysicsSearchResult(BaseModel):
    """Complete physics search result."""

    query: str
    physics_matches: List[PhysicsMatch] = Field(default_factory=list)
    concept_suggestions: List[ConceptSuggestion] = Field(default_factory=list)
    unit_suggestions: List[UnitSuggestion] = Field(default_factory=list)
    symbol_suggestions: List[SymbolSuggestion] = Field(default_factory=list)
    imas_path_suggestions: List[str] = Field(default_factory=list)


class ConceptExplanation(BaseModel):
    """Explanation of a physics concept with domain context."""

    concept: str
    domain: PhysicsDomain
    description: str
    phenomena: List[str] = Field(default_factory=list)
    typical_units: List[str] = Field(default_factory=list)
    measurement_methods: List[str] = Field(default_factory=list)
    related_domains: List[PhysicsDomain] = Field(default_factory=list)
    complexity_level: str


class UnitPhysicsContext(BaseModel):
    """Physics context for a unit."""

    unit: str
    context: Optional[str] = None
    category: Optional[str] = None
    physics_domains: List[PhysicsDomain] = Field(default_factory=list)


class DomainConceptsResult(BaseModel):
    """Result containing all concepts for a physics domain."""

    domain: PhysicsDomain
    concepts: List[str] = Field(default_factory=list)


# Physics Semantic Search Models


class PhysicsEmbeddingDocument(BaseModel):
    """Document model for physics concepts that can be embedded and searched."""

    concept_id: str
    concept_type: str  # "domain", "phenomenon", "unit", "measurement_method"
    domain_name: str
    title: str
    description: str
    content: str  # Rich content for embedding
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PhysicsSemanticResult(BaseModel):
    """Result from physics semantic search."""

    document: PhysicsEmbeddingDocument
    similarity_score: float
    rank: int

    @property
    def concept_id(self) -> str:
        return self.document.concept_id

    @property
    def domain_name(self) -> str:
        return self.document.domain_name


class PhysicsSearchRequest(BaseModel):
    """Request parameters for physics semantic search."""

    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0)
    concept_types: Optional[List[str]] = None
    domains: Optional[List[str]] = None


class PhysicsSearchResponse(BaseModel):
    """Response from physics semantic search."""

    query: str
    results: List[PhysicsSemanticResult] = Field(default_factory=list)
    total_results: int
    max_results: int
    min_similarity: float
