"""Pydantic models for physics search results."""

from typing import List, Optional
from pydantic import BaseModel, Field


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
