"""
Data models for graph-led discovery pipeline.

These models define runtime structures for the LLM-based scoring phase.
Schema-derived types (PathPurpose, DiscoveryStatus) are imported from
the generated graph/models.py module.

Note: DirectoryEvidence, ScoredDirectory, ScoredBatch are transient runtime
structures for the scorer, NOT graph node types. They are converted to
graph updates via frontier.mark_paths_scored().

The Pydantic models (EvidenceSchema, DirectoryScoringResult, DirectoryScoringBatch)
are used for LLM structured output - LiteLLM parses responses directly into these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Import schema-derived enums from generated models
from imas_codex.graph.models import DiscoveryStatus, PathPurpose

__all__ = [
    "PathPurpose",
    "DiscoveryStatus",
    "DirectoryEvidence",
    "ScoredDirectory",
    "ScoredBatch",
    "EvidenceSchema",
    "DirectoryScoringResult",
    "DirectoryScoringBatch",
    "PathPurposeEnum",
]


# Pydantic enum for LLM output (must be a regular Enum for Pydantic/LiteLLM)
class PathPurposeEnum(str, Enum):
    """Path purpose classification for LLM output.

    Score semantics vary by category:
    - Code/data categories: score = ingestion priority
    - Container: score = exploration potential of children
    - Skip categories: always low score, skip subtree
    """

    # Code categories (ingest based on score)
    modeling_code = "modeling_code"
    diagnostic_code = "diagnostic_code"
    data_interface = "data_interface"
    workflow = "workflow"
    visualization = "visualization"
    # Data categories
    simulation_data = "simulation_data"
    diagnostic_data = "diagnostic_data"
    # Support categories
    documentation = "documentation"
    configuration = "configuration"
    test_suite = "test_suite"
    # Structural categories
    container = "container"
    # Skip categories (low score forced)
    archive = "archive"
    build_artifact = "build_artifact"
    system = "system"


# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================


class EvidenceSchema(BaseModel):
    """Evidence collected about a directory's contents.

    Used by LLM for structured output - maps directly to DirectoryEvidence.
    """

    code_indicators: list[str] = Field(
        default_factory=list,
        description="Programming file extensions found (e.g., ['py', 'f90', 'cpp'])",
    )
    data_indicators: list[str] = Field(
        default_factory=list,
        description="Data file extensions found (e.g., ['nc', 'h5', 'csv'])",
    )
    imas_indicators: list[str] = Field(
        default_factory=list,
        description="IMAS-related patterns (e.g., ['put_slice', 'ids_properties'])",
    )
    physics_indicators: list[str] = Field(
        default_factory=list,
        description="Physics domain patterns (e.g., ['equilibrium', 'transport'])",
    )
    quality_indicators: list[str] = Field(
        default_factory=list,
        description="Project quality signals (e.g., ['has_readme', 'has_git'])",
    )


class DirectoryScoringResult(BaseModel):
    """LLM scoring result for a single directory.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.

    Note: ge/le constraints removed from float fields because Anthropic
    via OpenRouter doesn't support minimum/maximum JSON schema properties.
    Score clamping is done in the parser instead.
    """

    path: str = Field(description="The directory path (echo from input)")

    path_purpose: PathPurposeEnum = Field(
        description="Classification: modeling_code, diagnostic_code, data_interface, "
        "workflow, visualization, simulation_data, diagnostic_data, documentation, "
        "configuration, test_suite, container, archive, build_artifact, system"
    )

    description: str = Field(
        description="Concise description of directory contents (1-2 sentences)"
    )

    evidence: EvidenceSchema = Field(
        default_factory=EvidenceSchema,
        description="Structured evidence for scoring",
    )

    score_code: float = Field(description="Code discovery value (0.0-1.0)")

    score_data: float = Field(description="Data discovery value (0.0-1.0)")

    score_imas: float = Field(description="IMAS relevance (0.0-1.0)")

    should_expand: bool = Field(description="Whether to explore child directories")

    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5)",
    )

    physics_domain: str | None = Field(
        default=None,
        description="Primary physics domain (equilibrium, transport, etc.)",
    )

    expansion_reason: str | None = Field(
        default=None, description="Why to expand (if should_expand=true)"
    )

    skip_reason: str | None = Field(
        default=None, description="Why not to expand (if should_expand=false)"
    )


class DirectoryScoringBatch(BaseModel):
    """Batch of directory scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    The LLM returns a list of DirectoryScoringResult objects.
    """

    results: list[DirectoryScoringResult] = Field(
        description="List of scoring results, one per input directory, in order"
    )


@dataclass
class DirectoryEvidence:
    """Evidence collected by LLM about a directory's contents.

    Used for grounded scoring - the LLM collects evidence, then
    a deterministic function computes the final score.
    """

    code_indicators: list[str] = field(default_factory=list)
    """Programming file extensions found (e.g., ['py', 'f90', 'cpp'])"""

    data_indicators: list[str] = field(default_factory=list)
    """Data file extensions found (e.g., ['nc', 'h5', 'csv'])"""

    imas_indicators: list[str] = field(default_factory=list)
    """IMAS-related patterns found (e.g., ['put_slice', 'ids_properties'])"""

    physics_indicators: list[str] = field(default_factory=list)
    """Physics domain patterns (e.g., ['equilibrium', 'transport'])"""

    quality_indicators: list[str] = field(default_factory=list)
    """Project quality signals (e.g., ['has_readme', 'has_makefile', 'has_git'])"""

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary for graph storage."""
        return {
            "code_indicators": self.code_indicators,
            "data_indicators": self.data_indicators,
            "imas_indicators": self.imas_indicators,
            "physics_indicators": self.physics_indicators,
            "quality_indicators": self.quality_indicators,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirectoryEvidence:
        """Create from dictionary."""
        return cls(
            code_indicators=data.get("code_indicators", []),
            data_indicators=data.get("data_indicators", []),
            imas_indicators=data.get("imas_indicators", []),
            physics_indicators=data.get("physics_indicators", []),
            quality_indicators=data.get("quality_indicators", []),
        )


def parse_path_purpose(value: str) -> PathPurpose:
    """Parse string to PathPurpose, defaulting to container."""
    try:
        return PathPurpose(value.lower())
    except ValueError:
        return PathPurpose.container


@dataclass
class ScoredDirectory:
    """Result of LLM scoring for a single directory.

    Contains all scores, evidence, and expansion decision.
    Enriches the graph with metadata beyond just scores.
    """

    path: str
    """Absolute path to the directory."""

    path_purpose: PathPurpose
    """Classified purpose of the directory."""

    description: str
    """One-sentence description of the directory's contents."""

    evidence: DirectoryEvidence
    """Collected evidence for grounded scoring."""

    score_code: float
    """Code interest dimension (0.0-1.0)."""

    score_data: float
    """Data interest dimension (0.0-1.0)."""

    score_imas: float
    """IMAS relevance dimension (0.0-1.0)."""

    score: float = 0.0
    """Combined score computed by grounded scoring function."""

    should_expand: bool = False
    """Whether to explore children of this directory."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords for this directory (max 5)."""

    physics_domain: str | None = None
    """Primary physics domain if applicable (equilibrium, transport, etc)."""

    expansion_reason: str | None = None
    """Why this directory should be expanded."""

    skip_reason: str | None = None
    """Why this directory should not be expanded."""

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence."""
        import json

        return {
            "path": self.path,
            "path_purpose": self.path_purpose.value,
            "description": self.description,
            "evidence": json.dumps(self.evidence.to_dict()),
            "score_code": self.score_code,
            "score_data": self.score_data,
            "score_imas": self.score_imas,
            "score": self.score,
            "should_expand": self.should_expand,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain,
            "expansion_reason": self.expansion_reason,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ScoredBatch:
    """Result of scoring a batch of directories."""

    scored_dirs: list[ScoredDirectory]
    """List of scored directories."""

    total_cost: float
    """Estimated cost in USD for this batch."""

    model: str
    """Model used for scoring."""

    tokens_used: int
    """Total tokens (input + output) used."""

    @property
    def expanded_count(self) -> int:
        """Number of directories marked for expansion."""
        return sum(1 for d in self.scored_dirs if d.should_expand)
