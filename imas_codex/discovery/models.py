"""
Data models for graph-led discovery pipeline.

These models define runtime structures for the LLM-based scoring phase.
Schema-derived types (PathPurpose, DiscoveryStatus) are imported from
the generated graph/models.py module.

Note: DirectoryEvidence, ScoredDirectory, ScoredBatch are transient runtime
structures for the scorer, NOT graph node types. They are converted to
graph updates via frontier.mark_paths_scored().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Import schema-derived enums from generated models
from imas_codex.graph.models import DiscoveryStatus, PathPurpose

__all__ = [
    "PathPurpose",
    "DiscoveryStatus",
    "DirectoryEvidence",
    "ScoredDirectory",
    "ScoredBatch",
]


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
    """Parse string to PathPurpose, defaulting to unknown."""
    try:
        return PathPurpose(value.lower())
    except ValueError:
        return PathPurpose.unknown


@dataclass
class ScoredDirectory:
    """Result of LLM scoring for a single directory.

    Contains all scores, evidence, and expansion decision.
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
