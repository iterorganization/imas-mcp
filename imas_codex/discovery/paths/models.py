"""
Data models for graph-led discovery pipeline.

These models define runtime structures for the LLM-based triage and scoring phases.
Schema-derived types (ResourcePurpose) are imported from
the generated graph/models.py module.

Note: DirectoryEvidence, TriagedDirectory, TriagedBatch are transient runtime
structures for the triager, NOT graph node types. They are converted to
graph updates via frontier.mark_paths_triaged().

Pydantic models for LLM structured output:
- TriageResult/TriageBatch: First-pass triage (directory structure only)
- ScoreResult/ScoreBatch: Second-pass scoring (with enrichment evidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# Import schema-derived enums from generated models
from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.graph.models import ResourcePurpose, TerminalReason

__all__ = [
    "PhysicsDomain",
    "ResourcePurpose",
    "TerminalReason",
    "DirectoryEvidence",
    "TriagedDirectory",
    "TriagedBatch",
    "TriageResult",
    "TriageBatch",
    "ScoreResult",
    "ScoreBatch",
]


# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================
#
# Note: ResourcePurpose is imported from the generated LinkML models (graph/models.py).
# The generated enum is used directly for LLM structured output - no duplicate needed.
# LiteLLM/Pydantic correctly handle str-based enums from LinkML.
#
# SCHEMA SIMPLIFICATION (Feb 2026):
# Removed EvidenceSchema nested object to reduce JSON schema complexity for Gemini.
# Evidence is now derived from input data (has_readme, file_type_counts, etc.)
# and pattern matches from the enrichment worker. This eliminates 6 list fields
# and the nested object definition from the schema sent to the LLM.


class TriageResult(BaseModel):
    """LLM triage result for a single directory (first pass).

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.

    Triage uses directory structure only (file names, counts, quality indicators).
    High-triage-score paths proceed to enrichment and independent scoring.
    """

    path: str = Field(description="The directory path (echo from input)")

    path_purpose: ResourcePurpose = Field(
        description="Classification: modeling_code, analysis_code, operations_code, "
        "data_access, workflow, visualization, experimental_data, modeling_data, "
        "documentation, configuration, test_suite, container, archive, build_artifact, system"
    )

    description: str = Field(
        description="What this directory contains and its purpose (1-2 sentences). "
        "Describe the code, tools, or data — NOT the scoring rationale."
    )

    # Per-purpose scores (0.0-1.0 each)
    score_modeling_code: float = Field(
        default=0.0,
        description="Forward modeling/simulation code value (0.0-1.0)",
    )
    score_analysis_code: float = Field(
        default=0.0,
        description="Experimental analysis code value (0.0-1.0)",
    )
    score_operations_code: float = Field(
        default=0.0,
        description="Real-time operations code value (0.0-1.0)",
    )
    score_modeling_data: float = Field(
        default=0.0,
        description="Modeling outputs value (0.0-1.0)",
    )
    score_experimental_data: float = Field(
        default=0.0,
        description="Experimental shot data value (0.0-1.0)",
    )
    score_data_access: float = Field(
        default=0.0,
        description="Data access tools value (0.0-1.0)",
    )
    score_workflow: float = Field(
        default=0.0,
        description="Workflow/orchestration value (0.0-1.0)",
    )
    score_visualization: float = Field(
        default=0.0,
        description="Visualization tools value (0.0-1.0)",
    )
    score_documentation: float = Field(
        default=0.0,
        description="Documentation value (0.0-1.0)",
    )
    score_imas: float = Field(
        default=0.0,
        description="IMAS relevance (0.0-1.0)",
    )
    score_convention: float = Field(
        default=0.0,
        description="Convention handling value — COCOS, sign/coordinate conventions, unit conversion (0.0-1.0)",
    )

    should_expand: bool = Field(description="Whether to explore child directories")

    should_enrich: bool = Field(
        description="Whether to run deep analysis (du, tokei, patterns). "
        "False for huge dirs like /work, /home, or paths with no code files."
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5)",
    )

    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (use 'general' if no clear domain)",
    )

    expansion_reason: str = Field(
        default="", description="Why to expand (if should_expand=true)"
    )

    skip_reason: str = Field(
        default="", description="Why not to expand (if should_expand=false)"
    )

    enrich_skip_reason: str = Field(
        default="",
        description="Why not to enrich (if should_enrich=false). "
        "E.g., 'too large', 'no code files', 'container only'",
    )


class TriageBatch(BaseModel):
    """Batch of directory triage results from LLM (first pass)."""

    results: list[TriageResult] = Field(
        description="List of triage results, one per input directory, in order"
    )


# ============================================================================
# LLM Scoring Models (Second Pass)
# ============================================================================


class ScoreResult(BaseModel):
    """LLM scoring result for a single directory (second pass).

    Independent evaluation using enrichment evidence (pattern matches,
    language breakdown, LOC, multiformat detection) plus triage context
    (description, keywords, purpose) but WITHOUT numeric triage scores.
    This prevents anchoring bias and allows the full score range.
    """

    path: str = Field(description="The directory path (echo from input)")

    # Classification — evaluated with enrichment evidence
    path_purpose: ResourcePurpose = Field(
        description="Classification based on enrichment evidence"
    )

    # Description — what the directory IS (not why it scored this way)
    description: str = Field(
        description="What this directory contains and its purpose (1-2 sentences). "
        "Describe the code, tools, or data — NOT the scoring rationale. "
        "Example: 'Custom LIUQE equilibrium reconstruction interface with 2,500 lines of Python and MDSplus data readers'"
    )

    # Per-dimension scores (0.0-1.0, required)
    score_modeling_code: float = Field(
        default=0.0, description="Modeling code score (0.0-1.0)"
    )
    score_analysis_code: float = Field(
        default=0.0, description="Analysis code score (0.0-1.0)"
    )
    score_operations_code: float = Field(
        default=0.0, description="Operations code score (0.0-1.0)"
    )
    score_modeling_data: float = Field(
        default=0.0, description="Modeling data score (0.0-1.0)"
    )
    score_experimental_data: float = Field(
        default=0.0, description="Experimental data score (0.0-1.0)"
    )
    score_data_access: float = Field(
        default=0.0, description="Data access score (0.0-1.0)"
    )
    score_workflow: float = Field(default=0.0, description="Workflow score (0.0-1.0)")
    score_visualization: float = Field(
        default=0.0, description="Visualization score (0.0-1.0)"
    )
    score_documentation: float = Field(
        default=0.0, description="Documentation score (0.0-1.0)"
    )
    score_imas: float = Field(default=0.0, description="IMAS relevance score (0.0-1.0)")
    score_convention: float = Field(
        default=0.0, description="Convention handling score (0.0-1.0)"
    )

    # Combined score (max of dimension scores)
    new_score: float = Field(
        description="Combined score = max of all dimension scores (0.0-1.0)"
    )

    # Expansion decision — can be changed based on enrichment evidence
    should_expand: bool = Field(
        default=True,
        description="Whether to continue exploring child directories. "
        "Set false for data directories, archives, or dirs with no code.",
    )

    # Updated keywords based on enrichment evidence
    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords including pattern evidence (max 8)",
    )

    # Physics domain — can be updated based on pattern evidence
    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (updated from evidence if applicable)",
    )

    # Evidence for score adjustments (persisted to graph)
    primary_evidence: list[str] = Field(
        default_factory=list,
        description="Key pattern categories that informed the adjustment (e.g., ['mdsplus', 'imas_write'])",
    )
    evidence_summary: str = Field(
        default="",
        description="Brief summary of pattern evidence (e.g., '15 MDSplus, 3 IMAS writes')",
    )

    scoring_reason: str = Field(
        default="",
        description="Why the scores were assigned — the scoring rationale. "
        "Example: 'High data_access due to 15 MDSplus pattern matches; moderate analysis from profile fitting code'",
    )


class ScoreBatch(BaseModel):
    """Batch of scoring results from LLM (second pass)."""

    results: list[ScoreResult] = Field(
        description="List of scoring results, one per input directory"
    )


@dataclass
class DirectoryEvidence:
    """Evidence collected by LLM about a directory's contents.

    Used for grounded scoring - the LLM collects evidence, then
    a deterministic function computes the final score.

    Generic semantic signal container - reusable across FacilityPath,
    CodeFile, WikiPage, etc. Quantitative metrics belong on domain nodes.
    """

    code_indicators: list[str] = field(default_factory=list)
    """Programming file extensions found (e.g., ['py', 'f90', 'cpp'])"""

    data_indicators: list[str] = field(default_factory=list)
    """Data file extensions found (e.g., ['nc', 'h5', 'csv'])"""

    doc_indicators: list[str] = field(default_factory=list)
    """Documentation signals (e.g., ['README', 'docs/', 'pdf', 'tutorial'])"""

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
            "doc_indicators": self.doc_indicators,
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
            doc_indicators=data.get("doc_indicators", []),
            imas_indicators=data.get("imas_indicators", []),
            physics_indicators=data.get("physics_indicators", []),
            quality_indicators=data.get("quality_indicators", []),
        )


def parse_path_purpose(value: str) -> ResourcePurpose:
    """Parse string to ResourcePurpose, defaulting to container."""
    try:
        return ResourcePurpose(value.lower())
    except ValueError:
        return ResourcePurpose.container


@dataclass
class TriagedDirectory:
    """Result of LLM triage for a single directory (first pass).

    Contains per-purpose scores, evidence, and expansion decision.
    Enriches the graph with metadata beyond just scores.

    Per-purpose scores aligned with DiscoveryRootCategory taxonomy.
    """

    path: str
    """Absolute path to the directory."""

    path_purpose: ResourcePurpose
    """Classified purpose of the directory."""

    description: str
    """One-sentence description of the directory's contents."""

    evidence: DirectoryEvidence
    """Collected evidence for grounded scoring."""

    # Per-purpose scores (0.0-1.0 each)
    score_modeling_code: float = 0.0
    """Forward modeling/simulation code dimension."""

    score_analysis_code: float = 0.0
    """Experimental analysis code dimension."""

    score_operations_code: float = 0.0
    """Real-time operations code dimension."""

    score_modeling_data: float = 0.0
    """Modeling outputs dimension."""

    score_experimental_data: float = 0.0
    """Experimental shot data dimension."""

    score_data_access: float = 0.0
    """Data access tools dimension."""

    score_workflow: float = 0.0
    """Workflow/orchestration dimension."""

    score_visualization: float = 0.0
    """Visualization tools dimension."""

    score_documentation: float = 0.0
    """Documentation dimension."""

    score_imas: float = 0.0
    """IMAS relevance dimension (cross-cutting)."""

    score_convention: float = 0.0
    """Convention handling dimension (COCOS, sign/coordinate conventions, unit conversion)."""

    score: float = 0.0
    """Combined score computed by grounded scoring function."""

    should_expand: bool = False
    """Whether to explore children of this directory."""

    should_enrich: bool = True
    """Whether to run deep analysis (du, tokei, patterns) on this path."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords for this directory (max 5)."""

    physics_domain: PhysicsDomain | None = None
    """Primary physics domain from PhysicsDomain enum."""

    expansion_reason: str | None = None
    """Why this directory should be expanded."""

    skip_reason: str | None = None
    """Why this directory should not be expanded."""

    enrich_skip_reason: str | None = None
    """Why enrichment was skipped (e.g., 'too large', 'no code files')."""

    score_cost: float = 0.0
    """LLM cost in USD for scoring this path (batch cost / batch size)."""

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence.

        Note: terminal_reason is NOT set here - it's only for non-derivable
        cases (access_denied, empty, parent_terminal). For LLM-scored paths,
        the reason is derivable from has_git, path_purpose, score.
        """
        import json

        return {
            "path": self.path,
            "path_purpose": self.path_purpose.value,
            "description": self.description,
            "evidence": json.dumps(self.evidence.to_dict()),
            "score_modeling_code": self.score_modeling_code,
            "score_analysis_code": self.score_analysis_code,
            "score_operations_code": self.score_operations_code,
            "score_modeling_data": self.score_modeling_data,
            "score_experimental_data": self.score_experimental_data,
            "score_data_access": self.score_data_access,
            "score_workflow": self.score_workflow,
            "score_visualization": self.score_visualization,
            "score_documentation": self.score_documentation,
            "score_imas": self.score_imas,
            "score_convention": self.score_convention,
            "score": self.score,
            "should_expand": self.should_expand,
            "should_enrich": self.should_enrich,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "expansion_reason": self.expansion_reason,
            "skip_reason": self.skip_reason,
            "enrich_skip_reason": self.enrich_skip_reason,
            "score_cost": self.score_cost,
        }


@dataclass
class TriagedBatch:
    """Result of triaging a batch of directories."""

    triaged_dirs: list[TriagedDirectory]
    """List of triaged directories."""

    total_cost: float
    """Estimated cost in USD for this batch."""

    model: str
    """Model used for triage."""

    tokens_used: int
    """Total tokens (input + output) used."""

    @property
    def expanded_count(self) -> int:
        """Number of directories marked for expansion."""
        return sum(1 for d in self.triaged_dirs if d.should_expand)
