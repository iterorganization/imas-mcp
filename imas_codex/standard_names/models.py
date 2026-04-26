"""Pydantic models for standard name pipeline LLM responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StandardNameCandidate(BaseModel):
    """A single standard name candidate from LLM composition.

    Name-only: compose produces naming and grammar fields only.
    Documentation (description, tags, links, etc.) is added by ``sn enrich``.
    """

    source_id: str = Field(description="Source entity ID (DD path or signal ID)")
    standard_name: str = Field(description="Composed standard name in snake_case")
    kind: Literal["scalar", "vector", "metadata"] = Field(
        default="scalar", description="Entry kind"
    )
    dd_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    grammar_fields: dict[str, str] = Field(
        default_factory=dict, description="Grammar fields used"
    )
    confidence: float = Field(ge=0, le=1, description="Naming confidence")
    reason: str = Field(description="Brief justification")


class StandardNameVocabGap(BaseModel):
    """A path where naming requires vocabulary expansion."""

    source_id: str = Field(description="DD path that needs naming")
    segment: str = Field(
        description="Grammar segment missing a token (e.g., 'subject', 'position')"
    )
    needed_token: str = Field(
        description="Proposed token value for the grammar segment"
    )
    reason: str = Field(description="Why this token is needed for naming this path")


class StandardNameAttachment(BaseModel):
    """A DD path that should attach to an existing standard name without regeneration."""

    source_id: str = Field(description="DD path to attach")
    standard_name: str = Field(description="Existing standard name to attach to")
    reason: str = Field(description="Why this path maps to this existing name")


class StandardNameComposeBatch(BaseModel):
    """LLM response for a batch of standard name compositions."""

    candidates: list[StandardNameCandidate]
    attachments: list[StandardNameAttachment] = Field(
        default_factory=list,
        description=(
            "DD paths that map to existing standard names — attach without regeneration. "
            "Use when a path measures the exact same quantity as an existing name."
        ),
    )
    skipped: list[str] = Field(
        default_factory=list, description="Source IDs skipped (not physics quantities)"
    )
    vocab_gaps: list[StandardNameVocabGap] = Field(
        default_factory=list,
        description="Paths where naming requires vocabulary expansion in imas-standard-names",
    )


# =============================================================================
# Publish models — YAML catalog export (Feature 08)
# =============================================================================


class StandardNameProvenance(BaseModel):
    """Provenance metadata for a standard name entry."""

    source: str = Field(description="Source type: dd or signal")
    source_id: str = Field(description="Source entity ID")
    ids_name: str | None = Field(default=None, description="IDS name (for DD source)")
    confidence: float = Field(ge=0, le=1, description="Generation confidence")
    generated_by: str = Field(
        default="imas-codex", description="Tool that generated this"
    )


class StandardNamePublishEntry(BaseModel):
    """A single standard name entry ready for YAML catalog export."""

    name: str = Field(description="The standard name")
    kind: str = Field(
        default="scalar", description="Name kind: scalar, vector, or metadata"
    )
    unit: str | None = Field(default=None, description="SI unit string")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    status: str = Field(default="drafted", description="Entry status")
    physics_domain: str | None = Field(
        default=None, description="Physics domain classification"
    )
    description: str = Field(default="", description="Human-readable description")
    # Rich fields
    documentation: str | None = Field(
        default=None, description="Rich documentation with LaTeX"
    )
    links: list[str] = Field(default_factory=list, description="Related standard names")
    dd_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints"
    )
    validity_domain: str | None = Field(
        default=None, description="Physical region where valid"
    )
    cocos_transformation_type: str | None = Field(
        default=None,
        description="COCOS transformation type (e.g., psi_like, ip_like). Null for non-COCOS quantities.",
    )
    cocos: int | None = Field(
        default=None,
        description="COCOS convention index (e.g. 11, 17). Null for non-COCOS quantities.",
    )
    provenance: StandardNameProvenance = Field(description="Generation provenance")


class StandardNamePublishBatch(BaseModel):
    """A batch of entries to publish as a PR."""

    group_key: str = Field(description="Batch group key (IDS name or domain)")
    entries: list[StandardNamePublishEntry]
    confidence_tier: str = Field(description="high, medium, or low")


# =============================================================================
# Cross-model review models
# =============================================================================


class StandardNameReviewVerdict(StrEnum):
    """Review decision for a standard name candidate."""

    accept = "accept"
    reject = "reject"
    revise = "revise"


class StandardNameReviewItem(BaseModel):
    """Review of a single standard name candidate."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    verdict: StandardNameReviewVerdict = Field(description="Accept, reject, or revise")
    confidence: float = Field(ge=0, le=1, description="Review confidence")
    reason: str = Field(description="Justification for the verdict")
    revised_name: str | None = Field(
        default=None, description="Suggested revision if verdict is revise"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameReviewBatch(BaseModel):
    """LLM response for reviewing a batch of standard name candidates."""

    reviews: list[StandardNameReviewItem]


class StandardNameQualityComments(BaseModel):
    """Per-dimension comments for the full 6-dimensional review rubric."""

    grammar: str | None = Field(default=None, description="Comment on grammar score")
    semantic: str | None = Field(default=None, description="Comment on semantic score")
    documentation: str | None = Field(
        default=None, description="Comment on documentation score"
    )
    convention: str | None = Field(
        default=None, description="Comment on convention score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )
    compliance: str | None = Field(
        default=None, description="Comment on compliance score"
    )


class StandardNameQualityCommentsNameOnly(BaseModel):
    """Per-dimension comments for the 4-dimensional name-only review rubric."""

    grammar: str | None = Field(default=None, description="Comment on grammar score")
    semantic: str | None = Field(default=None, description="Comment on semantic score")
    convention: str | None = Field(
        default=None, description="Comment on convention score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )


class StandardNameQualityCommentsDocs(BaseModel):
    """Per-dimension comments for the 4-dimensional docs review rubric.

    Note: uses independent dimension names (description_quality etc.),
    NOT a subset of the full 6-dim names.
    """

    description_quality: str | None = Field(
        default=None, description="Comment on description quality score"
    )
    documentation_quality: str | None = Field(
        default=None, description="Comment on documentation quality score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )
    physics_accuracy: str | None = Field(
        default=None, description="Comment on physics accuracy score"
    )


# =============================================================================
# Unified quality review models (used by both mint and benchmark)
# =============================================================================


class StandardNameQualityScore(BaseModel):
    """6-dimensional quality score for a standard name entry."""

    grammar: int = Field(ge=0, le=20, description="Grammar correctness (0-20)")
    semantic: int = Field(ge=0, le=20, description="Semantic accuracy (0-20)")
    documentation: int = Field(ge=0, le=20, description="Documentation quality (0-20)")
    convention: int = Field(ge=0, le=20, description="Naming conventions (0-20)")
    completeness: int = Field(ge=0, le=20, description="Entry completeness (0-20)")
    compliance: int = Field(
        ge=0, le=20, description="Prompt instruction compliance (0-20)"
    )

    @property
    def total(self) -> int:
        return (
            self.grammar
            + self.semantic
            + self.documentation
            + self.convention
            + self.completeness
            + self.compliance
        )

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 6 dimensions / 120."""
        return self.total / 120.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReview(BaseModel):
    """Review of a single standard name with quality scoring."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScore = Field(description="6-dimensional quality scores")
    comments: StandardNameQualityComments | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    verdict: StandardNameReviewVerdict = Field(description="Accept, reject, or revise")
    reasoning: str = Field(description="Specific justification per dimension")
    revised_name: str | None = Field(
        default=None, description="Suggested revision if verdict is revise"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    suggested_name: str | None = Field(
        default=None,
        description=(
            "Reviewer-recommended improved name. Required for verdict=revise "
            "or verdict=reject; null for verdict=accept."
        ),
    )
    suggestion_justification: str | None = Field(
        default=None,
        description=(
            "1–3 sentence justification for suggested_name. Null when "
            "suggested_name is null."
        ),
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewBatch(BaseModel):
    """LLM response for quality-scored review of a batch."""

    reviews: list[StandardNameQualityReview]


# =============================================================================
# Name-only review — 4-dimensional rubric for --name-only cycles
# =============================================================================


class StandardNameQualityScoreNameOnly(BaseModel):
    """4-dimensional quality score for name-only review mode.

    Scores the name itself (grammar, semantic, convention, completeness)
    without penalising missing documentation or compliance, which are
    intentionally deferred in name-only generation cycles. Normalised
    over 80 rather than 120.
    """

    grammar: int = Field(ge=0, le=20, description="Grammar correctness (0-20)")
    semantic: int = Field(ge=0, le=20, description="Semantic accuracy (0-20)")
    convention: int = Field(ge=0, le=20, description="Naming conventions (0-20)")
    completeness: int = Field(ge=0, le=20, description="Entry completeness (0-20)")

    @property
    def total(self) -> int:
        return self.grammar + self.semantic + self.convention + self.completeness

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 4 dimensions / 80."""
        return self.total / 80.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReviewNameOnly(BaseModel):
    """Review of a single standard name using the 4-dimensional rubric."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScoreNameOnly = Field(
        description="4-dimensional quality scores"
    )
    comments: StandardNameQualityCommentsNameOnly | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    verdict: StandardNameReviewVerdict = Field(description="Accept, reject, or revise")
    reasoning: str = Field(description="Specific justification per dimension")
    revised_name: str | None = Field(
        default=None, description="Suggested revision if verdict is revise"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    suggested_name: str | None = Field(
        default=None,
        description=(
            "Reviewer-recommended improved name. Required for verdict=revise "
            "or verdict=reject; null for verdict=accept."
        ),
    )
    suggestion_justification: str | None = Field(
        default=None,
        description=(
            "1–3 sentence justification for suggested_name, grounded in ISN "
            "grammar and the per-item DD context. Null when suggested_name is null."
        ),
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewNameOnlyBatch(BaseModel):
    """LLM response for name-only quality-scored review of a batch."""

    reviews: list[StandardNameQualityReviewNameOnly]


# =============================================================================
# Docs review — 4-dimensional rubric for --target docs cycles
# =============================================================================


class StandardNameQualityScoreDocs(BaseModel):
    """4-dimensional quality score for docs review mode.

    Scores the generated documentation (description, documentation body,
    completeness of doc fields, and physics accuracy of prose) without
    re-scoring the name itself — the name was already reviewed in a prior
    ``--target names`` cycle. Normalised over 80 rather than 120.
    """

    description_quality: int = Field(
        ge=0, le=20, description="Clarity and precision of short description (0-20)"
    )
    documentation_quality: int = Field(
        ge=0,
        le=20,
        description="Documentation body: equations, variables, sign conventions (0-20)",
    )
    completeness: int = Field(
        ge=0,
        le=20,
        description="Required doc fields filled (links, aliases, cross-refs) (0-20)",
    )
    physics_accuracy: int = Field(
        ge=0,
        le=20,
        description="Physics correctness of documentation prose and equations (0-20)",
    )

    @property
    def total(self) -> int:
        return (
            self.description_quality
            + self.documentation_quality
            + self.completeness
            + self.physics_accuracy
        )

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 4 dimensions / 80."""
        return self.total / 80.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReviewDocs(BaseModel):
    """Review of a single standard name's docs using the 4-dimensional rubric."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScoreDocs = Field(
        description="4-dimensional docs quality scores"
    )
    comments: StandardNameQualityCommentsDocs | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    verdict: StandardNameReviewVerdict = Field(description="Accept, reject, or revise")
    reasoning: str = Field(description="Specific justification per dimension")
    revised_description: str | None = Field(
        default=None, description="Suggested revised description if verdict is revise"
    )
    revised_documentation: str | None = Field(
        default=None, description="Suggested revised documentation body"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewDocsBatch(BaseModel):
    """LLM response for docs quality-scored review of a batch."""

    reviews: list[StandardNameQualityReviewDocs]


# =============================================================================
# Enrichment models — documentation iteration (Phase 3D)
# =============================================================================


class StandardNameEnrichItem(BaseModel):
    """Enrichment result for a single standard name."""

    standard_name: str = Field(
        description="The standard name (must match input exactly)"
    )
    description: str = Field(description="One sentence definition, <120 chars")
    documentation: str = Field(
        description="Rich docs with LaTeX, links, typical values"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    links: list[str] = Field(
        default_factory=list, description="Related standard names (name:xxx format)"
    )
    validity_domain: str | None = Field(
        default=None, description="Physical region where quantity is meaningful"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints on the quantity"
    )


class StandardNameEnrichBatch(BaseModel):
    """LLM response for enriching a batch of standard names."""

    items: list[StandardNameEnrichItem]
