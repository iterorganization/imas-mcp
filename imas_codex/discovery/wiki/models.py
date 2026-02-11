"""
Data models for wiki discovery pipeline.

These models define runtime structures for the LLM-based wiki scoring phase.
Mirrors the structure of discovery/paths/models.py for consistency.

The Pydantic models (WikiScoreResult, WikiScoreBatch) are used for LLM
structured output - LiteLLM parses responses directly into these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.core.physics_domain import PhysicsDomain

__all__ = [
    "WikiPagePurpose",
    "WikiScoreResult",
    "WikiScoreBatch",
    "ScoredWikiPage",
    "ScoredWikiBatch",
    # Artifact scoring
    "ArtifactScoreResult",
    "ArtifactScoreBatch",
    "grounded_artifact_score",
    "ScoredArtifact",
    # Image captioning
    "ImageCaptionResult",
    "ImageCaptionBatch",
]


# ============================================================================
# Wiki Page Purpose Enum
# ============================================================================


class WikiPagePurpose(str, Enum):
    """Classification of wiki page content.

    Content-based categories that determine ingestion priority
    and how the page should be processed.
    """

    # High-value technical content
    data_source = "data_source"
    diagnostic = "diagnostic"
    code = "code"
    calibration = "calibration"
    data_access = "data_access"
    physics_analysis = "physics_analysis"

    # Medium-value content
    experimental_procedure = "experimental_procedure"
    tutorial = "tutorial"
    reference = "reference"

    # Low-value content
    administrative = "administrative"
    personal = "personal"
    other = "other"

    @property
    def description(self) -> str:
        """Get human-readable description for this purpose."""
        return _WIKI_PURPOSE_DESCRIPTIONS.get(
            self, self.value.replace("_", " ").title()
        )


# Human-readable descriptions for each WikiPagePurpose
_WIKI_PURPOSE_DESCRIPTIONS: dict[WikiPagePurpose, str] = {
    WikiPagePurpose.data_source: "Signal tables, node lists, shot databases - direct IMAS mapping targets",
    WikiPagePurpose.diagnostic: "Diagnostic documentation (Thomson, CXRS, ECE, bolometry, etc.)",
    WikiPagePurpose.code: "Software/code documentation (LIUQE, ASTRA, CHEASE, etc.)",
    WikiPagePurpose.calibration: "Calibration procedures, conversion factors, sensor specifications",
    WikiPagePurpose.data_access: "How to access data: MDSplus paths, TDI expressions, APIs",
    WikiPagePurpose.physics_analysis: "Physics methodology, analysis techniques, interpretation guides",
    WikiPagePurpose.experimental_procedure: "Experimental procedures, operational guides, safety protocols",
    WikiPagePurpose.tutorial: "Tutorials, how-tos, getting started guides",
    WikiPagePurpose.reference: "General reference material, tables, constants",
    WikiPagePurpose.administrative: "Meeting notes, schedules, project management",
    WikiPagePurpose.personal: "User pages, personal notes, sandboxes",
    WikiPagePurpose.other: "Uncategorized content",
}


# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================


class WikiScoreResult(BaseModel):
    """LLM scoring result for a single wiki page.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.

    Scoring dimensions focus on content value, not graph metrics:
    - Data documentation: Signal tables, node references, shot data
    - Physics content: Physics explanations, methodology, theory
    - Code documentation: Software docs, API references, usage guides
    - Data access: MDSplus paths, TDI expressions, access methods
    - Calibration: Calibration data, conversion factors, sensor specs
    - IMAS relevance: IMAS integration, IDS references, mapping hints
    """

    id: str = Field(description="The page ID (echo from input)")

    page_purpose: WikiPagePurpose = Field(
        description="Classification: data_source, diagnostic, code, calibration, "
        "data_access, physics_analysis, experimental_procedure, tutorial, "
        "reference, administrative, personal, other"
    )

    description: str = Field(
        description="Concise description of page contents (1-2 sentences, max 150 chars)"
    )

    # Per-dimension scores (0.0-1.0 each)
    score_data_documentation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Signal tables, node lists, shot databases (0.0-1.0)",
    )
    score_physics_content: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Physics explanations, methodology, theory (0.0-1.0)",
    )
    score_code_documentation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Software docs, API references, usage guides (0.0-1.0)",
    )
    score_data_access: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="MDSplus paths, TDI expressions, access methods (0.0-1.0)",
    )
    score_calibration: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calibration info, conversion factors, sensor specs (0.0-1.0)",
    )
    score_imas_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="IMAS integration, IDS references, mapping hints (0.0-1.0)",
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation for the score (max 80 chars)",
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5)",
    )

    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (use 'general' if no clear domain)",
    )

    should_ingest: bool = Field(
        description="Whether to ingest full content and create embeddings"
    )

    skip_reason: str = Field(
        default="",
        description="Why not to ingest (if should_ingest=false)",
    )


class WikiScoreBatch(BaseModel):
    """Batch of wiki page scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    The LLM returns a list of WikiScoreResult objects.
    """

    results: list[WikiScoreResult] = Field(
        description="List of scoring results, one per input page, in order"
    )


# ============================================================================
# Runtime Dataclasses
# ============================================================================


def grounded_wiki_score(
    scores: dict[str, float],
    purpose: WikiPagePurpose,
) -> float:
    """Compute combined score from per-dimension scores.

    Uses MAX of per-dimension scores (not weighted average) so that pages
    excelling in a single dimension are not penalized.

    Purpose-based multipliers:
    - High-value (data_source, diagnostic, code, calibration, data_access): 1.0
    - Medium-value (physics_analysis, experimental_procedure, tutorial, reference): 0.8
    - Low-value (administrative, personal, other): 0.3

    Args:
        scores: Dict of per-dimension scores
        purpose: Classified purpose

    Returns:
        Combined score (0.0-1.0)
    """
    # Use max of all per-dimension scores
    base_score = max(scores.values()) if scores else 0.0

    # Purpose-based multipliers
    high_value = {
        WikiPagePurpose.data_source,
        WikiPagePurpose.diagnostic,
        WikiPagePurpose.code,
        WikiPagePurpose.calibration,
        WikiPagePurpose.data_access,
    }
    medium_value = {
        WikiPagePurpose.physics_analysis,
        WikiPagePurpose.experimental_procedure,
        WikiPagePurpose.tutorial,
        WikiPagePurpose.reference,
    }
    low_value = {
        WikiPagePurpose.administrative,
        WikiPagePurpose.personal,
        WikiPagePurpose.other,
    }

    if purpose in high_value:
        multiplier = 1.0
    elif purpose in medium_value:
        multiplier = 0.8
    elif purpose in low_value:
        multiplier = 0.3
    else:
        multiplier = 0.5

    return max(0.0, min(1.0, base_score * multiplier))


@dataclass
class ScoredWikiPage:
    """Result of LLM scoring for a single wiki page.

    Contains per-dimension scores and ingestion decision.
    """

    id: str
    """Wiki page ID (facility:page_name)."""

    page_purpose: WikiPagePurpose
    """Classified purpose of the page."""

    description: str
    """One-sentence description of the page's contents."""

    # Per-dimension scores (0.0-1.0 each)
    score_data_documentation: float = 0.0
    score_physics_content: float = 0.0
    score_code_documentation: float = 0.0
    score_data_access: float = 0.0
    score_calibration: float = 0.0
    score_imas_relevance: float = 0.0

    score: float = 0.0
    """Combined score computed by grounded scoring function."""

    reasoning: str = ""
    """Brief explanation for the score."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords for this page (max 5)."""

    physics_domain: PhysicsDomain | None = None
    """Primary physics domain."""

    should_ingest: bool = False
    """Whether to ingest full content."""

    skip_reason: str | None = None
    """Why this page should not be ingested."""

    score_cost: float = 0.0
    """LLM cost in USD for scoring this page (batch cost / batch size)."""

    @classmethod
    def from_llm_result(
        cls,
        result: WikiScoreResult,
        cost_per_page: float = 0.0,
    ) -> ScoredWikiPage:
        """Create from LLM structured output result."""
        scores = {
            "score_data_documentation": result.score_data_documentation,
            "score_physics_content": result.score_physics_content,
            "score_code_documentation": result.score_code_documentation,
            "score_data_access": result.score_data_access,
            "score_calibration": result.score_calibration,
            "score_imas_relevance": result.score_imas_relevance,
        }

        combined = grounded_wiki_score(scores, result.page_purpose)

        return cls(
            id=result.id,
            page_purpose=result.page_purpose,
            description=result.description[:150],
            score_data_documentation=result.score_data_documentation,
            score_physics_content=result.score_physics_content,
            score_code_documentation=result.score_code_documentation,
            score_data_access=result.score_data_access,
            score_calibration=result.score_calibration,
            score_imas_relevance=result.score_imas_relevance,
            score=combined,
            reasoning=result.reasoning[:80],
            keywords=result.keywords[:5],
            physics_domain=result.physics_domain,
            should_ingest=result.should_ingest,
            skip_reason=result.skip_reason or None,
            score_cost=cost_per_page,
        )

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence."""
        return {
            "id": self.id,
            "page_purpose": self.page_purpose.value,
            "description": self.description,
            "score_data_documentation": self.score_data_documentation,
            "score_physics_content": self.score_physics_content,
            "score_code_documentation": self.score_code_documentation,
            "score_data_access": self.score_data_access,
            "score_calibration": self.score_calibration,
            "score_imas_relevance": self.score_imas_relevance,
            "score": self.score,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "should_ingest": self.should_ingest,
            "skip_reason": self.skip_reason,
            "score_cost": self.score_cost,
        }


@dataclass
class ScoredWikiBatch:
    """Result of scoring a batch of wiki pages."""

    scored_pages: list[ScoredWikiPage]
    """List of scored wiki pages."""

    total_cost: float
    """Estimated cost in USD for this batch."""

    model: str
    """Model used for scoring."""

    tokens_used: int
    """Total tokens (input + output) used."""

    @property
    def ingested_count(self) -> int:
        """Number of pages marked for ingestion."""
        return sum(1 for p in self.scored_pages if p.should_ingest)


# ============================================================================
# Artifact Scoring Pydantic Models (LLM Structured Output)
# ============================================================================


class ArtifactScoreResult(BaseModel):
    """LLM scoring result for a single wiki artifact.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.

    Uses same scoring dimensions as WikiScoreResult for consistency.
    """

    id: str = Field(description="The artifact ID (echo from input)")

    artifact_purpose: WikiPagePurpose = Field(
        description="Classification: data_source, diagnostic, code, calibration, "
        "data_access, physics_analysis, experimental_procedure, tutorial, "
        "reference, administrative, personal, other"
    )

    description: str = Field(
        description="Concise description of artifact contents (1-2 sentences, max 150 chars)"
    )

    # Per-dimension scores (0.0-1.0 each) - same as WikiScoreResult
    score_data_documentation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Signal tables, node lists, shot databases (0.0-1.0)",
    )
    score_physics_content: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Physics explanations, methodology, theory (0.0-1.0)",
    )
    score_code_documentation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Software docs, API references, usage guides (0.0-1.0)",
    )
    score_data_access: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="MDSplus paths, TDI expressions, access methods (0.0-1.0)",
    )
    score_calibration: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calibration info, conversion factors, sensor specs (0.0-1.0)",
    )
    score_imas_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="IMAS integration, IDS references, mapping hints (0.0-1.0)",
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation for the score (max 80 chars)",
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5)",
    )

    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (use 'general' if no clear domain)",
    )

    should_ingest: bool = Field(
        description="Whether to download full content and create embeddings"
    )

    skip_reason: str = Field(
        default="",
        description="Why not to ingest (if should_ingest=false)",
    )


class ArtifactScoreBatch(BaseModel):
    """Batch of artifact scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[ArtifactScoreResult] = Field(
        description="List of scoring results, one per input artifact, in order"
    )


def grounded_artifact_score(
    scores: dict[str, float],
    purpose: WikiPagePurpose,
) -> float:
    """Compute combined score from per-dimension scores for artifacts.

    Uses same logic as grounded_wiki_score for consistency.
    MAX of per-dimension scores with purpose-based multipliers.

    Args:
        scores: Dict of per-dimension scores
        purpose: Classified purpose

    Returns:
        Combined score (0.0-1.0)
    """
    # Reuse the same logic as wiki scoring
    return grounded_wiki_score(scores, purpose)


@dataclass
class ScoredArtifact:
    """Result of LLM scoring for a single wiki artifact.

    Runtime dataclass containing per-dimension scores and ingestion decision.
    """

    id: str
    """Artifact ID (facility:filename)."""

    artifact_purpose: WikiPagePurpose
    """Classified purpose of the artifact."""

    description: str
    """One-sentence description of the artifact's contents."""

    # Per-dimension scores (0.0-1.0 each)
    score_data_documentation: float = 0.0
    score_physics_content: float = 0.0
    score_code_documentation: float = 0.0
    score_data_access: float = 0.0
    score_calibration: float = 0.0
    score_imas_relevance: float = 0.0

    score: float = 0.0
    """Combined score computed by grounded scoring function."""

    reasoning: str = ""
    """Brief explanation for the score."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords for this artifact (max 5)."""

    physics_domain: PhysicsDomain | None = None
    """Primary physics domain."""

    should_ingest: bool = False
    """Whether to download full content and ingest."""

    skip_reason: str | None = None
    """Why this artifact should not be ingested."""

    score_cost: float = 0.0
    """LLM cost in USD for scoring this artifact (batch cost / batch size)."""

    @classmethod
    def from_llm_result(
        cls,
        result: ArtifactScoreResult,
        cost_per_artifact: float = 0.0,
    ) -> ScoredArtifact:
        """Create from LLM structured output result."""
        scores = {
            "score_data_documentation": result.score_data_documentation,
            "score_physics_content": result.score_physics_content,
            "score_code_documentation": result.score_code_documentation,
            "score_data_access": result.score_data_access,
            "score_calibration": result.score_calibration,
            "score_imas_relevance": result.score_imas_relevance,
        }

        combined = grounded_artifact_score(scores, result.artifact_purpose)

        return cls(
            id=result.id,
            artifact_purpose=result.artifact_purpose,
            description=result.description[:150],
            score_data_documentation=result.score_data_documentation,
            score_physics_content=result.score_physics_content,
            score_code_documentation=result.score_code_documentation,
            score_data_access=result.score_data_access,
            score_calibration=result.score_calibration,
            score_imas_relevance=result.score_imas_relevance,
            score=combined,
            reasoning=result.reasoning[:80],
            keywords=result.keywords[:5],
            physics_domain=result.physics_domain,
            should_ingest=result.should_ingest,
            skip_reason=result.skip_reason or None,
            score_cost=cost_per_artifact,
        )

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence."""
        return {
            "id": self.id,
            "artifact_purpose": self.artifact_purpose.value,
            "description": self.description,
            "score_data_documentation": self.score_data_documentation,
            "score_physics_content": self.score_physics_content,
            "score_code_documentation": self.score_code_documentation,
            "score_data_access": self.score_data_access,
            "score_calibration": self.score_calibration,
            "score_imas_relevance": self.score_imas_relevance,
            "score": self.score,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "should_ingest": self.should_ingest,
            "skip_reason": self.skip_reason,
            "score_cost": self.score_cost,
        }


# ============================================================================
# Image Captioning Pydantic Models (VLM Structured Output)
# ============================================================================


class ImageCaptionResult(BaseModel):
    """VLM captioning result for a single image.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the VLM.

    The VLM receives the image bytes + context (page_title, section,
    surrounding_text) and returns a fusion-domain-aware caption.
    """

    id: str = Field(description="The image ID (echo from input)")

    caption: str = Field(
        description="Detailed physics-aware description of image content. "
        "Include specific quantities, diagnostics, tree paths, conventions. "
        "Describe what the image shows in fusion physics terms, not visual appearance."
    )

    ocr_text: str = Field(
        default="",
        description="All visible text in the image: axis labels, legends, titles, "
        "MDSplus paths, parameter values. Empty string if no text visible.",
    )

    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (use 'general' if ambiguous)",
    )


class ImageCaptionBatch(BaseModel):
    """Batch of image captioning results from VLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[ImageCaptionResult] = Field(
        description="List of captioning results, one per input image, in order"
    )
