"""
Data models for wiki discovery pipeline.

These models define runtime structures for the LLM-based wiki scoring phase.
Mirrors the structure of discovery/paths/models.py for consistency.

The Pydantic models (WikiScoreResult, WikiScoreBatch) are used for LLM
structured output - LiteLLM parses responses directly into these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.discovery.base.scoring import (
    ContentScoreFields,
    purpose_weighted_composite,
)
from imas_codex.graph.models import ContentPurpose

# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================


class WikiScoreResult(ContentScoreFields):
    """LLM scoring result for a single wiki page.

    Inherits 6 content score dimensions from ContentScoreFields.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.
    """

    id: str = Field(description="The page ID (echo from input)")

    page_purpose: ContentPurpose = Field(
        description="Classification: data_source, diagnostic, code, calibration, "
        "data_access, physics_analysis, experimental_procedure, tutorial, "
        "reference, administrative, personal, other"
    )

    description: str = Field(
        description="Concise description of page contents (1-2 sentences)"
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation for the score",
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
    purpose: ContentPurpose,
) -> float:
    """Compute combined score from per-dimension scores.

    Delegates to the shared ``purpose_weighted_composite`` function.

    Args:
        scores: Dict of per-dimension scores
        purpose: Classified purpose

    Returns:
        Combined score (0.0-1.0)
    """
    return purpose_weighted_composite(scores, purpose)


@dataclass
class ScoredWikiPage:
    """Result of LLM scoring for a single wiki page.

    Contains per-dimension scores and ingestion decision.
    """

    id: str
    """Wiki page ID (facility:page_name)."""

    page_purpose: ContentPurpose
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

    score_composite: float = 0.0
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
        combined = grounded_wiki_score(result.get_score_dict(), result.page_purpose)

        return cls(
            id=result.id,
            page_purpose=result.page_purpose,
            description=result.description,
            score_data_documentation=result.score_data_documentation,
            score_physics_content=result.score_physics_content,
            score_code_documentation=result.score_code_documentation,
            score_data_access=result.score_data_access,
            score_calibration=result.score_calibration,
            score_imas_relevance=result.score_imas_relevance,
            score_composite=combined,
            reasoning=result.reasoning,
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
            "purpose": self.page_purpose.value,
            "description": self.description,
            "score_data_documentation": self.score_data_documentation,
            "score_physics_content": self.score_physics_content,
            "score_code_documentation": self.score_code_documentation,
            "score_data_access": self.score_data_access,
            "score_calibration": self.score_calibration,
            "score_imas_relevance": self.score_imas_relevance,
            "score_composite": self.score_composite,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "should_ingest": self.should_ingest,
            "skip_reason": self.skip_reason,
            "score_cost": self.score_cost,
            "scored_at": datetime.now(UTC).isoformat(),
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
# Document Scoring Pydantic Models (LLM Structured Output)
# ============================================================================


class DocumentScoreResult(ContentScoreFields):
    """LLM scoring result for a single wiki document.

    Inherits 6 content score dimensions from ContentScoreFields.

    This Pydantic model is passed to LiteLLM's response_format parameter
    to ensure structured, parseable output from the LLM.
    """

    id: str = Field(description="The document ID (echo from input)")

    document_purpose: ContentPurpose = Field(
        description="Classification: data_source, diagnostic, code, calibration, "
        "data_access, physics_analysis, experimental_procedure, tutorial, "
        "reference, administrative, personal, other"
    )

    description: str = Field(
        description="Concise description of document contents (1-2 sentences)"
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation for the score",
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


class DocumentScoreBatch(BaseModel):
    """Batch of document scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[DocumentScoreResult] = Field(
        description="List of scoring results, one per input document, in order"
    )


def grounded_document_score(
    scores: dict[str, float],
    purpose: ContentPurpose,
) -> float:
    """Compute combined score from per-dimension scores for documents.

    Delegates to the shared ``purpose_weighted_composite`` function.
    """
    return purpose_weighted_composite(scores, purpose)


@dataclass
class ScoredDocument:
    """Result of LLM scoring for a single wiki document.

    Runtime dataclass containing per-dimension scores and ingestion decision.
    """

    id: str
    """Document ID (facility:filename)."""

    document_purpose: ContentPurpose
    """Classified purpose of the document."""

    description: str
    """One-sentence description of the document's contents."""

    # Per-dimension scores (0.0-1.0 each)
    score_data_documentation: float = 0.0
    score_physics_content: float = 0.0
    score_code_documentation: float = 0.0
    score_data_access: float = 0.0
    score_calibration: float = 0.0
    score_imas_relevance: float = 0.0

    score_composite: float = 0.0
    """Combined score computed by grounded scoring function."""

    reasoning: str = ""
    """Brief explanation for the score."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords for this document (max 5)."""

    physics_domain: PhysicsDomain | None = None
    """Primary physics domain."""

    should_ingest: bool = False
    """Whether to download full content and ingest."""

    skip_reason: str | None = None
    """Why this document should not be ingested."""

    score_cost: float = 0.0
    """LLM cost in USD for scoring this document (batch cost / batch size)."""

    @classmethod
    def from_llm_result(
        cls,
        result: DocumentScoreResult,
        cost_per_document: float = 0.0,
    ) -> ScoredDocument:
        """Create from LLM structured output result."""
        combined = grounded_document_score(
            result.get_score_dict(), result.document_purpose
        )

        return cls(
            id=result.id,
            document_purpose=result.document_purpose,
            description=result.description,
            score_data_documentation=result.score_data_documentation,
            score_physics_content=result.score_physics_content,
            score_code_documentation=result.score_code_documentation,
            score_data_access=result.score_data_access,
            score_calibration=result.score_calibration,
            score_imas_relevance=result.score_imas_relevance,
            score_composite=combined,
            reasoning=result.reasoning,
            keywords=result.keywords[:5],
            physics_domain=result.physics_domain,
            should_ingest=result.should_ingest,
            skip_reason=result.skip_reason or None,
            score_cost=cost_per_document,
        )

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence."""
        return {
            "id": self.id,
            "document_purpose": self.document_purpose.value,
            "purpose": self.document_purpose.value,
            "description": self.description,
            "score_data_documentation": self.score_data_documentation,
            "score_physics_content": self.score_physics_content,
            "score_code_documentation": self.score_code_documentation,
            "score_data_access": self.score_data_access,
            "score_calibration": self.score_calibration,
            "score_imas_relevance": self.score_imas_relevance,
            "score_composite": self.score_composite,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "should_ingest": self.should_ingest,
            "skip_reason": self.skip_reason,
            "score_cost": self.score_cost,
            "scored_at": datetime.now(UTC).isoformat(),
        }


# ============================================================================
# Image Scoring Pydantic Models (VLM Structured Output)
# ============================================================================


class ImageScoreResult(ContentScoreFields):
    """VLM scoring + captioning result for a single image.

    Inherits 6 content score dimensions from ContentScoreFields.

    This Pydantic model is passed to LiteLLM's response_format parameter.
    The VLM receives image bytes + context (page_title, section,
    surrounding_text) and returns caption + scoring in a single pass.
    """

    id: str = Field(description="The image ID (echo from input)")

    mermaid_diagram: str = Field(
        default="",
        description="Mermaid diagram representing the structure of schematics, "
        "block diagrams, or data flow images. Use graph LR or graph TD syntax. "
        "Empty string for non-schematic images (plots, photos, etc.).",
    )

    ocr_text: str = Field(
        default="",
        description="All visible text in the image: axis labels, legends, titles, "
        "MDSplus paths, parameter values. Empty string if no text visible.",
    )

    purpose: ContentPurpose = Field(
        description="Classification: data_source, diagnostic, code, calibration, "
        "data_access, physics_analysis, experimental_procedure, tutorial, "
        "reference, administrative, personal, other"
    )

    description: str = Field(
        description="Detailed physics-aware description of image content. "
        "Include specific quantities, diagnostics, tree paths, conventions. "
        "Describe what the image shows in fusion physics terms, not visual appearance. "
        "Length scales with content richness: 1-2 sentences for simple images, "
        "full paragraph for complex schematics or multi-panel plots."
    )

    reasoning: str = Field(default="", description="Brief explanation for the score")

    keywords: list[str] = Field(
        default_factory=list, description="Searchable keywords (max 5)"
    )

    physics_domain: PhysicsDomain = Field(
        default=PhysicsDomain.GENERAL,
        description="Primary physics domain (use 'general' if ambiguous)",
    )

    should_ingest: bool = Field(
        description="Whether this image is worth embedding for search"
    )

    skip_reason: str = Field(
        default="", description="Why not to ingest (if should_ingest=false)"
    )


class ImageScoreBatch(BaseModel):
    """Batch of image scoring results from VLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[ImageScoreResult] = Field(
        description="List of scoring results, one per input image, in order"
    )


def grounded_image_score(
    scores: dict[str, float],
    purpose: ContentPurpose,
) -> float:
    """Compute combined score for an image.

    Delegates to the shared ``purpose_weighted_composite`` function.
    """
    return purpose_weighted_composite(scores, purpose)


@dataclass
class ScoredImage:
    """Result of VLM scoring for a single image.

    Runtime dataclass combining caption + scoring from a single VLM pass.
    """

    id: str
    """Image ID (facility:sha256[:16])."""

    mermaid_diagram: str = ""
    """Mermaid diagram for schematics/data flow images."""

    ocr_text: str = ""
    """Text extracted via OCR from image."""

    purpose: ContentPurpose = ContentPurpose.other
    """Content classification."""

    description: str = ""
    """Physics-aware image description."""

    # Per-dimension scores (0.0-1.0 each)
    score_data_documentation: float = 0.0
    score_physics_content: float = 0.0
    score_code_documentation: float = 0.0
    score_data_access: float = 0.0
    score_calibration: float = 0.0
    score_imas_relevance: float = 0.0

    score_composite: float = 0.0
    """Combined score computed by grounded scoring function."""

    reasoning: str = ""
    """Brief explanation for the score."""

    keywords: list[str] = field(default_factory=list)
    """Searchable keywords (max 5)."""

    physics_domain: PhysicsDomain | None = None
    """Primary physics domain."""

    should_ingest: bool = False
    """Whether this image is worth embedding for search."""

    skip_reason: str | None = None
    """Why this image was skipped."""

    score_cost: float = 0.0
    """VLM cost in USD for scoring this image (batch cost / batch size)."""

    @classmethod
    def from_vlm_result(
        cls,
        result: ImageScoreResult,
        cost_per_image: float = 0.0,
    ) -> ScoredImage:
        """Create from VLM structured output result."""
        combined = grounded_image_score(result.get_score_dict(), result.purpose)

        return cls(
            id=result.id,
            mermaid_diagram=result.mermaid_diagram,
            ocr_text=result.ocr_text,
            purpose=result.purpose,
            description=result.description,
            score_data_documentation=result.score_data_documentation,
            score_physics_content=result.score_physics_content,
            score_code_documentation=result.score_code_documentation,
            score_data_access=result.score_data_access,
            score_calibration=result.score_calibration,
            score_imas_relevance=result.score_imas_relevance,
            score_composite=combined,
            reasoning=result.reasoning,
            keywords=result.keywords[:5],
            physics_domain=result.physics_domain,
            should_ingest=result.should_ingest,
            skip_reason=result.skip_reason or None,
            score_cost=cost_per_image,
        )

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph persistence."""
        return {
            "id": self.id,
            "mermaid_diagram": self.mermaid_diagram,
            "ocr_text": self.ocr_text,
            "purpose": self.purpose.value,
            "description": self.description,
            "score_data_documentation": self.score_data_documentation,
            "score_physics_content": self.score_physics_content,
            "score_code_documentation": self.score_code_documentation,
            "score_data_access": self.score_data_access,
            "score_calibration": self.score_calibration,
            "score_imas_relevance": self.score_imas_relevance,
            "score_composite": self.score_composite,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "physics_domain": self.physics_domain.value
            if self.physics_domain
            else None,
            "should_ingest": self.should_ingest,
            "skip_reason": self.skip_reason,
            "score_cost": self.score_cost,
            "scored_at": datetime.now(UTC).isoformat(),
        }
