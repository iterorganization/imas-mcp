"""Pydantic models for standard name pipeline LLM responses."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SNCandidate(BaseModel):
    """A single standard name candidate from LLM composition."""

    source_id: str = Field(description="Source entity ID (DD path or signal ID)")
    standard_name: str = Field(description="Composed standard name")
    fields: dict[str, str] = Field(description="Grammar fields used")
    confidence: float = Field(ge=0, le=1, description="Naming confidence")
    reason: str = Field(description="Brief justification")


class SNComposeBatch(BaseModel):
    """LLM response for a batch of standard name compositions."""

    candidates: list[SNCandidate]
    skipped: list[str] = Field(
        default_factory=list, description="Source IDs skipped (not physics quantities)"
    )


# =============================================================================
# Publish models — YAML catalog export (Feature 08)
# =============================================================================


class SNProvenance(BaseModel):
    """Provenance metadata for a standard name entry."""

    source: str = Field(description="Source type: dd or signal")
    source_id: str = Field(description="Source entity ID")
    ids_name: str | None = Field(default=None, description="IDS name (for DD source)")
    confidence: float = Field(ge=0, le=1, description="Generation confidence")
    generated_by: str = Field(
        default="imas-codex", description="Tool that generated this"
    )


class SNPublishEntry(BaseModel):
    """A single standard name entry ready for YAML catalog export."""

    name: str = Field(description="The standard name")
    kind: str = Field(
        default="physical", description="Name kind: physical or geometric"
    )
    unit: str | None = Field(default=None, description="SI unit string")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    status: str = Field(default="drafted", description="Entry status")
    description: str = Field(default="", description="Human-readable description")
    provenance: SNProvenance = Field(description="Generation provenance")


class SNPublishBatch(BaseModel):
    """A batch of entries to publish as a PR."""

    group_key: str = Field(description="Batch group key (IDS name or domain)")
    entries: list[SNPublishEntry]
    confidence_tier: str = Field(description="high, medium, or low")


# =============================================================================
# Cross-model review models
# =============================================================================


class SNReviewVerdict(StrEnum):
    """Review decision for a standard name candidate."""

    accept = "accept"
    reject = "reject"
    revise = "revise"


class SNReviewItem(BaseModel):
    """Review of a single standard name candidate."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    verdict: SNReviewVerdict = Field(description="Accept, reject, or revise")
    confidence: float = Field(ge=0, le=1, description="Review confidence")
    reason: str = Field(description="Justification for the verdict")
    revised_name: str | None = Field(
        default=None, description="Suggested revision if verdict is revise"
    )
    revised_fields: dict[str, str] | None = Field(
        default=None, description="Revised grammar fields"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class SNReviewBatch(BaseModel):
    """LLM response for reviewing a batch of standard name candidates."""

    reviews: list[SNReviewItem]
