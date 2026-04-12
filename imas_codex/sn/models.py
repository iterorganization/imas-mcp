"""Pydantic models for standard name pipeline LLM responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SNCandidate(BaseModel):
    """A single standard name candidate from LLM composition."""

    source_id: str = Field(description="Source entity ID (DD path or signal ID)")
    standard_name: str = Field(description="Composed standard name in snake_case")
    description: str = Field(default="", description="One sentence, <120 chars")
    documentation: str = Field(
        default="", description="Rich docs with LaTeX, links, typical values"
    )
    kind: Literal["scalar", "vector", "metadata"] = Field(
        default="scalar", description="Entry kind"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    links: list[str] = Field(default_factory=list, description="Related standard names")
    ids_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    grammar_fields: dict[str, str] = Field(
        default_factory=dict, description="Grammar fields used"
    )
    confidence: float = Field(ge=0, le=1, description="Naming confidence")
    reason: str = Field(description="Brief justification")
    validity_domain: str | None = Field(
        default=None, description="Physical region where quantity is valid"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints"
    )


class SNVocabGap(BaseModel):
    """A path where naming requires vocabulary expansion."""

    source_id: str = Field(description="DD path that needs naming")
    segment: str = Field(
        description="Grammar segment missing a token (e.g., 'subject', 'position')"
    )
    needed_token: str = Field(
        description="Proposed token value for the grammar segment"
    )
    reason: str = Field(description="Why this token is needed for naming this path")


class SNComposeBatch(BaseModel):
    """LLM response for a batch of standard name compositions."""

    candidates: list[SNCandidate]
    skipped: list[str] = Field(
        default_factory=list, description="Source IDs skipped (not physics quantities)"
    )
    vocab_gaps: list[SNVocabGap] = Field(
        default_factory=list,
        description="Paths where naming requires vocabulary expansion in imas-standard-names",
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
        default="scalar", description="Name kind: scalar, vector, or metadata"
    )
    unit: str | None = Field(default=None, description="SI unit string")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    status: str = Field(default="drafted", description="Entry status")
    description: str = Field(default="", description="Human-readable description")
    # Rich fields
    documentation: str | None = Field(
        default=None, description="Rich documentation with LaTeX"
    )
    links: list[str] = Field(default_factory=list, description="Related standard names")
    ids_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints"
    )
    validity_domain: str | None = Field(
        default=None, description="Physical region where valid"
    )
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
