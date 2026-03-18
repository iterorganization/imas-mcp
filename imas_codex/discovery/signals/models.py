"""
Data models for signal enrichment pipeline.

These Pydantic models define the LLM structured output for signal enrichment.
The models enforce valid PhysicsDomain enum values and provide safety constraints
for unit and sign convention fields.

Design principles:
- Units and sign conventions are ONLY set when extracted from authoritative sources
- LLM MUST NOT hallucinate units - if not discoverable, leave empty
- Batch processing: multiple signals per LLM call for efficiency
- Physics domain classification uses schema-derived enum values
- Context quality scoring enables re-enrichment of underspecified signals
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.graph.models import DiagnosticCategory

__all__ = [
    "ContextQuality",
    "SignalEnrichmentResult",
    "SignalEnrichmentBatch",
    "SignalSourceEnrichmentResult",
    "SignalSourceEnrichmentBatch",
    "SignalSourceCodeUnwind",
    "SignalSourceCodeUnwindBatch",
    "UnitConfidence",
]


class ContextQuality(StrEnum):
    """How much context was available for this signal's enrichment.

    Signals with 'low' context quality are marked as 'underspecified'
    in the graph and queued for re-enrichment when better context
    becomes available.
    """

    low = "low"
    medium = "medium"
    high = "high"


class UnitConfidence(BaseModel):
    """Confidence level for unit extraction.

    Units should only be trusted when extracted from authoritative sources.
    The LLM must NOT guess units - hallucinated units are dangerous.
    """

    value: str = Field(
        default="",
        description="Physical units (e.g., 'A', 'V', 'm^-3'). "
        "ONLY set if extracted from MDSplus metadata, code comments, or documentation. "
        "Leave empty if uncertain - do NOT guess.",
    )
    source: str = Field(
        default="",
        description="Where units came from: 'mdsplus_metadata', 'code_comment', "
        "'documentation', or '' if unknown. Required if value is set.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence 0.0-1.0. Only >0.8 if from authoritative source.",
    )


class SignalEnrichmentResult(BaseModel):
    """LLM enrichment result for a single signal.

    This model enforces:
    - Valid PhysicsDomain enum values
    - Safety constraints on units (no hallucination)
    - Structured metadata for graph storage
    """

    signal_index: int = Field(
        description="1-based index matching the input signal order (Signal 1 = index 1)",
        json_schema_extra={"examples": [1]},
    )

    physics_domain: PhysicsDomain = Field(
        description="Physics domain classification. Use the category that best "
        "describes this signal's primary physics purpose."
    )

    name: str = Field(
        description="Human-readable signal name (e.g., 'Plasma Current', "
        "'Electron Temperature Profile'). Should be concise and descriptive."
    )

    description: str = Field(
        description="Physics description of what this signal measures "
        "(2-4 sentences). Include measurement technique, spatial/temporal "
        "characteristics, and relevant physics context."
    )

    diagnostic: DiagnosticCategory | None = Field(
        default=None,
        description="Diagnostic system category if identifiable. "
        "Use one of the defined DiagnosticCategory enum values. "
        "Leave null if not a diagnostic signal.",
    )

    @field_validator("diagnostic", mode="before")
    @classmethod
    def normalize_diagnostic(cls, v: str | DiagnosticCategory | None) -> DiagnosticCategory | None:
        """Normalize diagnostic input to DiagnosticCategory enum or None."""
        if v is None or v == "":
            return None
        if isinstance(v, DiagnosticCategory):
            return v
        # Try to match by value (lowercase_snake_case)
        normalized = str(v).lower().replace(" ", "_").replace("-", "_")
        try:
            return DiagnosticCategory(normalized)
        except ValueError:
            return None

    analysis_code: str = Field(
        default="",
        description="Analysis code that produces this signal if identifiable "
        "(e.g., 'LIUQE', 'ASTRA', 'CHEASE'). Leave empty if raw diagnostic data.",
    )

    # Sign convention — only from code or wiki context
    sign_convention: str = Field(
        default="",
        description="Sign convention if discoverable from source code or wiki documentation. "
        "E.g., 'positive direction = co-current', 'positive = outward flux'. "
        "Leave empty if not explicitly stated in the provided context.",
    )

    # Classification confidence
    confidence: float = Field(
        default=0.8,
        description="Confidence in the physics_domain classification (0.0-1.0). "
        "Lower if signal name is ambiguous.",
    )

    # Context quality assessment
    context_quality: ContextQuality = Field(
        default=ContextQuality.medium,
        description="How much context was available to classify this signal. "
        "'low' = only accessor/name, no source code, wiki, tree context, or code chunks. "
        "'medium' = some context (tree path, group header, or partial wiki). "
        "'high' = rich context (source code, wiki description, code references, siblings).",
    )

    # Keywords for search
    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5). Include physics terms, "
        "diagnostic names, and common abbreviations.",
    )


class SignalEnrichmentBatch(BaseModel):
    """Batch of signal enrichment results for efficient LLM processing.

    Designed for batches of 50-200 signals per LLM call.
    Gemini 3 Flash supports up to 65k output tokens (~300 signals max).
    """

    results: list[SignalEnrichmentResult] = Field(
        description="Enrichment results, one per input signal in the same order"
    )


class SignalSourceEnrichmentResult(BaseModel):
    """LLM enrichment for a single SignalSource node.

    SignalSource nodes group signals that map identically to the same IMAS field.
    The source-level description explains what the collection represents and how
    individual members differ from each other.
    """

    source_index: int = Field(
        description="1-based index matching the input source order",
        json_schema_extra={"examples": [1]},
    )

    description: str = Field(
        description="What this collection of signals represents. "
        "Describe the shared measurement type and how members differ."
    )

    physics_domain: PhysicsDomain = Field(
        description="Physics domain classification for this source."
    )

    diagnostic: str = Field(
        default="",
        description="Diagnostic system name if identifiable.",
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (max 5).",
    )

    member_variation: str = Field(
        default="",
        description="How individual members differ within the source "
        "(e.g., 'spatial position', 'channel index', 'coil number').",
    )


class SignalSourceEnrichmentBatch(BaseModel):
    """Batch of SignalSource enrichments — multiple sources per LLM call."""

    results: list[SignalSourceEnrichmentResult] = Field(
        description="Enrichment results, one per input source in the same order"
    )


class SignalSourceCodeUnwind(BaseModel):
    """LLM-generated Python code to individualize group member descriptions.

    Instead of a simple {member_id} template, the LLM generates a Python
    function that produces physics-aware individualized descriptions using
    actual values from SignalNode descriptions (geometry, positions, angles).
    """

    source_index: int = Field(
        description="1-based index matching the input source order",
    )

    name_template: str = Field(
        description="Name template with {member_id} placeholder. "
        "E.g., 'Magnetic Probe {member_id} Radial Position'",
    )

    description_template: str = Field(
        description="Description template with {member_id} and optional "
        "{node_description} placeholder. Use {node_description} when the "
        "SignalNode description contains actual physics values that should "
        "be included verbatim. "
        "E.g., 'Radial position of magnetic probe {member_id}. {node_description}'",
    )

    variation_field: str = Field(
        description="Which part of the accessor varies across members "
        "(e.g., 'probe number', 'channel index', 'coil name')",
    )


class SignalSourceCodeUnwindBatch(BaseModel):
    """Batch of code-based unwinding results — multiple sources per LLM call."""

    results: list[SignalSourceCodeUnwind] = Field(
        description="Unwinding results, one per input source"
    )
