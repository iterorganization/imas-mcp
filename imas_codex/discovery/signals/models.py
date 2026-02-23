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
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from imas_codex.core.physics_domain import PhysicsDomain

__all__ = [
    "SignalEnrichmentResult",
    "SignalEnrichmentBatch",
    "UnitConfidence",
]


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
        description="Brief physics description of what this signal measures "
        "(1-2 sentences). Include relevant physics context."
    )

    diagnostic: str = Field(
        default="",
        description="Diagnostic system name if identifiable "
        "(e.g., 'thomson_scattering', 'bolometer_array', 'interferometer'). "
        "Use lowercase_snake_case. Leave empty if not a diagnostic signal.",
    )

    @field_validator("diagnostic")
    @classmethod
    def normalize_diagnostic(cls, v: str) -> str:
        """Normalize diagnostic to lowercase_snake_case."""
        if not v:
            return v
        return v.lower().replace(" ", "_").replace("-", "_")

    analysis_code: str = Field(
        default="",
        description="Analysis code that produces this signal if identifiable "
        "(e.g., 'LIUQE', 'ASTRA', 'CHEASE'). Leave empty if raw diagnostic data.",
    )

    # Unit handling with safety constraints
    units_extracted: str = Field(
        default="",
        description="Units ONLY if present in the input metadata. "
        "Do NOT infer or guess units - copy from input or leave empty.",
    )

    # Classification confidence
    confidence: float = Field(
        default=0.8,
        description="Confidence in the physics_domain classification (0.0-1.0). "
        "Lower if signal name is ambiguous.",
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
