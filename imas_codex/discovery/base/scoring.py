"""Shared scoring models and composite score functions for discovery pipelines.

Two scoring families exist:

- **Code/Path family**: 9 shared dimensions (modeling_code, analysis_code, etc.)
  with a max-of-dimensions composite formula. Paths adds 2 extra dimensions.
- **Content family** (Wiki/Document/Image): 6 shared dimensions
  (data_documentation, physics_content, etc.) with a purpose-weighted composite.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from imas_codex.graph.models import ContentPurpose

# ---------------------------------------------------------------------------
# Dimension name constants
# ---------------------------------------------------------------------------

CODE_SCORE_DIMENSIONS: list[str] = [
    "score_modeling_code",
    "score_analysis_code",
    "score_operations_code",
    "score_data_access",
    "score_workflow",
    "score_visualization",
    "score_documentation",
    "score_imas",
    "score_convention",
]
"""9 scoring dimensions shared by code and path discovery."""

PATH_EXTRA_DIMENSIONS: list[str] = [
    "score_modeling_data",
    "score_experimental_data",
]
"""2 additional dimensions used only by path discovery."""

PATH_SCORE_DIMENSIONS: list[str] = CODE_SCORE_DIMENSIONS + PATH_EXTRA_DIMENSIONS
"""All 11 path scoring dimensions (code dimensions + data dimensions)."""

CONTENT_SCORE_DIMENSIONS: list[str] = [
    "score_data_documentation",
    "score_physics_content",
    "score_code_documentation",
    "score_data_access",
    "score_calibration",
    "score_imas_relevance",
]
"""6 scoring dimensions shared by wiki page, document, and image scoring."""


# ---------------------------------------------------------------------------
# Composite score functions
# ---------------------------------------------------------------------------


def max_composite(scores: dict[str, float]) -> float:
    """Composite score = max of all dimension scores.

    Used by the paths and code families.
    """
    if not scores:
        return 0.0
    return min(1.0, max(scores.values()))


def purpose_weighted_composite(
    scores: dict[str, float],
    purpose: ContentPurpose,
) -> float:
    """Compute purpose-weighted composite: max(scores) * purpose_multiplier.

    Purpose multipliers:

    - High-value (data_source, diagnostic, code, calibration, data_access): 1.0
    - Medium-value (physics_analysis, experimental_procedure, tutorial, reference): 0.8
    - Low-value (administrative, personal, other): 0.3

    Used by the wiki, document, and image families.
    """
    base_score = max(scores.values()) if scores else 0.0

    high_value = {
        ContentPurpose.data_source,
        ContentPurpose.diagnostic,
        ContentPurpose.code,
        ContentPurpose.calibration,
        ContentPurpose.data_access,
    }
    medium_value = {
        ContentPurpose.physics_analysis,
        ContentPurpose.experimental_procedure,
        ContentPurpose.tutorial,
        ContentPurpose.reference,
    }

    if purpose in high_value:
        multiplier = 1.0
    elif purpose in medium_value:
        multiplier = 0.8
    else:
        multiplier = 0.3

    return max(0.0, min(1.0, base_score * multiplier))


# ---------------------------------------------------------------------------
# Code/Path scoring base model
# ---------------------------------------------------------------------------


class CodeScoreFields(BaseModel):
    """Base Pydantic model with the 9 code/path scoring dimensions.

    Inherit from this in LLM response models to get consistent field
    definitions and the ``get_score_dict()`` helper.
    """

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

    def get_score_dict(self) -> dict[str, float]:
        """Return the 9 code score dimensions as a dict."""
        return {name: getattr(self, name) for name in CODE_SCORE_DIMENSIONS}


# ---------------------------------------------------------------------------
# Content scoring base model
# ---------------------------------------------------------------------------


class ContentScoreFields(BaseModel):
    """Base Pydantic model with the 6 content scoring dimensions.

    Inherit from this in LLM response models for wiki pages, documents,
    and images to get consistent field definitions.
    """

    score_data_documentation: float = Field(
        default=0.0,
        description="Signal tables, node lists, shot databases (0.0-1.0)",
    )
    score_physics_content: float = Field(
        default=0.0,
        description="Physics explanations, methodology, theory (0.0-1.0)",
    )
    score_code_documentation: float = Field(
        default=0.0,
        description="Software docs, API references, usage guides (0.0-1.0)",
    )
    score_data_access: float = Field(
        default=0.0,
        description="MDSplus paths, TDI expressions, access methods (0.0-1.0)",
    )
    score_calibration: float = Field(
        default=0.0,
        description="Calibration info, conversion factors, sensor specs (0.0-1.0)",
    )
    score_imas_relevance: float = Field(
        default=0.0,
        description="IMAS integration, IDS references, mapping hints (0.0-1.0)",
    )

    def get_score_dict(self) -> dict[str, float]:
        """Return the 6 content score dimensions as a dict."""
        return {name: getattr(self, name) for name in CONTENT_SCORE_DIMENSIONS}
