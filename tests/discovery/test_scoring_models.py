"""Tests for shared scoring models and composite score functions."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from imas_codex.discovery.base.scoring import (
    CODE_SCORE_DIMENSIONS,
    CONTENT_SCORE_DIMENSIONS,
    PATH_EXTRA_DIMENSIONS,
    PATH_SCORE_DIMENSIONS,
    CodeScoreFields,
    ContentScoreFields,
    max_composite,
    purpose_weighted_composite,
)
from imas_codex.graph.models import ContentPurpose

# ============================================================================
# Max composite (code/path family)
# ============================================================================


class TestMaxComposite:
    """Tests for max_composite: max of dimension scores."""

    def test_single_dimension(self):
        scores = {"dim_a": 0.8}
        assert max_composite(scores) == pytest.approx(0.8)

    def test_multiple_dimensions(self):
        scores = {"dim_a": 0.9, "dim_b": 0.5}
        assert max_composite(scores) == pytest.approx(0.9)

    def test_all_zeros(self):
        scores = {"dim_a": 0.0, "dim_b": 0.0}
        assert max_composite(scores) == 0.0

    def test_empty_dict(self):
        assert max_composite({}) == 0.0

    def test_capped_at_one(self):
        scores = {"dim_a": 1.0, "dim_b": 1.0, "dim_c": 1.0}
        assert max_composite(scores) <= 1.0

    def test_single_high_dimension_preserved(self):
        """A single high dimension is the composite — no reduction."""
        scores = {"dim_a": 0.9, "dim_b": 0.0, "dim_c": 0.0}
        assert max_composite(scores) == pytest.approx(0.9)

    def test_max_wins(self):
        """Composite equals the highest dimension regardless of others."""
        assert max_composite({"a": 0.9, "b": 0.3, "c": 0.0}) == pytest.approx(0.9)
        assert max_composite({"a": 0.9, "b": 0.9, "c": 0.9}) == pytest.approx(0.9)


# ============================================================================
# Purpose-weighted composite (content family)
# ============================================================================


class TestPurposeWeightedComposite:
    """Tests for the purpose-weighted composite: max * purpose_multiplier."""

    def test_high_value_purpose(self):
        scores = {"a": 0.8, "b": 0.3}
        result = purpose_weighted_composite(scores, ContentPurpose.data_source)
        # max=0.8, multiplier=1.0 → 0.8
        assert result == pytest.approx(0.8, abs=0.01)

    def test_medium_value_purpose(self):
        scores = {"a": 0.8, "b": 0.3}
        result = purpose_weighted_composite(scores, ContentPurpose.tutorial)
        # max=0.8, multiplier=0.8 → 0.64
        assert result == pytest.approx(0.64, abs=0.01)

    def test_low_value_purpose(self):
        scores = {"a": 0.8, "b": 0.3}
        result = purpose_weighted_composite(scores, ContentPurpose.other)
        # max=0.8, multiplier=0.3 → 0.24
        assert result == pytest.approx(0.24, abs=0.01)

    def test_all_high_value_purposes(self):
        """All high-value purpose types get multiplier 1.0."""
        scores = {"a": 0.7}
        for purpose in [
            ContentPurpose.data_source,
            ContentPurpose.diagnostic,
            ContentPurpose.code,
            ContentPurpose.calibration,
            ContentPurpose.data_access,
        ]:
            assert purpose_weighted_composite(scores, purpose) == pytest.approx(
                0.7, abs=0.01
            )

    def test_all_medium_value_purposes(self):
        """All medium-value purpose types get multiplier 0.8."""
        scores = {"a": 1.0}
        for purpose in [
            ContentPurpose.physics_analysis,
            ContentPurpose.experimental_procedure,
            ContentPurpose.tutorial,
            ContentPurpose.reference,
        ]:
            assert purpose_weighted_composite(scores, purpose) == pytest.approx(
                0.8, abs=0.01
            )

    def test_empty_scores(self):
        assert purpose_weighted_composite({}, ContentPurpose.code) == 0.0

    def test_clamped_to_unit_range(self):
        scores = {"a": 1.5}
        result = purpose_weighted_composite(scores, ContentPurpose.data_source)
        assert result <= 1.0


# ============================================================================
# Dimension constants
# ============================================================================


class TestDimensionConstants:
    def test_code_dimensions_count(self):
        assert len(CODE_SCORE_DIMENSIONS) == 9

    def test_path_extra_dimensions_count(self):
        assert len(PATH_EXTRA_DIMENSIONS) == 2

    def test_path_dimensions_count(self):
        assert len(PATH_SCORE_DIMENSIONS) == 11

    def test_content_dimensions_count(self):
        assert len(CONTENT_SCORE_DIMENSIONS) == 6

    def test_path_includes_code_dimensions(self):
        for dim in CODE_SCORE_DIMENSIONS:
            assert dim in PATH_SCORE_DIMENSIONS

    def test_all_dimensions_start_with_score(self):
        for dim in CODE_SCORE_DIMENSIONS + CONTENT_SCORE_DIMENSIONS:
            assert dim.startswith("score_")

    def test_no_overlap_between_families(self):
        """Code and content families share only score_data_access."""
        overlap = set(CODE_SCORE_DIMENSIONS) & set(CONTENT_SCORE_DIMENSIONS)
        assert overlap == {"score_data_access"}


# ============================================================================
# CodeScoreFields base model
# ============================================================================


class TestCodeScoreFields:
    def test_all_dimensions_present(self):
        model = CodeScoreFields()
        for dim in CODE_SCORE_DIMENSIONS:
            assert hasattr(model, dim)
            assert getattr(model, dim) == 0.0

    def test_get_score_dict(self):
        model = CodeScoreFields(score_imas=0.9, score_data_access=0.5)
        d = model.get_score_dict()
        assert len(d) == 9
        assert d["score_imas"] == 0.9
        assert d["score_data_access"] == 0.5
        assert d["score_modeling_code"] == 0.0

    def test_schema_is_flat(self):
        """JSON schema must be flat (no $ref, no allOf) for LiteLLM."""
        schema = CodeScoreFields.model_json_schema()
        assert schema["type"] == "object"
        assert "$ref" not in str(schema)
        # All 9 dimensions in properties
        for dim in CODE_SCORE_DIMENSIONS:
            assert dim in schema["properties"]

    def test_inheritance_schema_flat(self):
        """Schema remains flat when a child model inherits CodeScoreFields."""

        class ChildModel(CodeScoreFields):
            path: str = Field(description="Test path")
            extra_field: int = Field(default=0)

        schema = ChildModel.model_json_schema()
        assert schema["type"] == "object"
        assert "$ref" not in str(schema)
        # Parent fields present
        for dim in CODE_SCORE_DIMENSIONS:
            assert dim in schema["properties"]
        # Child fields present
        assert "path" in schema["properties"]
        assert "extra_field" in schema["properties"]

    def test_default_composite(self):
        """Composite of all-zero fields produces 0."""
        model = CodeScoreFields()
        assert max_composite(model.get_score_dict()) == 0.0


# ============================================================================
# ContentScoreFields base model
# ============================================================================


class TestContentScoreFields:
    def test_all_dimensions_present(self):
        model = ContentScoreFields()
        for dim in CONTENT_SCORE_DIMENSIONS:
            assert hasattr(model, dim)
            assert getattr(model, dim) == 0.0

    def test_get_score_dict(self):
        model = ContentScoreFields(score_physics_content=0.7, score_calibration=0.4)
        d = model.get_score_dict()
        assert len(d) == 6
        assert d["score_physics_content"] == 0.7
        assert d["score_calibration"] == 0.4

    def test_schema_is_flat(self):
        schema = ContentScoreFields.model_json_schema()
        assert schema["type"] == "object"
        assert "$ref" not in str(schema)
        for dim in CONTENT_SCORE_DIMENSIONS:
            assert dim in schema["properties"]

    def test_inheritance_schema_flat(self):
        class ChildModel(ContentScoreFields):
            id: str = Field(description="Page ID")
            should_ingest: bool = Field(default=False)

        schema = ChildModel.model_json_schema()
        assert schema["type"] == "object"
        assert "$ref" not in str(schema)
        for dim in CONTENT_SCORE_DIMENSIONS:
            assert dim in schema["properties"]
        assert "id" in schema["properties"]
        assert "should_ingest" in schema["properties"]

    def test_composite_with_purpose(self):
        model = ContentScoreFields(score_data_documentation=0.8)
        result = purpose_weighted_composite(
            model.get_score_dict(), ContentPurpose.data_source
        )
        assert result == pytest.approx(0.8, abs=0.01)
