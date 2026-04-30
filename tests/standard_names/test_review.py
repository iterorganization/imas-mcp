"""Tests for the cross-model review phase of the SN build pipeline."""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.models import (
    StandardNameReviewBatch,
    StandardNameReviewItem,
    StandardNameReviewVerdict,
)

# =============================================================================
# Model instantiation tests
# =============================================================================


class TestSNReviewModels:
    """Test review Pydantic model instantiation and validation."""

    def test_review_verdict_enum_values(self):
        """All three verdict values are valid."""
        assert StandardNameReviewVerdict.accept == "accept"
        assert StandardNameReviewVerdict.reject == "reject"
        assert StandardNameReviewVerdict.revise == "revise"

    def test_review_item_accept(self):
        """Accept verdict with minimal fields."""
        item = StandardNameReviewItem(
            source_id="equilibrium/time_slice/profiles_1d/psi",
            standard_name="poloidal_flux",
            verdict=StandardNameReviewVerdict.accept,
            reason="Name correctly captures the physics quantity",
        )
        assert item.verdict == StandardNameReviewVerdict.accept
        assert item.revised_name is None
        assert item.revised_fields is None
        assert item.issues == []

    def test_review_item_reject(self):
        """Reject verdict with issues."""
        item = StandardNameReviewItem(
            source_id="magnetics/flux_loop/flux/data",
            standard_name="invalid_name",
            verdict=StandardNameReviewVerdict.reject,
            reason="Name does not represent a valid physics quantity",
            issues=["Invalid physical_base", "No matching grammar rule"],
        )
        assert item.verdict == StandardNameReviewVerdict.reject
        assert len(item.issues) == 2

    def test_review_item_revise(self):
        """Revise verdict with revised name and fields."""
        item = StandardNameReviewItem(
            source_id="core_profiles/profiles_1d/electrons/temperature",
            standard_name="electron_temp",
            verdict=StandardNameReviewVerdict.revise,
            reason="Physical base should be 'temperature' not 'temp'",
            revised_name="electron_temperature",
            revised_fields={"physical_base": "temperature", "subject": "electron"},
            issues=["Abbreviated physical_base"],
        )
        assert item.verdict == StandardNameReviewVerdict.revise
        assert item.revised_name == "electron_temperature"
        assert item.revised_fields == {
            "physical_base": "temperature",
            "subject": "electron",
        }

    def test_review_batch(self):
        """StandardNameReviewBatch wraps a list of review items."""
        batch = StandardNameReviewBatch(
            reviews=[
                StandardNameReviewItem(
                    source_id="src1",
                    standard_name="electron_temperature",
                    verdict=StandardNameReviewVerdict.accept,
                    reason="Good",
                ),
                StandardNameReviewItem(
                    source_id="src2",
                    standard_name="bad_name",
                    verdict=StandardNameReviewVerdict.reject,
                    reason="Invalid",
                ),
            ]
        )
        assert len(batch.reviews) == 2
        assert batch.reviews[0].verdict == StandardNameReviewVerdict.accept
        assert batch.reviews[1].verdict == StandardNameReviewVerdict.reject

    def test_empty_review_batch(self):
        """Empty review batch is valid."""
        batch = StandardNameReviewBatch(reviews=[])
        assert batch.reviews == []


# =============================================================================
# Unified quality review model tests
# =============================================================================


class TestSNQualityReviewModels:
    """Test the unified 6-dimensional quality review models."""

    def test_quality_score_total(self):
        """Total is the sum of all six dimensions."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=18,
            semantic=16,
            documentation=14,
            convention=17,
            completeness=15,
            compliance=12,
        )
        assert score.total == 92

    def test_quality_score_tier_outstanding(self):
        """Score >= 102 is outstanding."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=20,
            semantic=19,
            documentation=18,
            convention=17,
            completeness=16,
            compliance=15,
        )
        assert score.total == 105
        assert score.tier == "outstanding"

    def test_quality_score_tier_good(self):
        """Score 78-101 is good."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=15,
            semantic=14,
            documentation=12,
            convention=13,
            completeness=11,
            compliance=13,
        )
        assert score.total == 78
        assert score.tier == "good"

    def test_quality_score_tier_adequate(self):
        """Score 48-77 is inadequate."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=10,
            semantic=10,
            documentation=8,
            convention=10,
            completeness=8,
            compliance=6,
        )
        assert score.total == 52
        assert score.tier == "inadequate"

    def test_quality_score_tier_poor(self):
        """Score < 48 is poor."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=5,
            semantic=5,
            documentation=3,
            convention=3,
            completeness=2,
            compliance=2,
        )
        assert score.total == 20
        assert score.tier == "poor"

    def test_quality_score_max_120(self):
        """Max possible total is 120."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=20,
            semantic=20,
            documentation=20,
            convention=20,
            completeness=20,
            compliance=20,
        )
        assert score.total == 120
        assert score.tier == "outstanding"

    def test_quality_score_model_dump(self):
        """model_dump() includes all dimension fields."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
            grammar=18,
            semantic=16,
            documentation=14,
            convention=17,
            completeness=15,
            compliance=12,
        )
        d = score.model_dump()
        assert set(d.keys()) == {
            "grammar",
            "semantic",
            "documentation",
            "convention",
            "completeness",
            "compliance",
        }

    def test_quality_review_full(self):
        """StandardNameQualityReview with all fields populated."""
        from imas_codex.standard_names.models import (
            StandardNameQualityReview,
            StandardNameQualityScore,
            StandardNameReviewVerdict,
        )

        review = StandardNameQualityReview(
            source_id="core_profiles/profiles_1d/electrons/temperature",
            standard_name="electron_temperature",
            scores=StandardNameQualityScore(
                grammar=20,
                semantic=19,
                documentation=18,
                convention=17,
                completeness=16,
                compliance=15,
            ),
            verdict=StandardNameReviewVerdict.accept,
            reasoning="Excellent entry with rich documentation",
        )
        assert review.scores.total == 105
        assert review.scores.tier == "outstanding"
        assert review.verdict == StandardNameReviewVerdict.accept

    def test_quality_review_batch(self):
        """StandardNameQualityReviewBatch wraps quality reviews."""
        from imas_codex.standard_names.models import (
            StandardNameQualityReview,
            StandardNameQualityReviewBatch,
            StandardNameQualityScore,
            StandardNameReviewVerdict,
        )

        batch = StandardNameQualityReviewBatch(
            reviews=[
                StandardNameQualityReview(
                    source_id="src1",
                    standard_name="electron_temperature",
                    scores=StandardNameQualityScore(
                        grammar=20,
                        semantic=20,
                        documentation=18,
                        convention=18,
                        completeness=16,
                        compliance=15,
                    ),
                    verdict=StandardNameReviewVerdict.accept,
                    reasoning="Good",
                ),
            ]
        )
        assert len(batch.reviews) == 1
        assert batch.reviews[0].scores.total == 107

    def test_quality_score_validation(self):
        """Scores must be in 0-20 range."""
        from pydantic import ValidationError

        from imas_codex.standard_names.models import StandardNameQualityScore

        with pytest.raises(ValidationError):
            StandardNameQualityScore(
                grammar=25,  # exceeds 20
                semantic=10,
                documentation=10,
                convention=10,
                completeness=10,
                compliance=10,
            )

        with pytest.raises(ValidationError):
            StandardNameQualityScore(
                grammar=-1,  # below 0
                semantic=10,
                documentation=10,
                convention=10,
                completeness=10,
                compliance=10,
            )


# =============================================================================
# State tests
# =============================================================================


class TestSNBuildStateReview:
    """Test review-related state fields after pipeline refactor.

    Review has been removed from the generate pipeline. These tests
    verify the state still works correctly without review fields.
    """

    def test_state_has_compose_model(self):
        """StandardNameBuildState includes compose model configuration."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd")
        assert state.compose_model is None

    def test_total_cost_is_compose_only(self):
        """total_cost only includes compose cost (no review)."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd")
        state.compose_stats.cost = 0.5
        assert state.total_cost == pytest.approx(0.5)

    def test_model_override_configuration(self):
        """compose_model can be set at construction."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(
            facility="dd",
            compose_model="test/compose",
        )
        assert state.compose_model == "test/compose"


# =============================================================================
# Pipeline wiring tests
# =============================================================================


class TestPipelineWiring:
    """Test that the pipeline correctly wires phases (no review in generate)."""

    @pytest.fixture(autouse=True)
    def _requires_imas_sn(self):
        pytest.importorskip("imas_standard_names")

    def test_validate_depends_on_compose_phase(self):
        """Validate worker should depend on compose_phase (review removed)."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd")

        assert not state.compose_phase.done
        assert not state.validate_phase.done

    def test_validate_reads_composed_buffer(self):
        """Validate worker reads from state.composed directly."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd", dry_run=True)
        state.composed = [
            {"id": "electron_temperature", "source_id": "a"},
        ]

        from imas_codex.standard_names.workers import validate_worker

        asyncio.run(validate_worker(state))
        assert state.validate_phase.done

    def test_pipeline_has_no_review_step(self):
        """Pool adapter should not include a review worker."""
        import importlib

        mod = importlib.import_module("imas_codex.standard_names.pool_adapter")
        source = importlib.util.find_spec(mod.__name__).origin
        with open(source) as f:
            content = f.read()

        assert "review_worker" not in content
        assert "review_phase" not in content
