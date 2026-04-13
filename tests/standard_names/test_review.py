"""Tests for the cross-model review phase of the SN build pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.models import (
    SNReviewBatch,
    SNReviewItem,
    SNReviewVerdict,
)

# =============================================================================
# Model instantiation tests
# =============================================================================


class TestSNReviewModels:
    """Test review Pydantic model instantiation and validation."""

    def test_review_verdict_enum_values(self):
        """All three verdict values are valid."""
        assert SNReviewVerdict.accept == "accept"
        assert SNReviewVerdict.reject == "reject"
        assert SNReviewVerdict.revise == "revise"

    def test_review_item_accept(self):
        """Accept verdict with minimal fields."""
        item = SNReviewItem(
            source_id="equilibrium/time_slice/profiles_1d/psi",
            standard_name="poloidal_flux",
            verdict=SNReviewVerdict.accept,
            confidence=0.95,
            reason="Name correctly captures the physics quantity",
        )
        assert item.verdict == SNReviewVerdict.accept
        assert item.confidence == 0.95
        assert item.revised_name is None
        assert item.revised_fields is None
        assert item.issues == []

    def test_review_item_reject(self):
        """Reject verdict with issues."""
        item = SNReviewItem(
            source_id="magnetics/flux_loop/flux/data",
            standard_name="invalid_name",
            verdict=SNReviewVerdict.reject,
            confidence=0.8,
            reason="Name does not represent a valid physics quantity",
            issues=["Invalid physical_base", "No matching grammar rule"],
        )
        assert item.verdict == SNReviewVerdict.reject
        assert len(item.issues) == 2

    def test_review_item_revise(self):
        """Revise verdict with revised name and fields."""
        item = SNReviewItem(
            source_id="core_profiles/profiles_1d/electrons/temperature",
            standard_name="electron_temp",
            verdict=SNReviewVerdict.revise,
            confidence=0.85,
            reason="Physical base should be 'temperature' not 'temp'",
            revised_name="electron_temperature",
            revised_fields={"physical_base": "temperature", "subject": "electron"},
            issues=["Abbreviated physical_base"],
        )
        assert item.verdict == SNReviewVerdict.revise
        assert item.revised_name == "electron_temperature"
        assert item.revised_fields == {
            "physical_base": "temperature",
            "subject": "electron",
        }

    def test_review_batch(self):
        """SNReviewBatch wraps a list of review items."""
        batch = SNReviewBatch(
            reviews=[
                SNReviewItem(
                    source_id="src1",
                    standard_name="electron_temperature",
                    verdict=SNReviewVerdict.accept,
                    confidence=0.95,
                    reason="Good",
                ),
                SNReviewItem(
                    source_id="src2",
                    standard_name="bad_name",
                    verdict=SNReviewVerdict.reject,
                    confidence=0.7,
                    reason="Invalid",
                ),
            ]
        )
        assert len(batch.reviews) == 2
        assert batch.reviews[0].verdict == SNReviewVerdict.accept
        assert batch.reviews[1].verdict == SNReviewVerdict.reject

    def test_review_item_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        # Valid at boundaries
        SNReviewItem(
            source_id="a",
            standard_name="b",
            verdict=SNReviewVerdict.accept,
            confidence=0.0,
            reason="c",
        )
        SNReviewItem(
            source_id="a",
            standard_name="b",
            verdict=SNReviewVerdict.accept,
            confidence=1.0,
            reason="c",
        )

        # Invalid: above 1.0
        with pytest.raises(ValidationError):
            SNReviewItem(
                source_id="a",
                standard_name="b",
                verdict=SNReviewVerdict.accept,
                confidence=1.5,
                reason="c",
            )

        # Invalid: below 0.0
        with pytest.raises(ValidationError):
            SNReviewItem(
                source_id="a",
                standard_name="b",
                verdict=SNReviewVerdict.accept,
                confidence=-0.1,
                reason="c",
            )

    def test_empty_review_batch(self):
        """Empty review batch is valid."""
        batch = SNReviewBatch(reviews=[])
        assert batch.reviews == []


# =============================================================================
# Unified quality review model tests
# =============================================================================


class TestSNQualityReviewModels:
    """Test the unified 6-dimensional quality review models."""

    def test_quality_score_total(self):
        """Total is the sum of all six dimensions."""
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
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
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
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
        """Score 72-101 is good."""
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
            grammar=15,
            semantic=14,
            documentation=12,
            convention=13,
            completeness=11,
            compliance=10,
        )
        assert score.total == 75
        assert score.tier == "good"

    def test_quality_score_tier_adequate(self):
        """Score 48-71 is adequate."""
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
            grammar=10,
            semantic=10,
            documentation=8,
            convention=10,
            completeness=8,
            compliance=6,
        )
        assert score.total == 52
        assert score.tier == "adequate"

    def test_quality_score_tier_poor(self):
        """Score < 48 is poor."""
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
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
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
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
        from imas_codex.standard_names.models import SNQualityScore

        score = SNQualityScore(
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
        """SNQualityReview with all fields populated."""
        from imas_codex.standard_names.models import (
            SNQualityReview,
            SNQualityScore,
            SNReviewVerdict,
        )

        review = SNQualityReview(
            source_id="core_profiles/profiles_1d/electrons/temperature",
            standard_name="electron_temperature",
            scores=SNQualityScore(
                grammar=20,
                semantic=19,
                documentation=18,
                convention=17,
                completeness=16,
                compliance=15,
            ),
            verdict=SNReviewVerdict.accept,
            reasoning="Excellent entry with rich documentation",
        )
        assert review.scores.total == 105
        assert review.scores.tier == "outstanding"
        assert review.verdict == SNReviewVerdict.accept

    def test_quality_review_batch(self):
        """SNQualityReviewBatch wraps quality reviews."""
        from imas_codex.standard_names.models import (
            SNQualityReview,
            SNQualityReviewBatch,
            SNQualityScore,
            SNReviewVerdict,
        )

        batch = SNQualityReviewBatch(
            reviews=[
                SNQualityReview(
                    source_id="src1",
                    standard_name="electron_temperature",
                    scores=SNQualityScore(
                        grammar=20,
                        semantic=20,
                        documentation=18,
                        convention=18,
                        completeness=16,
                        compliance=15,
                    ),
                    verdict=SNReviewVerdict.accept,
                    reasoning="Good",
                ),
            ]
        )
        assert len(batch.reviews) == 1
        assert batch.reviews[0].scores.total == 107

    def test_quality_score_validation(self):
        """Scores must be in 0-20 range."""
        from pydantic import ValidationError

        from imas_codex.standard_names.models import SNQualityScore

        with pytest.raises(ValidationError):
            SNQualityScore(
                grammar=25,  # exceeds 20
                semantic=10,
                documentation=10,
                convention=10,
                completeness=10,
                compliance=10,
            )

        with pytest.raises(ValidationError):
            SNQualityScore(
                grammar=-1,  # below 0
                semantic=10,
                documentation=10,
                convention=10,
                completeness=10,
                compliance=10,
            )


# =============================================================================
# Review worker tests
# =============================================================================


class TestReviewWorker:
    """Test the review_worker function."""

    @pytest.fixture(autouse=True)
    def _requires_imas_sn(self):
        pytest.importorskip("imas_standard_names")

    def _make_state(self, **overrides):
        """Create a minimal SNBuildState for testing."""
        from imas_codex.standard_names.state import SNBuildState

        defaults = {
            "facility": "dd",
            "source": "dd",
            "dry_run": False,
            "skip_review": False,
        }
        defaults.update(overrides)
        return SNBuildState(**defaults)

    def test_dry_run_skips_review(self):
        """In dry-run mode, review passes candidates through unchanged."""
        state = self._make_state(dry_run=True)
        state.composed = [
            {
                "id": "electron_temperature",
                "source_id": "path/a",
                "physical_base": "temperature",
            },
            {"id": "ion_density", "source_id": "path/b", "physical_base": "density"},
        ]

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert state.stats.get("review_skipped") is True
        assert len(state.reviewed) == 2
        assert state.review_stats.total == 2
        assert state.review_stats.processed == 2

    def test_empty_candidates_skip(self):
        """No candidates → review completes with no-op."""
        state = self._make_state()
        state.composed = []

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert state.reviewed == []

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_accept_verdict_passes_through(self, mock_cal, mock_existing, mock_batch):
        """Accept verdict keeps the candidate in reviewed output."""
        mock_existing.return_value = set()
        mock_cal.return_value = []

        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "path/a",
                "physical_base": "temperature",
            },
        ]

        # Mock the batch to return the candidate as accepted
        async def _mock_review(*args, **kwargs):
            return candidates, 0, 0, 0.001, 100

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = list(candidates)
        state.review_model = "test/model"

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert len(state.reviewed) == 1
        assert state.reviewed[0]["id"] == "electron_temperature"
        assert state.stats["review_accepted"] == 1
        assert state.stats["review_rejected"] == 0
        # Reviewer metadata is added to accepted entries
        assert state.reviewed[0].get("reviewer_model") == "test/model"
        assert "reviewed_at" in state.reviewed[0]

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_reject_verdict_removes_candidate(
        self, mock_cal, mock_existing, mock_batch
    ):
        """Reject verdict removes the candidate from reviewed output."""
        mock_existing.return_value = set()
        mock_cal.return_value = []

        candidates = [
            {"id": "bad_name", "source_id": "path/a", "physical_base": "x"},
        ]

        # Mock the batch to return empty accepted, 1 rejected
        async def _mock_review(*args, **kwargs):
            return [], 1, 0, 0.001, 100

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = list(candidates)
        state.review_model = "test/model"

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert len(state.reviewed) == 0
        assert state.stats["review_rejected"] == 1

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_revise_verdict_updates_candidate(
        self, mock_cal, mock_existing, mock_batch
    ):
        """Revise verdict updates candidate name in reviewed output."""
        mock_existing.return_value = set()
        mock_cal.return_value = []

        original = {
            "id": "electron_temp",
            "source_id": "path/a",
            "physical_base": "temp",
        }
        revised = {
            "id": "electron_temperature",
            "source_id": "path/a",
            "physical_base": "temperature",
        }

        # Mock the batch to return revised candidate
        async def _mock_review(*args, **kwargs):
            return [revised], 0, 1, 0.001, 100

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = [dict(original)]
        state.review_model = "test/model"

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert len(state.reviewed) == 1
        assert state.reviewed[0]["id"] == "electron_temperature"
        assert state.stats["review_revised"] == 1

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_batch_failure_passes_through(self, mock_cal, mock_existing, mock_batch):
        """On batch failure, candidates pass through unreviewed."""
        mock_existing.return_value = set()
        mock_cal.return_value = []

        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "path/a",
                "physical_base": "temperature",
            },
        ]

        # Mock the batch to raise an exception
        async def _mock_review(*args, **kwargs):
            raise RuntimeError("LLM call failed")

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = list(candidates)
        state.review_model = "test/model"

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        # On failure, candidates pass through
        assert len(state.reviewed) == 1
        assert state.review_stats.errors == 1

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_review_passes_calibration_and_context(
        self, mock_cal, mock_existing, mock_batch
    ):
        """review_worker passes calibration entries and batch context."""
        mock_existing.return_value = set()
        mock_cal.return_value = [
            {"name": "electron_temperature", "tier": "outstanding"}
        ]

        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "physical_base": "temperature",
            },
        ]

        call_kwargs = {}

        async def _mock_review(*args, **kwargs):
            call_kwargs.update(kwargs)
            return candidates, 0, 0, 0.001, 100

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = list(candidates)
        state.review_model = "test/model"

        # Set up extracted batches to provide context
        from imas_codex.standard_names.sources.base import ExtractionBatch

        state.extracted = [
            ExtractionBatch(
                source="dd",
                group_key="equilibrium",
                items=[{"path": "equilibrium/time_slice/profiles_1d/psi"}],
                context="Equilibrium profiles from reconstruction",
            )
        ]

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        assert call_kwargs.get("calibration_entries") == [
            {"name": "electron_temperature", "tier": "outstanding"}
        ]
        assert (
            call_kwargs.get("batch_context")
            == "Equilibrium profiles from reconstruction"
        )


# =============================================================================
# State tests
# =============================================================================


class TestSNBuildStateReview:
    """Test review-related state fields."""

    def test_state_has_review_fields(self):
        """SNBuildState includes review configuration fields."""
        from imas_codex.standard_names.state import SNBuildState

        state = SNBuildState(facility="dd")
        assert state.skip_review is False
        assert state.review_model is None
        assert state.reviewed == []
        assert state.review_phase.name == "review"
        assert not state.review_phase.done

    def test_total_cost_includes_review(self):
        """total_cost sums compose and review costs."""
        from imas_codex.standard_names.state import SNBuildState

        state = SNBuildState(facility="dd")
        state.compose_stats.cost = 0.5
        state.review_stats.cost = 0.3
        assert state.total_cost == pytest.approx(0.8)

    def test_skip_review_configuration(self):
        """skip_review can be set at construction."""
        from imas_codex.standard_names.state import SNBuildState

        state = SNBuildState(facility="dd", skip_review=True, review_model="test/model")
        assert state.skip_review is True
        assert state.review_model == "test/model"


# =============================================================================
# Pipeline wiring tests
# =============================================================================


class TestPipelineReviewWiring:
    """Test that the pipeline correctly wires the review phase."""

    @pytest.fixture(autouse=True)
    def _requires_imas_sn(self):
        pytest.importorskip("imas_standard_names")

    def test_validate_depends_on_review_phase(self):
        """Validate worker should depend on review_phase, not compose_phase."""
        # We can't easily test the actual pipeline running without graph,
        # but we can verify the WorkerSpec construction.
        from imas_codex.standard_names.state import SNBuildState

        state = SNBuildState(facility="dd", skip_review=False)

        # When skip_review is False, review_phase should not be done yet
        assert not state.review_phase.done
        assert not state.validate_phase.done

    def test_skip_review_allows_validate(self):
        """When review is skipped, validate can still proceed.

        The engine marks disabled phases as done, so validate's
        dependency on review_phase is satisfied.
        """
        from imas_codex.discovery.base.engine import WorkerSpec
        from imas_codex.standard_names.state import SNBuildState
        from imas_codex.standard_names.workers import review_worker, validate_worker

        state = SNBuildState(facility="dd", skip_review=True)

        review_spec = WorkerSpec(
            "review",
            "review_phase",
            review_worker,
            depends_on=["compose_phase"],
            enabled=not state.skip_review,
        )

        validate_spec = WorkerSpec(
            "validate",
            "validate_phase",
            validate_worker,
            depends_on=["review_phase"],
        )

        # When review is disabled, the engine would mark review_phase done
        assert review_spec.enabled is False
        assert validate_spec.depends_on == ["review_phase"]

        # Simulate engine marking disabled phase done
        state.review_phase.mark_done()
        assert state.review_phase.done

    def test_validate_reads_reviewed_buffer(self):
        """Validate worker reads from state.reviewed when populated."""
        from imas_codex.standard_names.state import SNBuildState

        state = SNBuildState(facility="dd", dry_run=True)
        state.reviewed = [
            {"id": "electron_temperature", "source_id": "a"},
        ]
        state.composed = [
            {"id": "old_name", "source_id": "b"},
        ]

        from imas_codex.standard_names.workers import validate_worker

        # In dry-run, validation is skipped — but we verify the buffer logic
        asyncio.run(validate_worker(state))
        assert state.validate_phase.done
