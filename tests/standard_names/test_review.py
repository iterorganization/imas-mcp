"""Tests for the cross-model review phase of the SN build pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

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
            confidence=0.95,
            reason="Name correctly captures the physics quantity",
        )
        assert item.verdict == StandardNameReviewVerdict.accept
        assert item.confidence == 0.95
        assert item.revised_name is None
        assert item.revised_fields is None
        assert item.issues == []

    def test_review_item_reject(self):
        """Reject verdict with issues."""
        item = StandardNameReviewItem(
            source_id="magnetics/flux_loop/flux/data",
            standard_name="invalid_name",
            verdict=StandardNameReviewVerdict.reject,
            confidence=0.8,
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
            confidence=0.85,
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
                    confidence=0.95,
                    reason="Good",
                ),
                StandardNameReviewItem(
                    source_id="src2",
                    standard_name="bad_name",
                    verdict=StandardNameReviewVerdict.reject,
                    confidence=0.7,
                    reason="Invalid",
                ),
            ]
        )
        assert len(batch.reviews) == 2
        assert batch.reviews[0].verdict == StandardNameReviewVerdict.accept
        assert batch.reviews[1].verdict == StandardNameReviewVerdict.reject

    def test_review_item_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        # Valid at boundaries
        StandardNameReviewItem(
            source_id="a",
            standard_name="b",
            verdict=StandardNameReviewVerdict.accept,
            confidence=0.0,
            reason="c",
        )
        StandardNameReviewItem(
            source_id="a",
            standard_name="b",
            verdict=StandardNameReviewVerdict.accept,
            confidence=1.0,
            reason="c",
        )

        # Invalid: above 1.0
        with pytest.raises(ValidationError):
            StandardNameReviewItem(
                source_id="a",
                standard_name="b",
                verdict=StandardNameReviewVerdict.accept,
                confidence=1.5,
                reason="c",
            )

        # Invalid: below 0.0
        with pytest.raises(ValidationError):
            StandardNameReviewItem(
                source_id="a",
                standard_name="b",
                verdict=StandardNameReviewVerdict.accept,
                confidence=-0.1,
                reason="c",
            )

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
        """Score 72-101 is good."""
        from imas_codex.standard_names.models import StandardNameQualityScore

        score = StandardNameQualityScore(
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
        assert score.tier == "adequate"

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
# Review worker tests
# =============================================================================


class TestReviewWorker:
    """Test the review_worker function (standalone, not part of generate pipeline).

    The review_worker function exists for future ``sn review`` CLI tool.
    These tests use a mock state with the required attributes.
    """

    @pytest.fixture(autouse=True)
    def _requires_imas_sn(self):
        pytest.importorskip("imas_standard_names")

    def _make_state(self, **overrides):
        """Create a minimal state with review-specific fields for testing."""
        from imas_codex.discovery.base.progress import WorkerStats
        from imas_codex.discovery.base.supervision import PipelinePhase
        from imas_codex.standard_names.state import StandardNameBuildState

        defaults = {
            "facility": "dd",
            "source": "dd",
            "dry_run": False,
        }
        defaults.update(overrides)
        state = StandardNameBuildState(**defaults)
        # Add review-specific fields that were removed from generate pipeline
        state.reviewed = None
        state.review_stats = WorkerStats()
        state.review_phase = PipelinePhase("review")
        state.review_model = None
        return state

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
        assert state.stats["review_scored"] == 1
        # Reviewer metadata is added to scored entries
        assert state.reviewed[0].get("reviewer_model") == "test/model"
        assert "reviewed_at" in state.reviewed[0]

    @patch("imas_codex.standard_names.workers._review_batch")
    @patch("imas_codex.standard_names.workers._get_existing_names_for_review")
    @patch("imas_codex.standard_names.workers._load_calibration_entries")
    def test_all_names_scored_no_rejection(self, mock_cal, mock_existing, mock_batch):
        """All names are scored and persisted — no rejection step."""
        mock_existing.return_value = set()
        mock_cal.return_value = []

        candidates = [
            {"id": "low_score_name", "source_id": "path/a", "physical_base": "x"},
        ]

        # Mock the batch to return the candidate scored (even if low quality)
        async def _mock_review(*args, **kwargs):
            scored = list(args[0])
            for c in scored:
                c["reviewer_score"] = 0.3  # Low score but still passes through
            return scored, 0, 0, 0.001, 100

        mock_batch.side_effect = _mock_review

        state = self._make_state()
        state.composed = list(candidates)
        state.review_model = "test/model"

        from imas_codex.standard_names.workers import review_worker

        asyncio.run(review_worker(state))

        assert state.review_phase.done
        # Low-scoring name still passes through — no rejection
        assert len(state.reviewed) == 1
        assert state.reviewed[0]["id"] == "low_score_name"
        assert state.stats["review_scored"] == 1

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
        """Generate pipeline should not include a review worker."""
        import importlib

        mod = importlib.import_module("imas_codex.standard_names.pipeline")
        source = importlib.util.find_spec(mod.__name__).origin
        with open(source) as f:
            content = f.read()

        assert "review_worker" not in content
        assert "review_phase" not in content
