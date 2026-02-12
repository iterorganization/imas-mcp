"""Tests for directory scorer, especially container expansion logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.paths.models import ResourcePurpose, ScoreBatch, ScoreResult
from imas_codex.discovery.paths.scorer import (
    CONTAINER_PURPOSES,
    CONTAINER_THRESHOLD,
    DirectoryScorer,
    grounded_score,
)


class TestGroundedScore:
    """Tests for the grounded_score() function."""

    def test_max_of_dimensions(self):
        """Score is max of all dimension scores."""
        scores = {
            "score_modeling_code": 0.0,
            "score_analysis_code": 0.8,
            "score_operations_code": 0.0,
            "score_modeling_data": 0.0,
            "score_experimental_data": 0.0,
            "score_data_access": 0.0,
            "score_workflow": 0.0,
            "score_visualization": 0.0,
            "score_documentation": 0.0,
            "score_imas": 0.0,
        }
        result = grounded_score(scores, {}, ResourcePurpose.analysis_code)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_container_with_zero_scores(self):
        """Container with all-zero scores gets 0.0."""
        scores = dict.fromkeys(
            [
                "score_modeling_code",
                "score_analysis_code",
                "score_operations_code",
                "score_modeling_data",
                "score_experimental_data",
                "score_data_access",
                "score_workflow",
                "score_visualization",
                "score_documentation",
                "score_imas",
            ],
            0.0,
        )
        result = grounded_score(scores, {}, ResourcePurpose.container)
        assert result == 0.0

    def test_suppressed_purpose_gets_penalized(self):
        """System directories get 0.3 multiplier."""
        scores = {"score_modeling_code": 0.5}
        result = grounded_score(scores, {}, ResourcePurpose.system)
        assert result < 0.5 * 0.4  # 0.3 multiplier


class TestContainerExpansion:
    """Test that containers with many subdirectories are always expanded.

    This was the root cause of the TCV /home directory not being scanned:
    /home was classified as 'container' with all 0.0 per-dimension scores,
    giving grounded_score=0.0 < CONTAINER_THRESHOLD=0.1, so should_expand
    was set to False and the 1037 user home directories were never discovered.
    """

    def _make_score_result(
        self,
        path: str = "/home",
        purpose: ResourcePurpose = ResourcePurpose.container,
        should_expand: bool = False,
        should_enrich: bool = False,
        **score_overrides,
    ) -> ScoreResult:
        """Create a ScoreResult with default zero scores."""
        return ScoreResult(
            path=path,
            path_purpose=purpose,
            description="Test directory",
            should_expand=should_expand,
            should_enrich=should_enrich,
            **score_overrides,
        )

    def _score_batch(self, results: list[ScoreResult]) -> ScoreBatch:
        return ScoreBatch(results=results)

    def test_container_with_many_dirs_always_expanded(self):
        """Container with total_dirs >= 5 should always expand."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result("/home", should_expand=False),
            ]
        )
        directories = [
            {
                "path": "/home",
                "total_files": 0,
                "total_dirs": 1037,  # Many user home dirs
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        assert results[0].should_expand is True

    def test_container_with_few_dirs_respects_llm(self):
        """Container with < 5 dirs respects LLM should_expand decision."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result("/tmp/few", should_expand=False),
            ]
        )
        directories = [
            {
                "path": "/tmp/few",
                "total_files": 2,
                "total_dirs": 3,  # Few dirs
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # With 0.0 scores and should_expand=False from LLM, stays False
        assert results[0].should_expand is False

    def test_container_with_git_not_expanded(self):
        """Container with .git should not be expanded (git override wins)."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result("/home/user/repo", should_expand=True),
            ]
        )
        directories = [
            {
                "path": "/home/user/repo",
                "total_files": 50,
                "total_dirs": 10,
                "has_readme": True,
                "has_makefile": True,
                "has_git": True,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # Git repos should never be expanded even if container
        assert results[0].should_expand is False

    def test_container_home_with_zero_scores_expanded(self):
        """/home with all 0.0 scores but 1037 dirs MUST expand.

        This is the exact scenario from the TCV bug: LLM gave /home all
        0.0 per-dimension scores, grounded_score was 0.0, which was below
        CONTAINER_THRESHOLD (0.1), so should_expand was False.
        """
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/home",
                    should_expand=False,  # LLM said no
                    score_modeling_code=0.0,
                    score_analysis_code=0.0,
                    score_operations_code=0.0,
                    score_modeling_data=0.0,
                    score_experimental_data=0.0,
                    score_data_access=0.0,
                    score_workflow=0.0,
                    score_visualization=0.0,
                    score_documentation=0.0,
                    score_imas=0.0,
                ),
            ]
        )
        directories = [
            {
                "path": "/home",
                "total_files": 4,
                "total_dirs": 1037,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # MUST expand despite 0.0 score and LLM saying should_expand=False
        assert results[0].should_expand is True

    def test_non_container_with_many_dirs_respects_threshold(self):
        """Non-container purposes with many dirs still respect score threshold."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/work/simulation_runs",
                    purpose=ResourcePurpose.modeling_data,
                    should_expand=True,
                ),
            ]
        )
        directories = [
            {
                "path": "/work/simulation_runs",
                "total_files": 0,
                "total_dirs": 500,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # modeling_data directories are explicitly blocked from expansion
        assert results[0].should_expand is False
