"""Tests for directory scorer, especially container expansion logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.paths.models import ResourcePurpose, ScoreBatch, ScoreResult
from imas_codex.discovery.paths.scorer import (
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

    def test_suppressed_purpose_no_penalty(self):
        """System directories get no penalty — LLM scores them low via prompt."""
        scores = {"score_modeling_code": 0.5}
        result = grounded_score(scores, {}, ResourcePurpose.system)
        # No multiplier — score is just max(scores)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_score_is_max_of_dimensions(self):
        """Score is simply max of dimension scores, no boosts or caps."""
        scores = {
            "score_modeling_code": 0.95,
            "score_imas": 0.5,
        }
        input_data = {
            "has_readme": True,
            "has_makefile": True,
            "has_git": True,
        }
        result = grounded_score(scores, input_data, ResourcePurpose.modeling_code)
        # No boosts — score is just max(0.95, 0.5) = 0.95
        assert result == pytest.approx(0.95, abs=0.01)


class TestContainerExpansion:
    """Test that container expansion is driven by the LLM's should_expand decision.

    The scorer prompt instructs the LLM to set should_expand=true for containers
    like /home, /work, etc. The code trusts this decision for containers without
    requiring a score threshold, unlike non-container paths which must also meet
    a minimum score.
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

    def test_container_expands_when_llm_says_yes(self):
        """Container expands when LLM sets should_expand=true, even with 0 scores."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result("/home", should_expand=True),
            ]
        )
        directories = [
            {
                "path": "/home",
                "total_files": 0,
                "total_dirs": 1037,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        assert results[0].should_expand is True

    def test_container_respects_llm_no_expand(self):
        """Container does not expand when LLM sets should_expand=false."""
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
                "total_dirs": 3,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
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

    def test_container_no_score_threshold_required(self):
        """Containers bypass score threshold — only LLM should_expand matters.

        Non-containers require score >= threshold AND should_expand=true.
        Containers only require should_expand=true (the prompt guides this).
        """
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/home",
                    should_expand=True,  # Prompt-guided decision
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
        # Expands because LLM said yes — root containers always expand
        assert results[0].should_expand is True

    def test_non_root_container_expands_when_llm_says_yes(self):
        """Non-root containers expand when LLM says yes — no minimum score."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/home/empty_user",
                    should_expand=True,  # LLM says yes
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
                "path": "/home/empty_user",
                "depth": 1,  # Not root
                "total_files": 0,
                "total_dirs": 2,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # LLM decision is trusted directly — no minimum score for containers
        assert results[0].should_expand is True

    def test_non_container_trusts_llm_expand(self):
        """Non-container paths expand when LLM says yes — no score threshold."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/work/codes/analysis",
                    purpose=ResourcePurpose.analysis_code,
                    should_expand=True,
                    score_analysis_code=0.3,  # Below old 0.7 threshold
                ),
            ]
        )
        directories = [
            {
                "path": "/work/codes/analysis",
                "total_files": 10,
                "total_dirs": 5,
                "has_readme": False,
                "has_makefile": False,
                "has_git": False,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # LLM said expand — trusted directly regardless of score vs threshold
        assert results[0].should_expand is True

    def test_non_container_data_never_expands(self):
        """Data containers are blocked from expansion regardless."""
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
