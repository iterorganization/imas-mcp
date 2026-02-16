"""Tests for directory scorer, especially container expansion logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.paths.models import ResourcePurpose, ScoreBatch, ScoreResult
from imas_codex.discovery.paths.scorer import (
    CONTAINER_PURPOSES,
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

    def test_score_capped_at_1(self):
        """Score with quality boosts is capped at 1.0."""
        scores = {
            "score_modeling_code": 0.95,
            "score_imas": 0.5,  # trigger IMAS boost (+0.10)
        }
        input_data = {
            "has_readme": True,  # +0.05
            "has_makefile": True,  # +0.05
            "has_git": True,  # +0.05
        }
        result = grounded_score(scores, input_data, ResourcePurpose.modeling_code)
        # base=0.95 + boosts=0.25 = 1.20 → capped at 1.0
        assert result == 1.0


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

    def test_non_root_container_requires_minimum_score(self):
        """Non-root containers need score >= 0.3 to expand, even if LLM says yes."""
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
        # Does NOT expand — score 0.0 < 0.3 threshold for non-root containers
        assert results[0].should_expand is False

    def test_non_container_requires_score_threshold(self):
        """Non-container paths must meet score threshold to expand."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/work/codes/analysis",
                    purpose=ResourcePurpose.analysis_code,
                    should_expand=True,
                    score_analysis_code=0.3,  # Below 0.7 threshold
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
        # LLM said expand, but score 0.3 < threshold 0.7 → no expand
        assert results[0].should_expand is False

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
