"""Tests for directory scorer, especially container expansion logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.paths.models import ResourcePurpose, ScoreBatch, ScoreResult
from imas_codex.discovery.paths.scorer import (
    DirectoryScorer,
    combined_score,
)


class TestGroundedScore:
    """Tests for the combined_score() function."""

    def test_max_of_dimensions(self):
        """Score uses breadth-weighted formula: max × (1 + mean_nonzero) / 2."""
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
        result = combined_score(scores, {}, ResourcePurpose.analysis_code)
        # Single-dim: 0.8 × (1 + 0.8) / 2 = 0.72
        assert result == pytest.approx(0.72, abs=0.01)

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
        result = combined_score(scores, {}, ResourcePurpose.container)
        assert result == 0.0

    def test_suppressed_purpose_no_penalty(self):
        """System directories get no penalty — LLM scores them low via prompt."""
        scores = {"score_modeling_code": 0.5}
        result = combined_score(scores, {}, ResourcePurpose.system)
        # Single-dim: 0.5 × (1 + 0.5) / 2 = 0.375
        assert result == pytest.approx(0.375, abs=0.01)

    def test_score_rewards_breadth(self):
        """Score rewards paths that excel across multiple dimensions."""
        scores = {
            "score_modeling_code": 0.95,
            "score_imas": 0.5,
        }
        input_data = {
            "has_readme": True,
            "has_makefile": True,
            "has_git": True,
        }
        result = combined_score(scores, input_data, ResourcePurpose.modeling_code)
        # Multi-dim: 0.95 × (1 + 0.725) / 2 = 0.819
        assert result == pytest.approx(0.819, abs=0.01)

    def test_single_outlier_reduced(self):
        """A single high dimension with all others zero is significantly reduced."""
        scores = {
            "score_modeling_code": 0.9,
            "score_analysis_code": 0.0,
            "score_data_access": 0.0,
        }
        result = combined_score(scores, {}, ResourcePurpose.modeling_code)
        # 0.9 × (1 + 0.9) / 2 = 0.855 (less than pure max of 0.9)
        assert result == pytest.approx(0.855, abs=0.01)
        assert result < 0.9  # Must be less than the pure max


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

    def test_container_with_git_passes_through_llm_decision(self):
        """Container with .git passes through LLM's should_expand decision.

        VCS accessibility overrides are applied in mark_paths_scored (frontier),
        not in the scorer. The scorer only computes scores.
        """
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
        # Scorer passes through LLM decision; VCS override is in frontier
        assert results[0].should_expand is True

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

    def test_data_container_passes_through_llm_decision(self):
        """Data containers pass through LLM decision (override in frontier)."""
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
        # Scorer passes through LLM decision; data container override is in frontier
        assert results[0].should_expand is True

    @pytest.mark.parametrize("vcs_type", ["svn", "hg", "bzr"])
    def test_non_git_vcs_repos_pass_through_llm_decision(self, vcs_type):
        """SVN/Hg/Bzr repos pass through LLM decision (override in frontier)."""
        scorer = DirectoryScorer(facility="test")
        batch = self._score_batch(
            [
                self._make_score_result(
                    "/work/codes/legacy_code",
                    purpose=ResourcePurpose.modeling_code,
                    should_expand=True,
                    score_modeling_code=0.9,
                ),
            ]
        )
        directories = [
            {
                "path": "/work/codes/legacy_code",
                "total_files": 50,
                "total_dirs": 10,
                "has_readme": True,
                "has_makefile": True,
                "has_git": False,
                "vcs_type": vcs_type,
            }
        ]

        results = scorer._map_scored_directories(batch, directories, threshold=0.7)
        assert len(results) == 1
        # Scorer passes through LLM decision; VCS override is in frontier
        assert results[0].should_expand is True


class TestIsRepoAccessibleElsewhere:
    """Tests for the standalone is_repo_accessible_elsewhere function."""

    def test_none_url_returns_false(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert is_repo_accessible_elsewhere(None) is False

    def test_github_iterorganization_is_accessible(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere("git@github.com/iterorganization/repo.git")
            is True
        )

    def test_git_iter_org_is_accessible(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere("https://git.iter.org/scm/imas/repo.git")
            is True
        )

    def test_unknown_host_not_accessible_without_scanner(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere("ssh://internal.example.com/repo.git") is False
        )

    def test_scanner_true_overrides_unknown_host(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere(
                "ssh://internal.example.com/repo.git", scanner_accessible=True
            )
            is True
        )

    def test_scanner_false_does_not_override(self):
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere(
                "ssh://internal.example.com/repo.git", scanner_accessible=False
            )
            is False
        )

    def test_config_pattern_takes_precedence_over_scanner_false(self):
        """Config match wins even if scanner says False."""
        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        assert (
            is_repo_accessible_elsewhere(
                "https://github.com/iterorganization/imas-codex.git",
                scanner_accessible=False,
            )
            is True
        )
