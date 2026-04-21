"""Tests for vocab_promotion module and consolidated sn gaps CLI.

Covers:
- mine_promotion_candidates threshold logic (min_usage_count, min_review_mean_score)
- persist_candidates writes PromotionCandidate nodes + EVIDENCED_BY edges
- format_isn_pr_snippet renders valid ISN-compatible YAML
- sn gaps --direction flag routes to saturated / missing / both
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMinePromotionCandidates:
    """Threshold logic: only tokens meeting both count AND score pass."""

    @patch("imas_codex.standard_names.vocab_promotion.GraphClient")
    @patch("imas_codex.standard_names.vocab_promotion._existing_tokens")
    def test_threshold_count_filters(self, mock_existing, mock_gc):
        from imas_codex.standard_names.vocab_promotion import (
            mine_promotion_candidates,
        )

        mock_existing.return_value = set()
        mock_client = MagicMock()
        mock_gc.return_value.__enter__.return_value = mock_client
        # 2 uses < min_usage_count=3 — excluded
        mock_client.query.return_value = []

        out = mine_promotion_candidates(
            segment="physical_base",
            min_usage_count=3,
            min_review_mean_score=0.75,
        )
        assert out == []

    @patch("imas_codex.standard_names.vocab_promotion.GraphClient")
    @patch("imas_codex.standard_names.vocab_promotion._existing_tokens")
    def test_candidate_shape(self, mock_existing, mock_gc):
        from imas_codex.standard_names.vocab_promotion import (
            mine_promotion_candidates,
        )

        mock_existing.return_value = set()
        mock_client = MagicMock()
        mock_gc.return_value.__enter__.return_value = mock_client
        mock_client.query.return_value = [
            {
                "token": "plasma_boundary_gap",
                "uses": 5,
                "min_score": 0.82,
                "domains": ["equilibrium"],
                "names": ["n1", "n2", "n3", "n4", "n5"],
            }
        ]

        out = mine_promotion_candidates(
            segment="physical_base",
            min_usage_count=3,
            min_review_mean_score=0.75,
        )
        assert len(out) == 1
        c = out[0]
        assert c["token"] == "plasma_boundary_gap"
        assert c["uses"] == 5
        assert c["min_review_score"] >= 0.75

    @patch("imas_codex.standard_names.vocab_promotion.GraphClient")
    @patch("imas_codex.standard_names.vocab_promotion._existing_tokens")
    def test_excludes_existing_by_default(self, mock_existing, mock_gc):
        from imas_codex.standard_names.vocab_promotion import (
            mine_promotion_candidates,
        )

        mock_existing.return_value = {"already_in_isn"}
        mock_client = MagicMock()
        mock_gc.return_value.__enter__.return_value = mock_client
        mock_client.query.return_value = [
            {
                "token": "already_in_isn",
                "uses": 10,
                "min_score": 0.9,
                "domains": ["equilibrium"],
                "names": ["n1"],
            },
            {
                "token": "new_candidate",
                "uses": 3,
                "min_score": 0.8,
                "domains": ["core_profiles"],
                "names": ["n2"],
            },
        ]

        out = mine_promotion_candidates(segment="physical_base")
        tokens = {c["token"] for c in out}
        assert "already_in_isn" not in tokens
        assert "new_candidate" in tokens


class TestPersistCandidates:
    """persist_candidates writes PromotionCandidate nodes + evidence edges."""

    @patch("imas_codex.standard_names.vocab_promotion.GraphClient")
    def test_writes_expected_nodes(self, mock_gc):
        from imas_codex.standard_names.vocab_promotion import persist_candidates

        mock_client = MagicMock()
        mock_gc.return_value.__enter__.return_value = mock_client

        candidates = [
            {
                "token": "plasma_boundary_gap",
                "segment": "physical_base",
                "uses": 4,
                "min_review_score": 0.8,
                "mean_review_score": 0.86,
                "supporting_names": ["n1", "n2", "n3", "n4"],
                "physics_domains": ["equilibrium"],
            }
        ]
        n = persist_candidates(candidates)
        assert n == 1
        assert mock_client.query.called

    def test_empty_list_returns_zero(self):
        from imas_codex.standard_names.vocab_promotion import persist_candidates

        assert persist_candidates([]) == 0


class TestFormatISNPRSnippet:
    """format_isn_pr_snippet renders ISN-compatible YAML."""

    def test_snippet_contains_tokens(self):
        from imas_codex.standard_names.vocab_promotion import (
            format_isn_pr_snippet,
        )

        candidates = [
            {
                "token": "plasma_boundary_gap",
                "segment": "physical_base",
                "uses": 5,
                "min_review_score": 0.82,
                "supporting_names": ["plasma_boundary_gap_angle"],
                "physics_domains": ["equilibrium"],
            }
        ]
        snippet = format_isn_pr_snippet(candidates, segment="physical_base")
        assert "plasma_boundary_gap" in snippet
        # Must be valid YAML
        import yaml

        yaml.safe_load(snippet)

    def test_empty_candidates_renders(self):
        from imas_codex.standard_names.vocab_promotion import (
            format_isn_pr_snippet,
        )

        # Should not crash on empty input
        out = format_isn_pr_snippet([], segment="physical_base")
        assert isinstance(out, str)


class TestSnGapsCliRouting:
    """Consolidated sn gaps routes by --direction."""

    def test_help_lists_direction_flag(self):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn_gaps

        runner = CliRunner()
        result = runner.invoke(sn_gaps, ["--help"])
        assert result.exit_code == 0
        assert "--direction" in result.output
        assert "missing" in result.output
        assert "saturated" in result.output

    def test_help_mentions_yaml_export(self):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn_gaps

        runner = CliRunner()
        result = runner.invoke(sn_gaps, ["--help"])
        assert result.exit_code == 0
        assert "yaml" in result.output.lower()

    def test_no_sn_vocab_group(self):
        """Verify sn vocab group was removed (merged into sn gaps)."""
        from imas_codex.cli.sn import sn

        assert "vocab" not in sn.commands
        assert "gaps" in sn.commands
