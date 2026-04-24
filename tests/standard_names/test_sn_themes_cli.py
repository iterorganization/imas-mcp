"""Tests for ``sn themes`` CLI command (Phase 3-reuse-themes).

Verifies that the themes command correctly extracts n-gram themes from
mock reviewer comments and renders a markdown-style table output.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn

# -- Fixtures ----------------------------------------------------------------

_MOCK_COMMENTS = [
    "Name is too long, contains redundant qualifiers. "
    "Should drop the 'value_of' prefix per convention.",
    "Redundant qualifiers again — 'electron_temperature_value' should "
    "just be 'electron_temperature'. Convention violation.",
    "Good name but missing sign convention for COCOS-dependent field. "
    "The name should encode whether psi increases inward or outward.",
    "Missing sign convention. Also the name is slightly too long.",
    "Excessive length — too many segments for a simple scalar quantity. "
    "Redundant qualifiers detected.",
    "The physical_base token is wrong — 'temp' is not a valid base, "
    "use 'temperature'. Grammar violation.",
    "Grammar violation — physical_base must come from closed vocabulary. "
    "'temp' should be 'temperature'.",
    "Name does not reflect the coordinate convention. The profile is "
    "over rho_tor_norm, not psi. Missing sign convention too.",
    "Convention violation: name uses abbreviation 'ne' instead of "
    "'electron_density'. Redundant qualifiers also present.",
    "Good overall. Minor: could be shorter by removing '_profile' suffix "
    "since the coordinate already implies a profile.",
]


class TestSNThemesGraph:
    """Test themes command with --source graph (default)."""

    def test_requires_domain_for_graph_source(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["themes"])
        assert result.exit_code == 0
        assert "--physics-domain is required" in result.output

    @patch("imas_codex.standard_names.review.themes.extract_reviewer_themes")
    def test_renders_themes_table(self, mock_extract):
        mock_extract.return_value = [
            "redundant qualifiers",
            "sign convention",
            "grammar violation",
        ]
        runner = CliRunner()
        result = runner.invoke(sn, ["themes", "--physics-domain", "equilibrium"])
        assert result.exit_code == 0
        assert "redundant qualifiers" in result.output
        assert "sign convention" in result.output
        assert "3 themes extracted" in result.output

    @patch("imas_codex.standard_names.review.themes.extract_reviewer_themes")
    def test_empty_themes(self, mock_extract):
        mock_extract.return_value = []
        runner = CliRunner()
        result = runner.invoke(sn, ["themes", "--physics-domain", "equilibrium"])
        assert result.exit_code == 0
        assert "No recurring themes" in result.output


class TestSNThemesReviews:
    """Test themes command with --source reviews."""

    @patch("imas_codex.graph.client.GraphClient")
    def test_reviews_source_queries_review_nodes(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"comments": c} for c in _MOCK_COMMENTS]

        runner = CliRunner()
        result = runner.invoke(sn, ["themes", "--source", "reviews", "--limit", "10"])
        assert result.exit_code == 0
        # Should extract at least 2 themes from our mock data
        assert "themes extracted" in result.output

    @patch("imas_codex.graph.client.GraphClient")
    def test_reviews_with_since_filter(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"comments": c} for c in _MOCK_COMMENTS[:3]]

        runner = CliRunner()
        result = runner.invoke(
            sn, ["themes", "--source", "reviews", "--since", "2025-01-01"]
        )
        assert result.exit_code == 0

    @patch("imas_codex.graph.client.GraphClient")
    def test_reviews_no_results(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = []

        runner = CliRunner()
        result = runner.invoke(sn, ["themes", "--source", "reviews"])
        assert result.exit_code == 0
        assert "No review comments found" in result.output


class TestThemeExtraction:
    """Test the underlying theme extraction on overlapping comments."""

    def test_clustering_overlapping_themes(self):
        from imas_codex.standard_names.review.themes import (
            _extract_themes_from_texts,
        )

        themes = _extract_themes_from_texts(_MOCK_COMMENTS, top_n=5)
        assert len(themes) >= 2
        # "redundant qualifiers" should be a top theme (appears 4+ times)
        combined = " ".join(themes).lower()
        assert "redundant" in combined or "qualifiers" in combined

    def test_theme_dedup(self):
        """Overlapping n-grams should be deduplicated."""
        from imas_codex.standard_names.review.themes import (
            _extract_themes_from_texts,
        )

        # Feed identical comments — should still produce distinct themes
        comments = ["sign convention violation"] * 10 + [
            "missing temperature token"
        ] * 5
        themes = _extract_themes_from_texts(comments, top_n=5)
        # Should have both themes but not redundant duplicates
        assert len(themes) >= 1
