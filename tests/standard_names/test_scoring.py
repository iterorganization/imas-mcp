"""Tests for 0-1 normalized quality scoring."""

from __future__ import annotations


def test_quality_score_normalized():
    """StandardNameQualityScore.score returns float in 0-1 range."""
    from imas_codex.standard_names.models import StandardNameQualityScore

    score = StandardNameQualityScore(
        grammar=15,
        semantic=16,
        documentation=14,
        convention=15,
        completeness=13,
        compliance=14,
    )
    assert isinstance(score.score, float)
    assert 0.0 <= score.score <= 1.0
    # 87/120 = 0.725
    assert abs(score.score - 87 / 120.0) < 0.001


def test_quality_score_perfect():
    """Perfect scores give 1.0."""
    from imas_codex.standard_names.models import StandardNameQualityScore

    score = StandardNameQualityScore(
        grammar=20,
        semantic=20,
        documentation=20,
        convention=20,
        completeness=20,
        compliance=20,
    )
    assert score.score == 1.0
    assert score.tier == "outstanding"


def test_quality_score_zero():
    """Zero scores give 0.0."""
    from imas_codex.standard_names.models import StandardNameQualityScore

    score = StandardNameQualityScore(
        grammar=0,
        semantic=0,
        documentation=0,
        convention=0,
        completeness=0,
        compliance=0,
    )
    assert score.score == 0.0
    assert score.tier == "poor"


def test_tier_boundaries():
    """Tier thresholds at 0.85/0.65/0.40."""
    from imas_codex.standard_names.models import StandardNameQualityScore

    # Outstanding boundary: 0.85 * 120 = 102
    outstanding = StandardNameQualityScore(
        grammar=17,
        semantic=17,
        documentation=17,
        convention=17,
        completeness=17,
        compliance=17,
    )
    assert outstanding.score == 102 / 120.0
    assert outstanding.tier == "outstanding"

    # Just below outstanding: 101/120 = 0.8417
    good = StandardNameQualityScore(
        grammar=17,
        semantic=17,
        documentation=17,
        convention=17,
        completeness=17,
        compliance=16,
    )
    assert good.tier == "good"

    # Good boundary: 0.65 * 120 = 78
    good_boundary = StandardNameQualityScore(
        grammar=13,
        semantic=13,
        documentation=13,
        convention=13,
        completeness=13,
        compliance=13,
    )
    assert good_boundary.score == 78 / 120.0
    assert good_boundary.tier == "good"

    # Adequate boundary: 0.40 * 120 = 48
    adequate = StandardNameQualityScore(
        grammar=8,
        semantic=8,
        documentation=8,
        convention=8,
        completeness=8,
        compliance=8,
    )
    assert adequate.score == 48 / 120.0
    assert adequate.tier == "adequate"

    # Poor: below 0.40
    poor = StandardNameQualityScore(
        grammar=7,
        semantic=7,
        documentation=7,
        convention=7,
        completeness=7,
        compliance=7,
    )
    assert poor.tier == "poor"


def test_total_preserved():
    """The .total property still returns integer sum for internal use."""
    from imas_codex.standard_names.models import StandardNameQualityScore

    score = StandardNameQualityScore(
        grammar=15,
        semantic=16,
        documentation=14,
        convention=15,
        completeness=13,
        compliance=14,
    )
    assert score.total == 87
    assert isinstance(score.total, int)
