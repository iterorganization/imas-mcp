"""Phase C: verify review worker promotes low-tier names to needs_revision.

Covers ``_match_reviews_to_entries`` in the review pipeline: when the LLM
returns a review whose ``scores.tier`` is ``"poor"`` or ``"adequate"``, the
matched entry dict must have ``validation_status == "needs_revision"``
attached so the subsequent ``write_standard_names`` coalesce picks it up.

Entries reviewed as ``"good"`` or ``"outstanding"`` must NOT be demoted.
"""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.models import (
    StandardNameQualityReview,
    StandardNameQualityScore,
    StandardNameReviewVerdict,
)
from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

# Score values that yield the target tier (computed from 6-dim sum / 120).
# poor <0.40 (< 48 sum); adequate 0.40-0.60; good 0.60-0.85; outstanding >=0.85.
_TIER_DIM_SCORES = {
    "poor": 5,  # 6 * 5 = 30 / 120 = 0.25
    "adequate": 10,  # 60 / 120 = 0.50
    "good": 15,  # 90 / 120 = 0.75
    "outstanding": 18,  # 108 / 120 = 0.90
}


def _make_review(
    *,
    name: str,
    source_id: str,
    tier: str,
    verdict: StandardNameReviewVerdict = StandardNameReviewVerdict.accept,
    revised_name: str | None = None,
) -> StandardNameQualityReview:
    dim = _TIER_DIM_SCORES[tier]
    scores = StandardNameQualityScore(
        grammar=dim,
        semantic=dim,
        documentation=dim,
        convention=dim,
        completeness=dim,
        compliance=dim,
    )
    assert scores.tier == tier, f"constructed score yielded {scores.tier}, want {tier}"
    return StandardNameQualityReview(
        source_id=source_id,
        standard_name=name,
        scores=scores,
        verdict=verdict,
        reasoning=f"Tier {tier} review of {name}",
        revised_name=revised_name,
    )


@pytest.fixture
def wlog() -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger("test"), {})


class TestTierToNeedsRevision:
    """Low tier → needs_revision on the matched entry."""

    @pytest.mark.parametrize("tier", ["poor", "adequate"])
    def test_low_tier_sets_needs_revision(
        self, tier: str, wlog: logging.LoggerAdapter
    ) -> None:
        entry = {
            "id": "bad_name",
            "source_id": "dd:some/path",
            "validation_status": "valid",
        }
        review = _make_review(
            name="bad_name",
            source_id="dd:some/path",
            tier=tier,
        )
        scored, _unmatched, _revised = _match_reviews_to_entries(
            [review], [entry], wlog
        )
        assert len(scored) == 1
        assert scored[0]["validation_status"] == "needs_revision"
        assert scored[0]["review_tier"] == tier

    @pytest.mark.parametrize("tier", ["good", "outstanding"])
    def test_high_tier_preserves_validation_status(
        self, tier: str, wlog: logging.LoggerAdapter
    ) -> None:
        entry = {
            "id": "good_name",
            "source_id": "dd:some/path",
            "validation_status": "valid",
        }
        review = _make_review(
            name="good_name",
            source_id="dd:some/path",
            tier=tier,
        )
        scored, _unmatched, _revised = _match_reviews_to_entries(
            [review], [entry], wlog
        )
        assert scored[0]["validation_status"] == "valid"

    def test_revised_entry_inherits_needs_revision_flag(
        self, wlog: logging.LoggerAdapter
    ) -> None:
        """When the reviewer proposes a revision for a poor-tier name, the
        revised entry dict (a copy of original) should also carry
        needs_revision so the next regen run picks it up."""
        entry = {
            "id": "old_name",
            "source_id": "dd:some/path",
            "validation_status": "valid",
        }
        review = _make_review(
            name="old_name",
            source_id="dd:some/path",
            tier="poor",
            verdict=StandardNameReviewVerdict.revise,
            revised_name="better_name",
        )
        scored, _unmatched, revised_count = _match_reviews_to_entries(
            [review], [entry], wlog
        )
        assert revised_count == 1
        # The revised entry is returned (not the original) — check it inherited
        # the needs_revision flag set on original before dict(original) copy.
        assert scored[0]["id"] == "better_name"
        assert scored[0]["validation_status"] == "needs_revision"
