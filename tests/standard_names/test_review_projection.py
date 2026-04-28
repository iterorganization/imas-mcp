"""Tests for canonical-review projection helper.

All tests are offline — the ``GraphClient`` is fully mocked via
``gc.query`` returning pre-canned rows that mimic the shape returned
by the ``MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview) RETURN ...`` Cypher.

Test inventory (≥9 cases, covers all 4 source branches + edge cases)
----------------------------------------------------------------------
1.  test_escalator_overrides_quorum            — 3 cycles; cycle-2 wins
2.  test_quorum_mean_when_no_escalator         — cycles 0+1 quorum; returns mean
3.  test_single_review_returned_as_is          — cycle 0 only; source="single"
4.  test_no_reviews_returns_none               — empty graph; returns None
5.  test_retry_item_treated_as_single          — cycle-0 retry_item; source="single"
6.  test_axis_isolation                        — names vs docs axis
7.  test_escalator_with_max_cycles_reached_falls_back_to_quorum   (RD #3)
8.  test_quorum_skips_retry_item_cycle         (RD #4)
9.  test_multiple_review_groups_picks_latest   (RD #5)
10. test_only_cycle1_present_falls_to_none     — edge case: no cycle 0
11. test_quorum_mean_scores_averaged_correctly  — arithmetic validation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.review.projection import (
    CanonicalReview,
    project_canonical_review,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_GROUP_A = "group-aaaa"
_GROUP_B = "group-bbbb"  # lexicographically later → "most recent" in ORDER BY DESC
_SN_ID = "electron_temperature"


def _make_gc(rows: list[dict]) -> MagicMock:
    """Return a mock GraphClient whose ``query`` method yields *rows*."""
    gc = MagicMock()
    gc.query = MagicMock(return_value=rows)
    return gc


def _row(
    *,
    group: str = _GROUP_A,
    cycle: int,
    method: str | None,
    score: float = 0.75,
    scores_json: str | None = None,
    model: str = "openrouter/anthropic/claude-opus-4.6",
    axis: str = "names",
) -> dict:
    """Build a minimal Review query-row dict."""
    if scores_json is None:
        scores_json = json.dumps({"grammar": 15, "semantic": 15})
    return {
        "review_group_id": group,
        "cycle_index": cycle,
        "resolution_method": method,
        "score": score,
        "scores_json": scores_json,
        "model": model,
        "comments": f"cycle-{cycle} comment",
        "comments_per_dim": None,
        "tier": "good",
    }


# ---------------------------------------------------------------------------
# 1. Escalator overrides quorum
# ---------------------------------------------------------------------------


class TestEscalatorBranch:
    def test_escalator_overrides_quorum(self) -> None:
        """Cycle-2 with authoritative_escalation is returned; cycles 0+1 ignored."""
        rows = [
            _row(cycle=0, method="quorum_consensus", score=0.80),
            _row(cycle=1, method="quorum_consensus", score=0.70),
            _row(cycle=2, method="authoritative_escalation", score=0.65),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "escalator"
        assert result.score == pytest.approx(0.65)

    def test_escalator_with_max_cycles_reached_falls_back_to_quorum(self) -> None:
        """Cycle-2 carrying max_cycles_reached is NOT an escalator (RD #3).

        The quorum-mean branch should fire on cycles {0, 1}.
        """
        rows = [
            _row(cycle=0, method="quorum_consensus", score=0.80),
            _row(cycle=1, method="quorum_consensus", score=0.70),
            _row(cycle=2, method="max_cycles_reached", score=0.99),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "quorum_mean"
        assert result.score == pytest.approx((0.80 + 0.70) / 2.0)


# ---------------------------------------------------------------------------
# 2. Quorum-mean branch
# ---------------------------------------------------------------------------


class TestQuorumMeanBranch:
    def test_quorum_mean_when_no_escalator(self) -> None:
        """Cycles 0 and 1 both quorum_consensus → mean returned."""
        rows = [
            _row(
                cycle=0,
                method="quorum_consensus",
                score=0.80,
                scores_json=json.dumps({"grammar": 16, "semantic": 16}),
            ),
            _row(
                cycle=1,
                method="quorum_consensus",
                score=0.70,
                scores_json=json.dumps({"grammar": 12, "semantic": 14}),
            ),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "quorum_mean"
        assert result.score == pytest.approx((0.80 + 0.70) / 2.0)

    def test_quorum_mean_scores_averaged_correctly(self) -> None:
        """Per-dimension scores must be the element-wise mean."""
        rows = [
            _row(
                cycle=0,
                method="quorum_consensus",
                score=0.90,
                scores_json=json.dumps({"grammar": 18, "semantic": 18}),
            ),
            _row(
                cycle=1,
                method="quorum_consensus",
                score=0.60,
                scores_json=json.dumps({"grammar": 12, "semantic": 12}),
            ),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        dims = json.loads(result.scores_json or "{}")
        assert dims["grammar"] == pytest.approx(15.0)
        assert dims["semantic"] == pytest.approx(15.0)

    def test_quorum_skips_retry_item_cycle(self) -> None:
        """If cycle 0 has retry_item, quorum-mean branch is skipped (RD #4).

        Projection should fall through to single branch (cycle 0).
        """
        rows = [
            _row(cycle=0, method="retry_item", score=0.75),
            _row(cycle=1, method="quorum_consensus", score=0.70),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        # quorum-mean skipped because cycle-0 has retry_item → falls to single
        assert result.source == "single"
        assert result.score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 3. Single branch
# ---------------------------------------------------------------------------


class TestSingleBranch:
    def test_single_review_returned_as_is(self) -> None:
        """Only cycle 0 present → source='single'."""
        rows = [_row(cycle=0, method="single_review", score=0.80)]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "single"
        assert result.score == pytest.approx(0.80)

    def test_retry_item_treated_as_single(self) -> None:
        """cycle-0 with retry_item and no cycle-1 → source='single'."""
        rows = [_row(cycle=0, method="retry_item", score=0.55)]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "single"

    def test_only_cycle1_present_falls_to_none(self) -> None:
        """No cycle 0, only cycle 1 present → returns None (can't project)."""
        rows = [_row(cycle=1, method="quorum_consensus", score=0.72)]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is None


# ---------------------------------------------------------------------------
# 4. None branch
# ---------------------------------------------------------------------------


class TestNoneBranch:
    def test_no_reviews_returns_none(self) -> None:
        """Empty Review set → None."""
        gc = _make_gc([])
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Axis isolation
# ---------------------------------------------------------------------------


class TestAxisIsolation:
    def test_axis_isolation(self) -> None:
        """Projection for each axis is independent of the other axis.

        The mock returns rows only when the correct axis is requested;
        the other axis query returns an empty list.
        """
        names_row = _row(cycle=0, method="single_review", score=0.85, axis="names")
        docs_row = _row(cycle=0, method="single_review", score=0.60, axis="docs")

        # We track which axis was requested by inspecting the kwarg
        def _side_effect(cypher: str, **kwargs: object) -> list[dict]:
            ax = kwargs.get("axis")
            if ax == "names":
                return [names_row]
            if ax == "docs":
                return [docs_row]
            return []

        gc = MagicMock()
        gc.query = MagicMock(side_effect=_side_effect)

        names_result = project_canonical_review(_SN_ID, "names", gc)
        docs_result = project_canonical_review(_SN_ID, "docs", gc)

        assert names_result is not None
        assert names_result.score == pytest.approx(0.85)
        assert names_result.source == "single"

        assert docs_result is not None
        assert docs_result.score == pytest.approx(0.60)
        assert docs_result.source == "single"


# ---------------------------------------------------------------------------
# 6. Multiple review groups — latest wins
# ---------------------------------------------------------------------------


class TestMultipleReviewGroups:
    def test_multiple_review_groups_picks_latest(self) -> None:
        """When two review_group_ids exist, the most-recent group is used (RD #5).

        The Cypher sorts by review_group_id DESC so the first row returned
        determines the latest group.  We simulate two groups: GROUP_B (newer,
        first in DESC order) with score 0.90, and GROUP_A (older) with score
        0.50.  The projection must return 0.90.
        """
        rows = [
            # GROUP_B rows first (ORDER BY review_group_id DESC)
            _row(group=_GROUP_B, cycle=0, method="single_review", score=0.90),
            # GROUP_A rows follow
            _row(group=_GROUP_A, cycle=0, method="single_review", score=0.50),
        ]
        gc = _make_gc(rows)
        result = project_canonical_review(_SN_ID, "names", gc)
        assert result is not None
        assert result.source == "single"
        assert result.score == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# 7. CanonicalReview dataclass smoke test
# ---------------------------------------------------------------------------


class TestCanonicalReviewDataclass:
    def test_source_field_values(self) -> None:
        """Accepted source literals round-trip correctly."""
        for src in ("escalator", "quorum_mean", "single", "none"):
            cr = CanonicalReview(
                score=0.5,
                scores_json=None,
                model=None,
                comments=None,
                comments_per_dim=None,
                tier=None,
                source=src,  # type: ignore[arg-type]
            )
            assert cr.source == src
