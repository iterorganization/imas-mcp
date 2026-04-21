"""Parametrised gate-C matrix — all permutations of score thresholds.

Phase 6d of plan 35: exhaustive gate C filtering verification across
combinations of min_score, include_unreviewed, min_description_score,
and force. Tests _run_gate_c directly.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.export import _run_gate_c

# ============================================================================
# Fixture candidates with varied scores
# ============================================================================


def _make_candidate(
    name: str,
    reviewer_score: float | None = None,
    reviewer_description_score: float | None = None,
) -> dict:
    return {
        "id": name,
        "reviewer_score": reviewer_score,
        "reviewer_description_score": reviewer_description_score,
    }


# Fixed candidate pool for matrix tests
_CANDIDATES = [
    _make_candidate("high_both", reviewer_score=0.90, reviewer_description_score=0.85),
    _make_candidate(
        "high_score_low_desc", reviewer_score=0.80, reviewer_description_score=0.30
    ),
    _make_candidate("low_score", reviewer_score=0.40, reviewer_description_score=0.80),
    _make_candidate("no_review", reviewer_score=None, reviewer_description_score=None),
    _make_candidate(
        "edge_at_threshold", reviewer_score=0.65, reviewer_description_score=0.50
    ),
    _make_candidate(
        "barely_below", reviewer_score=0.64, reviewer_description_score=0.90
    ),
]


# ============================================================================
# Parametrised matrix
# ============================================================================


class TestGateCMatrix:
    """Exhaustive gate C permutations."""

    @pytest.mark.parametrize(
        "min_score, include_unreviewed, min_desc_score, expected_names",
        [
            # ── Default thresholds, exclude unreviewed ──────────
            (
                0.65,
                False,
                None,
                {"high_both", "high_score_low_desc", "edge_at_threshold"},
            ),
            # ── Default thresholds, include unreviewed ──────────
            (
                0.65,
                True,
                None,
                {"high_both", "high_score_low_desc", "edge_at_threshold", "no_review"},
            ),
            # ── Zero threshold, include all scored ──────────────
            (
                0.0,
                False,
                None,
                {
                    "high_both",
                    "high_score_low_desc",
                    "low_score",
                    "edge_at_threshold",
                    "barely_below",
                },
            ),
            # ── Zero threshold, include all ─────────────────────
            (
                0.0,
                True,
                None,
                {
                    "high_both",
                    "high_score_low_desc",
                    "low_score",
                    "no_review",
                    "edge_at_threshold",
                    "barely_below",
                },
            ),
            # ── High threshold ──────────────────────────────────
            (
                0.85,
                False,
                None,
                {"high_both"},
            ),
            # ── Very high threshold — excludes all scored ───────
            (
                0.95,
                False,
                None,
                set(),
            ),
            # ── Very high threshold but include unreviewed ──────
            (
                0.95,
                True,
                None,
                {"no_review"},
            ),
            # ── With description score filter ───────────────────
            (
                0.65,
                False,
                0.50,
                {"high_both", "edge_at_threshold"},
            ),
            # ── Description score excludes low desc ─────────────
            (
                0.65,
                False,
                0.80,
                {"high_both"},
            ),
            # ── Description score + include unreviewed ──────────
            (
                0.65,
                True,
                0.50,
                {"high_both", "edge_at_threshold", "no_review"},
            ),
            # ── Zero score + description threshold ──────────────
            (
                0.0,
                False,
                0.50,
                {
                    "high_both",
                    "low_score",
                    "edge_at_threshold",
                    "barely_below",
                },
            ),
            # ── Zero score + high description threshold ─────────
            (
                0.0,
                False,
                0.85,
                {"high_both", "barely_below"},
            ),
            # ── Exact boundary: score == threshold ──────────────
            (
                0.64,
                False,
                None,
                {
                    "high_both",
                    "high_score_low_desc",
                    "edge_at_threshold",
                    "barely_below",
                },
            ),
        ],
        ids=[
            "default_no_unreviewed",
            "default_with_unreviewed",
            "zero_threshold_no_unrev",
            "zero_threshold_all",
            "high_threshold",
            "very_high_excludes_all",
            "very_high_plus_unreviewed",
            "desc_score_050",
            "desc_score_080",
            "desc_score_050_plus_unrev",
            "zero_plus_desc_050",
            "zero_plus_desc_085",
            "exact_boundary",
        ],
    )
    def test_gate_c_permutation(
        self,
        min_score: float,
        include_unreviewed: bool,
        min_desc_score: float | None,
        expected_names: set[str],
    ) -> None:
        """Gate C filters correctly for the given parameter combination."""
        result, filtered, below, unrev = _run_gate_c(
            list(_CANDIDATES),  # copy to avoid mutation
            min_score,
            include_unreviewed,
            min_desc_score,
        )

        actual_names = {c["id"] for c in filtered}
        assert actual_names == expected_names, (
            f"Expected {expected_names}, got {actual_names}"
        )

        # Gate C always passes (it's a filter, not a blocker)
        assert result.passed is True

        # Count invariant: filtered + excluded = total
        total_input = len(_CANDIDATES)
        assert len(filtered) + below + unrev == total_input, (
            f"filtered={len(filtered)}, below={below}, unrev={unrev}, "
            f"total input={total_input}"
        )


class TestGateCEdgeCases:
    """Gate C edge cases not covered by the matrix."""

    def test_empty_candidates(self) -> None:
        result, filtered, below, unrev = _run_gate_c(
            [],
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 0
        assert below == 0
        assert unrev == 0
        assert result.passed is True

    def test_all_unreviewed_excluded(self) -> None:
        candidates = [
            _make_candidate("a", reviewer_score=None),
            _make_candidate("b", reviewer_score=None),
        ]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 0
        assert unrev == 2
        assert below == 0

    def test_all_unreviewed_included(self) -> None:
        candidates = [
            _make_candidate("a", reviewer_score=None),
            _make_candidate("b", reviewer_score=None),
        ]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=True,
            min_description_score=None,
        )
        assert len(filtered) == 2
        assert unrev == 0
        assert below == 0

    def test_description_score_none_passes(self) -> None:
        """When candidate has no description score, desc threshold doesn't exclude."""
        candidates = [
            _make_candidate(
                "no_desc", reviewer_score=0.80, reviewer_description_score=None
            ),
        ]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=0.90,
        )
        assert len(filtered) == 1
        assert below == 0

    def test_description_score_issues_logged(self) -> None:
        """Gate C logs issues for candidates excluded by description score."""
        candidates = [
            _make_candidate(
                "low_desc", reviewer_score=0.80, reviewer_description_score=0.20
            ),
        ]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=0.50,
        )
        assert len(filtered) == 0
        assert below == 1
        assert len(result.issues) == 1
        assert result.issues[0]["type"] == "below_description_score"
        assert result.issues[0]["name"] == "low_desc"
