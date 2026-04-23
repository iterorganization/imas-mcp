"""Tests for export gate C — min_score threshold filtering.

Plan 35 §3d: names below threshold not emitted unless --include-unreviewed.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.export import _run_gate_c


def _make_candidate(
    name: str,
    reviewer_score: float | None = None,
    reviewer_description_score: float | None = None,
) -> dict:
    return {
        "id": name,
        "reviewer_score_name": reviewer_score,
        "reviewer_description_score": reviewer_description_score,
    }


class TestGateCMinScore:
    """Gate C filters candidates by reviewer_score."""

    def test_above_threshold_passes(self) -> None:
        candidates = [_make_candidate("good_name", reviewer_score=0.80)]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 1
        assert below == 0
        assert unrev == 0

    def test_below_threshold_excluded(self) -> None:
        candidates = [_make_candidate("weak_name", reviewer_score=0.40)]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 0
        assert below == 1
        assert unrev == 0

    def test_unreviewed_excluded_by_default(self) -> None:
        candidates = [_make_candidate("unreviewed_name", reviewer_score=None)]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 0
        assert unrev == 1

    def test_unreviewed_included_with_flag(self) -> None:
        candidates = [_make_candidate("unreviewed_name", reviewer_score=None)]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=True,
            min_description_score=None,
        )
        assert len(filtered) == 1
        assert unrev == 0

    def test_mixed_candidates(self) -> None:
        candidates = [
            _make_candidate("good", reviewer_score=0.90),
            _make_candidate("marginal", reviewer_score=0.65),
            _make_candidate("bad", reviewer_score=0.30),
            _make_candidate("unreviewed", reviewer_score=None),
        ]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 2  # good + marginal
        assert below == 1  # bad
        assert unrev == 1  # unreviewed

    def test_exact_threshold_passes(self) -> None:
        candidates = [_make_candidate("exact", reviewer_score=0.65)]
        result, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
        )
        assert len(filtered) == 1

    def test_description_score_filter(self) -> None:
        candidates = [
            _make_candidate(
                "thin_docs",
                reviewer_score=0.80,
                reviewer_description_score=0.30,
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
