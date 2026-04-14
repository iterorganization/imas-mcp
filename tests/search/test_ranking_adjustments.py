"""Unit tests for accessor de-ranking and IDS preference boost (Phase 1A + 1B).

These tests exercise the scoring adjustments in isolation — no Neo4j or
embedding server required.  We replicate the same logic that lives in
`imas_codex/tools/graph_search.py` after the segment-tiebreaker block.
"""

from __future__ import annotations

import pytest

from imas_codex.tools.graph_search import _ACCESSOR_TERMINALS, _CONCEPT_IDS_PREFERENCE

# ---------------------------------------------------------------------------
# Helpers that mirror the production scoring adjustments
# ---------------------------------------------------------------------------


def apply_accessor_deranking(scores: dict[str, float]) -> dict[str, float]:
    """Apply accessor de-ranking (Phase 1A) to a scores dict (mutates copy)."""
    scores = dict(scores)
    for pid in scores:
        terminal = pid.rsplit("/", 1)[-1].lower()
        if terminal in _ACCESSOR_TERMINALS:
            scores[pid] = round(scores[pid] * 0.95, 4)
    return scores


def apply_ids_preference(
    scores: dict[str, float],
    query_words: list[str],
    normalized_filter: str | list[str] | None,
) -> dict[str, float]:
    """Apply IDS preference boost (Phase 1B) to a scores dict (mutates copy)."""
    scores = dict(scores)
    if not normalized_filter and query_words:
        matched_ids_prefs: set[str] = set()
        for w in query_words:
            if w in _CONCEPT_IDS_PREFERENCE:
                matched_ids_prefs.add(_CONCEPT_IDS_PREFERENCE[w])
        if matched_ids_prefs:
            for pid in scores:
                result_ids = pid.split("/", 1)[0]
                if result_ids in matched_ids_prefs:
                    scores[pid] = round(scores[pid] * 1.03, 4)
    return scores


# ---------------------------------------------------------------------------
# Phase 1A — Accessor de-ranking
# ---------------------------------------------------------------------------


class TestAccessorDeranking:
    def test_data_terminal_penalized(self):
        scores = {"core_profiles/profiles_1d/electrons/temperature/data": 1.0}
        result = apply_accessor_deranking(scores)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature/data"
        ] == pytest.approx(0.95)

    def test_value_terminal_penalized(self):
        scores = {"equilibrium/time_slice/global_quantities/ip/value": 1.0}
        result = apply_accessor_deranking(scores)
        assert result[
            "equilibrium/time_slice/global_quantities/ip/value"
        ] == pytest.approx(0.95)

    def test_time_terminal_penalized(self):
        scores = {"core_profiles/time": 1.0}
        result = apply_accessor_deranking(scores)
        assert result["core_profiles/time"] == pytest.approx(0.95)

    def test_validity_terminal_penalized(self):
        scores = {"magnetics/flux_loop/flux/validity": 1.0}
        result = apply_accessor_deranking(scores)
        assert result["magnetics/flux_loop/flux/validity"] == pytest.approx(0.95)

    def test_fit_terminal_penalized(self):
        scores = {"core_profiles/profiles_1d/electrons/temperature/fit": 1.0}
        result = apply_accessor_deranking(scores)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature/fit"
        ] == pytest.approx(0.95)

    def test_coefficients_terminal_penalized(self):
        scores = {"core_profiles/profiles_1d/electrons/temperature/coefficients": 1.0}
        result = apply_accessor_deranking(scores)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature/coefficients"
        ] == pytest.approx(0.95)

    def test_non_accessor_terminal_unchanged(self):
        scores = {"core_profiles/profiles_1d/electrons/temperature": 1.0}
        result = apply_accessor_deranking(scores)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_concept_path_not_penalized(self):
        scores = {
            "equilibrium/time_slice/global_quantities/ip": 0.85,
            "core_profiles/profiles_1d/electrons/density": 0.80,
        }
        result = apply_accessor_deranking(scores)
        assert result["equilibrium/time_slice/global_quantities/ip"] == pytest.approx(
            0.85
        )
        assert result["core_profiles/profiles_1d/electrons/density"] == pytest.approx(
            0.80
        )

    def test_multiple_accessor_paths_all_penalized(self):
        scores = {
            "core_profiles/profiles_1d/electrons/temperature/data": 1.0,
            "equilibrium/time_slice/global_quantities/ip/data": 0.90,
            "magnetics/flux_loop/flux/time": 0.80,
        }
        result = apply_accessor_deranking(scores)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature/data"
        ] == pytest.approx(0.95)
        assert result[
            "equilibrium/time_slice/global_quantities/ip/data"
        ] == pytest.approx(0.855)
        assert result["magnetics/flux_loop/flux/time"] == pytest.approx(0.76)

    def test_root_level_accessor_name_penalized(self):
        """Even a bare 'data' path (no slash) should be penalized."""
        scores = {"data": 1.0}
        result = apply_accessor_deranking(scores)
        assert result["data"] == pytest.approx(0.95)

    def test_case_insensitive_terminal(self):
        """Terminal matching should be case-insensitive."""
        scores = {"some_ids/path/DATA": 1.0}
        result = apply_accessor_deranking(scores)
        assert result["some_ids/path/DATA"] == pytest.approx(0.95)

    def test_accessor_in_middle_of_path_not_penalized(self):
        """A segment named 'data' in the middle of the path is not the terminal."""
        scores = {"core_profiles/data/electrons/temperature": 1.0}
        result = apply_accessor_deranking(scores)
        # 'temperature' is not an accessor terminal
        assert result["core_profiles/data/electrons/temperature"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 1B — IDS preference boost
# ---------------------------------------------------------------------------


class TestIDSPreferenceBoost:
    def test_temperature_boosts_core_profiles(self):
        scores = {
            "core_profiles/profiles_1d/electrons/temperature": 1.0,
            "equilibrium/time_slice/profiles_1d/electrons/temperature": 1.0,
        }
        result = apply_ids_preference(scores, ["temperature"], normalized_filter=None)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.03)
        assert result[
            "equilibrium/time_slice/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_psi_boosts_equilibrium(self):
        scores = {
            "equilibrium/time_slice/profiles_1d/psi": 1.0,
            "core_profiles/profiles_1d/grid/psi": 1.0,
        }
        result = apply_ids_preference(scores, ["psi"], normalized_filter=None)
        assert result["equilibrium/time_slice/profiles_1d/psi"] == pytest.approx(1.03)
        assert result["core_profiles/profiles_1d/grid/psi"] == pytest.approx(1.0)

    def test_density_boosts_core_profiles(self):
        scores = {
            "core_profiles/profiles_1d/electrons/density": 0.80,
            "equilibrium/time_slice/profiles_2d/electrons/density": 0.80,
        }
        result = apply_ids_preference(scores, ["density"], normalized_filter=None)
        assert result["core_profiles/profiles_1d/electrons/density"] == pytest.approx(
            0.824
        )
        assert result[
            "equilibrium/time_slice/profiles_2d/electrons/density"
        ] == pytest.approx(0.80)

    def test_ids_filter_active_no_boost(self):
        """When an explicit IDS filter is active, boost must NOT be applied."""
        scores = {
            "core_profiles/profiles_1d/electrons/temperature": 1.0,
        }
        # String filter — should suppress boost
        result = apply_ids_preference(
            scores, ["temperature"], normalized_filter="core_profiles"
        )
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_ids_filter_as_list_active_no_boost(self):
        """A list-valued filter should also suppress the boost."""
        scores = {"core_profiles/profiles_1d/electrons/temperature": 1.0}
        result = apply_ids_preference(
            scores, ["temperature"], normalized_filter=["core_profiles"]
        )
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_unknown_query_word_no_boost(self):
        """Query words not in the preference dict must not trigger any boost."""
        scores = {
            "core_profiles/profiles_1d/electrons/temperature": 1.0,
        }
        result = apply_ids_preference(scores, ["flux"], normalized_filter=None)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_empty_query_words_no_boost(self):
        scores = {"core_profiles/profiles_1d/electrons/temperature": 1.0}
        result = apply_ids_preference(scores, [], normalized_filter=None)
        assert result[
            "core_profiles/profiles_1d/electrons/temperature"
        ] == pytest.approx(1.0)

    def test_non_matching_ids_not_boosted(self):
        scores = {
            "magnetics/flux_loop/flux/data": 0.70,
        }
        result = apply_ids_preference(scores, ["temperature"], normalized_filter=None)
        assert result["magnetics/flux_loop/flux/data"] == pytest.approx(0.70)

    def test_current_boosts_equilibrium(self):
        scores = {
            "equilibrium/time_slice/global_quantities/ip": 0.9,
            "core_profiles/profiles_1d/j_total": 0.9,
        }
        result = apply_ids_preference(scores, ["current"], normalized_filter=None)
        assert result["equilibrium/time_slice/global_quantities/ip"] == pytest.approx(
            0.927
        )
        assert result["core_profiles/profiles_1d/j_total"] == pytest.approx(0.9)

    def test_boundary_boosts_equilibrium(self):
        scores = {
            "equilibrium/time_slice/boundary/outline/r": 0.85,
            "mhd/time_slice/boundary/outline/r": 0.85,
        }
        result = apply_ids_preference(scores, ["boundary"], normalized_filter=None)
        assert result["equilibrium/time_slice/boundary/outline/r"] == pytest.approx(
            0.8755
        )
        assert result["mhd/time_slice/boundary/outline/r"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Combined tests (Phase 1A + 1B together)
# ---------------------------------------------------------------------------


class TestCombinedRankingAdjustments:
    def test_accessor_and_ids_preference_combined(self):
        """Apply both adjustments in order: de-rank then boost."""
        scores = {
            "core_profiles/profiles_1d/electrons/temperature": 1.0,
            "core_profiles/profiles_1d/electrons/temperature/data": 1.0,
            "equilibrium/time_slice/profiles_1d/electrons/temperature": 1.0,
        }
        # Step 1: accessor de-ranking
        scores = apply_accessor_deranking(scores)
        # Step 2: IDS preference boost
        scores = apply_ids_preference(scores, ["temperature"], normalized_filter=None)

        parent = scores["core_profiles/profiles_1d/electrons/temperature"]
        child = scores["core_profiles/profiles_1d/electrons/temperature/data"]
        other = scores["equilibrium/time_slice/profiles_1d/electrons/temperature"]

        # Parent gets boosted (no de-rank, +3%)  → 1.03
        assert parent == pytest.approx(1.03)
        # Child gets de-ranked first (×0.95) then boosted (×1.03) → 0.9785
        assert child == pytest.approx(0.9785)
        # Other IDS path: no accessor penalty, no IDS boost → 1.0
        assert other == pytest.approx(1.0)

        # The parent must rank above the child accessor
        assert parent > child

    def test_parent_concept_beats_accessor_sibling(self):
        """A concept path beats its /data sibling even when both start equal."""
        base_score = 0.80
        scores = {
            "core_profiles/profiles_1d/electrons/temperature": base_score,
            "core_profiles/profiles_1d/electrons/temperature/data": base_score,
        }
        scores = apply_accessor_deranking(scores)
        scores = apply_ids_preference(scores, ["temperature"], normalized_filter=None)

        concept = scores["core_profiles/profiles_1d/electrons/temperature"]
        accessor = scores["core_profiles/profiles_1d/electrons/temperature/data"]
        assert concept > accessor

    def test_all_constants_are_exported(self):
        """The module-level constants must be importable and non-empty."""
        assert len(_ACCESSOR_TERMINALS) > 0
        assert len(_CONCEPT_IDS_PREFERENCE) > 0
        assert "data" in _ACCESSOR_TERMINALS
        assert "temperature" in _CONCEPT_IDS_PREFERENCE
