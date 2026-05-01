"""Trigger predicate + arm assignment unit tests (plan 39 §5.1, §8.4)."""

from __future__ import annotations

import json

import pytest

from imas_codex.standard_names.fanout.trigger import (
    assign_arm,
    extract_reviewer_excerpt,
    should_trigger_fanout,
)

# ---------------------------------------------------------------------
# extract_reviewer_excerpt
# ---------------------------------------------------------------------

DIMS = ("clarity", "disambiguation")
KEYWORDS = ("unclear", "ambiguous", "duplicate", "consider", "compare")


class TestExtractExcerpt:
    def test_dim_allowlist_filters_out_other_dims(self) -> None:
        comments = {
            "clarity": "name is unclear",
            "convention": "should be lowercase",  # not in allow-list
            "grammar": "missing physical_base",  # not in allow-list
        }
        excerpt = extract_reviewer_excerpt(comments, dims=DIMS, char_cap=800)
        assert "clarity" in excerpt
        assert "unclear" in excerpt
        assert "convention" not in excerpt
        assert "grammar" not in excerpt

    def test_json_string_input_parsed(self) -> None:
        comments = json.dumps(
            {"clarity": "is unclear", "disambiguation": "needs context"}
        )
        excerpt = extract_reviewer_excerpt(comments, dims=DIMS, char_cap=800)
        assert "unclear" in excerpt
        assert "needs context" in excerpt

    def test_none_returns_empty(self) -> None:
        assert extract_reviewer_excerpt(None, dims=DIMS, char_cap=800) == ""

    def test_invalid_json_returns_empty(self) -> None:
        assert extract_reviewer_excerpt("not-json{", dims=DIMS, char_cap=800) == ""

    def test_excerpt_truncated(self) -> None:
        # S3: ``refine-trigger-comment-chars`` cap respected.
        long_value = "x" * 2000
        comments = {"clarity": long_value}
        excerpt = extract_reviewer_excerpt(comments, dims=DIMS, char_cap=120)
        assert len(excerpt) == 120

    def test_only_disallowed_dims_returns_empty(self) -> None:
        comments = {"convention": "lowercase please"}
        assert extract_reviewer_excerpt(comments, dims=DIMS, char_cap=800) == ""


# ---------------------------------------------------------------------
# should_trigger_fanout
# ---------------------------------------------------------------------


class TestShouldTrigger:
    def _kwargs(self, **overrides):
        base = {
            "reviewer_comments_per_dim": {
                "clarity": "the name is ambiguous",
            },
            "chain_length": 1,
            "chain_history": [{"name": "old"}],
            "keywords": KEYWORDS,
            "dims": DIMS,
            "char_cap": 800,
        }
        base.update(overrides)
        return base

    def test_fires_on_keyword_in_allowed_dim(self) -> None:
        fire, excerpt = should_trigger_fanout(**self._kwargs())
        assert fire is True
        assert "ambiguous" in excerpt

    def test_does_not_fire_on_keyword_in_disallowed_dim(self) -> None:
        kwargs = self._kwargs(
            reviewer_comments_per_dim={
                "convention": "the name is ambiguous",
            }
        )
        fire, excerpt = should_trigger_fanout(**kwargs)
        assert fire is False
        # excerpt is still empty because no allow-listed dim has a value
        assert excerpt == ""

    def test_does_not_fire_when_chain_length_zero(self) -> None:
        fire, _ = should_trigger_fanout(**self._kwargs(chain_length=0))
        assert fire is False

    def test_does_not_fire_without_chain_history(self) -> None:
        fire, _ = should_trigger_fanout(**self._kwargs(chain_history=[]))
        assert fire is False

    def test_does_not_fire_without_trigger_keyword(self) -> None:
        kwargs = self._kwargs(
            reviewer_comments_per_dim={"clarity": "score below threshold"}
        )
        fire, _ = should_trigger_fanout(**kwargs)
        assert fire is False

    def test_keyword_matching_is_case_insensitive(self) -> None:
        kwargs = self._kwargs(
            reviewer_comments_per_dim={"clarity": "Name is UNCLEAR here"}
        )
        fire, _ = should_trigger_fanout(**kwargs)
        assert fire is True


# ---------------------------------------------------------------------
# assign_arm
# ---------------------------------------------------------------------


class TestAssignArm:
    def test_deterministic_for_same_inputs(self) -> None:
        a = assign_arm("electron_temperature", 1)
        b = assign_arm("electron_temperature", 1)
        assert a == b

    def test_arm_percent_zero_kills_all(self) -> None:
        for i in range(20):
            assert assign_arm(f"sn_{i}", i, arm_percent=0) == "off"

    def test_arm_percent_hundred_forces_all_on(self) -> None:
        for i in range(20):
            assert assign_arm(f"sn_{i}", i, arm_percent=100) == "on"

    def test_50_50_distribution_over_synthetic_cohort(self) -> None:
        # Plan 39 §8.4 I2: hash routing yields ~50/50 across N
        # synthetic items.  Tolerance: 30..70 / 100 — blake2b is
        # well-distributed so this is comfortable.
        on = 0
        off = 0
        for i in range(200):
            arm = assign_arm(f"electron_temperature_{i}", i % 3)
            if arm == "on":
                on += 1
            else:
                off += 1
        assert 70 <= on <= 130, f"on={on} not within 70..130"
        assert on + off == 200

    def test_different_cycle_indices_independent(self) -> None:
        # Same sn_id, different cycle_index → not perfectly correlated.
        # We just assert at least one disagreement across 10 cycles
        # for one sn_id (probability of all-equal under a true 50/50
        # hash is 2^-9 ≈ 0.2%).
        arms = {assign_arm("test_sn", c) for c in range(10)}
        assert len(arms) == 2  # both "on" and "off" appear

    @pytest.mark.parametrize("percent", [25, 75])
    def test_percentage_skews_split(self, percent: int) -> None:
        on = sum(
            1
            for i in range(400)
            if assign_arm(f"sn_{i}", 0, arm_percent=percent) == "on"
        )
        # Loose tolerance — confirm direction, not exact split.
        if percent == 25:
            assert 60 <= on <= 160
        else:
            assert 240 <= on <= 340
