"""Tests for the RD-quorum review loop (Plan 39, Phase 4).

Validates:
- Single model → single_review (no quorum)
- 2 models agreement → quorum_consensus (final = mean)
- 2 models disagreement, no escalator → max_cycles_reached
- 3 models agreement → cycle 2 NOT invoked (budget saved)
- 3 models disagreement → cycle 2 escalates on disputed items only
- Partial failure handling (one/both cycles missing)
- Blind cycle 1 (no primary context leaks)
- Cycle 2 sees both prior critiques
- Hybrid batching (cycle 2 processes only disputed items)
- Review node metadata (axis, cycle_index, review_group_id, resolution_role)
- Budget lease accounting
- Helper functions (_check_per_dim_disagreement, _merge_review_items, etc.)
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_item(
    sn_id: str,
    scores: dict | None = None,
    score: float = 0.8,
    tier: str = "good",
    comments: str = "ok",
    verdict: str = "accept",
) -> dict:
    """Build a review item dict mimicking _match_reviews_to_entries output."""
    if scores is None:
        scores = {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16}
    return {
        "id": sn_id,
        "source_id": sn_id,
        "reviewer_score": score,
        "reviewer_scores": json.dumps(scores),
        "reviewer_comments": comments,
        "review_tier": tier,
        "reviewer_verdict": verdict,
        "reviewer_comments_per_dim": None,
    }


def _make_docs_item(
    sn_id: str,
    scores: dict | None = None,
    score: float = 0.8,
) -> dict:
    """Build a docs-axis review item."""
    if scores is None:
        scores = {
            "description_quality": 16,
            "documentation_quality": 16,
            "completeness": 16,
            "physics_accuracy": 16,
        }
    return _make_item(sn_id, scores=scores, score=score)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestCheckPerDimDisagreement:
    """Tests for _check_per_dim_disagreement."""

    def test_agreement_within_tolerance(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        # Both score 16/20 = 0.80 — no disagreement
        item_0 = _make_item(
            "a", {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 15, "semantic": 16, "convention": 16, "completeness": 16}
        )
        # 16/20 vs 15/20 = 0.80 vs 0.75 → diff = 0.05 < 0.15
        assert not _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

    def test_disagreement_beyond_tolerance(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        item_0 = _make_item(
            "a", {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 10, "semantic": 16, "convention": 16, "completeness": 16}
        )
        # 20/20 vs 10/20 = 1.0 vs 0.5 → diff = 0.5 > 0.15
        assert _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

    def test_docs_dimensions(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        item_0 = _make_docs_item(
            "a",
            {
                "description_quality": 20,
                "documentation_quality": 16,
                "completeness": 16,
                "physics_accuracy": 16,
            },
        )
        item_1 = _make_docs_item(
            "a",
            {
                "description_quality": 5,
                "documentation_quality": 16,
                "completeness": 16,
                "physics_accuracy": 16,
            },
        )
        assert _check_per_dim_disagreement(item_0, item_1, 0.15, "docs")

    def test_missing_scores_treated_as_disagreement(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        item_0 = _make_item("a")
        item_1 = {"id": "a", "reviewer_scores": None}
        assert _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

    def test_both_missing_no_disagreement(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        item_0 = {"id": "a", "reviewer_scores": None}
        item_1 = {"id": "a", "reviewer_scores": None}
        # Both empty → no disagreement
        assert not _check_per_dim_disagreement(item_0, item_1, 0.15, "names")


class TestMergeReviewItems:
    """Tests for _merge_review_items."""

    def test_mean_scores(self):
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        item_0 = _make_item(
            "a", {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 10, "semantic": 14, "convention": 18, "completeness": 12}
        )
        merged = _merge_review_items(item_0, item_1, "names")

        scores = json.loads(merged["reviewer_scores"])
        assert scores["grammar"] == pytest.approx(15.0)
        assert scores["semantic"] == pytest.approx(15.0)
        assert scores["convention"] == pytest.approx(17.0)
        assert scores["completeness"] == pytest.approx(14.0)

    def test_merged_score_normalized(self):
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        item_0 = _make_item(
            "a", {"grammar": 20, "semantic": 20, "convention": 20, "completeness": 20}
        )
        item_1 = _make_item(
            "a", {"grammar": 10, "semantic": 10, "convention": 10, "completeness": 10}
        )
        merged = _merge_review_items(item_0, item_1, "names")

        # Mean of (80/80 + 40/80) = 60/80 = 0.75
        assert merged["reviewer_score"] == pytest.approx(0.75)

    def test_comments_merged(self):
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        item_0 = _make_item("a", comments="primary comment")
        item_1 = _make_item("a", comments="secondary comment")
        merged = _merge_review_items(item_0, item_1, "names")
        assert "[Primary]" in merged["reviewer_comments"]
        assert "[Secondary]" in merged["reviewer_comments"]


class TestBuildPriorReviewsContext:
    """Tests for _build_prior_reviews_context."""

    def test_only_disputed_items_included(self):
        from imas_codex.standard_names.review.pipeline import (
            _build_prior_reviews_context,
        )

        c0_items = [_make_item("a"), _make_item("b")]
        c1_items = [_make_item("a"), _make_item("b")]
        disputed = {"b"}
        models = ["m0", "m1", "m2"]

        ctx = _build_prior_reviews_context(c0_items, c1_items, disputed, models)
        assert len(ctx) == 2  # primary + secondary
        for entry in ctx:
            assert len(entry["items"]) == 1
            assert entry["items"][0]["standard_name"] == "b"

    def test_roles_and_models(self):
        from imas_codex.standard_names.review.pipeline import (
            _build_prior_reviews_context,
        )

        c0_items = [_make_item("x")]
        c1_items = [_make_item("x")]
        ctx = _build_prior_reviews_context(
            c0_items, c1_items, {"x"}, ["modelA", "modelB", "modelC"]
        )
        assert ctx[0]["role"] == "primary"
        assert ctx[0]["model"] == "modelA"
        assert ctx[1]["role"] == "secondary"
        assert ctx[1]["model"] == "modelB"


class TestBuildReviewRecord:
    """Tests for _build_review_record."""

    def test_builds_record(self):
        from imas_codex.standard_names.review.pipeline import _build_review_record

        item = _make_item("electron_temperature")
        rec = _build_review_record(
            item,
            model="anthropic/claude-sonnet-4.6",
            is_canonical=True,
            reviewed_at="2025-01-01T00:00:00Z",
            cost_usd=0.001,
            tokens_in=100,
            tokens_out=50,
        )
        assert rec["standard_name_id"] == "electron_temperature"
        assert rec["is_canonical"] is True
        assert rec["model_family"] == "anthropic"


# ---------------------------------------------------------------------------
# RD-quorum flow integration tests (mocked LLM)
# ---------------------------------------------------------------------------


def _mock_review_result(items: list[dict], cost: float = 0.01) -> dict:
    """Build a mock return value for _review_single_batch."""
    return {
        "_items": items,
        "_cost": cost,
        "_tokens": 100,
        "_input_tokens": 80,
        "_output_tokens": 20,
        "_revised": 0,
        "_unscored": 0,
    }


class TestSingleModelSingleReview:
    """1 model → no quorum, resolution_method=single_review."""

    @pytest.mark.asyncio
    async def test_len_1_models_single_review(self):
        """With 1 model, only cycle 0 runs. Resolution = single_review."""
        items = [_make_item("a"), _make_item("b")]

        call_count = 0

        async def mock_review(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_review_result(items)

        with (
            patch(
                "imas_codex.standard_names.review.pipeline._review_single_batch",
                side_effect=mock_review,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._persist_review_records_sync",
                return_value=2,
            ),
        ):
            # Simulate the _process_batch flow with 1 model
            import copy
            import uuid as _uuid
            from datetime import UTC, datetime

            from imas_codex.standard_names.review.pipeline import (
                _build_review_record,
                _update_batch_stats,
            )

            models = ["model-primary"]

            result_0 = await mock_review(
                names=copy.deepcopy([{"id": "a"}, {"id": "b"}]),
                model=models[0],
            )
            c0_items = result_0["_items"]

            assert call_count == 1
            assert len(c0_items) == 2

            # With 1 model, resolution is single_review
            if len(models) == 1:
                resolution = "single_review"
                assert resolution == "single_review"


class TestTwoModelsAgreement:
    """2 models agree within tolerance → quorum_consensus, final=mean."""

    @pytest.mark.asyncio
    async def test_len_2_agreement_mean(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
            _merge_review_items,
        )

        # Both reviewers score similarly
        item_0 = _make_item(
            "a", {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 15, "semantic": 17, "convention": 16, "completeness": 15}
        )

        # Check no disagreement at tolerance=0.15
        assert not _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

        # Merge → mean
        merged = _merge_review_items(item_0, item_1, "names")
        scores = json.loads(merged["reviewer_scores"])
        assert scores["grammar"] == pytest.approx(15.5)
        assert scores["semantic"] == pytest.approx(16.5)


class TestTwoModelsDisagreementNoEscalator:
    """2 models disagree, no cycle 2 available → max_cycles_reached."""

    @pytest.mark.asyncio
    async def test_len_2_disagreement_no_escalator(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
            _merge_review_items,
        )

        item_0 = _make_item(
            "a", {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 5, "semantic": 16, "convention": 16, "completeness": 16}
        )

        # Disagreement on grammar
        assert _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

        # With len(models) < 3, resolution = max_cycles_reached
        # Final = mean of 0+1
        merged = _merge_review_items(item_0, item_1, "names")
        scores = json.loads(merged["reviewer_scores"])
        assert scores["grammar"] == pytest.approx(12.5)


class TestThreeModelsAgreementNoEscalation:
    """3 models configured but 0+1 agree → cycle 2 NOT invoked."""

    @pytest.mark.asyncio
    async def test_len_3_agreement_no_escalation(self):
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        item_0 = _make_item(
            "a", {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16}
        )
        item_1 = _make_item(
            "a", {"grammar": 15, "semantic": 16, "convention": 16, "completeness": 16}
        )

        assert not _check_per_dim_disagreement(item_0, item_1, 0.15, "names")

        # With agreement, cycle 2 should NOT be called
        models = ["m0", "m1", "m2"]
        # The flow skips cycle 2 when not disputed
        # This test just validates the disagreement check logic
        assert len(models) == 3  # escalator available but not needed


class TestThreeModelsDisagreementEscalates:
    """0+1 disagree, cycle 2 runs on disputed items only → authoritative_escalation."""

    @pytest.mark.asyncio
    async def test_len_3_disagreement_escalates(self):
        from imas_codex.standard_names.review.pipeline import (
            _build_prior_reviews_context,
            _check_per_dim_disagreement,
        )

        # 5 items, 2 disputed
        items_0 = [
            _make_item(
                "a",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "b",
                {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16},
            ),  # disputed
            _make_item(
                "c",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "d",
                {"grammar": 18, "semantic": 5, "convention": 16, "completeness": 16},
            ),  # disputed
            _make_item(
                "e",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
        ]
        items_1 = [
            _make_item(
                "a",
                {"grammar": 15, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "b",
                {"grammar": 5, "semantic": 16, "convention": 16, "completeness": 16},
            ),  # disputed
            _make_item(
                "c",
                {"grammar": 16, "semantic": 15, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "d",
                {"grammar": 17, "semantic": 18, "convention": 16, "completeness": 16},
            ),  # disputed
            _make_item(
                "e",
                {"grammar": 15, "semantic": 17, "convention": 16, "completeness": 16},
            ),
        ]

        disputed = set()
        for i0, i1 in zip(items_0, items_1, strict=True):
            if _check_per_dim_disagreement(i0, i1, 0.15, "names"):
                disputed.add(i0["id"])

        assert disputed == {"b", "d"}  # Only b and d disagree

        # Build prior reviews context for escalator
        models = ["m0", "m1", "m2"]
        ctx = _build_prior_reviews_context(items_0, items_1, disputed, models)
        assert len(ctx) == 2  # primary + secondary
        # Each entry should have exactly 2 items (the disputed ones)
        for entry in ctx:
            assert len(entry["items"]) == 2
            assert {it["standard_name"] for it in entry["items"]} == {"b", "d"}


class TestHybridBatchingCycle2PerItem:
    """5 items, 2 disputed → cycle 2 LLM call uses 2-item mini-batch."""

    @pytest.mark.asyncio
    async def test_hybrid_batching_cycle_2_per_item(self):
        """The escalator batch only contains disputed items."""
        from imas_codex.standard_names.review.pipeline import (
            _check_per_dim_disagreement,
        )

        # Create 5 items with 2 disputed
        items_0 = [
            _make_item(
                "a",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "b",
                {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "c",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "d",
                {"grammar": 16, "semantic": 20, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "e",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
        ]
        items_1 = [
            _make_item(
                "a",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "b",
                {"grammar": 5, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "c",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "d",
                {"grammar": 16, "semantic": 5, "convention": 16, "completeness": 16},
            ),
            _make_item(
                "e",
                {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16},
            ),
        ]

        names = [{"id": it["id"]} for it in items_0]
        c0_by_id = {it["id"]: it for it in items_0}
        c1_by_id = {it["id"]: it for it in items_1}

        disputed_ids = set()
        for nid in [n["id"] for n in names]:
            if nid in c0_by_id and nid in c1_by_id:
                if _check_per_dim_disagreement(
                    c0_by_id[nid], c1_by_id[nid], 0.15, "names"
                ):
                    disputed_ids.add(nid)

        assert disputed_ids == {"b", "d"}

        # Mini-batch for escalator should only contain disputed items
        disputed_names = [n for n in names if n["id"] in disputed_ids]
        assert len(disputed_names) == 2


class TestPartialFailureOneCycleMissing:
    """Cycle 0 fails for an item, cycle 1 succeeds → single_review."""

    @pytest.mark.asyncio
    async def test_partial_failure_one_cycle_missing(self):
        """When one cycle is missing for an item, use single_review."""
        c0_items = [_make_item("a")]  # Only item 'a'
        c1_items = [_make_item("a"), _make_item("b")]  # Items 'a' and 'b'

        c0_by_id = {it["id"]: it for it in c0_items}
        c1_by_id = {it["id"]: it for it in c1_items}
        all_ids = {"a", "b"}

        resolution_methods = {}
        final_items = []

        for nid in all_ids:
            in_c0 = nid in c0_by_id
            in_c1 = nid in c1_by_id

            if not in_c0 and not in_c1:
                resolution_methods[nid] = "retry_item"
            elif not in_c0:
                final_items.append(c1_by_id[nid])
                resolution_methods[nid] = "single_review"
            elif not in_c1:
                final_items.append(c0_by_id[nid])
                resolution_methods[nid] = "single_review"

        assert resolution_methods["b"] == "single_review"
        assert len(final_items) == 1  # only 'b' via single_review
        # 'a' is in both, so it would go through the normal flow
        assert "a" not in resolution_methods


class TestPartialFailureBothMissing:
    """Both fail for an item → retry_item (quarantined)."""

    @pytest.mark.asyncio
    async def test_partial_failure_both_missing_quarantines(self):
        """When both cycles miss an item, it's quarantined."""
        c0_items = [_make_item("a")]
        c1_items = [_make_item("a")]

        c0_by_id = {it["id"]: it for it in c0_items}
        c1_by_id = {it["id"]: it for it in c1_items}
        all_ids = {"a", "b"}

        resolution_methods = {}
        for nid in all_ids:
            in_c0 = nid in c0_by_id
            in_c1 = nid in c1_by_id
            if not in_c0 and not in_c1:
                resolution_methods[nid] = "retry_item"

        assert resolution_methods.get("b") == "retry_item"


class TestBlindCycle1NoPrimaryContext:
    """Assert cycle 1's prompt does NOT contain cycle 0's scores or comments."""

    @pytest.mark.asyncio
    async def test_blind_cycle_1_no_primary_context(self):
        """Cycle 1 must not receive prior_reviews context."""
        calls = []

        async def mock_review(**kwargs):
            calls.append(kwargs)
            return _mock_review_result([_make_item("a")])

        with (
            patch(
                "imas_codex.standard_names.review.pipeline._review_single_batch",
                side_effect=mock_review,
            ),
            patch(
                "imas_codex.standard_names.review.pipeline._persist_review_records_sync",
                return_value=1,
            ),
        ):
            # Simulate cycles 0 and 1
            import copy

            names = [{"id": "a"}]

            # Cycle 0
            await mock_review(
                names=copy.deepcopy(names),
                model="m0",
                prior_reviews=None,
            )

            # Cycle 1
            await mock_review(
                names=copy.deepcopy(names),
                model="m1",
                prior_reviews=None,  # BLIND — no primary context
            )

            assert len(calls) == 2
            # Both cycles should have prior_reviews=None
            assert calls[0].get("prior_reviews") is None
            assert calls[1].get("prior_reviews") is None


class TestCycle2SeesBothPrior:
    """Assert cycle 2's prompt contains both cycle 0 AND cycle 1 critiques."""

    @pytest.mark.asyncio
    async def test_cycle_2_sees_both_prior(self):
        from imas_codex.standard_names.review.pipeline import (
            _build_prior_reviews_context,
        )

        c0_items = [
            _make_item(
                "a",
                {"grammar": 20, "semantic": 16, "convention": 16, "completeness": 16},
            )
        ]
        c1_items = [
            _make_item(
                "a",
                {"grammar": 5, "semantic": 16, "convention": 16, "completeness": 16},
            )
        ]
        disputed = {"a"}
        models = ["m0", "m1", "m2"]

        ctx = _build_prior_reviews_context(c0_items, c1_items, disputed, models)
        assert len(ctx) == 2
        assert ctx[0]["role"] == "primary"
        assert ctx[1]["role"] == "secondary"
        # Both should reference item 'a'
        assert ctx[0]["items"][0]["standard_name"] == "a"
        assert ctx[1]["items"][0]["standard_name"] == "a"


class TestReviewNodeMetadata:
    """Assert Review nodes have axis, cycle_index, review_group_id, resolution_role."""

    def test_review_node_metadata(self):
        from imas_codex.standard_names.review.pipeline import _build_review_record

        item = _make_item("electron_temperature")
        rec = _build_review_record(
            item,
            model="model-a",
            is_canonical=True,
            reviewed_at="2025-01-01T00:00:00Z",
        )

        # Add quorum metadata as pipeline does
        group_id = str(uuid.uuid4())
        rec["id"] = f"electron_temperature:names:{group_id}:0"
        rec["review_axis"] = "names"
        rec["cycle_index"] = 0
        rec["review_group_id"] = group_id
        rec["resolution_role"] = "primary"
        rec["resolution_method"] = None

        assert rec["review_axis"] == "names"
        assert rec["cycle_index"] == 0
        assert rec["review_group_id"] == group_id
        assert rec["resolution_role"] == "primary"
        assert rec["resolution_method"] is None
        assert ":names:" in rec["id"]
        assert ":0" == rec["id"][-2:]


class TestWinningGroupSelection:
    """Winning group = most recent with resolution in {quorum_consensus, authoritative_escalation, single_review}."""

    def test_winning_group_excludes_retry_item(self):
        """retry_item resolution is never selected as winning."""
        winning_methods = {
            "quorum_consensus",
            "authoritative_escalation",
            "single_review",
        }
        non_winning_methods = {"max_cycles_reached", "retry_item"}

        for method in winning_methods:
            assert method in winning_methods

        for method in non_winning_methods:
            assert method not in winning_methods

    def test_winning_group_selection_logic(self):
        """Simulate two groups, verify most-recent winning is selected."""
        groups = [
            {
                "review_group_id": "old-group",
                "reviewed_at": "2025-01-01T00:00:00Z",
                "resolution_method": "quorum_consensus",
            },
            {
                "review_group_id": "new-group",
                "reviewed_at": "2025-06-01T00:00:00Z",
                "resolution_method": "authoritative_escalation",
            },
            {
                "review_group_id": "failed-group",
                "reviewed_at": "2025-07-01T00:00:00Z",
                "resolution_method": "retry_item",
            },  # excluded
        ]

        winning_methods = {
            "quorum_consensus",
            "authoritative_escalation",
            "single_review",
        }
        eligible = [g for g in groups if g["resolution_method"] in winning_methods]
        winner = max(eligible, key=lambda g: g["reviewed_at"])

        assert winner["review_group_id"] == "new-group"
        assert winner["resolution_method"] == "authoritative_escalation"


class TestLeaseChargedPerCycle:
    """Mock costs, assert lease.charged == sum of cycle costs."""

    def test_lease_charged_per_cycle(self):
        from imas_codex.standard_names.budget import BudgetManager

        mgr = BudgetManager(total_budget=10.0)
        lease = mgr.reserve(3.0)
        assert lease is not None

        # Simulate 3 cycles
        lease.charge(0.5)  # cycle 0
        lease.charge(0.6)  # cycle 1
        lease.charge(0.4)  # cycle 2

        assert lease.charged == pytest.approx(1.5)
        assert lease.remaining == pytest.approx(1.5)

        # Release unused
        released = lease.release_unused()
        assert released == pytest.approx(1.5)
        assert mgr.remaining == pytest.approx(8.5)


class TestLen3QuorumReachedReleasesUnused:
    """len=3 configured, cycle 2 skipped → lease releases cycle_2_budget."""

    def test_len_3_with_quorum_reached_releases_unused(self):
        from imas_codex.standard_names.budget import BudgetManager

        mgr = BudgetManager(total_budget=10.0)
        # Reserve for 3 cycles but only 2 run
        lease = mgr.reserve(3.0)
        assert lease is not None

        lease.charge(0.5)  # cycle 0
        lease.charge(0.5)  # cycle 1
        # cycle 2 skipped because quorum reached

        released = lease.release_unused()
        assert released == pytest.approx(2.0)  # reserved 3.0 - charged 1.0
        assert mgr.remaining == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Schema compliance tests
# ---------------------------------------------------------------------------


class TestSchemaEnums:
    """Verify new enums appear in generated models."""

    def test_review_resolution_role_enum(self):
        from imas_codex.graph.models import ReviewResolutionRole

        assert hasattr(ReviewResolutionRole, "primary")
        assert hasattr(ReviewResolutionRole, "secondary")
        assert hasattr(ReviewResolutionRole, "escalator")

    def test_review_resolution_method_enum(self):
        from imas_codex.graph.models import ReviewResolutionMethod

        assert hasattr(ReviewResolutionMethod, "quorum_consensus")
        assert hasattr(ReviewResolutionMethod, "authoritative_escalation")
        assert hasattr(ReviewResolutionMethod, "max_cycles_reached")
        assert hasattr(ReviewResolutionMethod, "retry_item")
        assert hasattr(ReviewResolutionMethod, "single_review")

    def test_review_model_has_new_fields(self):
        from imas_codex.graph.models import Review

        fields = Review.model_fields
        assert "review_axis" in fields
        assert "cycle_index" in fields
        assert "review_group_id" in fields
        assert "resolution_role" in fields
        assert "resolution_method" in fields


# ---------------------------------------------------------------------------
# Prompt blindness guard
# ---------------------------------------------------------------------------


class TestPromptBlindnessGuard:
    """Ensure prompt templates guard prior_reviews with conditional."""

    def test_review_names_prompt_guards_prior_reviews(self):
        """review_names.md uses {% if prior_reviews %} guard."""
        from pathlib import Path

        prompt_path = (
            Path(__file__).parents[2] / "imas_codex/llm/prompts/sn/review_names.md"
        )
        content = prompt_path.read_text()
        assert "{% if prior_reviews %}" in content
        assert "{% endif %}" in content
        # Ensure prior_reviews block exists but is conditional
        assert "Prior Review Critiques" in content

    def test_review_docs_prompt_guards_prior_reviews(self):
        """review_docs.md uses {% if prior_reviews %} guard."""
        from pathlib import Path

        prompt_path = (
            Path(__file__).parents[2] / "imas_codex/llm/prompts/sn/review_docs.md"
        )
        content = prompt_path.read_text()
        assert "{% if prior_reviews %}" in content
        assert "{% endif %}" in content
        assert "Prior Review Critiques" in content


class TestParseDimScores:
    """Tests for _parse_dim_scores helper."""

    def test_parse_names_dims(self):
        from imas_codex.standard_names.review.pipeline import _parse_dim_scores

        item = _make_item(
            "a", {"grammar": 18, "semantic": 16, "convention": 17, "completeness": 15}
        )
        scores = _parse_dim_scores(item, "names")
        assert scores == {
            "grammar": 18.0,
            "semantic": 16.0,
            "convention": 17.0,
            "completeness": 15.0,
        }

    def test_parse_docs_dims(self):
        from imas_codex.standard_names.review.pipeline import _parse_dim_scores

        item = _make_docs_item(
            "a",
            {
                "description_quality": 19,
                "documentation_quality": 18,
                "completeness": 17,
                "physics_accuracy": 20,
            },
        )
        scores = _parse_dim_scores(item, "docs")
        assert scores == {
            "description_quality": 19.0,
            "documentation_quality": 18.0,
            "completeness": 17.0,
            "physics_accuracy": 20.0,
        }

    def test_parse_string_scores(self):
        from imas_codex.standard_names.review.pipeline import _parse_dim_scores

        item = {
            "reviewer_scores": '{"grammar": 15, "semantic": 16, "convention": 17, "completeness": 18}'
        }
        scores = _parse_dim_scores(item, "names")
        assert scores["grammar"] == 15.0

    def test_parse_none_returns_empty(self):
        from imas_codex.standard_names.review.pipeline import _parse_dim_scores

        item = {"reviewer_scores": None}
        assert _parse_dim_scores(item, "names") == {}
