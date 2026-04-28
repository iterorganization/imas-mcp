"""Regression tests for Review-node write correctness.

Covers the bug where ``reviewer_model`` and ``verdict`` were NULL on all
Review nodes even though the parent StandardName stored the model name
correctly.  Root cause: ``write_reviews`` stored ``r.model`` but consumers
queried ``rv.reviewer_model``; ``verdict`` was never written at all.

All tests are offline — GraphClient is fully mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    sn_id: str = "electron_temperature",
    model: str = "openrouter/anthropic/claude-opus-4.6",
    reviewer_model: str | None = None,
    verdict: str = "accept",
    score: float = 0.725,
    tier: str = "good",
    comments: str = "Looks fine",
) -> dict:
    """Minimal Review record matching ``_build_review_record`` output."""
    import uuid
    from datetime import UTC, datetime

    group_id = str(uuid.uuid4())
    reviewed_at = datetime.now(UTC).isoformat()
    rid = f"{sn_id}:names:{group_id}:0"
    return {
        "id": rid,
        "standard_name_id": sn_id,
        "model": model,
        # reviewer_model is the consumer-facing alias — should equal model
        "reviewer_model": reviewer_model if reviewer_model is not None else model,
        "model_family": "anthropic",
        "is_canonical": True,
        "score": score,
        "scores_json": json.dumps({"clarity": 18, "physics": 20}),
        "tier": tier,
        "verdict": verdict,
        "comments": comments,
        "comments_per_dim_json": None,
        "reviewed_at": reviewed_at,
        "review_axis": "names",
        "cycle_index": 0,
        "review_group_id": group_id,
        "resolution_role": "primary",
        "resolution_method": "single_review",
        "llm_model": model,
        "llm_cost": 0.001,
        "llm_tokens_in": 500,
        "llm_tokens_out": 100,
        "llm_tokens_cached_read": 0,
        "llm_tokens_cached_write": 0,
        "llm_at": reviewed_at,
        "llm_service": "standard-names",
    }


def _call_write_reviews(records: list[dict]) -> tuple[int, list]:
    """Call ``write_reviews`` with a fully mocked GraphClient.

    Returns ``(return_value, query_call_args_list)`` so callers can
    inspect the Cypher and parameters that were passed.
    """
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_reviews

        result = write_reviews(records)

    return result, mock_gc.query.call_args_list


def _extract_batch(call_args_list: list) -> list[dict]:
    """Pull the ``batch`` kwarg from the first MERGE StandardNameReview query call."""
    for c in call_args_list:
        kwargs = c.kwargs
        if "batch" in kwargs:
            return kwargs["batch"]
    raise AssertionError("No 'batch' kwarg found in any query call")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteReviewsReviewerModel:
    """reviewer_model must be persisted on the StandardNameReview node."""

    def test_reviewer_model_populated_from_model(self) -> None:
        """reviewer_model is set to the same value as model."""
        record = _make_record(model="openrouter/anthropic/claude-opus-4.6")
        _, calls = _call_write_reviews([record])

        batch = _extract_batch(calls)
        assert len(batch) == 1
        row = batch[0]

        assert row["reviewer_model"] == "openrouter/anthropic/claude-opus-4.6", (
            f"Expected reviewer_model='openrouter/anthropic/claude-opus-4.6', "
            f"got {row['reviewer_model']!r}"
        )

    def test_reviewer_model_falls_back_to_model_when_absent(self) -> None:
        """If record omits reviewer_model, the model field is used as fallback."""
        record = _make_record(model="openrouter/google/gemini-2.5-pro")
        # Simulate a record built without reviewer_model (pre-fix code path)
        record.pop("reviewer_model", None)

        _, calls = _call_write_reviews([record])
        batch = _extract_batch(calls)
        row = batch[0]

        assert row["reviewer_model"] == "openrouter/google/gemini-2.5-pro", (
            f"Fallback to model failed: got {row['reviewer_model']!r}"
        )

    def test_reviewer_model_in_cypher_set_clause(self) -> None:
        """The Cypher must SET r.reviewer_model so the graph node gets it."""
        record = _make_record()
        _, calls = _call_write_reviews([record])

        # The first call with a non-empty string arg is the MERGE StandardNameReview query
        merge_cypher = None
        for c in calls:
            args = c.args
            if args and "MERGE (r:StandardNameReview" in args[0]:
                merge_cypher = args[0]
                break

        assert merge_cypher is not None, "No MERGE StandardNameReview query found"
        assert "r.reviewer_model = b.reviewer_model" in merge_cypher, (
            "Cypher is missing SET r.reviewer_model = b.reviewer_model"
        )

    def test_reviewer_model_not_null_for_all_model_slugs(self) -> None:
        """Spot-check several model slug formats to guard against edge cases."""
        model_slugs = [
            "openrouter/anthropic/claude-opus-4.6",
            "openrouter/google/gemini-2.5-pro",
            "openai/gpt-4o",
            "mistralai/mistral-large-2411",
        ]
        for model in model_slugs:
            record = _make_record(model=model)
            _, calls = _call_write_reviews([record])
            batch = _extract_batch(calls)
            row = batch[0]
            assert row["reviewer_model"] == model, (
                f"reviewer_model mismatch for model={model!r}: got {row['reviewer_model']!r}"
            )


class TestWriteReviewsVerdict:
    """verdict must be persisted on the StandardNameReview node."""

    def test_verdict_populated_from_record(self) -> None:
        """verdict is forwarded from the record into the Cypher batch."""
        for verdict in ("accept", "reject", "revise"):
            record = _make_record(verdict=verdict)
            _, calls = _call_write_reviews([record])
            batch = _extract_batch(calls)
            row = batch[0]
            assert row["verdict"] == verdict, (
                f"Expected verdict={verdict!r}, got {row['verdict']!r}"
            )

    def test_verdict_defaults_to_empty_string_when_absent(self) -> None:
        """Records without verdict must not raise — defaults to ''."""
        record = _make_record()
        record.pop("verdict", None)

        _, calls = _call_write_reviews([record])
        batch = _extract_batch(calls)
        row = batch[0]

        # Should be empty string (falsy), not None (which would be NULL in graph)
        assert row["verdict"] == "", f"Expected empty string, got {row['verdict']!r}"

    def test_verdict_in_cypher_set_clause(self) -> None:
        """The Cypher must SET r.verdict so the graph node gets it."""
        record = _make_record()
        _, calls = _call_write_reviews([record])

        merge_cypher = None
        for c in calls:
            args = c.args
            if args and "MERGE (r:StandardNameReview" in args[0]:
                merge_cypher = args[0]
                break

        assert merge_cypher is not None, "No MERGE StandardNameReview query found"
        assert "r.verdict = b.verdict" in merge_cypher, (
            "Cypher is missing SET r.verdict = b.verdict"
        )


class TestBuildReviewRecordIncludesNewFields:
    """_build_review_record must produce records with reviewer_model and verdict."""

    def _build(self, **kwargs) -> dict:
        from datetime import UTC, datetime

        from imas_codex.standard_names.review.pipeline import _build_review_record

        item = {
            "id": "electron_temperature",
            "reviewer_score": 0.75,
            "review_tier": "good",
            "reviewer_comments": "Fine",
            "reviewer_verdict": kwargs.pop("reviewer_verdict", "accept"),
        }
        return _build_review_record(
            item,
            model=kwargs.pop("model", "openrouter/anthropic/claude-opus-4.6"),
            is_canonical=True,
            reviewed_at=datetime.now(UTC).isoformat(),
            **kwargs,
        )

    def test_reviewer_model_equals_model(self) -> None:
        model = "openrouter/anthropic/claude-opus-4.6"
        rec = self._build(model=model)
        assert rec["reviewer_model"] == model, (
            f"reviewer_model {rec['reviewer_model']!r} != model {model!r}"
        )

    def test_verdict_comes_from_item(self) -> None:
        rec = self._build(reviewer_verdict="reject")
        assert rec["verdict"] == "reject", (
            f"Expected verdict='reject', got {rec['verdict']!r}"
        )

    def test_verdict_accept_default(self) -> None:
        rec = self._build(reviewer_verdict="accept")
        assert rec["verdict"] == "accept"

    def test_reviewer_model_not_none(self) -> None:
        rec = self._build()
        assert rec.get("reviewer_model") is not None
        assert rec["reviewer_model"] != ""

    def test_verdict_not_none(self) -> None:
        rec = self._build(reviewer_verdict="revise")
        assert rec.get("verdict") is not None
