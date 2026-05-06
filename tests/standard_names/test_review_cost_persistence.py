"""Guardrail: every StandardNameReview node has non-NULL llm_cost.

Root cause: quarantined names called ``persist_reviewed_name`` without
``llm_cost`` — the parameter defaulted to ``None`` and was written as
NULL to Neo4j.  Normal (LLM-executed) reviews passed ``llm_cost=cost``
and were fine.

These tests verify:

1. After ``write_reviews``, every record has ``llm_cost`` as a float (not None).
2. Cache-hit case (cost=0) writes ``llm_cost=0.0``, not NULL.
3. Quarantine skip path passes ``llm_cost=0.0`` explicitly.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

# ---------------------------------------------------------------------------
# 1. write_reviews coalesces None → 0.0 for cost and token fields
# ---------------------------------------------------------------------------


def test_write_reviews_coalesces_none_cost_to_zero() -> None:
    """Review with llm_cost=None in the input dict → 0.0 in the Cypher batch."""
    from imas_codex.standard_names.graph_ops import write_reviews

    records = [
        {
            "id": "test_name:names:abc:0",
            "standard_name_id": "test_name",
            "model": "test-model",
            "reviewer_model": "test-model",
            "model_family": "other",
            "is_canonical": True,
            "score": 0.5,
            "scores_json": "{}",
            "tier": "inadequate",
            "comments": "",
            "reviewed_at": "2026-01-01T00:00:00+00:00",
            "review_axis": "names",
            "cycle_index": 0,
            "review_group_id": "abc",
            "resolution_role": "primary",
            # Deliberately omit llm_cost / token fields
        }
    ]

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
        MockGC.return_value.__enter__.return_value.query.return_value = []
        write_reviews(records)

    call = MockGC.return_value.__enter__.return_value.query.call_args
    batch = call.kwargs.get("batch") or call.args[1]
    assert batch[0]["llm_cost"] == 0.0, "None llm_cost should be coalesced to 0.0"
    assert batch[0]["llm_tokens_in"] == 0, "None llm_tokens_in should be coalesced to 0"
    assert batch[0]["llm_tokens_out"] == 0, (
        "None llm_tokens_out should be coalesced to 0"
    )
    assert batch[0]["llm_tokens_cached_read"] == 0
    assert batch[0]["llm_tokens_cached_write"] == 0


# ---------------------------------------------------------------------------
# 2. Cache-hit (cost=0.0) is preserved, not converted to NULL
# ---------------------------------------------------------------------------


def test_write_reviews_preserves_zero_cost() -> None:
    """A cost of exactly 0.0 (cache hit) must remain 0.0, not become NULL."""
    from imas_codex.standard_names.graph_ops import write_reviews

    records = [
        {
            "id": "cached_name:names:def:0",
            "standard_name_id": "cached_name",
            "model": "test-model",
            "reviewer_model": "test-model",
            "model_family": "other",
            "is_canonical": True,
            "score": 0.9,
            "scores_json": "{}",
            "tier": "outstanding",
            "comments": "cache hit",
            "reviewed_at": "2026-01-01T00:00:00+00:00",
            "review_axis": "names",
            "cycle_index": 0,
            "review_group_id": "def",
            "resolution_role": "primary",
            "llm_cost": 0.0,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
        }
    ]

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
        MockGC.return_value.__enter__.return_value.query.return_value = []
        write_reviews(records)

    call = MockGC.return_value.__enter__.return_value.query.call_args
    batch = call.kwargs.get("batch") or call.args[1]
    assert batch[0]["llm_cost"] == 0.0
    assert isinstance(batch[0]["llm_cost"], float)


# ---------------------------------------------------------------------------
# 3. Quarantine skip path supplies llm_cost=0.0
# ---------------------------------------------------------------------------


async def test_quarantine_skip_passes_zero_cost() -> None:
    """The quarantine-skip codepath in review_name passes llm_cost=0.0."""
    from unittest.mock import MagicMock

    from imas_codex.standard_names.workers import process_review_name_batch

    items = [
        {
            "id": "quarantined_electron_temperature",
            "claim_token": "tok-123",
            "validation_status": "quarantined",
        }
    ]

    persist_calls: list[dict] = []

    def _fake_persist(**kwargs):
        persist_calls.append(kwargs)
        return "exhausted"

    # Provide required positional args: a mock BudgetManager and stop_event
    mock_mgr = MagicMock()
    stop_event = asyncio.Event()

    # persist_reviewed_name is imported inside the function body, so
    # patch it at the source module where it's defined.
    with patch(
        "imas_codex.standard_names.graph_ops.persist_reviewed_name",
        side_effect=_fake_persist,
    ):
        result = await process_review_name_batch(items, mock_mgr, stop_event)

    assert result == 1, f"Expected 1 processed, got {result}"
    assert len(persist_calls) == 1

    call = persist_calls[0]
    assert call["llm_cost"] == 0.0, f"Expected llm_cost=0.0, got {call.get('llm_cost')}"
    assert call["llm_tokens_in"] == 0
    assert call["llm_tokens_out"] == 0
    assert call["llm_tokens_cached_read"] == 0
    assert call["llm_tokens_cached_write"] == 0
    assert call["llm_service"] == "standard-names"
    assert call["model"] == "(skipped: quarantined)"


# ---------------------------------------------------------------------------
# 4. persist_reviewed_name stamps llm_cost on the Review node
# ---------------------------------------------------------------------------


def test_persist_reviewed_name_stamps_cost_on_review_node() -> None:
    """persist_reviewed_name passes llm_cost through to write_reviews."""
    from imas_codex.standard_names.graph_ops import persist_reviewed_name

    write_review_calls: list = []

    def _fake_write_reviews(records, **kw):
        write_review_calls.append(records)
        return len(records)

    # Mock the graph query for the main transaction and write_reviews
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,  # noqa: N806
        patch(
            "imas_codex.standard_names.graph_ops.write_reviews",
            side_effect=_fake_write_reviews,
        ),
    ):
        gc_ctx = MockGC.return_value.__enter__.return_value
        # Simulate: claim_token matches, chain_length=0
        gc_ctx.query.return_value = [
            {"chain_length": 0, "name_stage": "drafted", "target_stage": "reviewed"}
        ]

        persist_reviewed_name(
            sn_id="test_name",
            claim_token="tok-abc",
            score=0.0,
            scores={"grammar": 0.0},
            comments="quarantined",
            model="(skipped: quarantined)",
            llm_cost=0.0,
            llm_tokens_in=0,
            llm_tokens_out=0,
        )

    assert len(write_review_calls) == 1
    review_record = write_review_calls[0][0]
    assert review_record["llm_cost"] == 0.0
    assert review_record["llm_tokens_in"] == 0
    assert review_record["llm_tokens_out"] == 0


# ---------------------------------------------------------------------------
# 5. Escalation (cycle_index > 0) reviews also carry cost
# ---------------------------------------------------------------------------


def test_escalation_review_carries_cost() -> None:
    """Review records with cycle_index > 0 preserve their cost."""
    from imas_codex.standard_names.graph_ops import write_reviews

    records = [
        {
            "id": "test_name:names:ghi:2",
            "standard_name_id": "test_name",
            "model": "openrouter/anthropic/claude-opus-4.6",
            "reviewer_model": "openrouter/anthropic/claude-opus-4.6",
            "model_family": "anthropic",
            "is_canonical": False,
            "score": 0.65,
            "scores_json": json.dumps({"grammar": 0.7}),
            "tier": "good",
            "comments": "escalation review",
            "reviewed_at": "2026-01-01T00:00:00+00:00",
            "review_axis": "names",
            "cycle_index": 2,
            "review_group_id": "ghi",
            "resolution_role": "escalator",
            "llm_cost": 0.042,
            "llm_tokens_in": 3000,
            "llm_tokens_out": 500,
        }
    ]

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
        MockGC.return_value.__enter__.return_value.query.return_value = []
        write_reviews(records)

    # First query call is the MERGE with the batch parameter
    calls = MockGC.return_value.__enter__.return_value.query.call_args_list
    merge_call = calls[0]
    batch = merge_call.kwargs.get("batch") or merge_call.args[1]
    assert batch[0]["llm_cost"] == 0.042
    assert batch[0]["llm_tokens_in"] == 3000
    assert batch[0]["llm_tokens_out"] == 500
    assert batch[0]["cycle_index"] == 2
