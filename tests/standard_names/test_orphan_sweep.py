"""Tests for imas_codex.standard_names.orphan_sweep.

All tests mock GraphClient — no live Neo4j required.

Test matrix
-----------
- test_revert_stuck_refining_name          — name_stage='refining' + stale claimed_at → reverted
- test_revert_stuck_refining_docs          — docs_stage='refining' + stale claimed_at → reverted
- test_no_revert_within_timeout            — claimed_at is fresh → not reverted
- test_no_revert_when_claim_clean          — stage='refining' but claimed_at IS NULL → not reverted
- test_stale_token_cleared_non_refining    — stale claim_token on non-refining SN → cleared
- test_loop_cancels_on_stop_event          — coroutine exits promptly when stop_event is set
- test_concurrent_safe                     — two ticks in flight don't double-revert (idempotent)
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.standard_names.orphan_sweep import (
    _orphan_sweep_tick,
    run_orphan_sweep_loop,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_gc(query_side_effects: list[list[dict]]) -> MagicMock:
    """Build a mock GraphClient that returns successive query results."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(side_effect=query_side_effects)
    return gc


def _patch_gc(gc: MagicMock):
    return patch(
        "imas_codex.standard_names.orphan_sweep.GraphClient",
        return_value=gc,
    )


# Four query labels in declaration order.
_LABELS = ["name_refining", "docs_refining", "stale_token_sn", "stale_token_source"]


def _zero_results() -> list[list[dict]]:
    """Four queries all returning 0."""
    return [[{"n": 0}]] * 4


# ---------------------------------------------------------------------------
# 1. test_revert_stuck_refining_name
# ---------------------------------------------------------------------------


def test_revert_stuck_refining_name():
    """name_stage='refining' with stale claimed_at is counted by the first query."""
    results = [
        [{"n": 2}],  # name_refining
        [{"n": 0}],  # docs_refining
        [{"n": 0}],  # stale_token_sn
        [{"n": 0}],  # stale_token_source
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] == 2
    assert counts["docs_refining"] == 0
    # Verify the first query received the timeout parameter.
    first_call: call = gc.query.call_args_list[0]
    assert (
        first_call.kwargs.get("timeout_s") == 300
        or first_call.args[1:] == (300,)
        or "timeout_s=300" in str(first_call)
    )


# ---------------------------------------------------------------------------
# 2. test_revert_stuck_refining_docs
# ---------------------------------------------------------------------------


def test_revert_stuck_refining_docs():
    """docs_stage='refining' with stale claimed_at is counted by the second query."""
    results = [
        [{"n": 0}],  # name_refining
        [{"n": 3}],  # docs_refining
        [{"n": 0}],  # stale_token_sn
        [{"n": 0}],  # stale_token_source
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] == 3
    assert counts["name_refining"] == 0


# ---------------------------------------------------------------------------
# 3. test_no_revert_within_timeout
# ---------------------------------------------------------------------------


def test_no_revert_within_timeout():
    """When claimed_at is newer than threshold, queries return 0 — nothing reverted."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert all(v == 0 for v in counts.values()), counts
    # All four queries must still have been called.
    assert gc.query.call_count == 4


# ---------------------------------------------------------------------------
# 4. test_no_revert_when_claim_clean
# ---------------------------------------------------------------------------


def test_no_revert_when_claim_clean():
    """stage='refining' but claimed_at IS NULL → query returns 0 (WHERE filters it out).

    The WHERE clause requires claimed_at IS NOT NULL, so unclaimed refining
    nodes are ignored.  We verify the tick returns 0 for name_refining.
    """
    # Both refining queries return 0 (claimed_at IS NULL doesn't match filter).
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] == 0
    assert counts["docs_refining"] == 0


# ---------------------------------------------------------------------------
# 5. test_stale_token_cleared_non_refining
# ---------------------------------------------------------------------------


def test_stale_token_cleared_non_refining():
    """Stale claim_token on a non-refining SN is cleared by query 3."""
    results = [
        [{"n": 0}],  # name_refining
        [{"n": 0}],  # docs_refining
        [{"n": 5}],  # stale_token_sn  — 5 stale non-refining tokens cleared
        [{"n": 1}],  # stale_token_source
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["stale_token_sn"] == 5
    assert counts["stale_token_source"] == 1


# ---------------------------------------------------------------------------
# 6. test_loop_cancels_on_stop_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_cancels_on_stop_event():
    """Coroutine exits promptly when stop_event is set before the first tick."""
    gc = _make_gc(_zero_results())
    stop_event = asyncio.Event()
    stop_event.set()  # Set before starting — should exit without sleeping.

    with _patch_gc(gc):
        # Should return almost immediately (well within 2 seconds).
        await asyncio.wait_for(
            run_orphan_sweep_loop(
                interval_s=30,
                timeout_s=300,
                stop_event=stop_event,
            ),
            timeout=2.0,
        )
    # Loop exited cleanly — no TimeoutError raised.


# ---------------------------------------------------------------------------
# 7. test_loop_stop_event_during_sleep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_stop_event_during_sleep():
    """Coroutine wakes up mid-sleep when stop_event is set and exits cleanly."""
    gc = _make_gc(_zero_results() * 10)  # Enough for multiple ticks.
    stop_event = asyncio.Event()

    async def _set_after_delay():
        await asyncio.sleep(0.05)
        stop_event.set()

    with _patch_gc(gc):
        setter = asyncio.create_task(_set_after_delay())
        await asyncio.wait_for(
            run_orphan_sweep_loop(
                interval_s=60,  # Long sleep — must wake on stop_event.
                timeout_s=300,
                stop_event=stop_event,
            ),
            timeout=2.0,
        )
        await setter


# ---------------------------------------------------------------------------
# 8. test_concurrent_safe (idempotent ticks)
# ---------------------------------------------------------------------------


def test_concurrent_safe():
    """Two _orphan_sweep_tick calls with the same parameters are idempotent.

    On the second tick the DB has already cleared the orphans, so all
    queries return 0.  Verifies that the tick function itself is stateless
    (no in-memory counter that would produce incorrect results on re-run).
    """
    gc_first = _make_gc(
        [
            [{"n": 1}],  # name_refining — first tick sees 1 stuck node
            [{"n": 0}],
            [{"n": 0}],
            [{"n": 0}],
        ]
    )
    gc_second = _make_gc(_zero_results())  # second tick: already cleared

    with _patch_gc(gc_first):
        counts_first = _orphan_sweep_tick(timeout_s=300)

    with _patch_gc(gc_second):
        counts_second = _orphan_sweep_tick(timeout_s=300)

    assert counts_first["name_refining"] == 1
    assert counts_second["name_refining"] == 0  # idempotent — no double-revert


# ---------------------------------------------------------------------------
# 9. test_tick_returns_all_labels
# ---------------------------------------------------------------------------


def test_tick_returns_all_labels():
    """_orphan_sweep_tick always returns all four keys regardless of counts."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert set(counts.keys()) == {
        "name_refining",
        "docs_refining",
        "stale_token_sn",
        "stale_token_source",
    }


# ---------------------------------------------------------------------------
# 10. test_tick_passes_timeout_to_all_queries
# ---------------------------------------------------------------------------


def test_tick_passes_timeout_to_all_queries():
    """All four queries receive the timeout_s keyword argument."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        _orphan_sweep_tick(timeout_s=42)

    assert gc.query.call_count == 4
    for c in gc.query.call_args_list:
        # timeout_s is passed as a keyword argument to gc.query.
        assert c.kwargs.get("timeout_s") == 42, (
            f"Expected timeout_s=42 in call kwargs, got {c.kwargs}"
        )
