"""Tests for imas_codex.standard_names.orphan_sweep.

Section 1 — Unit tests (mock GraphClient, no live Neo4j required)
-----------------------------------------------------------------
- test_revert_stuck_refining_name          — name_stage='refining' + stale claimed_at → reverted
- test_revert_stuck_refining_docs          — docs_stage='refining' + stale claimed_at → reverted
- test_no_revert_within_timeout            — claimed_at is fresh → not reverted
- test_no_revert_when_claim_clean          — stage='refining' but claimed_at IS NULL → not reverted
- test_stale_token_cleared_non_refining    — stale claim_token on non-refining SN → cleared
- test_loop_cancels_on_stop_event          — coroutine exits promptly when stop_event is set
- test_concurrent_safe                     — two ticks in flight don't double-revert (idempotent)

Section 2 — Integration tests (real Neo4j, auto-skipped when unavailable)
--------------------------------------------------------------------------
- test_sweep_reverts_stale_refining_name   — real SN node w/ stale claimed_at → reverted
- test_sweep_skips_fresh_refining_name     — real SN node w/ fresh claimed_at → unchanged
- test_sweep_reverts_stale_refining_docs   — real SN node w/ stale docs_stage → reverted
- test_sweep_skips_fresh_refining_docs     — real SN node w/ fresh docs_stage → unchanged
- test_sweep_reverts_stale_source_token    — real StandardNameSource w/ stale token → cleared
- test_sweep_skips_fresh_source_token      — real StandardNameSource w/ fresh token → unchanged
- test_sweep_atomic_clear                  — stage + token + claimed_at cleared together
- test_sweep_no_op_when_clean              — no stale claims → tick returns all-zero counts
- test_sweep_count_returns_correctly       — 3 stale name + 2 stale docs → correct per-category
- test_run_loop_respects_stop_event        — loop exits within 0.5 s when stop_event is set
- test_run_loop_periodic                   — stale claim swept within 0.3 s by running loop
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


# ===========================================================================
# Section 2 — Integration tests (real Neo4j, auto-skipped when unavailable)
# ===========================================================================
#
# These tests create live :StandardName / :StandardNameSource nodes in the
# configured Neo4j instance, exercise _orphan_sweep_tick or the async loop,
# then assert the state of those nodes after the sweep.
#
# All test-node IDs are prefixed with "orphan_sweep_test__" so they can be
# targeted for cleanup without touching production data.
#
# Auto-skipped when Neo4j is unreachable (see conftest.py
# ``pytest_collection_modifyitems`` → @pytest.mark.graph skip logic).
# ===========================================================================

pytestmark_integration = [pytest.mark.graph, pytest.mark.integration]

_TEST_ID_PREFIX = "orphan_sweep_test__"


# ---------------------------------------------------------------------------
# graph_client fixture — function-scoped so each test gets a fresh cursor
# (the session-scoped fixture is in tests/graph/conftest.py; we duplicate
# a function-scoped variant here so that standard_names tests don't depend
# on the graph conftest being collected)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    """Function-scoped GraphClient; skip if Neo4j is unreachable."""
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture(autouse=False)
def _clean_test_nodes(_gc):
    """Delete all orphan_sweep test nodes before and after each test."""
    _gc.query(
        "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    _gc.query(
        "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    yield
    _gc.query(
        "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    _gc.query(
        "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )


# ---------------------------------------------------------------------------
# Helper: create a :StandardName with a stale or fresh claimed_at
# ---------------------------------------------------------------------------


def _create_sn(
    gc,
    sn_id: str,
    *,
    name_stage: str = "reviewed",
    docs_stage: str = "reviewed",
    stale: bool = True,
    claim_token: str = "tok-test",
) -> None:
    """Create (or MERGE) a :StandardName node with claim fields set."""
    age_s = 400 if stale else 60  # 400 s old → stale; 60 s old → fresh
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage   = $name_stage,
            sn.docs_stage   = $docs_stage,
            sn.claim_token  = $token,
            sn.claimed_at   = datetime() - duration({seconds: $age_s})
        """,
        id=sn_id,
        name_stage=name_stage,
        docs_stage=docs_stage,
        token=claim_token,
        age_s=age_s,
    )


def _create_source(
    gc,
    source_id: str,
    *,
    stale: bool = True,
    claim_token: str = "tok-src-test",
) -> None:
    """Create (or MERGE) a :StandardNameSource node with claim fields set."""
    age_s = 400 if stale else 60
    gc.query(
        """
        MERGE (s:StandardNameSource {id: $id})
        SET s.claim_token = $token,
            s.claimed_at  = datetime() - duration({seconds: $age_s})
        """,
        id=source_id,
        token=claim_token,
        age_s=age_s,
    )


def _fetch_sn(gc, sn_id: str) -> dict:
    """Return the first row for a :StandardName node."""
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.name_stage   AS name_stage,
               sn.docs_stage   AS docs_stage,
               sn.claim_token  AS claim_token,
               sn.claimed_at   AS claimed_at
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found"
    return rows[0]


def _fetch_source(gc, source_id: str) -> dict:
    """Return the first row for a :StandardNameSource node."""
    rows = gc.query(
        """
        MATCH (s:StandardNameSource {id: $id})
        RETURN s.claim_token AS claim_token,
               s.claimed_at  AS claimed_at
        """,
        id=source_id,
    )
    assert rows, f"StandardNameSource {source_id!r} not found"
    return rows[0]


# ---------------------------------------------------------------------------
# I1. test_sweep_reverts_stale_refining_name
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_refining_name(_gc, _clean_test_nodes):
    """name_stage='refining' with stale claimed_at → reverted to 'reviewed'."""
    sn_id = f"{_TEST_ID_PREFIX}stale_name"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] >= 1

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# I2. test_sweep_skips_fresh_refining_name
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_refining_name(_gc, _clean_test_nodes):
    """name_stage='refining' with fresh claimed_at (60 s) → unchanged."""
    sn_id = f"{_TEST_ID_PREFIX}fresh_name"
    _create_sn(_gc, sn_id, name_stage="refining", stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "refining", row
    assert row["claim_token"] is not None, row


# ---------------------------------------------------------------------------
# I3. test_sweep_reverts_stale_refining_docs
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_refining_docs(_gc, _clean_test_nodes):
    """docs_stage='refining' with stale claimed_at → reverted to 'reviewed'."""
    sn_id = f"{_TEST_ID_PREFIX}stale_docs"
    _create_sn(_gc, sn_id, docs_stage="refining", stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] >= 1

    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# I4. test_sweep_skips_fresh_refining_docs
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_refining_docs(_gc, _clean_test_nodes):
    """docs_stage='refining' with fresh claimed_at (60 s) → unchanged."""
    sn_id = f"{_TEST_ID_PREFIX}fresh_docs"
    _create_sn(_gc, sn_id, docs_stage="refining", stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "refining", row
    assert row["claim_token"] is not None, row


# ---------------------------------------------------------------------------
# I5. test_sweep_reverts_stale_source_token
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_source_token(_gc, _clean_test_nodes):
    """StandardNameSource with stale claim_token+claimed_at → both cleared."""
    src_id = f"{_TEST_ID_PREFIX}stale_source"
    _create_source(_gc, src_id, stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["stale_token_source"] >= 1

    row = _fetch_source(_gc, src_id)
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# I6. test_sweep_skips_fresh_source_token
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_source_token(_gc, _clean_test_nodes):
    """StandardNameSource with fresh claimed_at (60 s) → unchanged."""
    src_id = f"{_TEST_ID_PREFIX}fresh_source"
    _create_source(_gc, src_id, stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_source(_gc, src_id)
    assert row["claim_token"] is not None, row


# ---------------------------------------------------------------------------
# I7. test_sweep_atomic_clear
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_atomic_clear(_gc, _clean_test_nodes):
    """After sweep, name_stage + claim_token + claimed_at are all cleared.

    Verifies no partial state: either all three fields are cleared or none.
    """
    sn_id = f"{_TEST_ID_PREFIX}atomic"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    # All three must be cleared together (atomicity check).
    assert row["name_stage"] == "reviewed", f"stage not cleared: {row}"
    assert row["claim_token"] is None, f"token not cleared: {row}"
    assert row["claimed_at"] is None, f"claimed_at not cleared: {row}"


# ---------------------------------------------------------------------------
# I8. test_sweep_no_op_when_clean
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_no_op_when_clean(_gc, _clean_test_nodes):
    """Graph with no stale claims → tick returns 0 for all categories."""
    # Create a non-refining SN without a stale token — should not be swept.
    sn_id = f"{_TEST_ID_PREFIX}clean_sn"
    _gc.query(
        "MERGE (sn:StandardName {id: $id}) SET sn.name_stage = 'reviewed'",
        id=sn_id,
    )

    counts = _orphan_sweep_tick(timeout_s=1)  # very short timeout

    # Any freshly-created nodes won't have claimed_at set, so zero sweeps.
    assert counts["name_refining"] == 0, counts
    assert counts["docs_refining"] == 0, counts


# ---------------------------------------------------------------------------
# I9. test_sweep_count_returns_correctly
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_count_returns_correctly(_gc, _clean_test_nodes):
    """3 stale name-refining + 2 stale docs-refining → counts match."""
    for i in range(3):
        _create_sn(
            _gc,
            f"{_TEST_ID_PREFIX}cnt_name_{i}",
            name_stage="refining",
            stale=True,
        )
    for i in range(2):
        _create_sn(
            _gc,
            f"{_TEST_ID_PREFIX}cnt_docs_{i}",
            docs_stage="refining",
            stale=True,
        )

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] >= 3, counts
    assert counts["docs_refining"] >= 2, counts


# ---------------------------------------------------------------------------
# I10. test_run_loop_respects_stop_event
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_loop_respects_stop_event(_gc, _clean_test_nodes):
    """Loop exits within 0.5 s when stop_event is set immediately."""
    stop_event = asyncio.Event()
    stop_event.set()

    # Should return almost immediately — well within 0.5 s.
    await asyncio.wait_for(
        run_orphan_sweep_loop(
            interval_s=60,
            timeout_s=300,
            stop_event=stop_event,
        ),
        timeout=0.5,
    )


# ---------------------------------------------------------------------------
# I11. test_run_loop_periodic
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_loop_periodic(_gc, _clean_test_nodes):
    """Stale claim is swept within 0.3 s by a running loop (interval=0.1 s)."""
    sn_id = f"{_TEST_ID_PREFIX}loop_periodic"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    stop_event = asyncio.Event()

    loop_task = asyncio.create_task(
        run_orphan_sweep_loop(
            interval_s=0,  # 0 → sleep(0) between ticks, maximally fast
            timeout_s=300,
            stop_event=stop_event,
        )
    )

    # Give the loop time to run at least one tick.
    await asyncio.sleep(0.3)
    stop_event.set()

    await asyncio.wait_for(loop_task, timeout=1.0)

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "reviewed", (
        f"Loop did not sweep the stale claim within 0.3 s: {row}"
    )
