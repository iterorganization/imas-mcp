"""Writer-loop resilience tests for BudgetManager (Phase A2).

Verifies:
- T1: hang → timeout → retry → continue (record_llm_cost wedge)
- T2: always-raise → _write_failed + _write_dropped
- T3: writer crash + recreation under concurrent enqueue (TOCTOU-safe)
- T4: heartbeat fires at _WRITER_HEARTBEAT_SEC with DEBUG when idle
- T5: heartbeat with pending cost → INFO log
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import patch

import pytest

from imas_codex.standard_names.budget import (
    _WRITER_CALL_TIMEOUT,
    _WRITER_HEARTBEAT_SEC,
    _WRITER_MAX_RETRIES,
    BudgetManager,
    LLMCostEvent,
)

_EVENT = LLMCostEvent(model="test", tokens_in=10, tokens_out=5, phase="test")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mgr(run_id: str = "test-run") -> BudgetManager:
    """Create a BudgetManager with a run_id (enables graph writes)."""
    return BudgetManager(10.0, run_id=run_id)


# ── T1: hang → timeout → continue ───────────────────────────────────


@pytest.mark.asyncio
async def test_write_single_timeout_then_succeed():
    """record_llm_cost hangs on first call, succeeds on retry.

    Verify _write_single returns within ~timeout + backoff,
    and the item is eventually written.
    """
    mgr = _make_mgr()
    call_count = 0

    def _mock_record(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate a wedged Neo4j call (longer than timeout)
            time.sleep(30)
        # Subsequent calls succeed

    await mgr.start()
    # Use a short timeout for test speed
    with (
        patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=_mock_record,
        ),
        patch("imas_codex.standard_names.budget._WRITER_CALL_TIMEOUT", 1.0),
    ):
        # Charge an event — goes through the writer loop
        lease = mgr.reserve(1.0, phase="test")
        assert lease is not None
        lease.charge_event(0.01, _EVENT)

        # Drain — should complete (writer retried after timeout)
        t0 = time.monotonic()
        result = await asyncio.wait_for(mgr.drain_pending(), timeout=15)
        elapsed = time.monotonic() - t0

    # Should have completed within timeout(1s) + backoff(0.1s) + second call
    assert elapsed < 10  # generous bound
    assert call_count >= 2  # at least one retry
    assert result is True
    assert mgr._write_dropped == 0


# ── T2: always-raise → _write_failed + _write_dropped ───────────────


@pytest.mark.asyncio
async def test_write_single_always_fails():
    """record_llm_cost raises every time → _write_failed + _write_dropped."""
    mgr = _make_mgr()

    def _mock_record(**kwargs):
        raise ConnectionError("Neo4j unavailable")

    await mgr.start()
    with patch(
        "imas_codex.standard_names.graph_ops.record_llm_cost",
        side_effect=_mock_record,
    ):
        lease = mgr.reserve(1.0, phase="test")
        assert lease is not None
        lease.charge_event(0.01, _EVENT)

        result = await asyncio.wait_for(mgr.drain_pending(), timeout=15)

    assert result is False
    assert mgr._write_failed is True
    assert mgr._write_dropped == 1


# ── T3: writer crash + recreation (TOCTOU-safe under lock) ──────────


@pytest.mark.asyncio
async def test_writer_crash_recreation():
    """Forcibly cancel _writer_task; next enqueue detects and recreates.

    Verify the writer is recreated and continues processing. The lock
    ensures only ONE recreation happens — verified by checking that a
    second enqueue (with writer alive) does NOT recreate.
    """
    mgr = _make_mgr()
    await mgr.start()

    # Forcibly cancel the writer task
    assert mgr._writer_task is not None
    old_task = mgr._writer_task
    mgr._writer_task.cancel()
    # Wait for the cancellation to take effect
    try:
        await mgr._writer_task
    except (asyncio.CancelledError, Exception):
        pass

    assert mgr._writer_task.done()

    # First enqueue — should detect dead writer and recreate
    with patch(
        "imas_codex.standard_names.graph_ops.record_llm_cost",
    ):
        mgr._enqueue_write(0.001, _EVENT, 0.0)

    # Writer should have been recreated
    assert mgr._writer_task is not None
    assert mgr._writer_task is not old_task
    assert not mgr._writer_task.done()

    new_task = mgr._writer_task

    # Second enqueue should NOT trigger recreation (writer is alive)
    with patch(
        "imas_codex.standard_names.graph_ops.record_llm_cost",
    ):
        mgr._enqueue_write(0.001, _EVENT, 0.0)

    assert mgr._writer_task is new_task  # same task, no re-creation

    # Clean up: drain the items we enqueued
    await asyncio.wait_for(mgr.drain_pending(), timeout=10)


# ── T4: heartbeat DEBUG when idle ────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_fires_debug_when_idle(caplog):
    """Empty queue → heartbeat fires at _WRITER_HEARTBEAT_SEC with DEBUG."""
    mgr = _make_mgr()

    # Patch heartbeat interval AND ensure DEBUG logs are captured
    with (
        patch("imas_codex.standard_names.budget._WRITER_HEARTBEAT_SEC", 0.1),
        caplog.at_level(logging.DEBUG, logger="imas_codex.standard_names.budget"),
    ):
        await mgr.start()
        # Let the writer idle for a bit
        await asyncio.sleep(0.4)
        # Send sentinel to stop — MUST be inside patch context
        await asyncio.wait_for(mgr.drain_pending(), timeout=5)

    heartbeat_msgs = [r for r in caplog.records if "writer_loop heartbeat" in r.message]
    assert len(heartbeat_msgs) >= 1
    # With pending=0 and qsize=0, should be DEBUG
    debug_hb = [r for r in heartbeat_msgs if r.levelno == logging.DEBUG]
    assert len(debug_hb) >= 1


# ── T5: heartbeat INFO when pending > 0 ─────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_fires_info_when_pending(caplog):
    """Pending cost > 0 → heartbeat fires at INFO level."""
    mgr = _make_mgr()

    # Artificially set pending cost > 0
    with mgr._pending_lock:
        mgr._pending_cost = 1.23

    with (
        patch("imas_codex.standard_names.budget._WRITER_HEARTBEAT_SEC", 0.1),
        caplog.at_level(logging.DEBUG, logger="imas_codex.standard_names.budget"),
    ):
        await mgr.start()
        await asyncio.sleep(0.4)
        # Reset pending so drain works cleanly
        with mgr._pending_lock:
            mgr._pending_cost = 0.0
        # Drain INSIDE the patch context
        await asyncio.wait_for(mgr.drain_pending(), timeout=5)

    heartbeat_msgs = [r for r in caplog.records if "writer_loop heartbeat" in r.message]
    assert len(heartbeat_msgs) >= 1
    info_hb = [r for r in heartbeat_msgs if r.levelno == logging.INFO]
    assert len(info_hb) >= 1
