"""Tests for polling-based compose and review claim functions.

Verifies the graph claim functions used by the refactored polling workers
(42-polling-workers design).  Tests mock :class:`GraphClient` — no live
Neo4j connection needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager

# =====================================================================
# claim_compose_sources
# =====================================================================


class TestClaimComposeSources:
    """Tests for claim_compose_sources()."""

    def test_empty_pool_returns_empty(self):
        """When no sources are eligible, claim returns empty list."""
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        # Step 1 (SET) returns nothing, Step 2 (verify) returns empty
        mock_gc.query = MagicMock(side_effect=[None, []])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import claim_compose_sources

            token, claimed = claim_compose_sources(limit=10)

        assert isinstance(token, str)
        assert len(token) > 0
        assert claimed == []

    def test_claims_returned_as_dicts(self):
        """Claimed sources are returned as plain dicts."""
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        fake_rows = [
            {
                "id": "src-1",
                "source_id": "eq/psi",
                "source_type": "dd",
                "batch_key": "equilibrium",
                "description": "Poloidal flux",
            },
            {
                "id": "src-2",
                "source_id": "cp/Te",
                "source_type": "dd",
                "batch_key": "core_profiles",
                "description": "Electron temp",
            },
        ]
        mock_gc.query = MagicMock(side_effect=[None, fake_rows])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import claim_compose_sources

            token, claimed = claim_compose_sources(limit=15)

        assert len(claimed) == 2
        assert all(isinstance(c, dict) for c in claimed)
        assert claimed[0]["id"] == "src-1"
        assert claimed[1]["source_type"] == "dd"


# =====================================================================
# claim_review_names
# =====================================================================


class TestClaimReviewNames:
    """Tests for claim_review_names()."""

    def test_empty_input_returns_empty(self):
        """Passing empty name_ids returns empty without touching graph."""
        from imas_codex.standard_names.graph_ops import claim_review_names

        token, claimed = claim_review_names([])
        assert token == ""
        assert claimed == []

    def test_claims_subset(self):
        """Only names successfully claimed (verified) are returned."""
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        # Step 1 (SET) returns nothing, Step 2 (verify) returns subset
        mock_gc.query = MagicMock(side_effect=[None, [{"id": "electron_temperature"}]])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import claim_review_names

            token, claimed = claim_review_names(["electron_temperature", "ion_density"])

        assert len(claimed) == 1
        assert "electron_temperature" in claimed


# =====================================================================
# release functions
# =====================================================================


class TestReleaseFunctions:
    """Tests for release_standard_name_source_claims and release_review_claims."""

    def test_release_source_claims(self):
        """release_standard_name_source_claims returns affected count."""
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(return_value=[{"affected": 3}])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import (
                release_standard_name_source_claims,
            )

            count = release_standard_name_source_claims("tok-123")

        assert count == 3

    def test_release_review_claims_empty_token(self):
        """release_review_claims with empty token returns 0 immediately."""
        from imas_codex.standard_names.graph_ops import release_review_claims

        assert release_review_claims("") == 0

    def test_release_review_claims(self):
        """release_review_claims returns affected count."""
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(return_value=[{"affected": 5}])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import release_review_claims

            count = release_review_claims("tok-456")

        assert count == 5


# =====================================================================
# count_eligible_compose_sources
# =====================================================================


class TestCountEligibleComposeSources:
    """Tests for count_eligible_compose_sources()."""

    def test_returns_count(self):
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(return_value=[{"cnt": 42}])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import (
                count_eligible_compose_sources,
            )

            assert count_eligible_compose_sources() == 42

    def test_returns_zero_when_empty(self):
        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query = MagicMock(return_value=[{"cnt": 0}])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.standard_names.graph_ops import (
                count_eligible_compose_sources,
            )

            assert count_eligible_compose_sources() == 0


# =====================================================================
# Polling worker smoke tests (in-process, no graph/LLM)
# =====================================================================


@dataclass
class _ComposeStats:
    total: int = 0
    processed: int = 0
    cost: float = 0.0
    errors: int = 0
    stream_queue: Any = field(default_factory=lambda: MagicMock())

    def record_batch(self, n: int) -> None:
        pass

    def freeze_rate(self) -> None:
        pass


class TestPollingWorkerBudgetRetry:
    """Verify the polling worker budget-retry + re-enqueue logic."""

    def test_requeue_on_budget_exhaustion(self):
        """When budget is exhausted, batch is re-enqueued (up to max retries)
        and the worker eventually stops."""

        async def _run():
            mgr = BudgetManager(total_budget=0.01)  # Almost nothing
            # Exhaust budget immediately
            lease = mgr.reserve(0.01)
            assert lease is not None
            lease.charge(0.01)
            lease.release_unused()

            queue: asyncio.Queue = asyncio.Queue()
            queue.put_nowait({"items": [1, 2, 3], "group_key": "test"})

            processed = 0
            skipped = 0
            max_retries = 3

            async def _worker():
                nonlocal processed, skipped
                budget_retries = 0
                while True:
                    try:
                        batch = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    estimated = len(batch["items"]) * 0.04 * 1.3
                    wl = mgr.reserve(estimated)
                    if wl is None:
                        budget_retries += 1
                        if budget_retries > max_retries or mgr.exhausted():
                            skipped += 1
                            break
                        queue.put_nowait(batch)
                        await asyncio.sleep(0.001)
                        continue

                    wl.release_unused()
                    processed += 1

            await _worker()
            assert processed == 0
            assert skipped == 1

        asyncio.run(_run())

    def test_successful_processing_resets_retry_counter(self):
        """After a successful batch, the budget retry counter resets."""

        async def _run():
            mgr = BudgetManager(total_budget=1.0)
            queue: asyncio.Queue = asyncio.Queue()
            for i in range(5):
                queue.put_nowait({"items": [i], "group_key": f"b{i}"})

            processed = 0

            async def _worker():
                nonlocal processed
                budget_retries = 0
                while True:
                    try:
                        _batch = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    wl = mgr.reserve(0.10)
                    if wl is None:
                        budget_retries += 1
                        break

                    wl.charge(0.05)
                    wl.release_unused()
                    processed += 1
                    budget_retries = 0  # Reset

            await _worker()
            assert processed == 5

        asyncio.run(_run())

    def test_multiple_workers_drain_queue(self):
        """Multiple workers cooperatively drain all items from the queue."""

        async def _run():
            queue: asyncio.Queue = asyncio.Queue()
            for i in range(20):
                queue.put_nowait(i)

            results: list[int] = []

            async def _worker(wid: int):
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    results.append(item)
                    await asyncio.sleep(0.001)  # Simulate work

            await asyncio.gather(*[_worker(i) for i in range(4)])
            assert sorted(results) == list(range(20))

        asyncio.run(_run())
