"""Tests for async SNRun counter updates per-persist.

Verifies that each persist function bumps the appropriate SNRun counter
via ``bump_sn_run_counter``, providing live progress visibility during
``sn status`` while the run is in progress.

All tests mock the GraphClient — no live Neo4j required.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_RUN_ID = "run-async-counter-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc_ctx():
    """Return (patcher, mock_gc) that patches GraphClient as context manager."""
    patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
    MockGC = patcher.start()
    mock_gc = MagicMock()
    MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
    MockGC.return_value.__exit__ = MagicMock(return_value=False)
    return patcher, mock_gc


def _extract_bump_calls(mock_gc: MagicMock) -> list[dict[str, Any]]:
    """Extract bump_sn_run_counter Cypher calls from mock_gc.query calls."""
    bumps = []
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0] if c[0] else ""
        if "coalesce(rr." in cypher and "+ $delta" in cypher:
            kwargs = c[1] if len(c) > 1 else {}
            for counter in [
                "names_composed",
                "names_reviewed",
                "names_regenerated",
                "names_enriched",
            ]:
                if f"rr.{counter}" in cypher:
                    bumps.append(
                        {
                            "counter": counter,
                            "delta": kwargs.get("delta", 1),
                            "run_id": kwargs.get("run_id"),
                        }
                    )
    return bumps


# ---------------------------------------------------------------------------
# Test 1: bump_sn_run_counter → names_composed
# ---------------------------------------------------------------------------


class TestGenerateNameBumpsComposed:
    """After bump_sn_run_counter with names_composed, query is issued."""

    def test_single_batch_bumps_composed(self):
        """Verify bump_sn_run_counter is called with correct counter and delta."""
        from imas_codex.standard_names.graph_ops import bump_sn_run_counter

        patcher, mock_gc = _mock_gc_ctx()
        try:
            bump_sn_run_counter(_RUN_ID, "names_composed", delta=3)

            bumps = _extract_bump_calls(mock_gc)
            assert len(bumps) == 1
            assert bumps[0]["counter"] == "names_composed"
            assert bumps[0]["delta"] == 3
            assert bumps[0]["run_id"] == _RUN_ID
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 2: Two bumps accumulate (delta=2 + delta=5 → two queries)
# ---------------------------------------------------------------------------


class TestAccumulation:
    """After two bump calls of 2 + 5, both queries are issued."""

    def test_two_bumps_accumulate(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            bump_sn_run_counter(_RUN_ID, "names_composed", delta=2)
            bump_sn_run_counter(_RUN_ID, "names_composed", delta=5)

            bumps = _extract_bump_calls(mock_gc)
            composed_bumps = [b for b in bumps if b["counter"] == "names_composed"]
            assert len(composed_bumps) == 2
            total_delta = sum(b["delta"] for b in composed_bumps)
            assert total_delta == 7
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 3: persist_reviewed_name → names_reviewed (names_composed unchanged)
# ---------------------------------------------------------------------------


class TestReviewNameBumpsReviewed:
    """persist_reviewed_name bumps names_reviewed but not names_composed."""

    def test_review_name_bumps_reviewed_only(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.side_effect = [
                [{"chain_length": 0}],
                [],
                [],
            ]
            with patch("imas_codex.standard_names.graph_ops.write_reviews"):
                from imas_codex.standard_names.graph_ops import persist_reviewed_name

                persist_reviewed_name(
                    sn_id="electron_temperature",
                    claim_token="tok-123",
                    score=0.85,
                    model="test-model",
                    run_id=_RUN_ID,
                )

            bumps = _extract_bump_calls(mock_gc)
            reviewed_bumps = [b for b in bumps if b["counter"] == "names_reviewed"]
            composed_bumps = [b for b in bumps if b["counter"] == "names_composed"]

            assert len(reviewed_bumps) == 1
            assert reviewed_bumps[0]["delta"] == 1
            assert len(composed_bumps) == 0
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 4a: persist_refined_name → names_regenerated
# ---------------------------------------------------------------------------


class TestRefineNameBumpsRegenerated:
    """persist_refined_name bumps names_regenerated."""

    def test_refine_name_bumps_regenerated(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_session.begin_transaction.return_value = mock_tx
            mock_tx.closed = True
            mock_tx.run.return_value = [
                {
                    "new_name": "new_electron_temperature",
                    "old_name": "electron_temperature",
                }
            ]
            mock_gc.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_gc.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []

            from imas_codex.standard_names.graph_ops import persist_refined_name

            persist_refined_name(
                old_name="electron_temperature",
                new_name="new_electron_temperature",
                description="refined Te",
                run_id=_RUN_ID,
            )

            bumps = _extract_bump_calls(mock_gc)
            regen_bumps = [b for b in bumps if b["counter"] == "names_regenerated"]
            assert len(regen_bumps) == 1
            assert regen_bumps[0]["delta"] == 1
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 4b: persist_generated_docs → names_enriched
# ---------------------------------------------------------------------------


class TestGenerateDocsBumpsEnriched:
    """persist_generated_docs bumps names_enriched."""

    def test_generate_docs_bumps_enriched(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.return_value = [{"docs_stage": "drafted"}]

            from imas_codex.standard_names.graph_ops import persist_generated_docs

            persist_generated_docs(
                sn_id="electron_temperature",
                claim_token="tok-456",
                description="short desc",
                documentation="long doc",
                model="test-model",
                run_id=_RUN_ID,
            )

            bumps = _extract_bump_calls(mock_gc)
            enriched_bumps = [b for b in bumps if b["counter"] == "names_enriched"]
            assert len(enriched_bumps) == 1
            assert enriched_bumps[0]["delta"] == 1
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 4c: persist_reviewed_docs → names_reviewed
# ---------------------------------------------------------------------------


class TestReviewDocsBumpsReviewed:
    """persist_reviewed_docs bumps names_reviewed."""

    def test_review_docs_bumps_reviewed(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.side_effect = [
                [{"docs_chain_length": 0}],
                [],
                [],
            ]
            with patch("imas_codex.standard_names.graph_ops.write_reviews"):
                from imas_codex.standard_names.graph_ops import persist_reviewed_docs

                persist_reviewed_docs(
                    sn_id="electron_temperature",
                    claim_token="tok-789",
                    score=0.90,
                    model="test-model",
                    run_id=_RUN_ID,
                )

            bumps = _extract_bump_calls(mock_gc)
            reviewed_bumps = [b for b in bumps if b["counter"] == "names_reviewed"]
            assert len(reviewed_bumps) == 1
            assert reviewed_bumps[0]["delta"] == 1
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 4d: persist_refined_docs → names_regenerated
# ---------------------------------------------------------------------------


class TestRefineDocsBumpsRegenerated:
    """persist_refined_docs bumps names_regenerated."""

    def test_refine_docs_bumps_regenerated(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_session.begin_transaction.return_value = mock_tx
            mock_tx.closed = True
            mock_tx.run.return_value = [
                {"docs_chain_length": 1, "revision_id": "electron_temperature#rev-0"}
            ]
            mock_gc.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_gc.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []

            from imas_codex.standard_names.graph_ops import persist_refined_docs

            persist_refined_docs(
                sn_id="electron_temperature",
                claim_token="tok-abc",
                description="new desc",
                documentation="new doc",
                model="test-model",
                current_description="old desc",
                current_documentation="old doc",
                run_id=_RUN_ID,
            )

            bumps = _extract_bump_calls(mock_gc)
            regen_bumps = [b for b in bumps if b["counter"] == "names_regenerated"]
            assert len(regen_bumps) == 1
            assert regen_bumps[0]["delta"] == 1
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 5: No run_id → no bump (backward compat)
# ---------------------------------------------------------------------------


class TestNoRunIdNoBump:
    """When run_id is None, no counter bump query is issued."""

    def test_no_run_id_skips_bump(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.return_value = [{"docs_stage": "drafted"}]

            from imas_codex.standard_names.graph_ops import persist_generated_docs

            persist_generated_docs(
                sn_id="electron_temperature",
                claim_token="tok-none",
                description="desc",
                documentation="doc",
                model="test-model",
            )

            bumps = _extract_bump_calls(mock_gc)
            assert len(bumps) == 0, "No bump should be issued when run_id is None"
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 6: bump_sn_run_counter unit tests
# ---------------------------------------------------------------------------


class TestBumpSNRunCounter:
    """Direct unit tests for bump_sn_run_counter."""

    def test_bump_issues_correct_cypher(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            bump_sn_run_counter(_RUN_ID, "names_composed", delta=5)

            assert mock_gc.query.called
            cypher = mock_gc.query.call_args[0][0]
            assert (
                "rr.names_composed = coalesce(rr.names_composed, 0) + $delta" in cypher
            )
            kwargs = mock_gc.query.call_args[1]
            assert kwargs["run_id"] == _RUN_ID
            assert kwargs["delta"] == 5
        finally:
            patcher.stop()

    def test_bump_rejects_unknown_counter(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            bump_sn_run_counter(_RUN_ID, "unknown_counter", delta=1)
            assert not mock_gc.query.called
        finally:
            patcher.stop()

    def test_bump_skips_zero_delta(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            bump_sn_run_counter(_RUN_ID, "names_composed", delta=0)
            assert not mock_gc.query.called
        finally:
            patcher.stop()

    def test_bump_skips_none_run_id(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            bump_sn_run_counter(None, "names_composed", delta=3)
            assert not mock_gc.query.called
        finally:
            patcher.stop()

    def test_bump_swallows_graph_errors(self):
        """Graph errors in bump are best-effort — should not propagate."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.side_effect = RuntimeError("Neo4j down")

            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            # Should NOT raise
            bump_sn_run_counter(_RUN_ID, "names_composed", delta=1)
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 7: Concurrent bumps (simulated via threads)
# ---------------------------------------------------------------------------


class TestConcurrentBumps:
    """Verify that concurrent bump_sn_run_counter calls all issue queries."""

    def test_concurrent_bumps_all_fire(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import bump_sn_run_counter

            n_threads = 10
            barrier = threading.Barrier(n_threads)

            def bump():
                barrier.wait()
                bump_sn_run_counter(_RUN_ID, "names_composed", delta=1)

            threads = [threading.Thread(target=bump) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            bumps = _extract_bump_calls(mock_gc)
            composed_bumps = [b for b in bumps if b["counter"] == "names_composed"]
            assert len(composed_bumps) == n_threads
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test 8: persist function signatures accept run_id
# ---------------------------------------------------------------------------


class TestPersistSignatures:
    """Verify all persist functions accept run_id keyword argument."""

    def test_persist_generated_name_batch_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        sig = inspect.signature(persist_generated_name_batch)
        assert "run_id" in sig.parameters

    def test_persist_reviewed_name_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        sig = inspect.signature(persist_reviewed_name)
        assert "run_id" in sig.parameters

    def test_persist_reviewed_docs_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_reviewed_docs

        sig = inspect.signature(persist_reviewed_docs)
        assert "run_id" in sig.parameters

    def test_persist_refined_name_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_refined_name

        sig = inspect.signature(persist_refined_name)
        assert "run_id" in sig.parameters

    def test_persist_refined_docs_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_refined_docs

        sig = inspect.signature(persist_refined_docs)
        assert "run_id" in sig.parameters

    def test_persist_generated_docs_accepts_run_id(self):
        import inspect

        from imas_codex.standard_names.graph_ops import persist_generated_docs

        sig = inspect.signature(persist_generated_docs)
        assert "run_id" in sig.parameters
