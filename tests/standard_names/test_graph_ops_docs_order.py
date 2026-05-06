"""Tests for parent-first docs ordering in ``claim_generate_docs_batch``.

T15 — Parents are claimed before children: verifies that the seed Cypher
      query contains priority ordering so nodes with incoming COMPONENT_OF
      edges (parents) are claimed first.

T16 — Children proceed when parent unclaimed: verifies the escape-hatch
      condition that allows a child to be claimed even if its parent is still
      in docs_stage='pending', as long as that parent has never been claimed
      (claimed_at IS NULL).

All tests mock :class:`GraphClient` — no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_claim_atomic.py pattern)
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient wired for single-transaction claims."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


def _patch_gc(mock_gc):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


# ---------------------------------------------------------------------------
# T15: Parent claimed before children
# ---------------------------------------------------------------------------


class TestParentDocsPriority:
    """T15 — Parents (nodes with incoming COMPONENT_OF) should be claimed first."""

    def test_seed_query_contains_priority_ordering(self):
        """The seed Cypher emitted by claim_generate_docs_batch must include
        priority-based ORDER BY so parents are picked before children."""
        from imas_codex.standard_names.graph_ops import claim_generate_docs_batch

        gc, tx = _mock_gc_tx()

        # Seed returns nothing — we only care about the Cypher structure.
        tx.run = MagicMock(return_value=[])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            result = claim_generate_docs_batch(batch_size=5)

        assert result == []

        # Inspect the seed query (first tx.run call).
        seed_cypher = tx.run.call_args_list[0].args[0]

        # Priority column: parents (nodes with incoming COMPONENT_OF) get 0.
        assert "COMPONENT_OF" in seed_cypher, (
            "Expected COMPONENT_OF reference for parent priority in seed query"
        )
        # ORDER BY must include priority column before rand().
        assert "_docs_priority" in seed_cypher, (
            "Expected _docs_priority alias in seed ORDER BY"
        )
        # rand() must still appear (for tiebreaking within same priority).
        assert "rand()" in seed_cypher, "Expected rand() tiebreaker in seed ORDER BY"

    def test_priority_ordering_parents_before_children(self):
        """Verify that the priority expression assigns 0 to parents and 1 to
        leaf nodes, and that ORDER BY sorts ASC (parents first)."""
        from imas_codex.standard_names.graph_ops import claim_generate_docs_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            claim_generate_docs_batch(batch_size=1)

        seed_cypher = tx.run.call_args_list[0].args[0]

        # THEN 0 → parent priority; ELSE 1 → leaf priority.
        assert "THEN 0" in seed_cypher, (
            "Expected THEN 0 (parent priority) in CASE expression"
        )
        assert "ELSE 1" in seed_cypher, (
            "Expected ELSE 1 (leaf priority) in CASE expression"
        )
        # ASC ensures lower (parent) priority values come first.
        assert "ASC" in seed_cypher, (
            "Expected ASC ordering so parents (priority 0) are claimed first"
        )


# ---------------------------------------------------------------------------
# T16: Children proceed when parent unclaimed
# ---------------------------------------------------------------------------


class TestChildProceedsWhenParentUnclaimed:
    """T16 — Escape hatch: children should not be blocked by an unclaimed parent.

    When a parent exists with docs_stage='pending' but claimed_at IS NULL
    (the parent was never picked up), children must still be claimable.
    The escape hatch blocks only parents that are *actively* in-progress
    (docs_stage='pending' AND claimed_at IS NOT NULL).
    """

    def test_escape_hatch_present_in_seed_query(self):
        """The seed query must contain a NOT EXISTS block that checks
        parent.claimed_at IS NOT NULL — only blocking actively-claimed parents."""
        from imas_codex.standard_names.graph_ops import claim_generate_docs_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            claim_generate_docs_batch(batch_size=5)

        seed_cypher = tx.run.call_args_list[0].args[0]

        # The escape hatch must use NOT EXISTS to filter out in-progress parents.
        assert "NOT EXISTS" in seed_cypher, (
            "Expected NOT EXISTS escape-hatch clause in seed query"
        )
        # Must check claimed_at IS NOT NULL (active claim guard).
        assert "claimed_at IS NOT NULL" in seed_cypher, (
            "Expected claimed_at IS NOT NULL condition in escape-hatch"
        )

    def test_escape_hatch_targets_docs_stage_pending(self):
        """The escape hatch must also verify that the parent's docs_stage is
        'pending' — so children aren't blocked by parents in other stages."""
        from imas_codex.standard_names.graph_ops import claim_generate_docs_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.chain_history.name_chain_history",
                return_value=[],
            ),
        ):
            claim_generate_docs_batch(batch_size=5)

        seed_cypher = tx.run.call_args_list[0].args[0]

        # The NOT EXISTS block must reference docs_stage = 'pending'.
        assert "docs_stage" in seed_cypher, (
            "Expected docs_stage check in escape-hatch condition"
        )
        assert "pending" in seed_cypher, (
            "Expected 'pending' value in escape-hatch docs_stage check"
        )

    def test_default_claim_unaffected(self):
        """Other claim functions that do not pass seed_extra_where must not
        be affected — the default seed query should use ORDER BY rand()."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'accepted'",
                query_params={},
                batch_size=1,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]

        # Default ordering — no priority column injected.
        assert "_docs_priority" not in seed_cypher, (
            "Default _claim_sn_atomic must not include _docs_priority"
        )
        assert "rand()" in seed_cypher, (
            "Default _claim_sn_atomic must still use rand() ordering"
        )
        # No escape hatch by default.
        assert "claimed_at IS NOT NULL" not in seed_cypher, (
            "Default _claim_sn_atomic must not include the escape-hatch condition"
        )
