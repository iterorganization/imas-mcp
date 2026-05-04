"""Parametric stage-transition coverage for every persist function.

Asserts the documented stage-transition table from AGENTS.md holds for all
persist functions in imas_codex.standard_names.graph_ops.

Stage table (from AGENTS.md)::

  name_stage:  pending → drafted → reviewed → accepted | refining | exhausted
                                               ↑
                                               └── superseded (predecessor in REFINED_FROM chain)
  docs_stage:  pending → drafted → reviewed → accepted | refining | exhausted

Cross-pipeline gate: persist_generated_docs fires only when name_stage='accepted'.

All tests mock GraphClient — no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─── Module paths ─────────────────────────────────────────────────────────────

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_WRITE_REVIEWS_PATH = "imas_codex.standard_names.graph_ops.write_reviews"

# ─── Production defaults (must match imas_codex/standard_names/defaults.py) ──

_MIN_SCORE: float = 0.75
_ROTATION_CAP: int = 3


# =============================================================================
# Shared mock helpers
# =============================================================================


def _mock_gc_query(
    return_values: list[list[dict[str, Any]]] | None = None,
) -> MagicMock:
    """Mock GraphClient whose .query() returns successive values.

    When *return_values* is provided, each call to ``gc.query()`` pops the
    next entry from the list.  Useful for functions that execute two separate
    ``GraphClient()`` context-manager blocks.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    if return_values is not None:
        gc.query = MagicMock(side_effect=return_values)
    else:
        gc.query = MagicMock(return_value=[])
    return gc


def _mock_gc_tx(
    tx_run_return: list[dict[str, Any]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Mock GraphClient with a controllable transaction.

    Returns ``(gc, tx)`` where *tx* is the mock Transaction.
    *tx_run_return* is the value returned by ``list(tx.run(...))``; defaults
    to an empty list (triggers the no-op / mismatch branch in persist callers).
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()
    tx.run = MagicMock(return_value=tx_run_return if tx_run_return is not None else [])

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


@contextmanager
def _patch_gc(gc: MagicMock):
    """Patch GraphClient so every ``GraphClient()`` call returns the same mock."""
    with patch(_GC_PATH, return_value=gc):
        yield


@contextmanager
def _no_write_reviews():
    """Suppress write_reviews calls to avoid secondary graph interactions."""
    with patch(_WRITE_REVIEWS_PATH):
        yield


# =============================================================================
# Section A — parametric review-stage decisions (6 cases)
#
# Both persist_reviewed_name and persist_reviewed_docs implement the same
# three-way stage decision:
#
#   score >= min_score                           → "accepted"
#   score <  min_score AND chain <  rotation_cap → "reviewed"
#   score <  min_score AND chain >= rotation_cap → "exhausted"
#
# Case layout:
#   (case_id, fn_name, chain_key, chain_len, score, expected_stage)
# =============================================================================

_REVIEW_STAGE_CASES: list[tuple[str, str, str, int, float, str]] = [
    # ── name_stage decisions ──────────────────────────────────────────────────
    # High score → accepted regardless of chain depth.
    (
        "name_drafted_to_accepted",
        "persist_reviewed_name",
        "chain_length",
        0,
        0.95,
        "accepted",
    ),
    # Low score, chain below cap → reviewed (eligible for refine).
    (
        "name_drafted_to_reviewed",
        "persist_reviewed_name",
        "chain_length",
        0,
        0.60,
        "reviewed",
    ),
    # Low score, chain at/above cap → exhausted (no further refine).
    (
        "name_drafted_to_exhausted",
        "persist_reviewed_name",
        "chain_length",
        _ROTATION_CAP,
        0.60,
        "exhausted",
    ),
    # ── docs_stage decisions ──────────────────────────────────────────────────
    (
        "docs_drafted_to_accepted",
        "persist_reviewed_docs",
        "docs_chain_length",
        0,
        0.95,
        "accepted",
    ),
    (
        "docs_drafted_to_reviewed",
        "persist_reviewed_docs",
        "docs_chain_length",
        0,
        0.60,
        "reviewed",
    ),
    (
        "docs_drafted_to_exhausted",
        "persist_reviewed_docs",
        "docs_chain_length",
        _ROTATION_CAP,
        0.60,
        "exhausted",
    ),
]


@pytest.mark.parametrize(
    "case_id,fn_name,chain_key,chain_len,score,expected_stage",
    _REVIEW_STAGE_CASES,
    ids=[c[0] for c in _REVIEW_STAGE_CASES],
)
def test_review_stage_decision(
    case_id: str,
    fn_name: str,
    chain_key: str,
    chain_len: int,
    score: float,
    expected_stage: str,
) -> None:
    """Single template: review-score × chain-depth → documented target stage.

    Each parametric case encodes one row from the AGENTS.md stage table.
    The mock supplies the current chain depth; the function computes the
    target stage; we assert it matches the documented value.
    """
    from imas_codex.standard_names import graph_ops

    fn = getattr(graph_ops, fn_name)

    # persist_reviewed_* makes TWO GraphClient() calls:
    #   Q1 — reads current chain_length (returns a row with the chain key)
    #   Q2 — writes the new stage (return value is not used by the caller)
    gc = _mock_gc_query(
        return_values=[
            [{chain_key: chain_len}],  # Q1: chain-length read succeeds
            [],  # Q2: stage write (return value ignored)
        ]
    )

    with _patch_gc(gc), _no_write_reviews():
        result = fn(
            sn_id="test_sn",
            claim_token="tok-test-abc",
            score=score,
            model="test/model",
            min_score=_MIN_SCORE,
            rotation_cap=_ROTATION_CAP,
        )

    assert result == expected_stage, (
        f"{fn_name}(score={score}, chain={chain_len}/{_ROTATION_CAP}) "
        f"→ got {result!r}, expected {expected_stage!r}"
    )


def test_review_stage_token_mismatch_is_noop() -> None:
    """Token mismatch (Q1 returns empty) → function returns empty string (no-op)."""
    from imas_codex.standard_names.graph_ops import persist_reviewed_name

    # Q1 returns nothing — simulates wrong token or wrong stage.
    gc = _mock_gc_query(return_values=[[]])
    with _patch_gc(gc), _no_write_reviews():
        result = persist_reviewed_name(
            sn_id="test_sn",
            claim_token="wrong-token",
            score=0.9,
            model="test/model",
        )

    assert result == ""


# =============================================================================
# Section B — compose: name_stage → drafted, docs_stage → pending
#
# _finalize_generated_name_stage is the atomic helper called inside
# persist_generated_name_batch after write_standard_names succeeds.
# It transitions name_stage = 'drafted' and docs_stage = 'pending'
# in a single transaction.
# =============================================================================


class TestComposeNameStageDrafted:
    """compose pipeline: new SN nodes start at name_stage='drafted', docs_stage='pending'."""

    def test_name_stage_drafted(self) -> None:
        """Cypher SET includes name_stage = 'drafted'."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "test_sn", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher: str = tx.run.call_args.args[0]
        assert "name_stage" in cypher and "'drafted'" in cypher

    def test_docs_stage_pending(self) -> None:
        """Cypher SET includes docs_stage = 'pending'."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "test_sn", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher: str = tx.run.call_args.args[0]
        assert "docs_stage" in cypher and "'pending'" in cypher

    def test_chain_length_zero(self) -> None:
        """Cypher SET includes chain_length = 0."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "test_sn", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher: str = tx.run.call_args.args[0]
        assert "chain_length" in cypher and "= 0" in cypher

    def test_single_transaction_committed(self) -> None:
        """All stage updates land in a single committed transaction."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "test_sn", "sns_id": "dd:p/q", "model": "m"}]
            )

        tx.run.assert_called_once()
        tx.commit.assert_called_once()


# =============================================================================
# Section C — generate_docs: docs_stage pending → drafted
#
# Cross-pipeline gate: persist_generated_docs fires only when
# name_stage = 'accepted'.  The WHERE clause enforces this — any node whose
# name_stage is not 'accepted' will not match and the function raises.
# =============================================================================


class TestGenerateDocsPendingToDrafted:
    """persist_generated_docs: docs_stage pending → drafted (gate: name_stage='accepted')."""

    def test_returns_drafted_on_success(self) -> None:
        """Returns 'drafted' when the token + name_stage='accepted' gate passes."""
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = _mock_gc_query(return_values=[[{"docs_stage": "drafted"}]])
        with _patch_gc(gc):
            result = persist_generated_docs(
                sn_id="test_sn",
                claim_token="tok-gen-docs",
                description="short desc",
                documentation="long doc",
                model="test/model",
            )

        assert result == "drafted"

    def test_cypher_gates_on_name_stage_accepted(self) -> None:
        """WHERE clause in Cypher requires name_stage = 'accepted' (cross-pipeline gate)."""
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = _mock_gc_query(return_values=[[{"docs_stage": "drafted"}]])
        with _patch_gc(gc):
            persist_generated_docs(
                sn_id="test_sn",
                claim_token="tok-gen-docs",
                description="desc",
                documentation="doc",
                model="test/model",
            )

        cypher: str = gc.query.call_args.args[0]
        assert "name_stage" in cypher and "'accepted'" in cypher

    def test_cypher_sets_docs_stage_drafted(self) -> None:
        """Cypher SET includes docs_stage = 'drafted'."""
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = _mock_gc_query(return_values=[[{"docs_stage": "drafted"}]])
        with _patch_gc(gc):
            persist_generated_docs(
                sn_id="test_sn",
                claim_token="tok-gen-docs",
                description="desc",
                documentation="doc",
                model="test/model",
            )

        cypher: str = gc.query.call_args.args[0]
        assert "docs_stage" in cypher and "'drafted'" in cypher

    def test_raises_when_node_not_found(self) -> None:
        """Raises ValueError when graph returns no rows (name_stage != 'accepted' gate blocked).

        This is the cross-pipeline enforcement: generate_docs only runs after
        name review is complete.
        """
        from imas_codex.standard_names.graph_ops import persist_generated_docs

        gc = _mock_gc_query(return_values=[[]])  # empty → gate not satisfied
        with _patch_gc(gc):
            with pytest.raises(ValueError, match="test_sn"):
                persist_generated_docs(
                    sn_id="test_sn",
                    claim_token="tok-gen-docs",
                    description="desc",
                    documentation="doc",
                    model="test/model",
                )


# =============================================================================
# Section D — refine_name: predecessor → superseded, new SN → drafted
#
# Option B chain: a new StandardName node is created with the refined name
# string as its id.  The predecessor is marked name_stage='superseded', and
# PRODUCED_NAME / HAS_STANDARD_NAME edges are migrated to the new node.
# chain_length on the new node is old_chain_length + 1.
# =============================================================================


class TestRefineNameStageMachine:
    """persist_refined_name: predecessor→superseded, new SN→drafted, edges migrated."""

    def _call_persist(
        self,
        tx_run_return: list[dict[str, Any]] | None = None,
        old_chain_length: int = 1,
    ) -> tuple[dict[str, Any], MagicMock]:
        from imas_codex.standard_names.graph_ops import persist_refined_name

        default_return = [{"new_name": "refined_test_sn", "old_name": "test_sn"}]
        gc, tx = _mock_gc_tx(
            tx_run_return=tx_run_return if tx_run_return is not None else default_return
        )
        with _patch_gc(gc):
            result = persist_refined_name(
                old_name="test_sn",
                new_name="refined_test_sn",
                description="improved description",
                kind="scalar",
                unit="eV",
                old_chain_length=old_chain_length,
                model="test/model",
            )
        return result, tx

    def test_cypher_marks_predecessor_superseded(self) -> None:
        """Cypher SET marks old node name_stage = 'superseded'."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "superseded" in cypher

    def test_cypher_sets_new_sn_name_stage_drafted(self) -> None:
        """Cypher ON CREATE SET sets new node name_stage = 'drafted'."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "drafted" in cypher

    def test_cypher_sets_new_sn_docs_stage_pending(self) -> None:
        """Cypher ON CREATE SET sets new node docs_stage = 'pending'."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "docs_stage" in cypher and "'pending'" in cypher

    def test_cypher_creates_refined_from_edge(self) -> None:
        """Cypher creates REFINED_FROM edge from new SN to old SN."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "REFINED_FROM" in cypher

    def test_chain_length_incremented(self) -> None:
        """new_chain_length passed to Cypher equals old_chain_length + 1."""
        _, tx = self._call_persist(old_chain_length=1)
        params: dict[str, Any] = tx.run.call_args.kwargs
        assert params["new_chain_length"] == 2  # 1 + 1

    def test_chain_length_incremented_from_zero(self) -> None:
        """First refine: chain_length 0 → 1."""
        _, tx = self._call_persist(old_chain_length=0)
        params: dict[str, Any] = tx.run.call_args.kwargs
        assert params["new_chain_length"] == 1

    def test_cypher_migrates_produced_name_edges(self) -> None:
        """Cypher deletes PRODUCED_NAME from old SN and re-creates on new SN."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "PRODUCED_NAME" in cypher

    def test_cypher_migrates_has_standard_name_edges(self) -> None:
        """Cypher deletes HAS_STANDARD_NAME from old SN and re-creates on new SN."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "HAS_STANDARD_NAME" in cypher

    def test_returns_new_and_old_name(self) -> None:
        """Returns dict with new_name and old_name."""
        result, _ = self._call_persist()
        assert result["new_name"] == "refined_test_sn"
        assert result["old_name"] == "test_sn"

    def test_single_committed_transaction(self) -> None:
        """All changes (new node + edge + migration) land in a single transaction."""
        _, tx = self._call_persist()
        tx.run.assert_called_once()
        tx.commit.assert_called_once()


# =============================================================================
# Section E — refine_docs: in-place docs refine with DocsRevision snapshot
#
# docs_stage transitions: refining → drafted (SN node updated in-place).
# Prior docs are snapshotted to a DocsRevision node linked via DOCS_REVISION_OF.
# docs_chain_length is incremented.
# Reviewer fields on the SN are cleared so new docs receive a fresh review.
# =============================================================================


class TestRefineDocsStageMachine:
    """persist_refined_docs: docs_stage refining → drafted, DocsRevision created."""

    def _call_persist(
        self,
        tx_run_return: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], MagicMock]:
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        default_return = [{"docs_chain_length": 2, "revision_id": "test_sn#rev-1"}]
        gc, tx = _mock_gc_tx(
            tx_run_return=tx_run_return if tx_run_return is not None else default_return
        )
        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="test_sn",
                claim_token="tok-refine-docs",
                description="new desc",
                documentation="new long doc",
                model="test/model",
                current_description="old desc",
                current_documentation="old long doc",
            )
        return result, tx

    def test_cypher_sets_docs_stage_drafted(self) -> None:
        """Cypher SET docs_stage = 'drafted' on the SN node after refine."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "docs_stage" in cypher and "'drafted'" in cypher

    def test_cypher_creates_docs_revision_node(self) -> None:
        """Cypher creates a DocsRevision snapshot of the pre-refine docs."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "DocsRevision" in cypher

    def test_cypher_creates_docs_revision_of_edge(self) -> None:
        """Cypher creates DOCS_REVISION_OF edge from SN to snapshot node."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "DOCS_REVISION_OF" in cypher

    def test_docs_chain_length_incremented(self) -> None:
        """Returns incremented docs_chain_length from the Cypher result."""
        result, _ = self._call_persist(
            tx_run_return=[{"docs_chain_length": 2, "revision_id": "test_sn#rev-1"}]
        )
        assert result["docs_chain_length"] == 2

    def test_cypher_clears_reviewer_docs_fields(self) -> None:
        """Cypher nulls reviewer_score_docs so revised docs get a fresh review."""
        _, tx = self._call_persist()
        cypher: str = tx.run.call_args.args[0]
        assert "reviewer_score_docs" in cypher
        assert "null" in cypher

    def test_token_mismatch_returns_noop_dict(self) -> None:
        """Token/stage mismatch → {'docs_chain_length': -1, 'revision_id': ''} (no-op)."""
        result, _ = self._call_persist(tx_run_return=[])  # empty = mismatch
        assert result["docs_chain_length"] == -1
        assert result["revision_id"] == ""

    def test_single_committed_transaction(self) -> None:
        """Snapshot + SN update land in a single committed transaction."""
        _, tx = self._call_persist()
        tx.run.assert_called_once()
        tx.commit.assert_called_once()
