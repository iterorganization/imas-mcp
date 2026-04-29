"""Tests for Phase 8 B3 — regen ↔ review claim exclusivity.

Acceptance criterion #4 (plan.md:870):
    Regen and review never claim the same name.

The three eligibility predicates (mirrored from Cypher WHERE clauses):

    claim_review_names:
        reviewed_name_at IS NULL
        AND validation_status = 'valid'
        AND coalesce(reviewer_score_name, 1.0) >= min_score

    claim_review_docs:
        reviewed_docs_at IS NULL
        AND enriched_at IS NOT NULL
        AND reviewed_name_at IS NOT NULL
        AND coalesce(reviewer_score_name, 1.0) >= min_score

    claim_regen:
        reviewer_score_name IS NOT NULL
        AND reviewer_score_name < min_score
        AND reviewed_name_at IS NOT NULL

All tests use an in-memory ``_InMemoryDB`` that mirrors these predicates
exactly — no live Neo4j required.  The pool orchestration is exercised
through real :class:`PoolSpec` / :func:`run_pools` objects with a
mock :class:`BudgetManager` that always admits (admission control is
not the subject under test here).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from imas_codex.standard_names.pools import PoolSpec, run_pools

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SCORE: float = 0.5
N_PER_GROUP: int = 12

# Sentinel timestamps (non-None means "has been reviewed / enriched")
_REVIEWED_AT = "2024-01-01T00:00:00"
_ENRICHED_AT = "2024-01-01T00:00:00"


# ---------------------------------------------------------------------------
# In-memory database
# ---------------------------------------------------------------------------


class _InMemoryDB:
    """Thread-safe in-memory StandardName store.

    Implements the same B3 eligibility predicates used by the three
    Cypher claim queries in ``graph_ops.py``, so tests can run without
    Neo4j while still verifying the predicate logic.
    """

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, dict[str, Any]] = {n["id"]: dict(n) for n in nodes}

    # ── eligibility predicates ─────────────────────────────────────────

    @staticmethod
    def _review_names_eligible(n: dict[str, Any], min_score: float) -> bool:
        score = n.get("reviewer_score_name")
        effective_score = score if score is not None else 1.0
        return (
            n.get("claimed_at") is None
            and n.get("reviewed_name_at") is None
            and n.get("validation_status") == "valid"
            and effective_score >= min_score
        )

    @staticmethod
    def _review_docs_eligible(n: dict[str, Any], min_score: float) -> bool:
        score = n.get("reviewer_score_name")
        effective_score = score if score is not None else 1.0
        return (
            n.get("claimed_at") is None
            and n.get("reviewed_docs_at") is None
            and n.get("enriched_at") is not None
            and n.get("reviewed_name_at") is not None
            and n.get("validation_status") == "valid"
            and effective_score >= min_score
        )

    @staticmethod
    def _regen_eligible(n: dict[str, Any], min_score: float) -> bool:
        score = n.get("reviewer_score_name")
        return (
            n.get("claimed_at") is None
            and score is not None
            and score < min_score
            and n.get("reviewed_name_at") is not None
        )

    # ── atomic claim ──────────────────────────────────────────────────

    def _claim_eligible(
        self,
        predicate: Any,
        min_score: float,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            eligible = [n for n in self._nodes.values() if predicate(n, min_score)]
            if not eligible:
                return []
            token = str(uuid.uuid4())
            batch = eligible[:batch_size]
            for n in batch:
                n["claimed_at"] = "now"
                n["claim_token"] = token
            return [
                {
                    "id": n["id"],
                    "_token": token,
                    "cluster_id": n.get("cluster_id"),
                    "unit": n.get("unit"),
                    "validation_status": n.get("validation_status"),
                }
                for n in batch
            ]

    # ── public claim methods ──────────────────────────────────────────

    def claim_review_names(
        self, *, min_score: float = MIN_SCORE, batch_size: int = 3
    ) -> list[dict[str, Any]]:
        return self._claim_eligible(self._review_names_eligible, min_score, batch_size)

    def claim_review_docs(
        self, *, min_score: float = MIN_SCORE, batch_size: int = 3
    ) -> list[dict[str, Any]]:
        return self._claim_eligible(self._review_docs_eligible, min_score, batch_size)

    def claim_regen(
        self, *, min_score: float = MIN_SCORE, batch_size: int = 3
    ) -> list[dict[str, Any]]:
        return self._claim_eligible(self._regen_eligible, min_score, batch_size)

    # ── release ───────────────────────────────────────────────────────

    def release_token(self, token: str) -> None:
        with self._lock:
            for n in self._nodes.values():
                if n.get("claim_token") == token:
                    n["claimed_at"] = None
                    n["claim_token"] = None

    # ── state advancement ─────────────────────────────────────────────

    def advance_reviewed_name_at(self, node_id: str) -> None:
        """Set reviewed_name_at and release any existing claim on *node_id*."""
        with self._lock:
            n = self._nodes[node_id]
            n["reviewed_name_at"] = _REVIEWED_AT
            n["claimed_at"] = None
            n["claim_token"] = None


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_db(
    *,
    groups: tuple[str, ...] = ("A", "B", "C"),
    n: int = N_PER_GROUP,
) -> _InMemoryDB:
    """Return an in-memory DB seeded with the requested groups.

    Group A — ``reviewed_name_at=None``, ``reviewer_score_name=None``
              → review_names eligible only.

    Group B — ``reviewed_name_at`` set, ``reviewer_score_name=0.3`` (<MIN_SCORE)
              → regen eligible only.

    Group C — ``reviewed_name_at`` set, ``reviewed_docs_at`` set,
              ``reviewer_score_name=0.9`` (≥MIN_SCORE, fully reviewed)
              → neither pool eligible.
    """
    nodes: list[dict[str, Any]] = []

    if "A" in groups:
        for i in range(n):
            nodes.append(
                {
                    "id": f"gA-{i:03d}",
                    "validation_status": "valid",
                    "reviewed_name_at": None,
                    "reviewed_docs_at": None,
                    "enriched_at": _ENRICHED_AT,
                    "reviewer_score_name": None,
                    "cluster_id": "cA",
                    "unit": "eV",
                    "claimed_at": None,
                    "claim_token": None,
                }
            )

    if "B" in groups:
        for i in range(n):
            nodes.append(
                {
                    "id": f"gB-{i:03d}",
                    "validation_status": "valid",
                    "reviewed_name_at": _REVIEWED_AT,
                    "reviewed_docs_at": None,
                    "enriched_at": _ENRICHED_AT,
                    "reviewer_score_name": 0.3,  # < MIN_SCORE
                    "cluster_id": "cB",
                    "unit": "T",
                    "claimed_at": None,
                    "claim_token": None,
                }
            )

    if "C" in groups:
        for i in range(n):
            nodes.append(
                {
                    "id": f"gC-{i:03d}",
                    "validation_status": "valid",
                    "reviewed_name_at": _REVIEWED_AT,
                    "reviewed_docs_at": _REVIEWED_AT,
                    "enriched_at": _ENRICHED_AT,
                    "reviewer_score_name": 0.9,  # >= MIN_SCORE
                    "cluster_id": "cC",
                    "unit": "m",
                    "claimed_at": None,
                    "claim_token": None,
                }
            )

    return _InMemoryDB(nodes)


def _mock_budget_manager() -> MagicMock:
    """BudgetManager that always admits and never charges."""
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.pool_admit = MagicMock(return_value=True)
    mgr.exhausted = MagicMock(return_value=False)
    mgr.drain_pending = AsyncMock(return_value=True)
    return mgr


def _make_pool(
    name: str,
    db_method: Any,
    claims_log: list[tuple[str, str, list[str]]],
    db: _InMemoryDB,
    *,
    backoff_base: float = 0.05,
    backoff_cap: float = 0.15,
) -> PoolSpec:
    """Build a :class:`PoolSpec` backed by an in-memory claim method.

    The process adapter records ``(pool_name, token, [item_id…])`` in
    *claims_log*, sleeps 50ms, releases the claim, and returns the count.
    """

    async def _claim() -> dict[str, Any] | None:
        items = db_method()
        if not items:
            return None
        return {"items": items}

    async def _process(batch: dict[str, Any]) -> int:
        items = batch["items"]
        if not items:
            return 0
        token = items[0].get("_token", "")
        ids = [item["id"] for item in items]
        claims_log.append((name, token, ids))
        await asyncio.sleep(0.05)
        db.release_token(token)
        return len(items)

    spec = PoolSpec(name=name, claim=_claim, process=_process)
    spec.health.pending_count = 100  # keep pool in active_pools_fn
    spec.backoff.base = backoff_base
    spec.backoff.cap = backoff_cap
    spec.backoff.reset()
    return spec


# ---------------------------------------------------------------------------
# Test 1 — main acceptance test: regen and review never share a name
# ---------------------------------------------------------------------------


class TestReviewAndRegenClaimDisjointNames:
    """Acceptance criterion #4: no single StandardName is claimed by both
    a review pool and the regen pool within the same run window.
    """

    @pytest.mark.asyncio
    async def test_review_and_regen_claim_disjoint_names(self) -> None:
        db = _make_db()
        mgr = _mock_budget_manager()
        stop = asyncio.Event()

        claims_log: list[tuple[str, str, list[str]]] = []

        group_a_ids = {f"gA-{i:03d}" for i in range(N_PER_GROUP)}
        group_b_ids = {f"gB-{i:03d}" for i in range(N_PER_GROUP)}
        group_c_ids = {f"gC-{i:03d}" for i in range(N_PER_GROUP)}

        pools = [
            _make_pool(
                "review_name",
                lambda: db.claim_review_names(min_score=MIN_SCORE),
                claims_log,
                db,
            ),
            _make_pool(
                "review_docs",
                lambda: db.claim_review_docs(min_score=MIN_SCORE),
                claims_log,
                db,
            ),
            _make_pool(
                "refine_name",
                lambda: db.claim_regen(min_score=MIN_SCORE),
                claims_log,
                db,
            ),
        ]

        async def _stop_after() -> None:
            # Run long enough for every eligible item to be processed.
            # With N=12 per group and batch_size=3, each pool needs at
            # most 4 rounds × 0.05s = 0.2s.  2s is generous.
            await asyncio.sleep(2.0)
            stop.set()

        await asyncio.gather(
            run_pools(pools, mgr, stop, grace_period=1.0),
            _stop_after(),
        )

        # ── Collect claimed IDs per pool category ─────────────────────
        review_claimed: set[str] = set()
        regen_claimed: set[str] = set()

        for pool_name, _token, ids in claims_log:
            if pool_name in {"review_name", "review_docs"}:
                review_claimed.update(ids)
            elif pool_name == "refine_name":
                regen_claimed.update(ids)

        # ── Primary assertion: disjoint ───────────────────────────────
        overlap = review_claimed & regen_claimed
        assert not overlap, (
            f"Names claimed by BOTH review and regen (B3 violation): {overlap}"
        )

        # ── Group A: unreviewed → review_name only, never refine_name ──
        regen_in_a = regen_claimed & group_a_ids
        assert not regen_in_a, (
            f"Refine_name claimed unreviewed (Group A) names: {regen_in_a}"
        )

        # ── Group B: low score → regen only, never review ─────────────
        review_in_b = review_claimed & group_b_ids
        assert not review_in_b, (
            f"Review claimed low-score (Group B) names: {review_in_b}"
        )

        # ── Group C: fully reviewed → neither pool should touch it ────
        review_in_c = review_claimed & group_c_ids
        regen_in_c = regen_claimed & group_c_ids
        assert not review_in_c, (
            f"Review claimed fully-reviewed (Group C) names: {review_in_c}"
        )
        assert not regen_in_c, (
            f"Regen claimed fully-reviewed (Group C) names: {regen_in_c}"
        )

        # ── Sanity: some work was done ────────────────────────────────
        # Group A should have been claimed by review_name.
        # Group B should have been claimed by refine_name.
        assert review_claimed & group_a_ids, (
            "review_name made zero progress on Group A — test may be misconfigured"
        )
        assert regen_claimed & group_b_ids, (
            "refine_name made zero progress on Group B — test may be misconfigured"
        )


# ---------------------------------------------------------------------------
# Test 2 — review excludes low-score names (Group B only)
# ---------------------------------------------------------------------------


class TestReviewExcludesLowScore:
    """claim_review_names and claim_review_docs return empty when the only
    available names have reviewer_score_name < min_score (Group B).
    """

    def test_review_names_returns_empty_for_group_b(self) -> None:
        db = _make_db(groups=("B",))
        result = db.claim_review_names(min_score=MIN_SCORE)
        assert result == [], (
            "claim_review_names must not claim low-score (Group B) names; "
            f"got: {[r['id'] for r in result]}"
        )

    def test_review_docs_returns_empty_for_group_b(self) -> None:
        db = _make_db(groups=("B",))
        result = db.claim_review_docs(min_score=MIN_SCORE)
        assert result == [], (
            "claim_review_docs must not claim low-score (Group B) names; "
            f"got: {[r['id'] for r in result]}"
        )


# ---------------------------------------------------------------------------
# Test 3 — regen excludes unreviewed names (Group A only)
# ---------------------------------------------------------------------------


class TestRegenExcludesUnreviewed:
    """claim_regen returns empty when the only available names have
    reviewed_name_at IS NULL (Group A).
    """

    def test_regen_returns_empty_for_group_a(self) -> None:
        db = _make_db(groups=("A",))
        result = db.claim_regen(min_score=MIN_SCORE)
        assert result == [], (
            "claim_regen must not claim unreviewed (Group A) names; "
            f"got: {[r['id'] for r in result]}"
        )


# ---------------------------------------------------------------------------
# Test 4 — review_names and review_docs CAN share the same name
# ---------------------------------------------------------------------------


class TestReviewNamesAndDocsCanShareAName:
    """review_names and review_docs are sequential review stages and ARE
    allowed to claim the same name — just not concurrently.

    Sequence:
      1. review_names claims the Group A node (reviewed_name_at IS NULL).
      2. After processing, reviewed_name_at is advanced (set to non-null).
      3. review_docs can now claim the same node
         (reviewed_docs_at IS NULL, enriched_at IS NOT NULL,
          reviewed_name_at IS NOT NULL, reviewer_score_name=NULL → 1.0 ≥ 0.5).
    """

    def test_review_names_then_review_docs_same_name(self) -> None:
        # Single Group A node, enriched.
        node_id = "gA-000"
        db = _InMemoryDB(
            [
                {
                    "id": node_id,
                    "validation_status": "valid",
                    "reviewed_name_at": None,
                    "reviewed_docs_at": None,
                    "enriched_at": _ENRICHED_AT,
                    "reviewer_score_name": None,  # coalesce → 1.0 >= MIN_SCORE
                    "cluster_id": "cA",
                    "unit": "eV",
                    "claimed_at": None,
                    "claim_token": None,
                }
            ]
        )

        # Step 1: review_names claims the node.
        batch_names = db.claim_review_names(min_score=MIN_SCORE)
        assert len(batch_names) == 1 and batch_names[0]["id"] == node_id, (
            "review_names should have claimed the Group A node"
        )
        token_names = batch_names[0]["_token"]

        # Step 2: release claim, simulate review_names persisting reviewed_name_at.
        db.release_token(token_names)
        db.advance_reviewed_name_at(node_id)

        # Now review_names should return empty (reviewed_name_at IS NOT NULL).
        assert db.claim_review_names(min_score=MIN_SCORE) == [], (
            "review_names must not re-claim a node that already has reviewed_name_at set"
        )

        # Step 3: review_docs should now claim the same node.
        batch_docs = db.claim_review_docs(min_score=MIN_SCORE)
        assert len(batch_docs) == 1 and batch_docs[0]["id"] == node_id, (
            "review_docs should claim the node after reviewed_name_at is set"
        )

        # Regen should NOT claim it (reviewer_score_name IS NULL → predicate fails).
        db.release_token(batch_docs[0]["_token"])
        assert db.claim_regen(min_score=MIN_SCORE) == [], (
            "regen must not claim a node with reviewer_score_name IS NULL"
        )
