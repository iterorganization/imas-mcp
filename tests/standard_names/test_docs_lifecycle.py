"""Integration lifecycle tests for the docs-side pipeline state machine.

These tests exercise multi-step docs_stage transitions end-to-end against a
real Neo4j graph.  LLM calls are intercepted by the MockLLM fixture defined
in conftest.py.  Each test builds its own isolated nodes (unique prefix),
runs a multi-step state sequence, and verifies the final graph state.

Specifically covered:
  1. Full acceptance path  (generate_docs → review_docs accept)
  2. Rotation to acceptance (generate → review reject → refine → review accept)
  3. Exhaustion path        (two refine rotations → exhausted at cap)
  4. Escalation model on final refine attempt
  5. generate_docs gates on name_stage=accepted
  6. Revision history preserved across multi-step chains
  7. chain_history walker correctness (2-deep revision chain)
  8. Acceptance overrides chain_length at cap
  9. Orphan sweep recovers stuck-refining docs nodes
 10. Concurrent review_docs does not double-advance

NOT individual unit tests — those live in the specific persist/claim test files.

Run these tests only when Neo4j is reachable:
    uv run pytest tests/standard_names/test_docs_lifecycle.py -m "graph and integration" -v
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.chain_history import docs_chain_history
from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.graph_ops import (
    claim_generate_docs_seed_and_expand,
    persist_generated_docs,
    persist_refined_docs,
    persist_reviewed_docs,
)
from imas_codex.standard_names.orphan_sweep import _orphan_sweep_tick

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ID_PREFIX = "dlc_test__"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    """Function-scoped real GraphClient; skip if Neo4j is unreachable."""
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    """Delete all lifecycle-test StandardName and DocsRevision nodes before/after."""

    def _wipe() -> None:
        # DocsRevision ids are "{sn_id}#rev-{n}" — same prefix as the SN id
        _gc.query(
            "MATCH (n:DocsRevision) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_TEST_ID_PREFIX,
        )
        _gc.query(
            "MATCH (n:StandardName) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_TEST_ID_PREFIX,
        )

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Graph helpers (create / fetch / manipulate)
# ---------------------------------------------------------------------------


def _uid(tag: str) -> str:
    """Return a unique, prefixed node id for the current test run."""
    return f"{_TEST_ID_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _create_sn_accepted(gc, sn_id: str) -> None:
    """Create a StandardName at name_stage='accepted', docs_stage='pending'."""
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage        = 'accepted',
            sn.docs_stage        = 'pending',
            sn.chain_length      = 0,
            sn.docs_chain_length = 0,
            sn.description       = 'Test quantity description',
            sn.documentation     = '',
            sn.kind              = 'scalar',
            sn.unit              = 'eV',
            sn.physics_domain    = ['core_profiles'],
            sn.reviewer_score_name   = 0.85,
            sn.reviewer_verdict_name = 'accept'
        """,
        id=sn_id,
    )


def _set_claim(gc, sn_id: str, token: str) -> None:
    gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.claim_token = $token,
            sn.claimed_at  = datetime()
        """,
        id=sn_id,
        token=token,
    )


def _set_docs_stage(gc, sn_id: str, stage: str) -> None:
    gc.query(
        "MATCH (sn:StandardName {id: $id}) SET sn.docs_stage = $stage",
        id=sn_id,
        stage=stage,
    )


def _fetch_sn_docs(gc, sn_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.docs_stage           AS docs_stage,
               sn.docs_chain_length    AS docs_chain_length,
               sn.description          AS description,
               sn.documentation        AS documentation,
               sn.reviewer_score_docs  AS reviewer_score_docs,
               sn.claim_token          AS claim_token,
               sn.claimed_at           AS claimed_at
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found in graph"
    return rows[0]


def _count_docs_revisions(gc, sn_id: str) -> int:
    """Count DocsRevision nodes linked from sn_id via DOCS_REVISION_OF."""
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:DOCS_REVISION_OF]->(rev:DocsRevision)
        RETURN count(rev) AS n
        """,
        id=sn_id,
    )
    return rows[0]["n"] if rows else 0


def _fetch_docs_revision(gc, rev_id: str) -> dict | None:
    rows = gc.query(
        """
        MATCH (rev:DocsRevision {id: $id})
        RETURN rev.description          AS description,
               rev.documentation        AS documentation,
               rev.revision_number      AS revision_number,
               rev.reviewer_score_docs  AS reviewer_score_docs,
               rev.reviewer_comments_docs AS reviewer_comments_docs
        """,
        id=rev_id,
    )
    return rows[0] if rows else None


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


# ---------------------------------------------------------------------------
# Helper: simulate a full generate→draft cycle directly
# ---------------------------------------------------------------------------


def _generate_docs(gc, sn_id: str) -> str:
    """Directly persist generated docs (bypasses LLM worker)."""
    token = f"tok-gen-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, token)
    new_stage = persist_generated_docs(
        sn_id=sn_id,
        claim_token=token,
        description="Generated description for test quantity",
        documentation="## Test Quantity\n\nGenerated documentation for testing.",
        model="test/model",
    )
    assert new_stage == "drafted", f"Expected 'drafted', got {new_stage!r}"
    return new_stage


def _review_docs(
    gc,
    sn_id: str,
    verdict: str,
    score: float,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
) -> str:
    """Directly persist a docs review (bypasses LLM worker)."""
    token = f"tok-rev-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, token)
    new_stage = persist_reviewed_docs(
        sn_id=sn_id,
        claim_token=token,
        score=score,
        verdict=verdict,
        comments="Test reviewer comment",
        model="test/model",
        min_score=0.75,
        rotation_cap=rotation_cap,
    )
    return new_stage


def _refine_docs(gc, sn_id: str, iteration: int = 0) -> dict:
    """Simulate a refine_docs cycle: set refining stage + call persist."""
    token = f"tok-rfn-{uuid.uuid4().hex[:8]}"
    # Fetch current docs for snapshotting (including reviewer feedback)
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.description            AS description,
               sn.documentation          AS documentation,
               sn.reviewer_score_docs    AS reviewer_score_docs,
               sn.reviewer_comments_docs AS reviewer_comments_docs,
               sn.reviewer_verdict_docs  AS reviewer_verdict_docs
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found"
    snap = rows[0]
    current_desc = snap["description"] or ""
    current_doc = snap["documentation"] or ""
    snap_score = snap.get("reviewer_score_docs")
    snap_comments = snap.get("reviewer_comments_docs")
    snap_verdict = snap.get("reviewer_verdict_docs")
    # Transition to 'refining' + set claim
    gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.docs_stage  = 'refining',
            sn.claim_token = $token,
            sn.claimed_at  = datetime()
        """,
        id=sn_id,
        token=token,
    )
    result = persist_refined_docs(
        sn_id=sn_id,
        claim_token=token,
        description=f"Refined description iteration {iteration}",
        documentation=f"## Refined docs\n\nIteration {iteration} of the documentation.",
        model="test/model",
        current_description=current_desc,
        current_documentation=current_doc,
        current_model="test/model",
        current_generated_at=None,
        reviewer_score_to_snapshot=snap_score,
        reviewer_comments_to_snapshot=snap_comments or "Needs improvement",
        reviewer_comments_per_dim_to_snapshot=None,
        reviewer_verdict_to_snapshot=snap_verdict or "reject",
    )
    return result


# ===========================================================================
# D1. test_full_docs_acceptance_path
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_full_docs_acceptance_path(_gc, _clean):
    """generate_docs → review_docs accept: docs_stage='accepted', chain=0, no revisions.

    Asserts docs_chain_length=0, reviewer_score_docs=0.85, claim cleared,
    and no DocsRevision nodes created.
    """
    sn_id = _uid("accept_v1")
    _create_sn_accepted(_gc, sn_id)

    # Step 1: generate docs
    _generate_docs(_gc, sn_id)

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "drafted"
    assert row["description"] == "Generated description for test quantity"
    assert row["documentation"] != ""
    assert row["claim_token"] is None  # cleared after generate

    # Step 2: review with accept
    result = _review_docs(_gc, sn_id, verdict="accept", score=0.85)

    assert result == "accepted", f"Expected 'accepted', got {result!r}"

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "accepted"
    assert row["docs_chain_length"] == 0
    assert row["reviewer_score_docs"] == pytest.approx(0.85)
    assert row["claim_token"] is None

    # No DocsRevision nodes
    assert _count_docs_revisions(_gc, sn_id) == 0


# ===========================================================================
# D2. test_docs_rotation_to_acceptance
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_docs_rotation_to_acceptance(_gc, _clean):
    """generate → review reject → refine → review accept.

    Asserts 1 DocsRevision snapshot created, DOCS_REVISION_OF edge present,
    and SN description differs from the revision's description.
    """
    sn_id = _uid("rotate_v1")
    _create_sn_accepted(_gc, sn_id)

    # Step 1: generate docs
    _generate_docs(_gc, sn_id)
    original_desc = _fetch_sn_docs(_gc, sn_id)["description"]

    # Step 2: reject review
    r1 = _review_docs(_gc, sn_id, verdict="reject", score=0.5)
    assert r1 == "reviewed", f"Expected 'reviewed', got {r1!r}"

    # Step 3: refine → DocsRevision_v0 created
    refine_result = _refine_docs(_gc, sn_id, iteration=0)
    assert refine_result["docs_chain_length"] == 1
    assert refine_result["revision_id"] != ""

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "drafted"
    assert row["docs_chain_length"] == 1

    # Step 4: accept review
    r2 = _review_docs(_gc, sn_id, verdict="accept", score=0.85)
    assert r2 == "accepted", f"Expected 'accepted', got {r2!r}"

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "accepted"

    # Verify DocsRevision was created
    assert _count_docs_revisions(_gc, sn_id) == 1

    # Verify SN description differs from what was snapshotted
    rev_id = refine_result["revision_id"]
    rev = _fetch_docs_revision(_gc, rev_id)
    assert rev is not None, f"DocsRevision {rev_id!r} not found"
    assert rev["description"] == original_desc  # snapshot has original
    assert row["description"] != original_desc  # SN has refined version


# ===========================================================================
# D3. test_docs_exhaustion_path
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_docs_exhaustion_path(_gc, _clean):
    """Two refine rotations → review_docs reject → docs_stage='exhausted'.

    With rotation_cap=3, docs_chain_length=2 is the final attempt.
    Asserts 2 DocsRevision nodes and SN not eligible for further refine.
    """
    sn_id = _uid("exhaust_v1")
    _create_sn_accepted(_gc, sn_id)

    # Cycle 1: generate → review reject → refine
    _generate_docs(_gc, sn_id)
    r1 = _review_docs(_gc, sn_id, verdict="reject", score=0.5)
    assert r1 == "reviewed"
    result1 = _refine_docs(_gc, sn_id, iteration=0)
    assert result1["docs_chain_length"] == 1

    # Cycle 2: review reject → refine
    r2 = _review_docs(_gc, sn_id, verdict="reject", score=0.5)
    assert r2 == "reviewed"
    result2 = _refine_docs(_gc, sn_id, iteration=1)
    assert result2["docs_chain_length"] == 2

    # Cycle 3: review reject at chain_length=2, rotation_cap=3 → exhausted
    r3 = _review_docs(_gc, sn_id, verdict="reject", score=0.5, rotation_cap=3)
    assert r3 == "exhausted", f"Expected 'exhausted', got {r3!r}"

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "exhausted"

    # Two DocsRevision snapshots
    assert _count_docs_revisions(_gc, sn_id) == 2

    # SN is NOT eligible for further refine (docs_stage != 'reviewed')
    rows = _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        WHERE sn.docs_stage = 'reviewed' OR sn.docs_stage = 'refining'
        RETURN sn.id AS id
        """,
        id=sn_id,
    )
    assert not rows, "Exhausted SN should not be eligible for refine_docs"


# ===========================================================================
# D4. test_docs_escalation_at_final_attempt
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_docs_escalation_at_final_attempt(_gc, _clean, mock_llm):
    """process_refine_docs_batch uses DEFAULT_ESCALATION_MODEL at docs_chain_length=2."""
    from imas_codex.standard_names.models import RefinedDocs
    from imas_codex.standard_names.workers import process_refine_docs_batch

    sn_id = _uid("esc_docs")
    token = f"tok-esc-{uuid.uuid4().hex[:8]}"

    # Create SN in 'refining' state at docs_chain_length=2 (final attempt)
    _create_sn_accepted(_gc, sn_id)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.docs_stage        = 'refining',
            sn.docs_chain_length = 2,
            sn.description       = 'Existing description',
            sn.documentation     = 'Existing documentation',
            sn.claim_token       = $token,
            sn.claimed_at        = datetime()
        """,
        id=sn_id,
        token=token,
    )

    mock_llm.add_response(
        "unknown",
        response=RefinedDocs(
            description="Escalation-refined description for test quantity",
            documentation="## Escalation-refined docs\n\nFinal attempt.",
        ),
    )

    item = {
        "id": sn_id,
        "claim_token": token,
        "docs_chain_length": 2,
        "docs_chain_history": [],
        "description": "Existing description",
        "documentation": "Existing documentation",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["core_profiles"],
        "physics_domain": ["core_profiles"],
        "docs_stage": "refining",
        "docs_model": "test/model",
        "docs_generated_at": None,
        "reviewer_score_docs": 0.5,
        "reviewer_comments_docs": "Needs improvement",
        "reviewer_comments_per_dim_docs": None,
        "reviewer_verdict_docs": "reject",
    }
    stop_event = asyncio.Event()

    with patch(
        "imas_codex.llm.prompt_loader.render_prompt",
        return_value="Refine these docs.",
    ):
        processed = asyncio.run(
            process_refine_docs_batch([item], _mock_budget_manager(), stop_event)
        )

    assert processed == 1, "Worker should have processed exactly one item"

    # LLM must have been called with the escalation model
    call_record = mock_llm.calls[0]
    assert call_record["model"] == DEFAULT_ESCALATION_MODEL, (
        f"Expected escalation model {DEFAULT_ESCALATION_MODEL!r}, "
        f"got {call_record['model']!r}"
    )


# ===========================================================================
# D5. test_generate_docs_gates_on_name_accepted
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_generate_docs_gates_on_name_accepted(_gc, _clean):
    """generate_docs eligibility gates on name_stage='accepted'.

    Verifies that ``claim_generate_docs_seed_and_expand`` returns only SNs
    with ``name_stage='accepted' AND docs_stage='pending'``.  The duplicate-
    field bug (Neo4j 42N38) that previously blocked this call has been fixed
    by removing ``description`` and ``kind`` from ``extra_return_fields``
    (they are already present in the base readback query of ``_claim_sn_atomic``).
    """
    sn_reviewed = _uid("gate_reviewed")
    sn_accepted = _uid("gate_accepted")

    # SN with name_stage='reviewed' — NOT eligible for generate_docs
    _gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage    = 'reviewed',
            sn.docs_stage    = 'pending',
            sn.kind          = 'scalar',
            sn.unit          = 'eV',
            sn.description   = 'Test',
            sn.physics_domain = ['core_profiles']
        """,
        id=sn_reviewed,
    )

    # SN with name_stage='accepted' — ELIGIBLE
    _create_sn_accepted(_gc, sn_accepted)

    # Call the real claim function — this would previously fail with Neo4j
    # GQL error 42N38 (duplicate return item name).
    claimed = claim_generate_docs_seed_and_expand(batch_size=10)
    claimed_ids = {item["id"] for item in claimed}

    assert sn_reviewed not in claimed_ids, (
        "name_stage='reviewed' SN must not be eligible for generate_docs"
    )
    assert sn_accepted in claimed_ids, (
        "name_stage='accepted' SN must be eligible for generate_docs"
    )


# ===========================================================================
# D6. test_revisions_preserve_full_history
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_revisions_preserve_full_history(_gc, _clean):
    """Two refine cycles preserve reviewer feedback on each DocsRevision snapshot.

    Builds: generate → review reject → refine (rev0) → review reject → refine (rev1)
    → review accept.  Asserts revision_number=0 and 1, both reachable, and
    reviewer feedback preserved per snapshot.
    """
    sn_id = _uid("history_v1")
    _create_sn_accepted(_gc, sn_id)

    # Cycle 1: generate → review reject → refine
    _generate_docs(_gc, sn_id)
    _review_docs(_gc, sn_id, verdict="reject", score=0.4)
    result0 = _refine_docs(_gc, sn_id, iteration=0)
    rev0_id = result0["revision_id"]

    # Cycle 2: review reject → refine
    _review_docs(_gc, sn_id, verdict="reject", score=0.55)
    result1 = _refine_docs(_gc, sn_id, iteration=1)
    rev1_id = result1["revision_id"]

    # Accept
    r_final = _review_docs(_gc, sn_id, verdict="accept", score=0.85)
    assert r_final == "accepted"

    # Both revisions exist
    assert _count_docs_revisions(_gc, sn_id) == 2

    rev0 = _fetch_docs_revision(_gc, rev0_id)
    rev1 = _fetch_docs_revision(_gc, rev1_id)

    assert rev0 is not None, f"DocsRevision {rev0_id!r} not found"
    assert rev1 is not None, f"DocsRevision {rev1_id!r} not found"

    assert rev0["revision_number"] == 0
    assert rev1["revision_number"] == 1

    # Reviewer feedback was snapshotted
    assert rev0["reviewer_score_docs"] == pytest.approx(0.4)
    assert rev1["reviewer_score_docs"] == pytest.approx(0.55)
    assert rev0["reviewer_comments_docs"] is not None
    assert rev1["reviewer_comments_docs"] is not None


# ===========================================================================
# D7. test_chain_history_walks_revisions
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_chain_history_walks_revisions(_gc, _clean):
    """docs_chain_history(sn_id) returns 2 entries with scores and comments."""
    sn_id = _uid("walker_v1")
    _create_sn_accepted(_gc, sn_id)

    # Build 2-revision chain
    _generate_docs(_gc, sn_id)
    _review_docs(_gc, sn_id, verdict="reject", score=0.3)
    _refine_docs(_gc, sn_id, iteration=0)
    _review_docs(_gc, sn_id, verdict="reject", score=0.55)
    _refine_docs(_gc, sn_id, iteration=1)

    history = docs_chain_history(sn_id)

    assert len(history) == 2, (
        f"Expected 2 history entries, got {len(history)}: {history}"
    )

    # Oldest first (created_at ASC ordering)
    assert "documentation" in history[0]
    assert "reviewer_score" in history[0]
    assert "reviewer_comments_per_dim" in history[0]
    assert "created_at" in history[0]

    # Scores are present and ordered: older revision (lower score) first
    assert history[0]["reviewer_score"] == pytest.approx(0.3)
    assert history[1]["reviewer_score"] == pytest.approx(0.55)


# ===========================================================================
# D8. test_docs_acceptance_overrides_chain_length_at_cap
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_docs_acceptance_overrides_chain_length_at_cap(_gc, _clean):
    """Accept verdict wins even at rotation-cap chain_length (not exhausted).

    SN at docs_chain_length=2, rotation_cap=3, score>=min → 'accepted', not 'exhausted'.
    """
    sn_id = _uid("cap_accept_docs")
    token = f"tok-cap-{uuid.uuid4().hex[:8]}"

    _create_sn_accepted(_gc, sn_id)
    # Directly set docs_chain_length=2 (would be exhausted on reject)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.docs_stage        = 'drafted',
            sn.docs_chain_length = 2,
            sn.description       = 'Test',
            sn.documentation     = 'Test docs'
        """,
        id=sn_id,
    )
    _set_claim(_gc, sn_id, token)

    result = persist_reviewed_docs(
        sn_id=sn_id,
        claim_token=token,
        score=0.85,
        verdict="accept",
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )

    assert result == "accepted", (
        f"Accept verdict must win over cap rule; got {result!r}"
    )
    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "accepted"


# ===========================================================================
# D9. test_docs_orphan_sweep_recovers_stuck_refining
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_docs_orphan_sweep_recovers_stuck_refining(_gc, _clean):
    """Orphan sweep reverts docs_stage='refining' with stale claimed_at → 'reviewed'."""
    sn_id = _uid("stuck_docs_refining")
    _gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage        = 'accepted',
            sn.docs_stage        = 'refining',
            sn.docs_chain_length = 0,
            sn.kind              = 'scalar',
            sn.claim_token       = $token,
            sn.claimed_at        = datetime() - duration({seconds: 400})
        """,
        id=sn_id,
        token="tok-stuck-docs",
    )

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] >= 1, (
        f"Sweep should have reverted at least one stuck docs node; counts={counts}"
    )

    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ===========================================================================
# D10. test_concurrent_review_docs_does_not_double_advance
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_concurrent_review_docs_does_not_double_advance(_gc, _clean):
    """Second persist_reviewed_docs with stale token is a no-op.

    Sequence:
      1. SN drafted, claimed by Worker A with token T1.
      2. Worker A persists → docs_stage='accepted', score=0.85.
      3. Worker B tries to persist with stale token T2 → no-op.
      4. Reviewer score reflects only Worker A's review.
    """
    sn_id = _uid("concurrent_docs")
    t1 = f"tok-DW1-{uuid.uuid4().hex[:8]}"
    t2 = f"tok-DW2-{uuid.uuid4().hex[:8]}"

    _create_sn_accepted(_gc, sn_id)
    # Set docs_stage='drafted' so review is eligible
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.docs_stage        = 'drafted',
            sn.docs_chain_length = 0,
            sn.description       = 'Test',
            sn.documentation     = 'Test docs'
        """,
        id=sn_id,
    )
    _set_claim(_gc, sn_id, t1)

    # Worker A persists first — succeeds
    r1 = persist_reviewed_docs(
        sn_id=sn_id,
        claim_token=t1,
        score=0.85,
        verdict="accept",
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r1 == "accepted"

    # Worker B tries to persist with a different (stale) token — must be no-op
    r2 = persist_reviewed_docs(
        sn_id=sn_id,
        claim_token=t2,
        score=0.30,
        verdict="reject",
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r2 == "", f"Stale-token persist must be a no-op; got {r2!r}"

    # Graph reflects only the first review
    row = _fetch_sn_docs(_gc, sn_id)
    assert row["docs_stage"] == "accepted", (
        f"docs_stage should be 'accepted'; got {row['docs_stage']!r}"
    )
    assert row["reviewer_score_docs"] == pytest.approx(0.85), (
        f"Score should reflect Worker A's 0.85; got {row['reviewer_score_docs']}"
    )
