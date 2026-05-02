"""Integration lifecycle tests for the standard-name pipeline state machine.

These tests exercise multi-step name_stage transitions end-to-end against a
real Neo4j graph.  LLM calls are intercepted by the MockLLM fixture defined
in conftest.py.  Each test builds its own isolated nodes (unique prefix),
runs a multi-step state sequence, and verifies the final graph state.

Specifically covered:
  1. Full acceptance path  (generate → review accept)
  2. Rotation to acceptance (generate → review reject → refine → review accept)
  3. Exhaustion path        (two refine rotations → review exhausted at cap)
  4. Escalation model on final refine attempt
  5. Acceptance overrides chain_length at cap
  6. Edge migration idempotent across multiple refines
  7. chain_history walker correctness (3-deep chain)
  8. Orphan sweep recovers stuck-refining nodes
  9. Token clobber prevented under race
 10. Concurrent review does not double-advance

NOT individual unit tests — those live in test_generate_name_persist.py,
test_refine_name_chain.py, and test_review_name_stages.py.

Run these tests only when Neo4j is reachable:
    uv run pytest tests/standard_names/test_name_lifecycle.py -m "graph and integration" -v
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.chain_history import name_chain_history
from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.graph_ops import (
    _finalize_generated_name_stage,
    persist_refined_name,
    persist_reviewed_name,
    release_review_names_failed_claims,
)
from imas_codex.standard_names.orphan_sweep import _orphan_sweep_tick

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ID_PREFIX = "lc_test__"


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
    """Delete all lifecycle-test nodes before and after each test."""

    def _wipe() -> None:
        _gc.query(
            "MATCH (n:StandardName) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_TEST_ID_PREFIX,
        )
        _gc.query(
            "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_TEST_ID_PREFIX,
        )

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Graph helpers (create / fetch / link)
# ---------------------------------------------------------------------------


def _uid(tag: str) -> str:
    """Return a unique, prefixed node id for the current test run."""
    return f"{_TEST_ID_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _create_source(gc, source_id: str) -> None:
    gc.query(
        """
        MERGE (sns:StandardNameSource {id: $id})
        SET sns.status        = 'extracted',
            sns.source_type   = 'dd',
            sns.source_id     = 'test/path',
            sns.description   = 'A test quantity',
            sns.physics_domain = 'core_profiles'
        """,
        id=source_id,
    )


def _create_sn(
    gc,
    sn_id: str,
    *,
    name_stage: str = "drafted",
    chain_length: int = 0,
    reviewer_score: float | None = None,
) -> None:
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage    = $name_stage,
            sn.chain_length  = $chain_length,
            sn.docs_stage    = 'pending',
            sn.description   = 'Test quantity',
            sn.kind          = 'scalar',
            sn.unit          = 'eV',
            sn.physics_domain = ['core_profiles']
        """,
        id=sn_id,
        name_stage=name_stage,
        chain_length=chain_length,
    )
    if reviewer_score is not None:
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.reviewer_score_name   = $score,
                sn.reviewer_verdict_name = 'revise'
            """,
            id=sn_id,
            score=reviewer_score,
        )


def _link_source_sn(gc, source_id: str, sn_id: str) -> None:
    gc.query(
        """
        MATCH (sns:StandardNameSource {id: $src_id})
        MATCH (sn:StandardName        {id: $sn_id})
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        src_id=source_id,
        sn_id=sn_id,
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


def _fetch_sn(gc, sn_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.name_stage            AS name_stage,
               sn.chain_length          AS chain_length,
               sn.reviewer_score_name   AS reviewer_score_name,
               sn.claim_token           AS claim_token,
               sn.claimed_at            AS claimed_at
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found in graph"
    return rows[0]


def _fetch_produced_name_target(gc, source_id: str) -> str | None:
    """Return the SN id that the source currently points to via PRODUCED_NAME."""
    rows = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $id})-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN sn.id AS sn_id
        """,
        id=source_id,
    )
    return rows[0]["sn_id"] if rows else None


def _count_produced_name_incoming(gc, sn_id: str) -> int:
    """Return how many PRODUCED_NAME edges point to sn_id."""
    rows = gc.query(
        """
        MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $id})
        RETURN count(*) AS n
        """,
        id=sn_id,
    )
    return rows[0]["n"] if rows else 0


def _has_refined_from_edge(gc, new_id: str, old_id: str) -> bool:
    rows = gc.query(
        """
        MATCH (n:StandardName {id: $new_id})-[:REFINED_FROM]->(o:StandardName {id: $old_id})
        RETURN count(*) AS n
        """,
        new_id=new_id,
        old_id=old_id,
    )
    return bool(rows and rows[0]["n"] > 0)


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


# ---------------------------------------------------------------------------
# Helper: build a refine-name item dict (matches claim_refine_name readback)
# ---------------------------------------------------------------------------


def _make_refine_item(
    sn_id: str,
    *,
    chain_length: int = 0,
    claim_token: str = "tok-refine-test",
) -> dict:
    return {
        "id": sn_id,
        "description": "Test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": 0.5,
        "reviewer_comments_per_dim_name": None,
        "chain_length": chain_length,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/test_quantity"],
        "tags": ["test"],
        "claim_token": claim_token,
        "chain_history": [],
    }


# ===========================================================================
# I1. test_full_acceptance_path
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_full_acceptance_path(_gc, _clean):
    """generate→review_accept: SN reaches 'accepted' in one step.

    Asserts chain_length=0, reviewer_score_name=0.85, claim cleared,
    and PRODUCED_NAME edge from source to SN still intact.
    """
    sn_id = _uid("accept_v1")
    src_id = _uid("src_accept")
    token = f"tok-accept-{uuid.uuid4().hex[:8]}"

    # Set up: source + drafted SN linked via PRODUCED_NAME
    _create_source(_gc, src_id)
    _create_sn(_gc, sn_id, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_id)
    _set_claim(_gc, sn_id, token)

    # Review with accept verdict, score above min
    result = persist_reviewed_name(
        sn_id=sn_id,
        claim_token=token,
        score=0.85,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )

    assert result == "accepted", f"Expected 'accepted', got {result!r}"

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "accepted"
    assert row["reviewer_score_name"] == pytest.approx(0.85)
    assert row["chain_length"] == 0
    assert row["claim_token"] is None
    assert row["claimed_at"] is None

    # PRODUCED_NAME edge preserved
    assert _fetch_produced_name_target(_gc, src_id) == sn_id


# ===========================================================================
# I2. test_rotation_to_acceptance
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_rotation_to_acceptance(_gc, _clean):
    """generate→review_reject→refine→review_accept: 2-node chain, SN_v2 accepted.

    Asserts: 2 SN nodes (one superseded, one accepted), REFINED_FROM edge,
    and source PRODUCED_NAME migrated to SN_v2.  SN_v1 has no incoming
    PRODUCED_NAME.
    """
    src_id = _uid("src_rotate")
    sn_v1 = _uid("rotate_v1")
    sn_v2 = _uid("rotate_v2")
    tok_v1 = f"tok-r1-{uuid.uuid4().hex[:8]}"
    tok_v2 = f"tok-r2-{uuid.uuid4().hex[:8]}"

    # Set up
    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)
    _set_claim(_gc, sn_v1, tok_v1)

    # Step 1: reject review → SN_v1 reviewed
    r1 = persist_reviewed_name(
        sn_id=sn_v1,
        claim_token=tok_v1,
        score=0.5,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r1 == "reviewed"

    # Step 2: refine SN_v1 → SN_v2 (new node, edge migration)
    persist_refined_name(
        old_name=sn_v1,
        new_name=sn_v2,
        description="Improved test quantity",
        kind="scalar",
        unit="eV",
        old_chain_length=0,
        model="test/model",
    )

    # Step 3: claim SN_v2, accept review
    _set_claim(_gc, sn_v2, tok_v2)
    r2 = persist_reviewed_name(
        sn_id=sn_v2,
        claim_token=tok_v2,
        score=0.8,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r2 == "accepted"

    # Verify graph state
    row_v1 = _fetch_sn(_gc, sn_v1)
    row_v2 = _fetch_sn(_gc, sn_v2)

    assert row_v1["name_stage"] == "superseded"
    assert row_v2["name_stage"] == "accepted"
    assert row_v2["chain_length"] == 1

    # REFINED_FROM edge: SN_v2 → SN_v1
    assert _has_refined_from_edge(_gc, sn_v2, sn_v1)

    # Source points to SN_v2 (migrated); SN_v1 has no PRODUCED_NAME
    assert _fetch_produced_name_target(_gc, src_id) == sn_v2
    assert _count_produced_name_incoming(_gc, sn_v1) == 0


# ===========================================================================
# I3. test_exhaustion_path
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_exhaustion_path(_gc, _clean):
    """Four-step rejection chain leads to 'exhausted' at rotation_cap=3.

    SN_v4 (chain_length=3) is rejected after three prior refine attempts
    (the final one routed through the Opus escalator at chain_length=2)
    → name_stage='exhausted'.  The SN is no longer eligible for
    refine_name.

    Pre-2026-05-03: exhaustion fired at chain=2 directly, pre-empting
    the escalator.  Post-fix: chain=2 stays 'reviewed' so the escalator
    can spend its final attempt; exhaustion fires only at chain=3.
    """
    src_id = _uid("src_exhaust")
    sn_v1 = _uid("exhaust_v1")
    sn_v2 = _uid("exhaust_v2")
    sn_v3 = _uid("exhaust_v3")
    sn_v4 = _uid("exhaust_v4")

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)

    # Review + refine loop × 3 (third cycle is the escalator attempt
    # that previously could not run because of premature exhaustion).
    for old_sn, new_sn, old_chain in [
        (sn_v1, sn_v2, 0),
        (sn_v2, sn_v3, 1),
        (sn_v3, sn_v4, 2),
    ]:
        tok = f"tok-ex-{uuid.uuid4().hex[:8]}"
        _set_claim(_gc, old_sn, tok)
        r = persist_reviewed_name(
            sn_id=old_sn,
            claim_token=tok,
            score=0.5,
            model="test/model",
            min_score=0.75,
            rotation_cap=3,
        )
        assert r == "reviewed", f"Expected 'reviewed' for {old_sn!r}, got {r!r}"
        persist_refined_name(
            old_name=old_sn,
            new_name=new_sn,
            description="Revised quantity",
            kind="scalar",
            unit="eV",
            old_chain_length=old_chain,
            model="test/model",
        )

    # Final review on SN_v4 (chain_length=3, rotation_cap=3 → exhausted)
    tok4 = f"tok-ex4-{uuid.uuid4().hex[:8]}"
    _set_claim(_gc, sn_v4, tok4)
    r4 = persist_reviewed_name(
        sn_id=sn_v4,
        claim_token=tok4,
        score=0.5,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r4 == "exhausted", f"Expected 'exhausted', got {r4!r}"

    row_v4 = _fetch_sn(_gc, sn_v4)
    assert row_v4["name_stage"] == "exhausted"
    assert row_v4["chain_length"] == 3

    # SN_v4 is NOT eligible for further refine_name (name_stage != 'reviewed')
    rows = _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        WHERE sn.name_stage = 'reviewed'
           OR sn.name_stage = 'refining'
        RETURN sn.id AS id
        """,
        id=sn_v4,
    )
    assert not rows, "Exhausted SN should not be eligible for refine_name"


# ===========================================================================
# I4. test_escalation_at_final_attempt
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_escalation_at_final_attempt(_gc, _clean, mock_llm):
    """Worker uses DEFAULT_ESCALATION_MODEL when chain_length >= rotation_cap - 1.

    Creates a real SN in the graph at chain_length=2 (final attempt before
    rotation_cap=3 exhaustion) and calls process_refine_name_batch.  Asserts
    the LLM was invoked with the escalation model.
    """
    from imas_codex.standard_names.models import RefinedName
    from imas_codex.standard_names.workers import process_refine_name_batch

    sn_old = _uid("esc_old")
    sn_new = _uid("esc_new")
    token = f"tok-esc-{uuid.uuid4().hex[:8]}"

    # Create a real graph node at chain_length=2 (the final attempt position)
    _create_sn(_gc, sn_old, name_stage="refining", chain_length=2)
    _set_claim(_gc, sn_old, token)

    # MockLLM will receive stage='unknown' because process_refine_name_batch
    # only sends a user-role message (no system message for stage inference).
    mock_llm.add_response(
        "unknown",
        response=RefinedName(
            name=sn_new,
            description="Escalation-refined test quantity",
            kind="scalar",
        ),
    )

    item = _make_refine_item(sn_old, chain_length=2, claim_token=token)
    stop_event = asyncio.Event()

    with patch(
        "imas_codex.llm.prompt_loader.render_prompt",
        return_value="Refine this name.",
    ):
        processed = asyncio.run(
            process_refine_name_batch([item], _mock_budget_manager(), stop_event)
        )

    assert processed == 1, "Worker should have processed exactly one item"
    assert mock_llm.calls_for("unknown") == 1

    # The LLM must have been called with the escalation model
    call_record = mock_llm.calls[0]
    assert call_record["model"] == DEFAULT_ESCALATION_MODEL, (
        f"Expected escalation model {DEFAULT_ESCALATION_MODEL!r}, "
        f"got {call_record['model']!r}"
    )


# ===========================================================================
# I5. test_acceptance_overrides_chain_length_at_cap
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_acceptance_overrides_chain_length_at_cap(_gc, _clean):
    """Accept verdict always wins even at the rotation-cap chain_length.

    SN at chain_length=3, rotation_cap=3 → would be 'exhausted' on
    reject, but a score-≥-min + accept verdict produces 'accepted'
    instead.  (Pre-2026-05-03 the gate was at chain=2 — moved to
    chain=3 to keep the Opus escalator at chain=2 reachable.)
    """
    sn_id = _uid("cap_accept")
    token = f"tok-cap-{uuid.uuid4().hex[:8]}"

    _create_sn(_gc, sn_id, name_stage="drafted", chain_length=3)
    _set_claim(_gc, sn_id, token)

    result = persist_reviewed_name(
        sn_id=sn_id,
        claim_token=token,
        score=0.85,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )

    assert result == "accepted", f"Acceptance must win over cap rule; got {result!r}"
    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "accepted"


# ===========================================================================
# I6. test_edge_migration_idempotent
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_edge_migration_idempotent(_gc, _clean):
    """Edges always migrate to the latest SN in a multi-refine chain.

    Builds a 3-level chain (v1 → v2 → v3).  After each refine the
    PRODUCED_NAME edge migrates forward.  Intermediate SNs end up as
    'superseded' with no incoming source edges.
    """
    src_id = _uid("src_migrate")
    sn_v1 = _uid("migrate_v1")
    sn_v2 = _uid("migrate_v2")
    sn_v3 = _uid("migrate_v3")

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)

    # First refine cycle (v1 → v2)
    tok1 = f"tok-m1-{uuid.uuid4().hex[:8]}"
    _set_claim(_gc, sn_v1, tok1)
    persist_reviewed_name(
        sn_id=sn_v1,
        claim_token=tok1,
        score=0.5,
        model="m",
        min_score=0.75,
        rotation_cap=3,
    )
    persist_refined_name(
        old_name=sn_v1,
        new_name=sn_v2,
        description="v2 quantity",
        kind="scalar",
        unit="eV",
        old_chain_length=0,
        model="m",
    )

    # After first refine: source → SN_v2
    assert _fetch_produced_name_target(_gc, src_id) == sn_v2
    assert _count_produced_name_incoming(_gc, sn_v1) == 0

    # Second refine cycle (v2 → v3)
    tok2 = f"tok-m2-{uuid.uuid4().hex[:8]}"
    _set_claim(_gc, sn_v2, tok2)
    persist_reviewed_name(
        sn_id=sn_v2,
        claim_token=tok2,
        score=0.5,
        model="m",
        min_score=0.75,
        rotation_cap=3,
    )
    persist_refined_name(
        old_name=sn_v2,
        new_name=sn_v3,
        description="v3 quantity",
        kind="scalar",
        unit="eV",
        old_chain_length=1,
        model="m",
    )

    # After second refine: source → SN_v3 only
    assert _fetch_produced_name_target(_gc, src_id) == sn_v3
    assert _count_produced_name_incoming(_gc, sn_v2) == 0
    assert _count_produced_name_incoming(_gc, sn_v1) == 0

    # SN_v1 and SN_v2 are superseded
    for sn_id in (sn_v1, sn_v2):
        row = _fetch_sn(_gc, sn_id)
        assert row["name_stage"] == "superseded", (
            f"{sn_id!r} should be superseded, got {row['name_stage']!r}"
        )

    # SN_v3 has draft stage (ready for review)
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["name_stage"] == "drafted"
    assert row_v3["chain_length"] == 2


# ===========================================================================
# I6b. test_refined_name_inherits_unit_and_cluster_edges
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_refined_name_inherits_unit_and_cluster_edges(_gc, _clean):
    """persist_refined_name must propagate HAS_UNIT and IN_CLUSTER edges.

    Without this, claim-expand (which scopes by cluster+unit) cannot
    surface chain>0 names — they are statistically excluded from
    review-pool batches even though name_stage='drafted'.

    Regression for the chain=1 review-starvation bug observed in the
    iter2 smoke (22 chain=1 / 0 reviewed despite 1126 review attempts).
    """
    src_id = _uid("src_inherit")
    sn_v1 = _uid("inherit_v1")
    sn_v2 = _uid("inherit_v2")

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)

    # Attach Unit + Cluster on predecessor (pipeline normally does this
    # via _write_standard_name_edges for chain=0 names).
    _gc.query(
        """
        MATCH (sn:StandardName {id: $sn})
        MERGE (u:Unit {id: 'eV'})
        MERGE (c:IMASSemanticCluster {id: 'test_cluster_inherit'})
        MERGE (sn)-[:HAS_UNIT]->(u)
        MERGE (sn)-[:IN_CLUSTER]->(c)
        """,
        sn=sn_v1,
    )
    _link_source_sn(_gc, src_id, sn_v1)

    tok = f"tok-inh-{uuid.uuid4().hex[:8]}"
    _set_claim(_gc, sn_v1, tok)
    persist_reviewed_name(
        sn_id=sn_v1,
        claim_token=tok,
        score=0.5,
        model="m",
        min_score=0.75,
        rotation_cap=3,
    )
    persist_refined_name(
        old_name=sn_v1,
        new_name=sn_v2,
        description="v2 quantity",
        kind="scalar",
        unit="eV",
        old_chain_length=0,
        model="m",
    )

    rows = list(
        _gc.query(
            """
        MATCH (sn:StandardName {id: $sn})
        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (sn)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        RETURN u.id AS unit, c.id AS cluster
        """,
            sn=sn_v2,
        )
    )
    assert rows, "v2 missing"
    assert rows[0]["unit"] == "eV", (
        f"v2 must inherit HAS_UNIT from predecessor, got {rows[0]['unit']!r}"
    )
    assert rows[0]["cluster"] == "test_cluster_inherit", (
        f"v2 must inherit IN_CLUSTER from predecessor, got {rows[0]['cluster']!r}"
    )


# ===========================================================================
# I7. test_chain_history_walks_full_chain
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_chain_history_walks_full_chain(_gc, _clean):
    """name_chain_history(v3) returns predecessors v2 and v1 with their scores.

    Builds a 3-deep chain by calling persist_refined_name twice, sets
    reviewer scores on the predecessors, and asserts the history walker
    returns 2 entries in REFINED_FROM order (oldest first) with the
    correct scores.
    """
    sn_v1 = _uid("hist_v1")
    sn_v2 = _uid("hist_v2")
    sn_v3 = _uid("hist_v3")

    # Build chain: v3 → v2 → v1
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    persist_refined_name(
        old_name=sn_v1,
        new_name=sn_v2,
        description="v2 description",
        kind="scalar",
        unit="eV",
        old_chain_length=0,
        model="m",
    )
    persist_refined_name(
        old_name=sn_v2,
        new_name=sn_v3,
        description="v3 description",
        kind="scalar",
        unit="eV",
        old_chain_length=1,
        model="m",
    )

    # Stamp reviewer scores onto v1 and v2 (predecessors)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.reviewer_score_name = $score
        """,
        id=sn_v1,
        score=0.42,
    )
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.reviewer_score_name = $score
        """,
        id=sn_v2,
        score=0.58,
    )

    # Walk the chain from v3
    history = name_chain_history(sn_v3)

    assert len(history) == 2, (
        f"Expected 2 history entries (v1, v2), got {len(history)}: {history}"
    )

    # Oldest first (v1 at index 0, v2 at index 1)
    assert history[0]["name"] == sn_v1
    assert history[1]["name"] == sn_v2

    # Scores are present
    assert history[0]["reviewer_score"] == pytest.approx(0.42)
    assert history[1]["reviewer_score"] == pytest.approx(0.58)


# ===========================================================================
# I8. test_orphan_sweep_recovers_stuck_refining
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_orphan_sweep_recovers_stuck_refining(_gc, _clean):
    """A node stuck in name_stage='refining' with a stale claim is reverted.

    Simulates a worker that crashed mid-refine by setting claimed_at to
    400 seconds ago.  The orphan sweep should revert the stage to 'reviewed'
    and clear the claim fields so the node is eligible for the next refine
    cycle.
    """
    sn_id = _uid("stuck_refining")
    _gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage    = 'refining',
            sn.chain_length  = 0,
            sn.docs_stage    = 'pending',
            sn.claim_token   = $token,
            sn.claimed_at    = datetime() - duration({seconds: 400})
        """,
        id=sn_id,
        token="tok-stuck",
    )

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] >= 1, (
        f"Sweep should have reverted at least one stuck node; counts={counts}"
    )

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ===========================================================================
# I9. test_token_clobber_prevented_under_race
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_token_clobber_prevented_under_race(_gc, _clean):
    """Late release with an obsolete token must not clobber a newer token.

    Sequence:
      1. Worker A holds token T1 on a drafted SN.
      2. Orphan sweep clears T1 (simulated by direct graph write).
      3. Worker B re-claims the SN with token T2.
      4. Worker A's release_review_names_failed_claims(T1) → no-op.
      5. SN still carries T2; claim state is intact.
    """
    sn_id = _uid("race_token")
    t1 = f"tok-T1-{uuid.uuid4().hex[:8]}"
    t2 = f"tok-T2-{uuid.uuid4().hex[:8]}"

    # Step 1: SN claimed by Worker A with T1
    _create_sn(_gc, sn_id, name_stage="drafted", chain_length=0)
    _set_claim(_gc, sn_id, t1)

    # Step 2: Orphan sweep clears T1 (simulate)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.claim_token = null, sn.claimed_at = null
        """,
        id=sn_id,
    )

    # Step 3: Worker B claims SN with T2
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.claim_token = $token, sn.claimed_at = datetime()
        """,
        id=sn_id,
        token=t2,
    )

    # Step 4: Worker A tries to release its stale token T1 — must be a no-op
    released = release_review_names_failed_claims(
        sn_ids=[sn_id],
        claim_token=t1,
        from_stage="drafted",
        to_stage="drafted",
    )

    assert released == 0, f"Stale-token release should be a no-op; released={released}"

    # Step 5: SN still has T2, claim intact
    row = _fetch_sn(_gc, sn_id)
    assert row["claim_token"] == t2, (
        f"SN should still have token T2={t2!r}, got {row['claim_token']!r}"
    )
    assert row["claimed_at"] is not None, "claimed_at should still be set"


# ===========================================================================
# I10. test_concurrent_review_does_not_double_advance
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_concurrent_review_does_not_double_advance(_gc, _clean):
    """A second persist_reviewed_name with an obsolete token is a no-op.

    Simulates two workers that both believe they hold a claim on the same
    drafted SN.  The first worker's persist succeeds (token matches) and
    advances name_stage to 'accepted'.  The second worker's persist with a
    different token finds no matching token → returns '' and leaves the
    accepted state intact.

    This verifies the token-mismatch guard in persist_reviewed_name, which
    prevents duplicate stage advances under concurrent access.
    """
    sn_id = _uid("concurrent_review")
    t1 = f"tok-W1-{uuid.uuid4().hex[:8]}"
    t2 = f"tok-W2-{uuid.uuid4().hex[:8]}"

    _create_sn(_gc, sn_id, name_stage="drafted", chain_length=0)
    _set_claim(_gc, sn_id, t1)

    # Worker A persists first — succeeds
    r1 = persist_reviewed_name(
        sn_id=sn_id,
        claim_token=t1,
        score=0.85,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r1 == "accepted"

    # Worker B tries to persist with a different (now-stale) token
    r2 = persist_reviewed_name(
        sn_id=sn_id,
        claim_token=t2,
        score=0.50,
        model="test/model",
        min_score=0.75,
        rotation_cap=3,
    )
    assert r2 == "", f"Stale-token persist must be a no-op; got {r2!r}"

    # Graph state reflects only the FIRST review
    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "accepted", (
        f"name_stage should be 'accepted' (first worker's result); "
        f"got {row['name_stage']!r}"
    )
    assert row["reviewer_score_name"] == pytest.approx(0.85), (
        f"Score should be 0.85 (first worker's score); got {row['reviewer_score_name']}"
    )
