"""End-to-end exhaustion path tests for the standard-name pipeline.

These tests verify behaviour when name or docs rotation reaches the rotation
cap without acceptance — and the cross-stage gates that prevent further work
on exhausted nodes.

Specifically covered:
  E1. Name exhausted → docs pipeline never starts (gate on name_stage='accepted')
  E2. Docs exhausted independently while name remains accepted
  E3. Acceptance at rotation cap always overrides exhaustion logic
  E4. Exhausted nodes are immune to orphan sweep (terminal state)
  E5. Mid-flight exhaustion with orphan-sweep race interleaving

Run only when Neo4j is reachable:
    uv run pytest tests/standard_names/test_e2e_exhaustion.py -m "graph and integration" -v
"""

from __future__ import annotations

import uuid

import pytest

from imas_codex.standard_names.graph_ops import (
    claim_generate_docs_seed_and_expand,
    claim_refine_name_seed_and_expand,
    persist_refined_docs,
    persist_refined_name,
    persist_reviewed_docs,
    persist_reviewed_name,
)
from imas_codex.standard_names.orphan_sweep import _orphan_sweep_tick

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ID_PREFIX = "ex_test__"
_CAP = 3  # DEFAULT_REFINE_ROTATIONS


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
    """Delete all exhaustion-test nodes before and after each test."""

    def _wipe() -> None:
        _gc.query(
            "MATCH (n:DocsRevision) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_TEST_ID_PREFIX,
        )
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
# Graph helpers
# ---------------------------------------------------------------------------


def _uid(tag: str) -> str:
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
) -> None:
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage     = $name_stage,
            sn.chain_length   = $chain_length,
            sn.docs_stage     = 'pending',
            sn.docs_chain_length = 0,
            sn.description    = 'Test quantity',
            sn.documentation  = '',
            sn.kind           = 'scalar',
            sn.unit           = 'eV',
            sn.physics_domain = ['core_profiles']
        """,
        id=sn_id,
        name_stage=name_stage,
        chain_length=chain_length,
    )


def _create_sn_accepted(gc, sn_id: str) -> None:
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
        RETURN sn.name_stage         AS name_stage,
               sn.docs_stage         AS docs_stage,
               sn.chain_length       AS chain_length,
               sn.docs_chain_length  AS docs_chain_length,
               sn.claim_token        AS claim_token,
               sn.claimed_at         AS claimed_at
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found in graph"
    return rows[0]


def _count_sn_nodes(gc, prefix: str) -> int:
    rows = gc.query(
        "MATCH (sn:StandardName) WHERE sn.id STARTS WITH $p RETURN count(sn) AS n",
        p=prefix,
    )
    return rows[0]["n"] if rows else 0


def _count_docs_revisions(gc, sn_id: str) -> int:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:DOCS_REVISION_OF]->(rev:DocsRevision)
        RETURN count(rev) AS n
        """,
        id=sn_id,
    )
    return rows[0]["n"] if rows else 0


def _fetch_produced_name_target(gc, source_id: str) -> str | None:
    rows = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $id})-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN sn.id AS sn_id
        """,
        id=source_id,
    )
    return rows[0]["sn_id"] if rows else None


def _has_refined_from_chain(gc, sn_id: str, hops: int) -> bool:
    """True if sn_id has a REFINED_FROM chain of exactly `hops` hops back."""
    rows = gc.query(
        """
        MATCH p = (sn:StandardName {id: $id})-[:REFINED_FROM*1..5]->(ancestor)
        RETURN length(p) AS depth
        ORDER BY depth DESC
        LIMIT 1
        """,
        id=sn_id,
    )
    if not rows:
        return hops == 0
    return rows[0]["depth"] == hops


def _reject_name(gc, sn_id: str, *, rotation_cap: int = _CAP) -> str:
    """Set a claim, persist a reject review, return new stage."""
    tok = f"tok-rj-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, tok)
    return persist_reviewed_name(
        sn_id=sn_id,
        claim_token=tok,
        score=0.5,
        verdict="reject",
        model="test/model",
        min_score=0.75,
        rotation_cap=rotation_cap,
    )


def _accept_name(gc, sn_id: str, *, rotation_cap: int = _CAP) -> str:
    tok = f"tok-ac-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, tok)
    return persist_reviewed_name(
        sn_id=sn_id,
        claim_token=tok,
        score=0.85,
        verdict="accept",
        model="test/model",
        min_score=0.75,
        rotation_cap=rotation_cap,
    )


def _refine_name(gc, old_sn: str, new_sn: str, old_chain: int) -> None:
    persist_refined_name(
        old_name=old_sn,
        new_name=new_sn,
        description="Refined test quantity",
        kind="scalar",
        unit="eV",
        old_chain_length=old_chain,
        model="test/model",
    )


def _generate_docs(gc, sn_id: str) -> None:
    from imas_codex.standard_names.graph_ops import persist_generated_docs

    tok = f"tok-gd-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, tok)
    stage = persist_generated_docs(
        sn_id=sn_id,
        claim_token=tok,
        description="Generated description",
        documentation="## Test\n\nGenerated documentation.",
        model="test/model",
    )
    assert stage == "drafted"


def _reject_docs(gc, sn_id: str, *, rotation_cap: int = _CAP) -> str:
    tok = f"tok-rd-{uuid.uuid4().hex[:8]}"
    _set_claim(gc, sn_id, tok)
    return persist_reviewed_docs(
        sn_id=sn_id,
        claim_token=tok,
        score=0.5,
        verdict="reject",
        comments="Needs improvement",
        model="test/model",
        min_score=0.75,
        rotation_cap=rotation_cap,
    )


def _refine_docs(gc, sn_id: str, iteration: int = 0) -> dict:
    tok = f"tok-rfnd-{uuid.uuid4().hex[:8]}"
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
    assert rows
    snap = rows[0]
    gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.docs_stage  = 'refining',
            sn.claim_token = $token,
            sn.claimed_at  = datetime()
        """,
        id=sn_id,
        token=tok,
    )
    return persist_refined_docs(
        sn_id=sn_id,
        claim_token=tok,
        description=f"Refined description iteration {iteration}",
        documentation=f"## Refined docs\n\nIteration {iteration}.",
        model="test/model",
        current_description=snap["description"] or "",
        current_documentation=snap["documentation"] or "",
        current_model="test/model",
        current_generated_at=None,
        reviewer_score_to_snapshot=snap.get("reviewer_score_docs"),
        reviewer_comments_to_snapshot=snap.get("reviewer_comments_docs")
        or "Needs improvement",
        reviewer_comments_per_dim_to_snapshot=None,
        reviewer_verdict_to_snapshot=snap.get("reviewer_verdict_docs") or "reject",
    )


# ===========================================================================
# E1. test_e2e_name_exhausted_blocks_docs_pipeline
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_name_exhausted_blocks_docs_pipeline(_gc, _clean):
    """Name exhausted → docs pipeline never starts (gate on name_stage='accepted').

    Builds a 3-node chain where every review rejects, exhausting the rotation
    cap.  Then verifies that claim_generate_docs and claim_refine_name both
    return 0 items for this exhausted SN.
    """
    # Use a unique test-local prefix so claim queries only find these nodes
    local_prefix = _uid("exh_name")
    src_id = f"{local_prefix}_src"
    sn_v1 = f"{local_prefix}_v1"
    sn_v2 = f"{local_prefix}_v2"
    sn_v3 = f"{local_prefix}_v3"

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)

    # Cycle 1: review reject → refine (v1 → v2)
    r1 = _reject_name(_gc, sn_v1)
    assert r1 == "reviewed", f"Expected 'reviewed', got {r1!r}"
    _refine_name(_gc, sn_v1, sn_v2, old_chain=0)

    # Cycle 2: review reject → refine (v2 → v3)
    r2 = _reject_name(_gc, sn_v2)
    assert r2 == "reviewed"
    _refine_name(_gc, sn_v2, sn_v3, old_chain=1)

    # Final review on v3 (chain_length=2, cap=3) → exhausted
    r3 = _reject_name(_gc, sn_v3)
    assert r3 == "exhausted", f"Expected 'exhausted' on cap, got {r3!r}"

    # --- Assertions on graph state ---
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["name_stage"] == "exhausted"
    assert row_v3["docs_stage"] == "pending", (
        "docs_stage must remain 'pending' — name never accepted"
    )
    assert row_v3["claim_token"] is None

    # No DocsRevision nodes for this chain
    for sn_id in (sn_v1, sn_v2, sn_v3):
        assert _count_docs_revisions(_gc, sn_id) == 0

    # generate_docs must NOT claim v3 (name_stage='exhausted', not 'accepted')
    claimed_docs = claim_generate_docs_seed_and_expand(batch_size=50)
    claimed_ids = {item["id"] for item in claimed_docs}
    assert sn_v3 not in claimed_ids, (
        "Exhausted SN must not appear in generate_docs claim"
    )

    # refine_name must NOT claim v3 (name_stage='exhausted', not 'reviewed')
    claimed_refine = claim_refine_name_seed_and_expand(
        min_score=0.75, rotation_cap=_CAP, batch_size=50
    )
    claimed_refine_ids = {item["id"] for item in claimed_refine}
    assert sn_v3 not in claimed_refine_ids, (
        "Exhausted SN must not appear in refine_name claim"
    )


# ===========================================================================
# E2. test_e2e_docs_exhausted_keeps_name_accepted
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_docs_exhausted_keeps_name_accepted(_gc, _clean):
    """Name accepted; docs exhaust independently. Name stage is preserved.

    Builds: generate_docs → 2× (review reject → refine) → review reject.
    The third rejection at docs_chain_length=2 exhausts the docs pipeline.
    name_stage remains 'accepted' throughout.
    """
    local_prefix = _uid("exh_docs")
    sn_id = f"{local_prefix}_sn"
    _create_sn_accepted(_gc, sn_id)

    # Step 1: generate docs
    _generate_docs(_gc, sn_id)

    # Cycle 1: review reject → refine (docs_chain_length → 1)
    rd1 = _reject_docs(_gc, sn_id)
    assert rd1 == "reviewed"
    result1 = _refine_docs(_gc, sn_id, iteration=0)
    assert result1["docs_chain_length"] == 1

    # Cycle 2: review reject → refine (docs_chain_length → 2)
    rd2 = _reject_docs(_gc, sn_id)
    assert rd2 == "reviewed"
    result2 = _refine_docs(_gc, sn_id, iteration=1)
    assert result2["docs_chain_length"] == 2

    # Cycle 3: final reject at docs_chain_length=2 → exhausted
    rd3 = _reject_docs(_gc, sn_id, rotation_cap=_CAP)
    assert rd3 == "exhausted", f"Expected 'exhausted', got {rd3!r}"

    # --- Assertions ---
    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "accepted", (
        "name_stage must remain 'accepted' after docs exhaustion"
    )
    assert row["docs_stage"] == "exhausted"
    assert row["docs_chain_length"] == 2
    assert row["claim_token"] is None

    # Both DocsRevision snapshots preserved
    assert _count_docs_revisions(_gc, sn_id) == 2, (
        f"Expected 2 DocsRevision nodes, got {_count_docs_revisions(_gc, sn_id)}"
    )

    # Exhausted SN must not be claimed by refine_docs (docs_stage != 'reviewed')
    from imas_codex.standard_names.graph_ops import claim_refine_docs_seed_and_expand

    claimed_refine = claim_refine_docs_seed_and_expand(batch_size=50)
    claimed_refine_ids = {item["id"] for item in claimed_refine}
    assert sn_id not in claimed_refine_ids, (
        "Exhausted SN must not appear in refine_docs claim"
    )

    # review_docs must NOT claim this SN (docs_stage='exhausted', not 'drafted')
    from imas_codex.standard_names.graph_ops import claim_review_docs_seed_and_expand

    claimed_review = claim_review_docs_seed_and_expand(batch_size=50)
    claimed_review_ids = {item["id"] for item in claimed_review}
    assert sn_id not in claimed_review_ids, (
        "Exhausted SN must not appear in review_docs claim"
    )


# ===========================================================================
# E3. test_e2e_acceptance_at_cap_overrides_exhaustion
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_acceptance_at_cap_overrides_exhaustion(_gc, _clean):
    """Accept verdict always wins even at the rotation cap.

    SN_v3 at chain_length=2 with rotation_cap=3 would be 'exhausted' on
    reject, but an 'accept' verdict with score >= min → 'accepted' instead.
    """
    local_prefix = _uid("cap_acc")
    src_id = f"{local_prefix}_src"
    sn_v1 = f"{local_prefix}_v1"
    sn_v2 = f"{local_prefix}_v2"
    sn_v3 = f"{local_prefix}_v3"

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)

    # Cycle 1: review reject → refine (v1 → v2)
    r1 = _reject_name(_gc, sn_v1)
    assert r1 == "reviewed"
    _refine_name(_gc, sn_v1, sn_v2, old_chain=0)

    # Cycle 2: review reject → refine (v2 → v3, chain_length=2)
    r2 = _reject_name(_gc, sn_v2)
    assert r2 == "reviewed"
    _refine_name(_gc, sn_v2, sn_v3, old_chain=1)

    # Final review: ACCEPT at chain_length=2 — acceptance overrides cap logic
    r3 = _accept_name(_gc, sn_v3)
    assert r3 == "accepted", f"Accept must win over exhaustion cap; got {r3!r}"

    # --- Assertions ---
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["name_stage"] == "accepted"
    assert row_v3["chain_length"] == 2
    assert row_v3["claim_token"] is None

    # Source PRODUCED_NAME → SN_v3
    assert _fetch_produced_name_target(_gc, src_id) == sn_v3

    # generate_docs CAN claim SN_v3 (name_stage='accepted', docs_stage='pending')
    claimed = claim_generate_docs_seed_and_expand(batch_size=50)
    claimed_ids = {item["id"] for item in claimed}
    assert sn_v3 in claimed_ids, (
        "Accepted SN at rotation cap should be eligible for generate_docs"
    )


# ===========================================================================
# E4. test_e2e_exhausted_excluded_from_orphan_recovery
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_exhausted_excluded_from_orphan_recovery(_gc, _clean):
    """Orphan sweep must NOT touch exhausted nodes — they are terminal.

    Creates a StandardName at name_stage='exhausted' with a stale claimed_at
    (simulating a crash mid-sweep).  Runs _orphan_sweep_tick once.  The node
    must remain exhausted with its claim cleared by the stale_token sweep
    (not reverted to 'reviewed').
    """
    sn_id = _uid("exh_sweep")

    # Create an exhausted node with a stale claim
    _gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage    = 'exhausted',
            sn.chain_length  = 2,
            sn.docs_stage    = 'pending',
            sn.docs_chain_length = 0,
            sn.kind          = 'scalar',
            sn.unit          = 'eV',
            sn.claim_token   = 'stale-token',
            sn.claimed_at    = datetime() - duration({seconds: 400})
        """,
        id=sn_id,
    )

    # Run the orphan sweep
    counts = _orphan_sweep_tick(timeout_s=300)

    # Sweep ran without error (just checking it didn't raise)
    assert isinstance(counts, dict)

    # Node must still be exhausted — not reverted to 'reviewed'
    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "exhausted", (
        f"Orphan sweep must not revert exhausted node; got {row['name_stage']!r}"
    )


# ===========================================================================
# E5. test_e2e_partial_pipeline_then_exhaustion
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_partial_pipeline_then_exhaustion(_gc, _clean):
    """Mid-flight exhaustion with orphan-sweep race interleaving.

    Sequence:
      1. generate_name → SN_v1 (chain=0, drafted)
      2. review_name reject → reviewed
      3. refine_name → SN_v2 (chain=1, drafted)
      4. Simulate stale claim on SN_v2, run orphan sweep → claim cleared,
         stage stays 'drafted' (sweep only touches 'refining', not 'drafted')
      5. Re-claim + review_name reject → reviewed
      6. refine_name → SN_v3 (chain=2, drafted)
      7. review_name reject → exhausted

    Final assertions:
      - 3 SN nodes; SN_v3 exhausted, no claim
      - SN_v3 has 2 REFINED_FROM hops back to SN_v1
    """
    local_prefix = _uid("mid_exh")
    src_id = f"{local_prefix}_src"
    sn_v1 = f"{local_prefix}_v1"
    sn_v2 = f"{local_prefix}_v2"
    sn_v3 = f"{local_prefix}_v3"

    _create_source(_gc, src_id)
    _create_sn(_gc, sn_v1, name_stage="drafted", chain_length=0)
    _link_source_sn(_gc, src_id, sn_v1)

    # Step 2: review reject → SN_v1 reviewed
    r1 = _reject_name(_gc, sn_v1)
    assert r1 == "reviewed"

    # Step 3: refine → SN_v2 (chain=1)
    _refine_name(_gc, sn_v1, sn_v2, old_chain=0)
    row_v2 = _fetch_sn(_gc, sn_v2)
    assert row_v2["name_stage"] == "drafted"
    assert row_v2["chain_length"] == 1

    # Step 4: Simulate a stale review claim on SN_v2 (worker crashed mid-review)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        SET sn.claim_token = 'stale-review-tok',
            sn.claimed_at  = datetime() - duration({seconds: 400})
        """,
        id=sn_v2,
    )
    # Run orphan sweep — should clear stale token but NOT revert 'drafted' stage
    counts = _orphan_sweep_tick(timeout_s=300)
    assert isinstance(counts, dict)

    # Stage must still be 'drafted' (sweep only reverts 'refining', not 'drafted')
    row_v2_after = _fetch_sn(_gc, sn_v2)
    assert row_v2_after["name_stage"] == "drafted", (
        f"Orphan sweep must not touch 'drafted' stage; got {row_v2_after['name_stage']!r}"
    )
    # Stale claim should have been cleared by stale_token sweep
    assert row_v2_after["claim_token"] is None, (
        "Orphan sweep should clear stale token from non-refining node"
    )

    # Step 5: Re-claim + review reject
    r2 = _reject_name(_gc, sn_v2)
    assert r2 == "reviewed"

    # Step 6: refine → SN_v3 (chain=2)
    _refine_name(_gc, sn_v2, sn_v3, old_chain=1)
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["chain_length"] == 2

    # Step 7: final reject → exhausted
    r3 = _reject_name(_gc, sn_v3)
    assert r3 == "exhausted", f"Expected 'exhausted', got {r3!r}"

    # --- Final assertions ---
    row_v3_final = _fetch_sn(_gc, sn_v3)
    assert row_v3_final["name_stage"] == "exhausted"
    assert row_v3_final["claim_token"] is None

    # SN_v3 should have a 2-hop REFINED_FROM chain back to SN_v1
    assert _has_refined_from_chain(_gc, sn_v3, hops=2), (
        "SN_v3 must have a 2-hop REFINED_FROM chain back to SN_v1"
    )

    # All 3 SN nodes exist
    node_count = _count_sn_nodes(_gc, local_prefix)
    assert node_count == 3, f"Expected 3 SN nodes in chain, got {node_count}"
