"""End-to-end rotation tests for the full 6-pool SN pipeline.

These tests exercise the complete state machine across both the name pipeline
(generate → review → refine → accept) and the docs pipeline (generate_docs →
review_docs → refine_docs → accept), including the cross-pipeline gating
handoff (docs starts only after name_stage='accepted').

Workers are called directly (bypassing the main event loop), LLM calls are
intercepted by the MockLLM fixture from conftest.py.

Tests:
  E1. Full rotation to acceptance  — 1 name refine + 1 docs refine
  E2. Immediate acceptance          — 0 rotations on either side
  E3. Double rotation               — 2 name refines + 2 docs refines

Run only when Neo4j is reachable:
    uv run pytest tests/standard_names/test_e2e_rotation.py -m "graph and integration" -v
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.chain_history import (
    docs_chain_history,
    name_chain_history,
)
from imas_codex.standard_names.defaults import DEFAULT_ESCALATION_MODEL
from imas_codex.standard_names.models import (
    GeneratedDocs,
    RefinedDocs,
    RefinedName,
    StandardNameQualityReviewDocs,
    StandardNameQualityReviewNameOnly,
    StandardNameQualityScoreDocs,
    StandardNameQualityScoreNameOnly,
)
from imas_codex.standard_names.workers import (
    process_generate_docs_batch,
    process_refine_docs_batch,
    process_refine_name_batch,
    process_review_docs_batch,
    process_review_name_batch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ID_PREFIX = "e2e_rot__"

# Score helpers — ScoreNameOnly: total/80; ScoreDocs: total/80
# DEFAULT_MIN_SCORE = 0.75 → threshold = 60/80

_SCORE_REJECT_NAME = StandardNameQualityScoreNameOnly(
    grammar=11, semantic=11, convention=11, completeness=11
)  # 44/80 = 0.55

_SCORE_REJECT_NAME_MED = StandardNameQualityScoreNameOnly(
    grammar=12, semantic=12, convention=12, completeness=12
)  # 48/80 = 0.60

_SCORE_ACCEPT_NAME = StandardNameQualityScoreNameOnly(
    grammar=17, semantic=17, convention=17, completeness=17
)  # 68/80 = 0.85

_SCORE_ACCEPT_NAME_HIGH = StandardNameQualityScoreNameOnly(
    grammar=18, semantic=18, convention=18, completeness=18
)  # 72/80 = 0.90

_SCORE_ACCEPT_NAME_MED = StandardNameQualityScoreNameOnly(
    grammar=16, semantic=17, convention=16, completeness=17
)  # 66/80 = 0.825

_SCORE_REJECT_DOCS_LOW = StandardNameQualityScoreDocs(
    description_quality=10,
    documentation_quality=10,
    completeness=10,
    physics_accuracy=10,
)  # 40/80 = 0.50

_SCORE_REJECT_DOCS = StandardNameQualityScoreDocs(
    description_quality=11,
    documentation_quality=11,
    completeness=11,
    physics_accuracy=11,
)  # 44/80 = 0.55

_SCORE_ACCEPT_DOCS = StandardNameQualityScoreDocs(
    description_quality=17,
    documentation_quality=17,
    completeness=17,
    physics_accuracy=17,
)  # 68/80 = 0.85


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
    """Delete all e2e-test nodes before and after each test."""

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
    """Return a unique prefixed id for the current test run."""
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
        SET sn.name_stage      = $name_stage,
            sn.chain_length    = $chain_length,
            sn.docs_stage      = 'pending',
            sn.docs_chain_length = 0,
            sn.description     = 'Test quantity',
            sn.documentation   = '',
            sn.kind            = 'scalar',
            sn.unit            = 'eV',
            sn.physics_domain  = ['core_profiles'],
            sn.tags            = ['test']
        """,
        id=sn_id,
        name_stage=name_stage,
        chain_length=chain_length,
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


def _set_claim_and_stage(
    gc, sn_id: str, token: str, stage_field: str, stage_value: str
) -> None:
    gc.query(
        f"""
        MATCH (sn:StandardName {{id: $id}})
        SET sn.claim_token = $token,
            sn.claimed_at  = datetime(),
            sn.{stage_field} = $stage_value
        """,
        id=sn_id,
        token=token,
        stage_value=stage_value,
    )


def _fetch_sn(gc, sn_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.name_stage          AS name_stage,
               sn.docs_stage          AS docs_stage,
               sn.chain_length        AS chain_length,
               sn.docs_chain_length   AS docs_chain_length,
               sn.description         AS description,
               sn.documentation       AS documentation,
               sn.reviewer_score_name AS reviewer_score_name,
               sn.reviewer_score_docs AS reviewer_score_docs,
               sn.reviewer_comments_docs AS reviewer_comments_docs
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found in graph"
    return rows[0]


def _fetch_produced_name_target(gc, source_id: str) -> str | None:
    rows = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $id})-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN sn.id AS sn_id
        """,
        id=source_id,
    )
    return rows[0]["sn_id"] if rows else None


def _count_refined_from_edges(gc, tip_id: str) -> int:
    """Count REFINED_FROM edges reachable back from the chain tip."""
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:REFINED_FROM*]->(pred:StandardName)
        RETURN count(pred) AS n
        """,
        id=tip_id,
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


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


# ---------------------------------------------------------------------------
# Item dict builders — match shapes expected by each worker
# ---------------------------------------------------------------------------


def _make_review_name_item(
    sn_id: str, *, claim_token: str, chain_length: int = 0
) -> dict:
    return {
        "id": sn_id,
        "description": "Test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": None,
        "claim_token": claim_token,
        "name": sn_id,
        "tags": ["test"],
        "chain_length": chain_length,
        "name_stage": "drafted",
    }


def _make_refine_name_item(
    sn_id: str,
    *,
    claim_token: str,
    chain_length: int,
    reviewer_score: float = 0.55,
) -> dict:
    return {
        "id": sn_id,
        "description": "Test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": None,
        "reviewer_score_name": reviewer_score,
        "reviewer_comments_per_dim_name": None,
        "chain_length": chain_length,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/test_quantity"],
        "tags": ["test"],
        "claim_token": claim_token,
        "chain_history": [],
    }


def _make_generate_docs_item(sn_id: str, *, claim_token: str) -> dict:
    return {
        "id": sn_id,
        "description": "Test quantity",
        "documentation": "",
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": None,
        "claim_token": claim_token,
        "tags": ["test"],
        "reviewer_score_name": 0.85,
        "reviewer_comments_name": "Good name",
        "chain_length": 0,
        "docs_stage": "pending",
        "name_stage": "accepted",
        "chain_history": [],
    }


def _make_review_docs_item(
    sn_id: str,
    *,
    claim_token: str,
    description: str,
    documentation: str,
    docs_chain_length: int = 0,
) -> dict:
    return {
        "id": sn_id,
        "description": description,
        "documentation": documentation,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": None,
        "claim_token": claim_token,
        "name": sn_id,
        "tags": ["test"],
        "docs_chain_length": docs_chain_length,
        "docs_stage": "drafted",
    }


def _make_refine_docs_item(
    sn_id: str,
    *,
    claim_token: str,
    description: str,
    documentation: str,
    docs_chain_length: int,
    reviewer_score: float = 0.55,
    reviewer_comments: str = "Needs improvement",
) -> dict:
    return {
        "id": sn_id,
        "description": description,
        "documentation": documentation,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": None,
        "claim_token": claim_token,
        "tags": ["test"],
        "docs_stage": "refining",
        "docs_chain_length": docs_chain_length,
        "docs_model": "test/model",
        "docs_generated_at": None,
        "reviewer_score_docs": reviewer_score,
        "reviewer_comments_docs": reviewer_comments,
        "reviewer_comments_per_dim_docs": None,
        "docs_chain_history": [],
    }


# ---------------------------------------------------------------------------
# Worker drivers — synchronous wrappers around asyncio.run
# ---------------------------------------------------------------------------


def _do_generate_name(gc, src_id: str, sn_id: str) -> None:
    """Simulate generate_name output: source → SN with name_stage='drafted'."""
    _create_source(gc, src_id)
    _create_sn(gc, sn_id, name_stage="drafted", chain_length=0)
    _link_source_sn(gc, src_id, sn_id)


def _do_review_name(
    gc,
    mock_llm,
    sn_id: str,
    *,
    scores: StandardNameQualityScoreNameOnly,
    chain_length: int = 0,
) -> str:
    """Claim + run review_name worker. Returns new name_stage."""
    mock_llm.add_response(
        "review_name",
        response=StandardNameQualityReviewNameOnly(
            source_id=sn_id,
            standard_name=sn_id,
            scores=scores,
            reasoning="E2E test reviewer reasoning",
        ),
    )
    token = str(uuid.uuid4())
    _set_claim(gc, sn_id, token)
    item = _make_review_name_item(sn_id, claim_token=token, chain_length=chain_length)
    stop_event = asyncio.Event()
    n = asyncio.run(
        process_review_name_batch([item], _mock_budget_manager(), stop_event)
    )
    assert n == 1, f"review_name processed {n}/1 for {sn_id!r}"
    return _fetch_sn(gc, sn_id)["name_stage"]


def _do_refine_name(
    gc,
    mock_llm,
    old_id: str,
    new_id: str,
    *,
    chain_length: int,
) -> None:
    """Set refining state + run refine_name worker (old_id → new_id)."""
    mock_llm.add_response(
        "unknown",
        response=RefinedName(
            name=new_id,
            description=f"E2E refined quantity chain-{chain_length + 1}",
            kind="scalar",
        ),
    )
    token = str(uuid.uuid4())
    _set_claim_and_stage(gc, old_id, token, "name_stage", "refining")
    item = _make_refine_name_item(old_id, claim_token=token, chain_length=chain_length)
    stop_event = asyncio.Event()
    with patch(
        "imas_codex.llm.prompt_loader.render_prompt",
        return_value=f"Refine this standard name (chain={chain_length}).",
    ):
        n = asyncio.run(
            process_refine_name_batch([item], _mock_budget_manager(), stop_event)
        )
    assert n == 1, f"refine_name processed {n}/1 for {old_id!r}"


def _do_generate_docs(gc, mock_llm, sn_id: str) -> None:
    """Claim + run generate_docs worker."""
    mock_llm.add_response(
        "unknown",
        response=GeneratedDocs(
            description=f"E2E generated description for {sn_id}",
            documentation=f"## {sn_id}\n\nE2E generated documentation body for testing purposes.",
        ),
    )
    token = str(uuid.uuid4())
    _set_claim(gc, sn_id, token)
    item = _make_generate_docs_item(sn_id, claim_token=token)
    stop_event = asyncio.Event()
    n = asyncio.run(
        process_generate_docs_batch([item], _mock_budget_manager(), stop_event)
    )
    assert n == 1, f"generate_docs processed {n}/1 for {sn_id!r}"


def _do_review_docs(
    gc,
    mock_llm,
    sn_id: str,
    *,
    scores: StandardNameQualityScoreDocs,
    docs_chain_length: int = 0,
) -> str:
    """Claim + run review_docs worker. Returns new docs_stage."""
    mock_llm.add_response(
        "review_docs",
        response=StandardNameQualityReviewDocs(
            source_id=sn_id,
            standard_name=sn_id,
            scores=scores,
            reasoning="E2E docs reviewer reasoning",
        ),
    )
    token = str(uuid.uuid4())
    _set_claim(gc, sn_id, token)
    # Fetch current description/documentation (written by generate_docs/refine_docs)
    row = _fetch_sn(gc, sn_id)
    item = _make_review_docs_item(
        sn_id,
        claim_token=token,
        description=row["description"] or "",
        documentation=row["documentation"] or "",
        docs_chain_length=docs_chain_length,
    )
    stop_event = asyncio.Event()
    n = asyncio.run(
        process_review_docs_batch([item], _mock_budget_manager(), stop_event)
    )
    assert n == 1, f"review_docs processed {n}/1 for {sn_id!r}"
    return _fetch_sn(gc, sn_id)["docs_stage"]


def _do_refine_docs(
    gc,
    mock_llm,
    sn_id: str,
    *,
    docs_chain_length: int,
    iteration: int = 0,
) -> None:
    """Set docs refining state + run refine_docs worker."""
    mock_llm.add_response(
        "unknown",
        response=RefinedDocs(
            description=f"E2E refined docs description iter-{iteration}",
            documentation=f"## Refined\n\nE2E refined documentation iteration {iteration}.",
        ),
    )
    token = str(uuid.uuid4())
    # Fetch current docs state before snapshotting
    row = _fetch_sn(gc, sn_id)
    current_desc = row["description"] or ""
    current_doc = row["documentation"] or ""
    reviewer_score = row.get("reviewer_score_docs")
    reviewer_comments = row.get("reviewer_comments_docs") or "Needs improvement"
    # Atomically transition to 'refining' + set claim
    _set_claim_and_stage(gc, sn_id, token, "docs_stage", "refining")
    item = _make_refine_docs_item(
        sn_id,
        claim_token=token,
        description=current_desc,
        documentation=current_doc,
        docs_chain_length=docs_chain_length,
        reviewer_score=reviewer_score if reviewer_score is not None else 0.55,
        reviewer_comments=reviewer_comments,
    )
    stop_event = asyncio.Event()
    n = asyncio.run(
        process_refine_docs_batch([item], _mock_budget_manager(), stop_event)
    )
    assert n == 1, f"refine_docs processed {n}/1 for {sn_id!r}"


# ===========================================================================
# E1. test_e2e_full_rotation_to_acceptance
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_full_rotation_to_acceptance(_gc, _clean, mock_llm):
    """Happy path with one name rotation and one docs rotation.

    Flow:
    1. generate_name: source → SN_v1 (name_stage='drafted', chain=0)
    2. review_name reject 0.55: SN_v1 → 'reviewed'
    3. refine_name: SN_v1 → SN_v2 (chain=1, REFINED_FROM SN_v1)
    4. review_name accept 0.85: SN_v2 → 'accepted'
    5. generate_docs: SN_v2 → docs_stage='drafted'
    6. review_docs reject 0.55: SN_v2 → docs_stage='reviewed'
    7. refine_docs: SN_v2 → DocsRevision_v0, docs_chain=1
    8. review_docs accept 0.85: SN_v2 → docs_stage='accepted'

    Assertions:
    - 2 SN nodes (SN_v1 superseded, SN_v2 accepted)
    - 1 REFINED_FROM edge
    - 1 DocsRevision node
    - Source PRODUCED_NAME → SN_v2
    - chain_length=1, docs_chain_length=1
    - Both pipelines at 'accepted' on SN_v2
    """
    src_id = _uid("src_rot1")
    sn_v1 = _uid("rot1_v1")
    sn_v2 = _uid("rot1_v2")

    # Step 1: generate_name
    _do_generate_name(_gc, src_id, sn_v1)
    row = _fetch_sn(_gc, sn_v1)
    assert row["name_stage"] == "drafted"
    assert row["chain_length"] == 0

    # Step 2: review_name reject → reviewed
    stage = _do_review_name(_gc, mock_llm, sn_v1, scores=_SCORE_REJECT_NAME)
    assert stage == "reviewed", f"Expected 'reviewed', got {stage!r}"

    # Step 3: refine_name SN_v1 → SN_v2
    _do_refine_name(_gc, mock_llm, sn_v1, sn_v2, chain_length=0)

    row_v1 = _fetch_sn(_gc, sn_v1)
    row_v2 = _fetch_sn(_gc, sn_v2)
    assert row_v1["name_stage"] == "superseded", (
        f"SN_v1 should be superseded, got {row_v1['name_stage']!r}"
    )
    assert row_v2["name_stage"] == "drafted", (
        f"SN_v2 should be drafted, got {row_v2['name_stage']!r}"
    )
    assert row_v2["chain_length"] == 1

    # PRODUCED_NAME migrated to SN_v2
    assert _fetch_produced_name_target(_gc, src_id) == sn_v2

    # Step 4: review_name accept → accepted (cross-pipeline gate unlocked)
    stage = _do_review_name(
        _gc,
        mock_llm,
        sn_v2,
        scores=_SCORE_ACCEPT_NAME,
        chain_length=1,
    )
    assert stage == "accepted", f"Expected 'accepted', got {stage!r}"

    # Step 5: generate_docs (gated on name_stage='accepted')
    _do_generate_docs(_gc, mock_llm, sn_v2)
    row_v2 = _fetch_sn(_gc, sn_v2)
    assert row_v2["docs_stage"] == "drafted"
    assert row_v2["description"] != "Test quantity"  # updated by generate_docs
    assert row_v2["documentation"] != ""

    # Step 6: review_docs reject → reviewed
    stage = _do_review_docs(_gc, mock_llm, sn_v2, scores=_SCORE_REJECT_DOCS)
    assert stage == "reviewed", f"Expected 'reviewed', got {stage!r}"

    # Step 7: refine_docs → DocsRevision_v0 created, docs_chain=1
    _do_refine_docs(_gc, mock_llm, sn_v2, docs_chain_length=0, iteration=0)
    row_v2 = _fetch_sn(_gc, sn_v2)
    assert row_v2["docs_stage"] == "drafted"
    assert row_v2["docs_chain_length"] == 1

    # Step 8: review_docs accept → accepted
    stage = _do_review_docs(
        _gc,
        mock_llm,
        sn_v2,
        scores=_SCORE_ACCEPT_DOCS,
        docs_chain_length=1,
    )
    assert stage == "accepted", f"Expected 'accepted', got {stage!r}"

    # ── Final assertions ─────────────────────────────────────────────────
    row_v1 = _fetch_sn(_gc, sn_v1)
    row_v2 = _fetch_sn(_gc, sn_v2)

    # 2 SN nodes: v1 superseded, v2 accepted
    assert row_v1["name_stage"] == "superseded"
    assert row_v2["name_stage"] == "accepted"
    assert row_v2["docs_stage"] == "accepted"

    # Chain lengths
    assert row_v2["chain_length"] == 1
    assert row_v2["docs_chain_length"] == 1

    # REFINED_FROM chain: SN_v2 → SN_v1
    assert _count_refined_from_edges(_gc, sn_v2) == 1, (
        "Expected 1 REFINED_FROM edge from SN_v2"
    )

    # 1 DocsRevision node linked from SN_v2
    assert _count_docs_revisions(_gc, sn_v2) == 1, (
        "Expected 1 DocsRevision node linked to SN_v2"
    )

    # Source PRODUCED_NAME → SN_v2
    assert _fetch_produced_name_target(_gc, src_id) == sn_v2

    # Reviewer fields populated for name review
    assert row_v2["reviewer_score_name"] == pytest.approx(0.85, abs=0.01)

    # Reviewer fields populated for docs review
    assert row_v2["reviewer_score_docs"] == pytest.approx(0.85, abs=0.01)

    # chain_history walker finds 1 predecessor (SN_v1)
    hist = name_chain_history(sn_v2)
    assert len(hist) == 1, f"Expected 1 chain history entry, got {len(hist)}"

    # docs_chain_history walker finds 1 revision
    docs_hist = docs_chain_history(sn_v2)
    assert len(docs_hist) == 1, f"Expected 1 docs history entry, got {len(docs_hist)}"


# ===========================================================================
# E2. test_e2e_immediate_acceptance_no_rotation
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_immediate_acceptance_no_rotation(_gc, _clean, mock_llm):
    """Both pipelines accepted on first review — zero rotations.

    Flow:
    1. generate_name: source → SN (drafted, chain=0)
    2. review_name accept 0.90: SN → 'accepted'
    3. generate_docs: SN → docs_stage='drafted'
    4. review_docs accept 0.85: SN → docs_stage='accepted'

    Assertions:
    - 1 SN node
    - 0 REFINED_FROM edges
    - 0 DocsRevision nodes
    - chain_length=0, docs_chain_length=0
    - Both stages='accepted'
    """
    src_id = _uid("src_imm")
    sn_id = _uid("imm_sn")

    # Step 1: generate_name
    _do_generate_name(_gc, src_id, sn_id)

    # Step 2: review_name accept immediately
    stage = _do_review_name(_gc, mock_llm, sn_id, scores=_SCORE_ACCEPT_NAME_HIGH)
    assert stage == "accepted", f"Expected 'accepted', got {stage!r}"

    # Step 3: generate_docs (cross-pipeline gate: name accepted)
    _do_generate_docs(_gc, mock_llm, sn_id)
    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "drafted"
    assert row["description"] != "Test quantity"
    assert row["documentation"] != ""

    # Step 4: review_docs accept immediately
    stage = _do_review_docs(_gc, mock_llm, sn_id, scores=_SCORE_ACCEPT_DOCS)
    assert stage == "accepted", f"Expected 'accepted', got {stage!r}"

    # ── Final assertions ─────────────────────────────────────────────────
    row = _fetch_sn(_gc, sn_id)

    assert row["name_stage"] == "accepted"
    assert row["docs_stage"] == "accepted"
    assert row["chain_length"] == 0
    assert row["docs_chain_length"] == 0

    # No rotations — no REFINED_FROM, no DocsRevision
    assert _count_refined_from_edges(_gc, sn_id) == 0, "Expected 0 REFINED_FROM edges"
    assert _count_docs_revisions(_gc, sn_id) == 0, "Expected 0 DocsRevision nodes"

    # Source still points to the single SN
    assert _fetch_produced_name_target(_gc, src_id) == sn_id

    # chain_history walkers both empty
    assert name_chain_history(sn_id) == []
    assert docs_chain_history(sn_id) == []

    # Reviewer fields populated for both axes
    assert row["reviewer_score_name"] == pytest.approx(0.90, abs=0.01)
    assert row["reviewer_score_docs"] == pytest.approx(0.85, abs=0.01)


# ===========================================================================
# E3. test_e2e_name_rotates_then_docs_rotates
# ===========================================================================


@pytest.mark.graph
@pytest.mark.integration
def test_e2e_name_rotates_then_docs_rotates(_gc, _clean, mock_llm):
    """Both pipelines require two rotations each.

    Name pipeline:
    1. generate_name → SN_v1 (chain=0)
    2. review_name reject 0.55 → reviewed
    3. refine_name → SN_v2 (chain=1)
    4. review_name reject 0.60 → reviewed
    5. refine_name → SN_v3 (chain=2)
    6. review_name accept 0.825 → accepted

    Docs pipeline (on SN_v3, gated on name_stage='accepted'):
    7. generate_docs → drafted
    8. review_docs reject 0.50 → reviewed
    9. refine_docs → DocsRev_v0, docs_chain=1
    10. review_docs reject 0.55 → reviewed
    11. refine_docs → DocsRev_v1, docs_chain=2
    12. review_docs accept 0.85 → accepted

    Assertions:
    - 3 SN nodes (v1, v2 superseded; v3 accepted)
    - 2 REFINED_FROM edges reachable from SN_v3
    - 2 DocsRevision nodes linked from SN_v3
    - chain_length=2, docs_chain_length=2 on SN_v3
    - Source PRODUCED_NAME → SN_v3
    - chain_history walker finds 2 prior SNs
    - docs_chain_history walker finds 2 prior DocsRevisions
    - refine_name models tracked via mock_llm.calls
    - refine_docs models tracked via mock_llm.calls
    """
    src_id = _uid("src_dbl")
    sn_v1 = _uid("dbl_v1")
    sn_v2 = _uid("dbl_v2")
    sn_v3 = _uid("dbl_v3")

    # ── Name pipeline ───────────────────────────────────────────────────

    # Step 1: generate_name
    _do_generate_name(_gc, src_id, sn_v1)

    # Step 2: review_name reject 0.55
    stage = _do_review_name(_gc, mock_llm, sn_v1, scores=_SCORE_REJECT_NAME)
    assert stage == "reviewed"

    # Step 3: refine_name v1 → v2 (chain_length of old=0)
    _do_refine_name(_gc, mock_llm, sn_v1, sn_v2, chain_length=0)
    assert _fetch_sn(_gc, sn_v1)["name_stage"] == "superseded"
    row_v2 = _fetch_sn(_gc, sn_v2)
    assert row_v2["name_stage"] == "drafted"
    assert row_v2["chain_length"] == 1

    # Step 4: review_name reject 0.60
    stage = _do_review_name(
        _gc,
        mock_llm,
        sn_v2,
        scores=_SCORE_REJECT_NAME_MED,
        chain_length=1,
    )
    assert stage == "reviewed"

    # Step 5: refine_name v2 → v3 (chain_length of old=1)
    _do_refine_name(_gc, mock_llm, sn_v2, sn_v3, chain_length=1)
    assert _fetch_sn(_gc, sn_v2)["name_stage"] == "superseded"
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["name_stage"] == "drafted"
    assert row_v3["chain_length"] == 2

    # Step 6: review_name accept 0.825
    stage = _do_review_name(
        _gc,
        mock_llm,
        sn_v3,
        scores=_SCORE_ACCEPT_NAME_MED,
        chain_length=2,
    )
    assert stage == "accepted"

    # Source PRODUCED_NAME now points to SN_v3
    assert _fetch_produced_name_target(_gc, src_id) == sn_v3

    # ── Docs pipeline (cross-pipeline gate: SN_v3 name_stage='accepted') ─

    # Step 7: generate_docs
    _do_generate_docs(_gc, mock_llm, sn_v3)
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["docs_stage"] == "drafted"
    assert row_v3["description"] != "Test quantity"

    # Step 8: review_docs reject 0.50
    stage = _do_review_docs(_gc, mock_llm, sn_v3, scores=_SCORE_REJECT_DOCS_LOW)
    assert stage == "reviewed"

    # Step 9: refine_docs → DocsRev_v0, docs_chain=1
    _do_refine_docs(_gc, mock_llm, sn_v3, docs_chain_length=0, iteration=0)
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["docs_stage"] == "drafted"
    assert row_v3["docs_chain_length"] == 1

    # Step 10: review_docs reject 0.55
    stage = _do_review_docs(
        _gc,
        mock_llm,
        sn_v3,
        scores=_SCORE_REJECT_DOCS,
        docs_chain_length=1,
    )
    assert stage == "reviewed"

    # Step 11: refine_docs → DocsRev_v1, docs_chain=2
    _do_refine_docs(_gc, mock_llm, sn_v3, docs_chain_length=1, iteration=1)
    row_v3 = _fetch_sn(_gc, sn_v3)
    assert row_v3["docs_stage"] == "drafted"
    assert row_v3["docs_chain_length"] == 2

    # Step 12: review_docs accept 0.85
    stage = _do_review_docs(
        _gc,
        mock_llm,
        sn_v3,
        scores=_SCORE_ACCEPT_DOCS,
        docs_chain_length=2,
    )
    assert stage == "accepted"

    # ── Final graph state assertions ────────────────────────────────────

    row_v1 = _fetch_sn(_gc, sn_v1)
    row_v2 = _fetch_sn(_gc, sn_v2)
    row_v3 = _fetch_sn(_gc, sn_v3)

    # 3 SN nodes: v1, v2 superseded; v3 accepted
    assert row_v1["name_stage"] == "superseded"
    assert row_v2["name_stage"] == "superseded"
    assert row_v3["name_stage"] == "accepted"
    assert row_v3["docs_stage"] == "accepted"

    # Chain lengths
    assert row_v3["chain_length"] == 2
    assert row_v3["docs_chain_length"] == 2

    # 2 REFINED_FROM edges reachable back from SN_v3
    assert _count_refined_from_edges(_gc, sn_v3) == 2, (
        "Expected 2 REFINED_FROM edges from SN_v3 chain"
    )

    # 2 DocsRevision nodes linked from SN_v3
    assert _count_docs_revisions(_gc, sn_v3) == 2, (
        "Expected 2 DocsRevision nodes linked to SN_v3"
    )

    # Source PRODUCED_NAME → SN_v3 (migrated twice)
    assert _fetch_produced_name_target(_gc, src_id) == sn_v3

    # chain_history walker finds 2 predecessors
    hist = name_chain_history(sn_v3)
    assert len(hist) == 2, f"Expected 2 chain_history entries, got {len(hist)}"

    # docs_chain_history walker finds 2 prior revisions
    docs_hist = docs_chain_history(sn_v3)
    assert len(docs_hist) == 2, (
        f"Expected 2 docs_chain_history entries, got {len(docs_hist)}"
    )

    # Reviewer fields on SN_v3
    assert row_v3["reviewer_score_name"] == pytest.approx(0.825, abs=0.02)
    assert row_v3["reviewer_score_docs"] == pytest.approx(0.85, abs=0.01)

    # Verify refine_name model calls (2 × language model, no escalation with
    # DEFAULT_REFINE_ROTATIONS=3 and chain lengths 0,1 both < 2)
    refine_name_calls = [
        c
        for c in mock_llm.calls
        if c["stage"] == "unknown" and c["response_model"] == "RefinedName"
    ]
    assert len(refine_name_calls) == 2, (
        f"Expected 2 refine_name LLM calls, got {len(refine_name_calls)}"
    )
    for call in refine_name_calls:
        assert call["model"] != DEFAULT_ESCALATION_MODEL, (
            f"Unexpected escalation model used at chain < 2: {call['model']!r}"
        )

    # Verify refine_docs model calls (2 × language model, no escalation)
    refine_docs_calls = [
        c
        for c in mock_llm.calls
        if c["stage"] == "unknown" and c["response_model"] == "RefinedDocs"
    ]
    assert len(refine_docs_calls) == 2, (
        f"Expected 2 refine_docs LLM calls, got {len(refine_docs_calls)}"
    )
    for call in refine_docs_calls:
        assert call["model"] != DEFAULT_ESCALATION_MODEL, (
            f"Unexpected escalation model used at docs_chain < 2: {call['model']!r}"
        )
