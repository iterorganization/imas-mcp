"""Integration and claim-safety tests for the SN enrich pipeline (Phase C.8).

Groups:
1. Full pipeline smoke test (live Neo4j + LLM).
2. Dry-run integration (live Neo4j, no LLM).
3. Claim-safety: concurrent workers (live Neo4j).
4. Stale claim recovery (live Neo4j).
5. Round-trip validation (mock-based, no Neo4j).
6. Failure isolation (mock-based, no Neo4j).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Shared helpers
# =============================================================================

TEST_PREFIX = "test_integration_sn_"
# Use a physics_domain with no production data so claim-safety tests isolate
# themselves from the live graph. ``runaway_electrons`` has 0 ``pipeline_status='named'``
# nodes; using a real enum value keeps schema-compliance tests happy.
_TEST_PHYSICS_DOMAIN = "runaway_electrons"


def _make_sn(suffix: str | int, **overrides: Any) -> dict[str, Any]:
    """Build a synthetic SN item matching the extract worker's output shape."""
    name = f"{TEST_PREFIX}{suffix}"
    base = {
        "id": name,
        "description": None,
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "tags": ["transport"],
        "links": None,
        "source_paths": [f"transport_solver_numerics/time_slice/profiles_1d/{name}"],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "physics_domain": "transport",
        "confidence": 0.85,
        "model": "test-model",
    }
    base.update(overrides)
    return base


def _make_enriched_item(item: dict[str, Any]) -> dict[str, Any]:
    """Add enrichment fields to a raw item (simulating document + validate)."""
    item["enriched_description"] = f"The {item['id'].replace('_', ' ')} in plasma."
    item["enriched_documentation"] = f"Detailed docs for {item['id']}."
    item["enriched_links"] = []
    item["enriched_tags"] = ["spatial-profile"]
    item["validation_status"] = "valid"
    item["validation_issues"] = []
    item["llm_model"] = "test-enrich-model"
    item["llm_cost"] = 0.003
    item["enrich_tokens"] = 200
    item["embedding"] = [0.1] * 384
    return item


def _make_batch(items: list[dict], token: str = "tok", idx: int = 0) -> dict[str, Any]:
    return {"items": items, "claim_token": token, "batch_index": idx}


# =============================================================================
# Mock LLM response builder
# =============================================================================


def _build_mock_llm_response(items: list[dict[str, Any]]):
    """Build a StandardNameEnrichBatch-like mock from item list."""
    from imas_codex.standard_names.models import (
        StandardNameEnrichBatch,
        StandardNameEnrichItem,
    )

    enrich_items = [
        StandardNameEnrichItem(
            standard_name=it["id"],
            description=f"The {it['id'].replace('_', ' ')} in plasma.",
            documentation=f"Detailed docs for {it['id']}.",
            tags=["spatial-profile"],
            links=[],
        )
        for it in items
    ]
    return StandardNameEnrichBatch(items=enrich_items)


# =============================================================================
# 5. Round-trip validation (mock-based, no Neo4j)
# =============================================================================


class TestRoundTripMocked:
    """Full pipeline round-trip with mocked graph, LLM, and embedder."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock_roundtrip(self) -> None:
        """Feed 5 SNs through all 5 workers end-to-end, all mocked."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        items = [_make_sn(i) for i in range(5)]
        token = "roundtrip-token"

        state = StandardNameEnrichState(
            facility="dd",
            domain=_TEST_PHYSICS_DOMAIN,
            cost_limit=5.0,
            dry_run=False,
        )

        # Build the LLM response from items
        mock_response = _build_mock_llm_response(items)

        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=(token, list(items)),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=5,
            ),
            # Contextualise: mock graph client
            patch("imas_codex.graph.client.GraphClient") as mock_gc_cls,
            # Document: mock LLM call
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=(mock_response, 0.01, 500),
            ),
            # Validate: mock link integrity graph call
            patch(
                "imas_codex.standard_names.enrich_workers._check_links_batch",
                return_value={it["id"]: [] for it in items},
            ),
            # Validate: mock pydantic construction
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_pydantic",
                return_value=[],
            ),
            # Validate: mock description quality check
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_description",
                return_value=[],
            ),
            # Persist: mock embedding
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=lambda items, *a, **kw: [
                    it.__setitem__("embedding", [0.1] * 384) for it in items
                ],
            ),
            # Persist: mock graph persist
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=lambda items: len(items),
            ),
        ):
            # Set up the mock graph client for contextualise
            mock_gc = MagicMock()
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []
            mock_gc_cls.return_value = mock_gc

            await run_sn_enrich_engine(state)

        # All phases completed
        assert state.extract_phase.done
        assert state.contextualise_phase.done
        assert state.document_phase.done
        assert state.validate_phase.done
        assert state.persist_phase.done

        # Worker stats populated
        assert state.extract_stats.processed == 5
        assert state.contextualise_stats.processed == 5
        assert state.document_stats.processed == 5
        assert state.validate_stats.processed == 5

        # Cost tracked
        assert state.cost > 0
        assert state.cost == pytest.approx(0.01)

        # All items got validation status
        all_items = [it for b in state.batches for it in b["items"]]
        for item in all_items:
            assert item["validation_status"] in ("valid", "quarantined")

    @pytest.mark.asyncio
    async def test_roundtrip_enrichment_fields_populated(self) -> None:
        """Verify enriched items have correct fields after full pipeline."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        items = [_make_sn(i) for i in range(3)]
        token = "fields-token"

        state = StandardNameEnrichState(
            facility="dd",
            cost_limit=5.0,
            dry_run=False,
        )

        mock_response = _build_mock_llm_response(items)

        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=(token, list(items)),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=3,
            ),
            patch("imas_codex.graph.client.GraphClient") as mock_gc_cls,
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=(mock_response, 0.005, 300),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._check_links_batch",
                return_value={it["id"]: [] for it in items},
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_pydantic",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_description",
                return_value=[],
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=lambda items, *a, **kw: [
                    it.__setitem__("embedding", [0.1] * 384) for it in items
                ],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=lambda items: len(items),
            ),
        ):
            mock_gc = MagicMock()
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []
            mock_gc_cls.return_value = mock_gc

            await run_sn_enrich_engine(state)

        # Verify enriched fields are populated
        all_items = [it for b in state.batches for it in b["items"]]
        for item in all_items:
            assert isinstance(item.get("enriched_description"), str)
            assert len(item["enriched_description"]) > 0
            assert isinstance(item.get("enriched_documentation"), str)
            assert len(item["enriched_documentation"]) > 0

    @pytest.mark.asyncio
    async def test_roundtrip_cost_accumulates(self) -> None:
        """Cost from multiple batches accumulates on state."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        # 15 items → 2 batches (batch_size=10)
        items = [_make_sn(i) for i in range(15)]
        token = "cost-token"

        state = StandardNameEnrichState(
            facility="dd",
            cost_limit=10.0,
            dry_run=False,
        )

        call_count = 0

        async def _mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return matching response for whichever batch
            resp = _build_mock_llm_response(items)
            return resp, 0.02, 600

        with (
            patch(
                "imas_codex.standard_names.enrich_workers.claim_names_for_enrichment",
                return_value=(token, list(items)),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=15,
            ),
            patch("imas_codex.graph.client.GraphClient") as mock_gc_cls,
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=_mock_llm,
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._check_links_batch",
                return_value={it["id"]: [] for it in items},
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_pydantic",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_description",
                return_value=[],
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=lambda items, *a, **kw: [
                    it.__setitem__("embedding", [0.1] * 384) for it in items
                ],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=lambda items: len(items),
            ),
        ):
            mock_gc = MagicMock()
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []
            mock_gc_cls.return_value = mock_gc

            await run_sn_enrich_engine(state)

        # 15 items / batch_size=10 → 2 batches → 2 LLM calls
        assert call_count == 2
        assert state.cost == pytest.approx(0.04)


# =============================================================================
# 6. Failure isolation (mock-based)
# =============================================================================


class TestFailureIsolation:
    """Verify that LLM failures in one batch don't poison other batches."""

    @pytest.mark.asyncio
    async def test_malformed_llm_batch_isolated(self) -> None:
        """One batch with malformed LLM response fails; the other succeeds.

        Tests workers directly (no engine) to avoid supervised_worker
        restart complexity.
        """
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import (
            enrich_document_worker,
            enrich_persist_worker,
            enrich_validate_worker,
        )

        # 12 items → batch 0 (10 items), batch 1 (2 items)
        items = [_make_sn(i) for i in range(12)]
        token = "fail-iso-token"

        state = StandardNameEnrichState(
            facility="dd",
            cost_limit=10.0,
            dry_run=False,
        )

        # Simulate post-extract: items batched and contextualised
        from imas_codex.standard_names.enrich_workers import _build_batches

        state.batches = _build_batches(items, batch_size=10, token=token)
        # Add minimal context (as contextualise would)
        for batch in state.batches:
            for item in batch["items"]:
                item["dd_paths"] = []
                item["nearby"] = []
                item["siblings"] = []
                item["grammar"] = {
                    "physical_base": "temperature",
                    "subject": "electron",
                }
                item["current"] = {
                    "description": None,
                    "documentation": None,
                    "tags": item.get("tags"),
                    "links": None,
                }
                item["cocos"] = None

        call_idx = 0

        async def _mock_llm(*args, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                raise ValueError("Malformed JSON from LLM")
            resp = _build_mock_llm_response(items)
            return resp, 0.01, 200

        # Run document worker
        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=_mock_llm,
            ),
        ):
            await enrich_document_worker(state)

        assert state.document_phase.done
        assert state.batches[0].get("failed") is True

        # Batch 0 errored (10 items), batch 1 processed (2 items)
        assert state.stats["document_errors"] == 10
        assert state.stats["document_processed"] == 2

        # Run validate worker
        with (
            patch(
                "imas_codex.standard_names.enrich_workers._check_links_batch",
                return_value={it["id"]: [] for it in items},
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_pydantic",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_description",
                return_value=[],
            ),
        ):
            await enrich_validate_worker(state)

        assert state.validate_phase.done

        # Batch 0 items have no enriched_description AND no pre-existing
        # description/documentation → quarantined (P0.1: prevents empty-doc
        # leaks into the valid pool). The validation_issues record captures
        # the reason for downstream triage.
        for item in state.batches[0]["items"]:
            assert item["validation_status"] == "quarantined"
            assert "empty_documentation" in (item.get("validation_issues") or [])
        # Batch 1 items were validated
        for item in state.batches[1]["items"]:
            assert item["validation_status"] in ("valid", "quarantined")

        # Run persist worker
        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=lambda items, *a, **kw: [
                    it.__setitem__("embedding", [0.1] * 384) for it in items
                ],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=lambda items: len(items),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=12,
            ),
        ):
            await enrich_persist_worker(state)

        assert state.persist_phase.done
        # Only batch 1's valid items were persisted
        assert state.stats["persist_written"] == 2
        assert state.stats["persist_skipped"] >= 10  # batch 0 items skipped

    @pytest.mark.asyncio
    async def test_failed_batch_skipped_in_validate(self) -> None:
        """Validate worker skips items from failed batches (no enriched_description)."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState
        from imas_codex.standard_names.enrich_workers import enrich_validate_worker

        # Simulate post-document state: batch 0 failed, batch 1 ok
        item_failed = _make_sn(0)  # No enriched_description → pending
        item_ok = _make_sn(1)
        item_ok["enriched_description"] = "Good description."
        item_ok["enriched_documentation"] = "Good docs."
        item_ok["enriched_links"] = []
        item_ok["enriched_tags"] = ["transport"]

        state = StandardNameEnrichState(facility="dd")
        state.batches = [
            {
                "items": [item_failed],
                "claim_token": "t1",
                "batch_index": 0,
                "failed": True,
            },
            {"items": [item_ok], "claim_token": "t2", "batch_index": 1},
        ]

        with (
            patch(
                "imas_codex.standard_names.enrich_workers._check_links_batch",
                return_value={item_failed["id"]: [], item_ok["id"]: []},
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_pydantic",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.enrich_workers._validate_item_description",
                return_value=[],
            ),
        ):
            await enrich_validate_worker(state)

        assert state.validate_phase.done
        # Failed item has no enriched_description AND no pre-existing
        # description/documentation → quarantined with ``empty_documentation``
        # (P0.1: empty-doc leak guard).
        assert item_failed["validation_status"] == "quarantined"
        assert "empty_documentation" in (item_failed.get("validation_issues") or [])
        # Good item validated
        assert item_ok["validation_status"] == "valid"

    @pytest.mark.asyncio
    async def test_embedding_failure_isolates_item(self) -> None:
        """One item's embedding failure doesn't block the entire batch."""
        from imas_codex.standard_names.enrich_workers import enrich_persist_worker

        item_ok = _make_enriched_item(_make_sn(0))
        item_bad = _make_enriched_item(_make_sn(1))

        state = MagicMock()
        state.batches = [_make_batch([item_ok, item_bad], token="embed-tok")]
        state.stop_requested = False
        state.dry_run = False
        state.persist_stats = MagicMock()
        state.persist_stats.total = 0
        state.persist_stats.processed = 0
        state.persist_stats.errors = 0
        state.persist_phase = MagicMock()
        state.stats = {}

        def _partial_embed(items, *args, **kwargs):
            """Embed first item, fail second."""
            items[0]["embedding"] = [0.2] * 384
            items[1]["embedding"] = None  # Simulate failure

        with (
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                side_effect=_partial_embed,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_enriched_batch",
                side_effect=lambda items: len(items),
            ),
            patch(
                "imas_codex.standard_names.enrich_workers.release_enrichment_claims",
                return_value=2,
            ),
        ):
            await enrich_persist_worker(state)

        # First item persisted, second quarantined
        assert item_ok["embedding"] == [0.2] * 384
        assert item_bad["validation_status"] == "quarantined"
        assert "embedding_failed" in item_bad["validation_issues"]
        assert state.stats["persist_written"] == 1
        assert state.stats["persist_errors"] >= 1


# =============================================================================
# 1–4: Live Neo4j integration tests
# =============================================================================


def _skip_no_neo4j():
    """Pytest skip decorator for tests needing Neo4j."""
    try:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        gc.get_stats()
        gc.close()
        return False
    except Exception:
        return True


def _skip_no_api_key():
    """Check if OPENROUTER_API_KEY is set."""
    return not os.environ.get("OPENROUTER_API_KEY")


def _create_test_nodes(names: list[str]) -> None:
    """Create test StandardName nodes in graph with pipeline_status='named'."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for name in names:
            gc.query(
                """
                MERGE (sn:StandardName {id: $name})
                SET sn.pipeline_status = 'named',
                    sn.kind = 'scalar',
                    sn.unit = 'eV',
                    sn.physics_domain = $domain,
                    sn.grammar_physical_base = 'temperature',
                    sn.grammar_subject = 'electron',
                    sn.tags = ['transport'],
                    sn.confidence = 0.85,
                    sn.model = 'test-model',
                    sn.source_paths = [$path],
                    sn.enrich_claimed_at = null,
                    sn.enrich_claim_token = null
                """,
                name=name,
                domain=_TEST_PHYSICS_DOMAIN,
                path=f"transport_solver_numerics/time_slice/profiles_1d/{name}",
            )


def _cleanup_test_nodes() -> None:
    """Remove all test nodes from graph (idempotent)."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE sn.id STARTS WITH '{TEST_PREFIX}'
                DETACH DELETE sn
                """
            )
    except Exception:
        pass  # Best-effort cleanup


@pytest.fixture
def clean_test_nodes():
    """Fixture that cleans up test nodes before and after the test."""
    _cleanup_test_nodes()
    yield
    _cleanup_test_nodes()


@pytest.mark.integration
class TestDryRunIntegration:
    """Dry-run integration: live Neo4j, no LLM calls."""

    @pytest.mark.asyncio
    async def test_dry_run_no_mutations(self, clean_test_nodes) -> None:
        """Dry-run claims and releases without writing enrichment data."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        names = [f"{TEST_PREFIX}dry_1", f"{TEST_PREFIX}dry_2"]
        _create_test_nodes(names)

        state = StandardNameEnrichState(
            facility="dd",
            domain=_TEST_PHYSICS_DOMAIN,
            cost_limit=0.10,
            dry_run=True,
        )

        await run_sn_enrich_engine(state)

        # Pipeline completed
        assert state.extract_phase.done
        assert state.persist_phase.done

        # Verify no enrichment was written to graph
        with GraphClient() as gc:
            for name in names:
                rows = gc.query(
                    """
                    MATCH (sn:StandardName {id: $name})
                    RETURN sn.pipeline_status AS status,
                           sn.description AS description,
                           sn.documentation AS documentation,
                           sn.enrich_claimed_at AS claimed
                    """,
                    name=name,
                )
                assert len(rows) == 1
                node = rows[0]
                assert node["status"] == "named"  # NOT enriched
                assert node["description"] is None
                assert node["documentation"] is None
                assert node["claimed"] is None  # Claims released

    @pytest.mark.asyncio
    async def test_dry_run_no_llm_cost(self, clean_test_nodes) -> None:
        """Dry-run incurs zero LLM cost."""
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        names = [f"{TEST_PREFIX}drycost_1"]
        _create_test_nodes(names)

        state = StandardNameEnrichState(
            facility="dd",
            cost_limit=0.10,
            dry_run=True,
        )

        await run_sn_enrich_engine(state)

        assert state.cost == 0.0
        assert state.tokens_in == 0


@pytest.mark.integration
class TestClaimSafetyConcurrent:
    """Claim-safety: concurrent claim behavior and sequential exclusion.

    The two-step claim-token pattern (SET claim_token, then verify) provides
    exclusion for sequential access.  Under true concurrent access from
    separate Neo4j sessions, the last-writer-wins due to session-level
    read-your-own-writes semantics — this is a known limitation documented
    in Phase C.8 testing.  In production, only one enrich engine runs at
    a time, so sequential exclusion is sufficient.
    """

    @pytest.mark.asyncio
    async def test_held_claims_block_second_engine(self, clean_test_nodes) -> None:
        """While claims are held, a sequential second engine cannot claim the same SNs."""
        from imas_codex.standard_names.enrich_workers import (
            claim_names_for_enrichment,
            release_enrichment_claims,
        )

        names = [f"{TEST_PREFIX}held_{i}" for i in range(5)]
        _create_test_nodes(names)

        # Engine 1: claim all 5
        token1, items1 = claim_names_for_enrichment(
            limit=5, domain=_TEST_PHYSICS_DOMAIN
        )
        assert len(items1) == 5

        try:
            # Engine 2: try to claim — should get 0 (all held)
            token2, items2 = claim_names_for_enrichment(
                limit=5, domain=_TEST_PHYSICS_DOMAIN
            )

            assert len(items2) == 0, (
                f"Expected 0 items (all held by engine 1), got {len(items2)}"
            )

            if token2:
                release_enrichment_claims(token2)
        finally:
            # Release engine 1's claims
            release_enrichment_claims(token1)

        # After release, a new claim should succeed
        token3, items3 = claim_names_for_enrichment(
            limit=5, domain=_TEST_PHYSICS_DOMAIN
        )
        assert len(items3) == 5
        release_enrichment_claims(token3)

    @pytest.mark.asyncio
    async def test_concurrent_claims_db_consistent(self, clean_test_nodes) -> None:
        """After two concurrent claims, the DB converges to one winner per node.

        This tests that even under concurrent access, the claim-token
        pattern leaves the database in a consistent state: each node has
        exactly one token, and no claims are lost.
        """
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.enrich_workers import (
            claim_names_for_enrichment,
            release_enrichment_claims,
        )

        names = [f"{TEST_PREFIX}conc_{i}" for i in range(10)]
        _create_test_nodes(names)

        barrier = asyncio.Barrier(2)
        tokens: list[str] = []

        async def _claim(idx: int):
            await barrier.wait()
            token, _items = await asyncio.to_thread(
                claim_names_for_enrichment, limit=10, domain=_TEST_PHYSICS_DOMAIN
            )
            tokens.append(token)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_claim(1))
                tg.create_task(_claim(2))

            # After concurrent claims, check DB state:
            # Each node has exactly one token (the last writer's)
            with GraphClient() as gc:
                rows = gc.query(
                    f"""
                    MATCH (sn:StandardName)
                    WHERE sn.id STARTS WITH '{TEST_PREFIX}conc_'
                    RETURN sn.id AS id,
                           sn.enrich_claim_token AS token
                    """,
                )

                # All nodes have a token assigned
                assert len(rows) == 10
                for r in rows:
                    assert r["token"] is not None, f"Node {r['id']} has no token"

                # All nodes have the SAME token (last writer wins)
                unique_tokens = {r["token"] for r in rows}
                assert len(unique_tokens) == 1, (
                    f"Expected 1 winning token, got {len(unique_tokens)}: {unique_tokens}"
                )
        finally:
            for token in tokens:
                release_enrichment_claims(token)


@pytest.mark.integration
class TestStaleClaimRecovery:
    """Stale claim recovery: engine picks up timed-out claims."""

    @pytest.mark.asyncio
    async def test_stale_claim_recovered(self, clean_test_nodes) -> None:
        """A claim older than _CLAIM_TIMEOUT is re-claimable."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        name = f"{TEST_PREFIX}stale_1"
        _create_test_nodes([name])

        # Manually set a stale claim (10 minutes ago)
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (sn:StandardName {id: $name})
                SET sn.enrich_claimed_at = datetime() - duration('PT600S'),
                    sn.enrich_claim_token = 'stale-old-token'
                """,
                name=name,
            )

        # Run engine — should reclaim the stale node
        state = StandardNameEnrichState(
            facility="dd",
            domain=_TEST_PHYSICS_DOMAIN,
            cost_limit=0.10,
            dry_run=True,  # dry_run to avoid LLM
            limit=1,
        )

        await run_sn_enrich_engine(state)

        # The stale node was picked up
        assert state.stats.get("extract_count", 0) >= 1

        # Verify the old stale token was overwritten (claims released in dry_run)
        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName {id: $name})
                RETURN sn.enrich_claim_token AS token,
                       sn.enrich_claimed_at AS claimed
                """,
                name=name,
            )
            assert len(rows) == 1
            # Claims released in dry_run
            assert rows[0]["token"] is None
            assert rows[0]["claimed"] is None


@pytest.mark.integration
class TestFullPipelineSmoke:
    """Full pipeline smoke test (live Neo4j + LLM)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        _skip_no_api_key(),
        reason="OPENROUTER_API_KEY not set",
    )
    async def test_live_enrich_pipeline(self, clean_test_nodes) -> None:
        """End-to-end: 3 test SNs enriched with real LLM + graph."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        names = [f"{TEST_PREFIX}live_{i}" for i in range(3)]
        _create_test_nodes(names)

        state = StandardNameEnrichState(
            facility="dd",
            domain=_TEST_PHYSICS_DOMAIN,
            cost_limit=0.20,
            dry_run=False,
            limit=3,
        )

        await run_sn_enrich_engine(state)

        # All 3 claimed
        assert state.stats.get("extract_count", 0) == 3

        # At least 2 successfully enriched (allow one flaky)
        with GraphClient() as gc:
            rows = gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE sn.id STARTS WITH '{TEST_PREFIX}live_'
                RETURN sn.id AS id,
                       sn.pipeline_status AS status,
                       sn.description AS description,
                       sn.documentation AS documentation,
                       sn.embedding IS NOT NULL AS has_embedding,
                       sn.enrich_claimed_at AS claimed
                """,
            )
            enriched = [
                r
                for r in rows
                if r["status"] == "enriched"
                and r["description"] is not None
                and r["documentation"] is not None
                and r["has_embedding"]
            ]
            assert len(enriched) >= 2, (
                f"Expected ≥2 enriched, got {len(enriched)}: "
                f"{[(r['id'], r['status']) for r in rows]}"
            )

            # Claims released on completion
            for r in rows:
                assert r["claimed"] is None, f"Claim not released for {r['id']}"
