"""Tests for IMAS IDS node enrichment and embedding pipeline.

Covers:
- IDS enrichment Pydantic models
- Context gathering logic
- Enrichment orchestration (mocked LLM)
- Embedding generation (mocked encoder)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.dd_ids_enrichment import (
    IDSEnrichmentBatch,
    IDSEnrichmentResult,
    _compute_enrichment_hash,
    _gather_ids_context,
    embed_ids_nodes,
    enrich_ids_nodes,
)


class TestIDSEnrichmentModels:
    """Test Pydantic response models."""

    def test_enrichment_result_roundtrip(self):
        result = IDSEnrichmentResult(
            ids_index=1,
            description=(
                "Contains MHD equilibrium reconstruction results including "
                "flux surfaces, safety factor profiles, and global quantities."
            ),
            keywords=["equilibrium", "MHD", "flux surfaces", "EFIT"],
        )
        assert result.ids_index == 1
        assert "equilibrium" in result.description
        assert len(result.keywords) == 4

    def test_batch_model(self):
        batch = IDSEnrichmentBatch(
            results=[
                IDSEnrichmentResult(
                    ids_index=1,
                    description="Test IDS 1",
                    keywords=["a"],
                ),
                IDSEnrichmentResult(
                    ids_index=2,
                    description="Test IDS 2",
                    keywords=["b", "c"],
                ),
            ]
        )
        assert len(batch.results) == 2

    def test_keywords_max_length(self):
        result = IDSEnrichmentResult(
            ids_index=1,
            description="Test",
            keywords=["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        assert len(result.keywords) == 8


class TestComputeEnrichmentHash:
    def test_deterministic(self):
        h1 = _compute_enrichment_hash("ctx", "model-a")
        h2 = _compute_enrichment_hash("ctx", "model-a")
        assert h1 == h2

    def test_changes_with_model(self):
        h1 = _compute_enrichment_hash("ctx", "model-a")
        h2 = _compute_enrichment_hash("ctx", "model-b")
        assert h1 != h2

    def test_changes_with_context(self):
        h1 = _compute_enrichment_hash("ctx-1", "model-a")
        h2 = _compute_enrichment_hash("ctx-2", "model-a")
        assert h1 != h2


class TestGatherIDSContext:
    """Test context gathering for IDS enrichment."""

    def test_gathers_sections_and_identifiers(self):
        client = MagicMock()
        # Setup query responses for sections, identifiers, domain groups
        client.query.side_effect = [
            # sections_query
            [
                {
                    "ids_name": "equilibrium",
                    "path": "equilibrium/time_slice",
                    "name": "time_slice",
                    "documentation": "Equilibrium time slice",
                    "data_type": "STRUCT_ARRAY",
                },
                {
                    "ids_name": "equilibrium",
                    "path": "equilibrium/vacuum_toroidal_field",
                    "name": "vacuum_toroidal_field",
                    "documentation": "Vacuum toroidal field",
                    "data_type": "STRUCTURE",
                },
            ],
            # ident_query
            [
                {
                    "ids_name": "equilibrium",
                    "schema_name": "coordinate_identifier",
                    "schema_documentation": "Coordinate system selection",
                    "option_count": 20,
                },
            ],
            # domain_query
            [
                {"name": "equilibrium", "domain": "equilibrium"},
                {"name": "mhd_linear", "domain": "equilibrium"},
            ],
        ]

        ids_list = [
            {
                "id": "equilibrium",
                "name": "equilibrium",
                "documentation": "MHD equilibrium",
                "physics_domain": "equilibrium",
            }
        ]

        result = _gather_ids_context(client, ids_list)

        assert len(result) == 1
        ctx = result[0]
        assert len(ctx["sections"]) == 2
        assert ctx["sections"][0]["name"] == "time_slice"
        assert len(ctx["identifier_schemas"]) == 1
        assert ctx["identifier_schemas"][0]["name"] == "coordinate_identifier"
        assert "mhd_linear" in ctx["domain_siblings"]
        assert "equilibrium" not in ctx["domain_siblings"]  # self excluded

    def test_empty_context(self):
        client = MagicMock()
        client.query.side_effect = [[], [], []]
        ids_list = [{"id": "test_ids", "name": "test_ids", "physics_domain": "general"}]
        result = _gather_ids_context(client, ids_list)
        assert len(result) == 1
        assert result[0]["sections"] == []
        assert result[0]["identifier_schemas"] == []
        assert result[0]["domain_siblings"] == []


class TestEnrichIDSNodes:
    """Test IDS enrichment orchestration (mocked LLM)."""

    def test_skips_when_no_ids(self):
        client = MagicMock()
        client.query.return_value = []
        stats = enrich_ids_nodes(client, model="test-model")
        assert stats["enriched"] == 0

    def test_enriches_ids_with_llm(self):
        client = MagicMock()
        # First query: IDS nodes needing enrichment
        # Then: 3 context queries (sections, identifiers, domains)
        # Then: update query
        client.query.side_effect = [
            # Main IDS query
            [
                {
                    "id": "equilibrium",
                    "name": "equilibrium",
                    "documentation": "MHD equilibrium",
                    "physics_domain": "equilibrium",
                    "path_count": 1500,
                    "leaf_count": 800,
                    "max_depth": 8,
                    "lifecycle_status": "active",
                    "ids_type": "dynamic",
                    "enrichment_hash": None,
                }
            ],
            # Context: sections
            [
                {
                    "ids_name": "equilibrium",
                    "path": "equilibrium/time_slice",
                    "name": "time_slice",
                    "documentation": "Time slice",
                    "data_type": "STRUCT_ARRAY",
                }
            ],
            # Context: identifiers
            [],
            # Context: domains
            [{"name": "equilibrium", "domain": "equilibrium"}],
            # Update query
            None,
        ]

        mock_result = IDSEnrichmentBatch(
            results=[
                IDSEnrichmentResult(
                    ids_index=1,
                    description=(
                        "Contains MHD equilibrium reconstruction results "
                        "including flux surfaces and safety factor profiles."
                    ),
                    keywords=["equilibrium", "MHD", "flux surfaces"],
                )
            ]
        )

        with patch(
            "imas_codex.discovery.base.llm.call_llm_structured",
            return_value=(mock_result, 0.01, 500),
        ):
            stats = enrich_ids_nodes(client, model="test-model")

        assert stats["enriched"] == 1
        assert stats["cost"] == 0.01
        # Verify graph update was called
        update_call = client.query.call_args_list[-1]
        updates = update_call.kwargs.get("updates") or update_call[1].get("updates")
        assert updates[0]["id"] == "equilibrium"
        assert "flux surfaces" in updates[0]["description"]

    def test_caches_enriched_ids(self):
        """Already-enriched IDS with matching hash are skipped."""
        client = MagicMock()
        # Return empty list (no IDS needing enrichment)
        client.query.return_value = []
        stats = enrich_ids_nodes(client, model="test-model")
        assert stats["enriched"] == 0
        assert stats["cached"] == 0


class TestEmbedIDSNodes:
    """Test IDS embedding (mocked encoder, no graph)."""

    def test_skips_when_no_enriched_ids(self):
        client = MagicMock()
        client.query.return_value = []
        stats = embed_ids_nodes(client)
        assert stats["updated"] == 0
        assert stats["cached"] == 0
