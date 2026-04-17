"""Tests for IMAS identifier schema enrichment and embedding pipeline.

Covers:
- Phase 1: XML-based option extraction with description/units
- Phase 2: Enrichment prompt construction and LLM response handling
- Phase 3: Embedding generation for identifier schemas
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.build_dd import _collect_identifier_schemas
from imas_codex.graph.dd_identifier_enrichment import (
    IdentifierEnrichmentBatch,
    IdentifierEnrichmentResult,
    _compute_enrichment_hash,
    embed_identifier_schemas,
    enrich_identifier_schemas,
)


class TestCollectIdentifierSchemas:
    """Test XML-based identifier schema extraction (Phase 1)."""

    def test_extracts_schemas_from_paths(self):
        """Schemas are collected for all referenced identifier enums."""
        paths = {
            "equilibrium/time_slice/profiles_1d/grid/identifier/index": {
                "identifier_enum_name": "coordinate_identifier",
            },
            "equilibrium/time_slice/boundary/type/index": {
                "identifier_enum_name": "coordinate_identifier",
            },
            "magnetics/method/identifier/index": {
                "identifier_enum_name": "magnetics_method_identifier",
            },
            "equilibrium/time_slice/profiles_1d/psi": {},
        }
        schemas = _collect_identifier_schemas(paths)

        assert "coordinate_identifier" in schemas
        # Two fields reference coordinate_identifier
        assert schemas["coordinate_identifier"]["field_count"] == 2

    def test_options_have_description_and_units(self):
        """Each option includes description and units from XML."""
        paths = {
            "equilibrium/grid/identifier/index": {
                "identifier_enum_name": "coordinate_identifier",
            },
        }
        schemas = _collect_identifier_schemas(paths)
        options = json.loads(schemas["coordinate_identifier"]["options"])

        assert len(options) > 0
        # Every option should have all four keys
        for opt in options:
            assert "index" in opt
            assert "name" in opt
            assert "description" in opt
            assert "units" in opt

        # coordinate_identifier should have units on at least some options
        units_present = [o for o in options if o["units"]]
        assert len(units_present) > 0, "Expected at least some options to have units"

    def test_schema_has_description_from_header(self):
        """Schema-level description is extracted from XML <header>."""
        paths = {
            "a/b/c": {"identifier_enum_name": "coordinate_identifier"},
        }
        schemas = _collect_identifier_schemas(paths)
        desc = schemas["coordinate_identifier"]["documentation"]
        assert desc, "Schema documentation should be populated from XML header"
        assert "coordinate" in desc.lower() or "translation" in desc.lower()

    def test_options_sorted_by_index(self):
        """Options are sorted by integer index."""
        paths = {
            "a/b": {"identifier_enum_name": "coordinate_identifier"},
        }
        schemas = _collect_identifier_schemas(paths)
        options = json.loads(schemas["coordinate_identifier"]["options"])
        indices = [o["index"] for o in options]
        assert indices == sorted(indices)

    def test_source_is_xml_path(self):
        """Source field uses utilities/<name>.xml convention."""
        paths = {
            "a/b": {"identifier_enum_name": "coordinate_identifier"},
        }
        schemas = _collect_identifier_schemas(paths)
        assert (
            schemas["coordinate_identifier"]["source"]
            == "utilities/coordinate_identifier.xml"
        )

    def test_no_identifier_paths_returns_empty(self):
        """Paths without identifier enums produce no schemas."""
        paths = {
            "equilibrium/time_slice/profiles_1d/psi": {},
            "equilibrium/time_slice/profiles_1d/pressure": {},
        }
        assert _collect_identifier_schemas(paths) == {}

    def test_unknown_enum_name_skipped(self):
        """Enum names not in dd_identifiers() are silently skipped."""
        paths = {
            "a/b": {"identifier_enum_name": "nonexistent_identifier_xyz"},
        }
        schemas = _collect_identifier_schemas(paths)
        assert "nonexistent_identifier_xyz" not in schemas


class TestEnrichmentHash:
    """Test enrichment hash computation."""

    def test_hash_deterministic(self):
        h1 = _compute_enrichment_hash("test context", "model-a")
        h2 = _compute_enrichment_hash("test context", "model-a")
        assert h1 == h2

    def test_hash_changes_with_model(self):
        h1 = _compute_enrichment_hash("test context", "model-a")
        h2 = _compute_enrichment_hash("test context", "model-b")
        assert h1 != h2

    def test_hash_changes_with_context(self):
        h1 = _compute_enrichment_hash("context-1", "model-a")
        h2 = _compute_enrichment_hash("context-2", "model-a")
        assert h1 != h2


class TestIdentifierEnrichmentModels:
    """Test Pydantic response models for enrichment."""

    def test_enrichment_result_roundtrip(self):
        result = IdentifierEnrichmentResult(
            schema_index=1,
            description="Test description of coordinate system selection.",
            keywords=["coordinate", "geometry"],
        )
        assert result.schema_index == 1
        assert "coordinate" in result.description
        assert len(result.keywords) == 2

    def test_batch_model(self):
        batch = IdentifierEnrichmentBatch(
            results=[
                IdentifierEnrichmentResult(
                    schema_index=1,
                    description="Test",
                    keywords=["a"],
                ),
                IdentifierEnrichmentResult(
                    schema_index=2,
                    description="Another",
                    keywords=["b", "c"],
                ),
            ]
        )
        assert len(batch.results) == 2
        assert batch.results[1].schema_index == 2


class TestEnrichIdentifierSchemas:
    """Test the enrichment orchestration logic (mocked LLM)."""

    def test_skips_when_no_schemas(self):
        """Returns zero stats when graph has no IdentifierSchema nodes."""
        client = MagicMock()
        client.query.return_value = []

        stats = enrich_identifier_schemas(client, model="test-model")
        assert stats["enriched"] == 0
        assert stats["cached"] == 0

    def test_enriches_schemas_with_llm(self):
        """Calls LLM and stores results for schemas needing enrichment."""
        client = MagicMock()
        # First query returns schemas
        client.query.side_effect = [
            [
                {
                    "id": "coordinate_identifier",
                    "name": "coordinate_identifier",
                    "documentation": "Translation table",
                    "options": json.dumps(
                        [
                            {
                                "index": 0,
                                "name": "unspecified",
                                "description": "unspecified",
                                "units": "m",
                            }
                        ]
                    ),
                    "option_count": 1,
                    "field_count": 5,
                    "source": "utilities/coordinate_identifier.xml",
                    "enrichment_hash": None,
                }
            ],
            None,  # update query
        ]

        mock_result = IdentifierEnrichmentBatch(
            results=[
                IdentifierEnrichmentResult(
                    schema_index=1,
                    description="Selects the coordinate system convention.",
                    keywords=["coordinate", "geometry"],
                )
            ]
        )

        with patch(
            "imas_codex.discovery.base.llm.call_llm_structured",
            return_value=(mock_result, 0.001, 100),
        ):
            stats = enrich_identifier_schemas(client, model="test-model")

        assert stats["enriched"] == 1
        # Verify graph update was called with description
        update_call = client.query.call_args_list[-1]
        updates = update_call.kwargs.get("updates") or update_call[1].get("updates")
        assert updates[0]["description"] == "Selects the coordinate system convention."


class TestIdentifierNodeEnrichmentModels:
    """Test Pydantic models for identifier node enrichment."""

    def test_node_result(self):
        from imas_codex.graph.dd_identifier_enrichment import (
            IdentifierNodeEnrichmentResult,
        )

        r = IdentifierNodeEnrichmentResult(
            path_index=1,
            description="Selects the grid type for equilibrium reconstruction.",
            keywords=["grid type", "equilibrium", "mesh"],
        )
        assert r.path_index == 1
        assert "grid" in r.description
        assert len(r.keywords) == 3

    def test_node_batch(self):
        from imas_codex.graph.dd_identifier_enrichment import (
            IdentifierNodeEnrichmentBatch,
            IdentifierNodeEnrichmentResult,
        )

        batch = IdentifierNodeEnrichmentBatch(
            results=[
                IdentifierNodeEnrichmentResult(
                    path_index=1, description="Desc 1", keywords=["a"]
                ),
                IdentifierNodeEnrichmentResult(
                    path_index=2, description="Desc 2", keywords=["b"]
                ),
            ]
        )
        assert len(batch.results) == 2


class TestEnrichIdentifierNodes:
    """Test enrichment of identifier IMASNodes (mocked LLM)."""

    def test_skips_when_no_nodes(self):
        from imas_codex.graph.dd_identifier_enrichment import enrich_identifier_nodes

        client = MagicMock()
        client.query.return_value = []
        stats = enrich_identifier_nodes(client, model="test-model")
        assert stats["enriched"] == 0
        assert stats["cached"] == 0

    def test_enriches_nodes_with_llm(self):
        from imas_codex.graph.dd_identifier_enrichment import (
            IdentifierNodeEnrichmentBatch,
            IdentifierNodeEnrichmentResult,
            enrich_identifier_nodes,
        )

        client = MagicMock()
        # First query returns identifier nodes
        client.query.side_effect = [
            [
                {
                    "id": "equilibrium/time_slice/profiles_2d/grid_type/index",
                    "name": "grid_type",
                    "ids": "equilibrium",
                    "documentation": "Integer identifier",
                    "enrichment_hash": None,
                    "parent_id": "equilibrium/time_slice/profiles_2d",
                    "parent_name": "profiles_2d",
                    "parent_description": "2D equilibrium profiles",
                    "schema_id": "grid_type_identifier",
                    "schema_name": "grid_type_identifier",
                    "schema_description": "Defines the grid type for 2D profiles.",
                    "schema_options": json.dumps(
                        [
                            {
                                "index": 1,
                                "name": "rectangular",
                                "description": "Rectangular grid",
                                "units": "",
                            }
                        ]
                    ),
                    "sibling_names": ["psi", "r", "z"],
                }
            ],
            None,  # update query
        ]

        mock_result = IdentifierNodeEnrichmentBatch(
            results=[
                IdentifierNodeEnrichmentResult(
                    path_index=1,
                    description="Selects the 2D mesh type for equilibrium reconstruction.",
                    keywords=["grid type", "equilibrium", "mesh"],
                )
            ]
        )

        with patch(
            "imas_codex.discovery.base.llm.call_llm_structured",
            return_value=(mock_result, 0.001, 100),
        ):
            stats = enrich_identifier_nodes(client, model="test-model")

        assert stats["enriched"] == 1
        # Verify graph update was called
        update_call = client.query.call_args_list[-1]
        updates = update_call.kwargs.get("updates") or update_call[1].get("updates")
        assert (
            updates[0]["description"]
            == "Selects the 2D mesh type for equilibrium reconstruction."
        )
        assert updates[0]["enrichment_model"] == "test-model"

    def test_caches_when_hash_matches(self):
        from imas_codex.graph.dd_identifier_enrichment import (
            _compute_enrichment_hash,
            enrich_identifier_nodes,
        )

        # Pre-compute the expected hash
        ctx_str = "test/path:schema_id:parent desc:doc"
        expected_hash = _compute_enrichment_hash(ctx_str, "test-model")

        client = MagicMock()
        client.query.return_value = [
            {
                "id": "test/path",
                "name": "identifier",
                "ids": "test",
                "documentation": "doc",
                "enrichment_hash": expected_hash,
                "parent_id": None,
                "parent_name": None,
                "parent_description": "parent desc",
                "schema_id": "schema_id",
                "schema_name": "schema_id",
                "schema_description": "Schema desc",
                "schema_options": None,
                "sibling_names": [],
            }
        ]

        stats = enrich_identifier_nodes(client, model="test-model")
        assert stats["cached"] == 1
        assert stats["enriched"] == 0
