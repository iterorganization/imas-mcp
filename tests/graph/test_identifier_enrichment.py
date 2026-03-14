"""Unit tests for identifier schema enrichment pipeline."""

import pytest

from imas_codex.graph.dd_identifier_enrichment import (
    IdentifierEnrichmentBatch,
    IdentifierEnrichmentResult,
    _compute_enrichment_hash,
)


class TestEnrichmentHash:
    """Hash computation for idempotency."""

    def test_same_input_same_hash(self):
        h1 = _compute_enrichment_hash("ctx", "model-a")
        h2 = _compute_enrichment_hash("ctx", "model-a")
        assert h1 == h2

    def test_different_model_different_hash(self):
        h1 = _compute_enrichment_hash("ctx", "model-a")
        h2 = _compute_enrichment_hash("ctx", "model-b")
        assert h1 != h2

    def test_different_context_different_hash(self):
        h1 = _compute_enrichment_hash("ctx-1", "model-a")
        h2 = _compute_enrichment_hash("ctx-2", "model-a")
        assert h1 != h2

    def test_hash_length(self):
        h = _compute_enrichment_hash("test", "model")
        assert len(h) == 16


class TestPydanticModels:
    """Validate response model schemas."""

    def test_single_result(self):
        r = IdentifierEnrichmentResult(
            schema_index=1,
            description="Test description",
            keywords=["kw1", "kw2"],
        )
        assert r.schema_index == 1
        assert len(r.keywords) == 2

    def test_batch_result(self):
        batch = IdentifierEnrichmentBatch(
            results=[
                IdentifierEnrichmentResult(
                    schema_index=1,
                    description="Desc 1",
                    keywords=["a"],
                ),
                IdentifierEnrichmentResult(
                    schema_index=2,
                    description="Desc 2",
                    keywords=["b", "c"],
                ),
            ]
        )
        assert len(batch.results) == 2
        assert batch.results[0].schema_index == 1
        assert batch.results[1].schema_index == 2

    def test_empty_keywords(self):
        r = IdentifierEnrichmentResult(schema_index=1, description="Test", keywords=[])
        assert r.keywords == []
