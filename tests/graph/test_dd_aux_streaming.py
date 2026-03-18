from typing import Any, cast

from imas_codex.graph.dd_identifier_enrichment import (
    IdentifierEnrichmentBatch,
    IdentifierEnrichmentResult,
    embed_identifier_schemas,
    enrich_identifier_schemas,
)
from imas_codex.graph.dd_ids_enrichment import (
    IDSEnrichmentBatch,
    IDSEnrichmentResult,
    embed_ids_nodes,
    enrich_ids_nodes,
)


class _Vec:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeEncoder:
    def __init__(self, config):
        self.config = config

    def embed_texts(self, texts):
        return [_Vec([float(i + 1)]) for i, _ in enumerate(texts)]


class _FakeClient:
    def __init__(self):
        self.writes = []

    def query(self, cypher: str, **params):
        self.writes.append((cypher, params))
        compact = " ".join(cypher.split())

        if "MATCH (s:IdentifierSchema)" in compact and "RETURN s.id AS id" in compact:
            return [
                {
                    "id": "coordinate_identifier",
                    "name": "coordinate_identifier",
                    "documentation": "Coordinate enum",
                    "options": "[]",
                    "option_count": 0,
                    "field_count": 1,
                    "source": "dd",
                    "enrichment_hash": None,
                    "existing_hash": None,
                    "description": "Coordinate enum description",
                    "keywords": ["coord"],
                }
            ]

        if "MATCH (i:IDS)" in compact and "RETURN i.id AS id" in compact:
            return [
                {
                    "id": "core_profiles",
                    "name": "core_profiles",
                    "documentation": "Profiles IDS",
                    "physics_domain": "transport",
                    "path_count": 10,
                    "leaf_count": 5,
                    "max_depth": 3,
                    "lifecycle_status": "active",
                    "ids_type": "dynamic",
                    "enrichment_hash": None,
                    "existing_hash": None,
                    "description": "Core profiles description",
                    "keywords": ["profiles"],
                }
            ]

        if "UNWIND $ids_names AS ids_name" in compact:
            return []
        if (
            "MATCH (p:IMASNode)-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)"
            in compact
        ):
            return []
        if "MATCH (i:IDS) WHERE i.physics_domain IS NOT NULL" in compact:
            return []

        return []


def test_enrich_identifier_schemas_emits_items(monkeypatch):
    emitted = []
    client = cast(Any, _FakeClient())

    monkeypatch.setattr(
        "imas_codex.discovery.base.llm.call_llm_structured",
        lambda **_kwargs: (
            IdentifierEnrichmentBatch(
                results=[
                    IdentifierEnrichmentResult(
                        schema_index=1,
                        description="Identifier description",
                        keywords=["enum"],
                    )
                ]
            ),
            0.25,
            100,
        ),
    )
    monkeypatch.setattr(
        "imas_codex.llm.prompt_loader.render_prompt",
        lambda *_args, **_kwargs: "prompt",
    )
    monkeypatch.setattr(
        "imas_codex.settings.get_model",
        lambda _section: "test-model",
    )

    stats = enrich_identifier_schemas(
        client,
        model="test-model",
        on_items=lambda items, _batch_time: emitted.extend(items),
    )

    assert stats["enriched"] == 1
    assert emitted[0]["primary_text"] == "coordinate_identifier"


def test_enrich_ids_nodes_emits_items(monkeypatch):
    emitted = []
    client = cast(Any, _FakeClient())

    monkeypatch.setattr(
        "imas_codex.discovery.base.llm.call_llm_structured",
        lambda **_kwargs: (
            IDSEnrichmentBatch(
                results=[
                    IDSEnrichmentResult(
                        ids_index=1,
                        description="IDS description",
                        keywords=["ids"],
                    )
                ]
            ),
            0.5,
            120,
        ),
    )
    monkeypatch.setattr(
        "imas_codex.llm.prompt_loader.render_prompt",
        lambda *_args, **_kwargs: "prompt",
    )
    monkeypatch.setattr(
        "imas_codex.settings.get_model",
        lambda _section: "test-model",
    )

    stats = enrich_ids_nodes(
        client,
        model="test-model",
        on_items=lambda items, _batch_time: emitted.extend(items),
    )

    assert stats["enriched"] == 1
    assert emitted[0]["primary_text"] == "core_profiles"


def test_embed_identifier_schemas_emits_items(monkeypatch):
    emitted = []
    client = cast(Any, _FakeClient())

    monkeypatch.setattr("imas_codex.settings.get_embedding_dimension", lambda: 1)
    monkeypatch.setattr("imas_codex.settings.get_embedding_model", lambda: "embed")
    monkeypatch.setattr(
        "imas_codex.embeddings.encoder.Encoder",
        _FakeEncoder,
    )

    stats = embed_identifier_schemas(
        client,
        force_reembed=True,
        on_items=lambda items, _batch_time: emitted.extend(items),
    )

    assert stats["updated"] == 1
    assert emitted[0]["primary_text"] == "coordinate_identifier"


def test_embed_ids_nodes_emits_items(monkeypatch):
    emitted = []
    client = cast(Any, _FakeClient())

    monkeypatch.setattr("imas_codex.settings.get_embedding_dimension", lambda: 1)
    monkeypatch.setattr("imas_codex.settings.get_embedding_model", lambda: "embed")
    monkeypatch.setattr(
        "imas_codex.embeddings.encoder.Encoder",
        _FakeEncoder,
    )

    stats = embed_ids_nodes(
        client,
        force_reembed=True,
        on_items=lambda items, _batch_time: emitted.extend(items),
    )

    assert stats["updated"] == 1
    assert emitted[0]["primary_text"] == "core_profiles"
