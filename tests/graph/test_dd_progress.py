from types import SimpleNamespace
from unittest.mock import patch

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.graph import dd_progress


def _state() -> SimpleNamespace:
    enrich_stats = WorkerStats()
    refine_stats = WorkerStats()
    embed_stats = WorkerStats()

    class _FakeState(SimpleNamespace):
        @property
        def total_cost(self) -> float:
            return self.enrich_stats.cost + self.refine_stats.cost

    return _FakeState(
        stats={},
        imas_node_status_counts={},
        embeddable_status_counts={},
        enrich_stats=enrich_stats,
        refine_stats=refine_stats,
        embed_stats=embed_stats,
        embed_phase=SimpleNamespace(is_done=False),
        cluster_phase=SimpleNamespace(is_done=False),
        cluster_stats=WorkerStats(),
    )


def test_build_stats_prefers_graph_embedded_count() -> None:
    state = _state()
    state.stats["embeddings_updated"] = 20067
    state.imas_node_status_counts["embedded"] = 20037
    state.embed_stats.processed = 20067

    stats = dd_progress._build_stats(state)

    assert ("embedded", "20.0K", "magenta") in stats
    assert ("embedded", "20.1K", "magenta") not in stats


def test_graph_refresh_overwrites_processed_with_graph_count(monkeypatch) -> None:
    state = _state()
    state.enrich_stats.processed = 999
    state.refine_stats.processed = 999
    state.embed_stats.processed = 999

    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.count_imas_nodes_by_status",
        lambda **kwargs: (
            {"built": 5, "refined": 3, "embedded": 7, "total": 15}
            if kwargs.get("node_categories") == dd_progress.ENRICHABLE_CATEGORIES
            else {"refined": 1, "embedded": 7, "total": 8}
        ),
    )

    with patch("imas_codex.graph.client.GraphClient") as mock_gc:
        mock_gc.return_value.__enter__ = lambda s: s
        mock_gc.return_value.__exit__ = lambda *a: None
        mock_gc.return_value.query.return_value = [
            {"enrich_cost": 5.50, "refine_cost": 8.25}
        ]
        dd_progress._graph_refresh(state, "imas")

    assert state.enrich_stats.total == 15
    assert state.enrich_stats.processed == 10  # total - built = 15 - 5
    assert state.refine_stats.total == 15
    assert state.refine_stats.processed == 10  # refined + embedded = 3 + 7
    assert state.embed_stats.total == 8  # embeddable only
    assert state.embed_stats.processed == 7
    assert state.enrich_stats.cost == 5.50
    assert state.refine_stats.cost == 8.25


def test_build_stats_include_auxiliary_ids_and_identifier_totals() -> None:
    state = _state()
    state.stats.update(
        {
            "identifier_schemas_total": 62,
            "identifier_schemas_enriched": 40,
            "identifier_schemas_cached": 22,
            "ids_total": 87,
            "ids_enriched": 50,
            "ids_cached": 37,
            "identifier_embeddings_updated": 62,
            "ids_embeddings_updated": 87,
        }
    )

    stats = dd_progress._build_stats(state)

    # Merged into single fields: identifiers and ids (no separate -emb fields)
    assert ("identifiers", "62/62", "cyan") in stats
    assert ("ids", "87/87", "green") in stats
    # Old separate fields must not appear
    assert not any(label == "ident" for label, _, _ in stats)
    assert not any(label == "ident-emb" for label, _, _ in stats)
    assert not any(label == "ids-emb" for label, _, _ in stats)


def test_build_stats_uses_max_of_enriched_and_embedded() -> None:
    """When enrichment returns 0 (prior run), embedding count is used."""
    state = _state()
    state.stats.update(
        {
            "identifier_schemas_total": 62,
            "identifier_schemas_enriched": 0,
            "identifier_schemas_cached": 0,
            "ids_total": 87,
            "ids_enriched": 0,
            "ids_cached": 0,
            "identifier_embeddings_updated": 0,
            "identifier_embeddings_cached": 62,
            "ids_embeddings_updated": 0,
            "ids_embeddings_cached": 87,
        }
    )

    stats = dd_progress._build_stats(state)

    assert ("identifiers", "62/62", "cyan") in stats
    assert ("ids", "87/87", "green") in stats


def test_build_stats_cost_includes_enrich_and_refine() -> None:
    """Cost stat shows total of enrich + refine costs."""
    state = _state()
    state.enrich_stats.cost = 13.65
    state.refine_stats.cost = 18.82

    stats = dd_progress._build_stats(state)

    cost_items = [(lbl, v) for lbl, v, _ in stats if lbl == "cost"]
    assert len(cost_items) == 1
    assert cost_items[0][1] == "$32.47"


def test_build_pending_includes_refine_stage() -> None:
    """Pending counts include enrich, refine, embed, and cluster."""
    state = _state()
    state.imas_node_status_counts = {
        "built": 100,
        "enriched": 30,
        "refined": 10,
        "embedded": 50,
    }
    state.embeddable_status_counts = {"refined": 5, "embedded": 50, "total": 55}

    pending = dd_progress._build_pending(state)

    labels = [p[0] for p in pending]
    assert labels == ["enrich", "refine", "embed", "cluster"]
    assert dict(pending)["enrich"] == 100
    assert dict(pending)["refine"] == 30
    assert dict(pending)["embed"] == 5  # embeddable only, not coordinate


def test_build_pending_embed_falls_back_to_enrichable_counts() -> None:
    """When embeddable counts unavailable, falls back to enrichable refined."""
    state = _state()
    state.imas_node_status_counts = {
        "built": 0,
        "enriched": 0,
        "refined": 15,
    }
    state.embeddable_status_counts = {}  # no embeddable query yet

    pending = dd_progress._build_pending(state)
    assert dict(pending)["embed"] == 15  # fallback to enrichable refined


def test_create_dd_build_display_has_refine_stage() -> None:
    """Display configuration includes REFINE stage between ENRICH and EMBED."""
    state = _state()
    display = dd_progress.create_dd_build_display(state)

    stage_names = [s.name for s in display.stages]
    assert "REFINE" in stage_names
    refine_idx = stage_names.index("REFINE")
    assert stage_names[refine_idx - 1] == "ENRICH"
    assert stage_names[refine_idx + 1] == "EMBED"
