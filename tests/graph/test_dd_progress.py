from types import SimpleNamespace

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.graph import dd_progress


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        stats={},
        imas_node_status_counts={},
        enrich_stats=WorkerStats(),
        embed_stats=WorkerStats(),
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
    state.embed_stats.processed = 999

    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.count_imas_nodes_by_status",
        lambda **_kwargs: {"built": 5, "embedded": 7, "total": 12},
    )

    dd_progress._graph_refresh(state, "imas")

    assert state.enrich_stats.total == 12
    assert state.enrich_stats.processed == 7
    assert state.embed_stats.total == 12
    assert state.embed_stats.processed == 7


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

    assert ("ident", "62/62", "cyan") in stats
    assert ("ids", "87/87", "green") in stats
    assert ("ident-emb", "62/62", "magenta") in stats
    assert ("ids-emb", "87/87", "magenta") in stats
