import pytest

from imas_codex.discovery.base.supervision import PipelinePhase
from imas_codex.graph.dd_workers import DDBuildState, embed_worker, enrich_worker


@pytest.mark.asyncio
async def test_enrich_worker_runs_aux_before_idle_completion(monkeypatch):
    state = DDBuildState(facility="imas")
    state.enrich_phase = PipelinePhase(
        "enrich", has_work_fn=lambda: False, idle_threshold=1
    )
    state.enrich_phase.refresh_has_work()

    order: list[str] = []
    original_record_idle = state.enrich_phase.record_idle

    def record_idle() -> None:
        order.append("idle")
        original_record_idle()

    state.enrich_phase.record_idle = record_idle  # type: ignore[method-assign]

    async def run_aux(test_state: DDBuildState) -> None:
        order.append("aux")
        test_state.aux_enrichment_done = True
        test_state.stop_requested = True

    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.claim_paths_for_enrichment",
        lambda limit, ids_filter=None: [],
    )
    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.count_imas_nodes_by_status",
        lambda node_category=None: {"total": 0},
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda _section: "test-model")
    monkeypatch.setattr("imas_codex.graph.dd_workers._run_aux_enrichment", run_aux)

    await enrich_worker(state)

    assert order[0] == "aux"
    assert "idle" not in order


@pytest.mark.asyncio
async def test_embed_worker_runs_aux_before_idle_completion(monkeypatch):
    state = DDBuildState(facility="imas")
    state.enrich_phase.mark_done()
    state.refine_phase.mark_done()
    state.embed_phase = PipelinePhase(
        "embed", has_work_fn=lambda: False, idle_threshold=1
    )
    state.embed_phase.refresh_has_work()

    order: list[str] = []
    original_record_idle = state.embed_phase.record_idle

    def record_idle() -> None:
        order.append("idle")
        original_record_idle()

    state.embed_phase.record_idle = record_idle  # type: ignore[method-assign]

    async def run_aux(test_state: DDBuildState) -> None:
        order.append("aux")
        test_state.aux_embedding_done = True
        test_state.stop_requested = True

    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.claim_paths_for_embedding",
        lambda limit: [],
    )
    monkeypatch.setattr(
        "imas_codex.graph.dd_graph_ops.count_imas_nodes_by_status",
        lambda node_category=None: {"total": 0},
    )
    monkeypatch.setattr("imas_codex.graph.dd_workers._run_aux_embedding", run_aux)

    await embed_worker(state)

    assert order[0] == "aux"
    assert "idle" not in order
