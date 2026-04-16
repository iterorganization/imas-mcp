from imas_codex.graph import dd_graph_ops


class _FakeGraphClient:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def query(self, cypher, **params):
        self._calls.append((cypher, params))
        return self._result


def test_count_imas_nodes_by_status_filters_node_category(monkeypatch):
    calls = []
    result = [
        {"status": "embedded", "cnt": 5},
        {"status": "enriched", "cnt": 2},
    ]

    monkeypatch.setattr(
        dd_graph_ops,
        "GraphClient",
        lambda: _FakeGraphClient(result, calls),
    )

    counts = dd_graph_ops.count_imas_nodes_by_status(node_categories=["data"])

    assert counts == {"embedded": 5, "enriched": 2, "total": 7}
    assert len(calls) == 1
    cypher, params = calls[0]
    assert "p.node_category IN $filter_categories" in cypher
    assert params["filter_categories"] == ["data"]


def test_count_imas_nodes_by_status_without_filter(monkeypatch):
    calls = []
    result = [{"status": "built", "cnt": 3}]

    monkeypatch.setattr(
        dd_graph_ops,
        "GraphClient",
        lambda: _FakeGraphClient(result, calls),
    )

    counts = dd_graph_ops.count_imas_nodes_by_status()

    assert counts == {"built": 3, "total": 3}
    assert len(calls) == 1
    _cypher, params = calls[0]
    assert params["filter_categories"] is None


# -------------------------------------------------------------------------
# mark_paths_enriched — cost tracking
# -------------------------------------------------------------------------


def test_mark_paths_enriched_includes_cost_field(monkeypatch):
    """enrich_llm_cost is SET atomically with other enrichment fields."""
    calls: list = []
    monkeypatch.setattr(
        dd_graph_ops,
        "GraphClient",
        lambda: _FakeGraphClient([{"updated": 2}], calls),
    )

    updates = [
        {
            "id": "equilibrium/time_slice/profiles_1d/psi",
            "description": "Poloidal flux (ψ)",
            "enrichment_source": "llm",
            "enrich_llm_cost": 0.00012,
        },
        {
            "id": "equilibrium/time_slice/profiles_1d/q",
            "description": "Safety factor (q)",
            "enrichment_source": "template",
            # No enrich_llm_cost — template nodes get null
        },
    ]
    count = dd_graph_ops.mark_paths_enriched(updates)

    assert count == 2
    assert len(calls) == 1
    cypher, params = calls[0]
    # Verify cost field is in the Cypher SET clause
    assert "enrich_llm_cost" in cypher
    assert "coalesce(item.enrich_llm_cost, p.enrich_llm_cost)" in cypher


def test_mark_paths_refined_includes_cost_field(monkeypatch):
    """refine_llm_cost is SET atomically with other refinement fields."""
    calls: list = []
    monkeypatch.setattr(
        dd_graph_ops,
        "GraphClient",
        lambda: _FakeGraphClient([{"updated": 1}], calls),
    )

    updates = [
        {
            "id": "equilibrium/time_slice/profiles_1d/psi",
            "description": "Refined: Poloidal flux (ψ)",
            "refinement_hash": "abc123",
            "refine_llm_cost": 0.00015,
        },
    ]
    count = dd_graph_ops.mark_paths_refined(updates)

    assert count == 1
    cypher, _ = calls[0]
    assert "refine_llm_cost" in cypher
    assert "coalesce(item.refine_llm_cost, p.refine_llm_cost)" in cypher


# -------------------------------------------------------------------------
# Reset clears cost fields
# -------------------------------------------------------------------------


def test_reset_to_built_clears_cost_fields():
    """Resetting to 'built' clears both enrich and refine cost fields."""
    fields = dd_graph_ops._RESET_CLEAR_FIELDS["built"]
    assert "enrich_llm_cost" in fields
    assert "refine_llm_cost" in fields


def test_reset_to_enriched_clears_refine_cost():
    """Resetting to 'enriched' clears refine cost but not enrich cost."""
    fields = dd_graph_ops._RESET_CLEAR_FIELDS["enriched"]
    assert "refine_llm_cost" in fields
    assert "enrich_llm_cost" not in fields


def test_reset_to_refined_no_cost_fields():
    """Resetting to 'refined' does not clear any cost fields."""
    fields = dd_graph_ops._RESET_CLEAR_FIELDS["refined"]
    assert "enrich_llm_cost" not in fields
    assert "refine_llm_cost" not in fields
