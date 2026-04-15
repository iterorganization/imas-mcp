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
