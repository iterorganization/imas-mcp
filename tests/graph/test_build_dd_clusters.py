from imas_codex.graph.build_dd import _compute_cluster_centroid_embeddings


class _FakeClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, cypher: str, **_params):
        self.queries.append(cypher)
        return [{"embeddings_set": 3}]


def test_compute_cluster_centroid_embeddings_uses_valid_cypher() -> None:
    client = _FakeClient()

    result = _compute_cluster_centroid_embeddings(client)

    assert result == 3
    assert len(client.queries) == 1
    assert "SET c.embedding = cluster_emb" in client.queries[0]
    assert "if _stopped():" not in client.queries[0]
