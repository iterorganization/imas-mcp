"""Raw graph query benchmarks.

Isolates Neo4j query performance from MCP overhead.  Uses
``GraphClient.query()`` directly against the loaded GHCR dump.
"""

from __future__ import annotations

from benchmarks.conftest_bench import IMAS_PATHS, SEARCH_QUERIES, _fixture


class GraphQueryBenchmarks:
    """Benchmark raw Cypher query patterns against a loaded graph."""

    timeout = 120

    def setup(self):
        """Prepare graph client and a reusable embedding vector."""
        self.gc = _fixture.graph_client

        # Pre-compute an embedding for vector search benchmarks
        from imas_codex.embeddings.encoder import Encoder

        self._encoder = Encoder()
        self._embedding = self._encoder.embed_texts([SEARCH_QUERIES["simple"]])[
            0
        ].tolist()

        # Verify vector index is usable (skip if dimension mismatch from dump)
        try:
            self.gc.query(
                "CALL db.index.vector.queryNodes("
                "'imas_node_embedding', 1, $embedding) "
                "YIELD node RETURN node.id LIMIT 1",
                embedding=self._embedding,
            )
            self._vector_ok = True
        except Exception:
            self._vector_ok = False

        # Warmup: simple query
        self.gc.query("MATCH (n:IMASNode) RETURN n.id LIMIT 1")

    def time_vector_search_imas(self):
        """Vector index latency."""
        if not self._vector_ok:
            raise NotImplementedError("Vector index not usable (dimension mismatch)")
        self.gc.query(
            "CALL db.index.vector.queryNodes("
            "'imas_node_embedding', $k, $embedding) "
            "YIELD node, score "
            "RETURN node.id AS id, score",
            k=10,
            embedding=self._embedding,
        )

    def time_fulltext_search(self):
        """BM25 text search."""
        self.gc.query(
            "CALL db.index.fulltext.queryNodes("
            "'imas_node_text', $search_text) "
            "YIELD node, score "
            "RETURN node.id AS id, score "
            "LIMIT 10",
            search_text="electron temperature",
        )

    def time_path_traversal_enrichment(self):
        """Multi-hop enrichment from paths."""
        self.gc.query(
            "UNWIND $paths AS p "
            "MATCH (n:IMASNode {path: p}) "
            "OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit) "
            "OPTIONAL MATCH (n)-[:IN_DD_VERSION]->(v:DDVersion) "
            "OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster) "
            "RETURN n.path AS path, u.symbol AS unit, "
            "v.version AS version, c.label AS cluster",
            paths=[IMAS_PATHS["leaf"], IMAS_PATHS["branch"]],
        )

    def time_ids_aggregation(self):
        """Full IDS scan with aggregation."""
        self.gc.query(
            "MATCH (i:IDS) "
            "OPTIONAL MATCH (i)<-[:IN_IDS]-(n:IMASNode) "
            "RETURN i.id AS ids, count(n) AS path_count "
            "ORDER BY path_count DESC"
        )

    def time_version_chain_traversal(self):
        """Recursive version chain walk."""
        self.gc.query(
            "MATCH (v:DDVersion) "
            "OPTIONAL MATCH path = (v)-[:HAS_PREDECESSOR*1..5]->() "
            "RETURN v.version AS version, length(path) AS chain_length "
            "ORDER BY v.version DESC "
            "LIMIT 20"
        )

    def time_prefix_scan(self):
        """B-tree index prefix scan on path."""
        self.gc.query(
            "MATCH (n:IMASNode) "
            "WHERE n.path STARTS WITH $prefix "
            "RETURN n.path "
            "LIMIT 50",
            prefix="equilibrium/time_slice/profiles_1d",
        )

    def time_cluster_expansion(self):
        """Cluster member expansion."""
        self.gc.query(
            "MATCH (c:IMASSemanticCluster)"
            "<-[:IN_CLUSTER]-(n:IMASNode) "
            "WITH c, collect(n.path) AS members "
            "RETURN c.label AS cluster, size(members) AS n_members "
            "ORDER BY n_members DESC "
            "LIMIT 10"
        )

    def time_cross_ids_relationships(self):
        """Cross-IDS relationship join."""
        self.gc.query(
            "MATCH (n:IMASNode)-[:IN_IDS]->(i:IDS) "
            "OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster) "
            "OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit) "
            "OPTIONAL MATCH (n)-[:HAS_COORDINATE_SPEC]->(cs:IMASCoordinateSpec) "
            "RETURN i.id AS ids, n.path AS path, "
            "c.label AS cluster, u.symbol AS unit, "
            "cs.coordinate_type AS coord_type "
            "LIMIT 20"
        )
