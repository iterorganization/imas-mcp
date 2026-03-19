"""Query builder benchmarks.

Pure Python — measures Cypher query generation overhead without
executing queries against Neo4j.  A stub ``gc`` intercepts the
final ``gc.query()`` call so only the build path is timed.
"""

from __future__ import annotations

from imas_codex.graph.query_builder import graph_search


class _StubGC:
    """Minimal stand-in for GraphClient — returns empty results."""

    def query(self, cypher: str, **params) -> list[dict]:  # noqa: ARG002
        return []


_gc = _StubGC()


class QueryBuilderBenchmarks:
    """Benchmark graph_search() query generation and schema validation."""

    timeout = 30

    def setup(self):
        """Warmup: run a simple query to trigger any lazy init."""
        graph_search("IMASNode", limit=1, gc=_gc)

    def time_basic_query_generation(self):
        """Minimal query generation."""
        graph_search("IMASNode", limit=10, gc=_gc)

    def time_filtered_query_generation(self):
        """Query with where filters and traversals."""
        graph_search(
            "IMASNode",
            where={"path__starts_with": "equilibrium"},
            traverse=["HAS_UNIT>Unit", "IN_IDS>IDS"],
            limit=25,
            gc=_gc,
        )

    def time_schema_validation(self):
        """Validation overhead on invalid label."""
        try:
            graph_search("NonExistentLabel", limit=1, gc=_gc)
        except ValueError:
            pass

    def time_traversal_expansion(self):
        """Multi-hop traversal expansion."""
        graph_search(
            "IMASNode",
            traverse=[
                "HAS_UNIT>Unit",
                "IN_IDS>IDS",
                "IN_CLUSTER>IMASSemanticCluster",
            ],
            limit=10,
            gc=_gc,
        )
