"""Tests for auto-generated expected path expansion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tests.search.benchmark_data import ALL_QUERIES, BenchmarkQuery
from tests.search.benchmark_helpers import QueryResult, run_benchmark


class TestQueryResultExpandedExpected:
    """Verify QueryResult uses expanded expected paths when provided."""

    def test_default_uses_query_paths(self):
        q = BenchmarkQuery(
            query_text="electron temperature",
            expected_paths=["core_profiles/profiles_1d/electrons/temperature"],
            category="exact_concept",
        )
        qr = QueryResult(
            query=q,
            returned_paths=["core_profiles/profiles_1d/electrons/temperature"],
        )
        assert qr.expected_set == {"core_profiles/profiles_1d/electrons/temperature"}
        assert qr.reciprocal_rank == 1.0

    def test_expanded_overrides_query_paths(self):
        q = BenchmarkQuery(
            query_text="electron temperature",
            expected_paths=["core_profiles/profiles_1d/electrons/temperature"],
            category="exact_concept",
        )
        expanded = {
            "core_profiles/profiles_1d/electrons/temperature",
            "edge_profiles/profiles_1d/electrons/temperature",
            "summary/local/itb/t_e",
        }
        qr = QueryResult(
            query=q,
            returned_paths=["summary/local/itb/t_e", "other/path"],
            expanded_expected=expanded,
        )
        assert qr.expected_set == expanded
        assert qr.reciprocal_rank == 1.0  # t_e is at rank 1

    def test_expanded_empty_set_counts_as_provided(self):
        q = BenchmarkQuery(
            query_text="test",
            expected_paths=["a/b/c"],
            category="exact_concept",
        )
        qr = QueryResult(
            query=q,
            returned_paths=["a/b/c"],
            expanded_expected=set(),
        )
        # Empty expanded set means nothing matches
        assert qr.expected_set == set()
        assert qr.reciprocal_rank == 0.0

    def test_none_expanded_falls_back(self):
        q = BenchmarkQuery(
            query_text="test",
            expected_paths=["a/b/c"],
            category="exact_concept",
        )
        qr = QueryResult(
            query=q,
            returned_paths=["a/b/c"],
            expanded_expected=None,
        )
        assert qr.expected_set == {"a/b/c"}
        assert qr.reciprocal_rank == 1.0


class TestRunBenchmarkWithExpanded:
    """Verify run_benchmark passes expanded paths through."""

    def test_without_expanded(self):
        q = BenchmarkQuery(
            query_text="test",
            expected_paths=["a/b"],
            category="exact_concept",
        )
        results = run_benchmark(
            "test",
            [q],
            lambda qt, limit: ["a/b"],
        )
        assert results.mrr == 1.0

    def test_with_expanded(self):
        q = BenchmarkQuery(
            query_text="test",
            expected_paths=["a/b"],
            category="exact_concept",
        )
        expanded = {"test": {"a/b", "c/d"}}
        results = run_benchmark(
            "test",
            [q],
            lambda qt, limit: ["c/d"],
            expanded_expected=expanded,
        )
        assert results.mrr == 1.0  # c/d is in expanded set

    def test_expanded_improves_mrr(self):
        """Expanded paths should always produce >= MRR vs hand-curated."""
        q = BenchmarkQuery(
            query_text="test",
            expected_paths=["a/b"],
            category="exact_concept",
        )
        returned = ["x/y", "c/d", "a/b"]

        # Without expanded: a/b at rank 3
        r1 = run_benchmark("base", [q], lambda qt, limit: returned)

        # With expanded including c/d: match at rank 2
        expanded = {"test": {"a/b", "c/d"}}
        r2 = run_benchmark(
            "expanded", [q], lambda qt, limit: returned, expanded_expected=expanded
        )

        assert r2.mrr >= r1.mrr


@pytest.mark.graph
class TestGenerateExpectedPaths:
    """Integration tests for expected path generation (requires graph)."""

    @pytest.fixture(scope="class")
    def graph_client(self):
        from imas_codex.graph.client import GraphClient

        try:
            gc = GraphClient()
            gc.get_stats()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        yield gc
        gc.close()

    def test_electron_temperature_expansion(self, graph_client):
        """electron temperature should expand to >=10 paths across IDSs."""
        from tests.search.generate_expected_paths import generate_expected_paths

        q = BenchmarkQuery(
            query_text="electron temperature",
            expected_paths=[
                "core_profiles/profiles_1d/electrons/temperature",
            ],
            category="exact_concept",
        )
        paths = generate_expected_paths(q, graph_client)
        assert len(paths) >= 10, f"Only {len(paths)} paths for 'electron temperature'"
        # Must include hand-curated
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_hand_curated_always_included(self, graph_client):
        """Hand-curated paths must always be in the expanded set."""
        from tests.search.generate_expected_paths import generate_expected_paths

        for q in ALL_QUERIES[:10]:
            paths = generate_expected_paths(q, graph_client)
            for ep in q.expected_paths:
                assert ep in paths, (
                    f"Hand-curated path {ep!r} missing for {q.query_text!r}"
                )


class TestCacheRoundtrip:
    """Test cache save/load cycle without graph."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        import tests.search.conftest as search_conftest

        cache_file = tmp_path / ".expected_paths_cache.json"
        monkeypatch.setattr(search_conftest, "_CACHE_FILE", cache_file)

        # Save
        paths = {"test query": ["a/b", "c/d"]}
        search_conftest._save_cache(paths, "test_key")
        assert cache_file.exists()

        # Load
        loaded = search_conftest._load_cache()
        assert loaded is not None
        assert loaded["cache_key"] == "test_key"
        assert loaded["paths"] == paths

    def test_stale_cache_returns_none(self, tmp_path, monkeypatch):
        import tests.search.conftest as search_conftest

        cache_file = tmp_path / ".expected_paths_cache.json"
        monkeypatch.setattr(search_conftest, "_CACHE_FILE", cache_file)

        # Save with old timestamp
        data = {
            "cache_key": "old",
            "cached_at": time.time() - (8 * 86400),  # 8 days ago
            "paths": {},
        }
        cache_file.write_text(json.dumps(data))

        loaded = search_conftest._load_cache()
        assert loaded is None  # Too old
