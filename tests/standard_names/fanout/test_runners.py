"""Async runner tests for fan-out (plan 39 §4.2, §12.2)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from imas_codex.standard_names.fanout.runners import (
    _run_with_timeout,
    run_find_related_dd_paths,
    run_search_dd_clusters,
    run_search_dd_paths,
    run_search_existing_names,
)
from imas_codex.standard_names.fanout.schemas import (
    FanoutScope,
    _FindRelatedDDPaths,
    _SearchDDClusters,
    _SearchDDPaths,
    _SearchExistingNames,
)

# ---------------------------------------------------------------------
# _run_with_timeout — primitive that all runners use
# ---------------------------------------------------------------------


class TestRunWithTimeout:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        def _ok():
            return [1, 2, 3]

        def _norm(raw):
            return []

        r = await _run_with_timeout(
            fn_id="search_dd_paths",
            args_dict={"fn_id": "search_dd_paths", "query": "x"},
            sync_callable=_ok,
            timeout_s=2.0,
            normalise=_norm,
        )
        assert r.ok is True
        assert r.error is None
        assert r.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self) -> None:
        def _slow():
            time.sleep(0.5)  # blocks worker thread; wait_for cancels at gate
            return []

        r = await _run_with_timeout(
            fn_id="search_dd_paths",
            args_dict={"fn_id": "search_dd_paths", "query": "x"},
            sync_callable=_slow,
            timeout_s=0.05,
            normalise=lambda raw: [],
        )
        assert r.ok is False
        assert r.error == "timeout"

    @pytest.mark.asyncio
    async def test_exception_caught(self) -> None:
        def _boom():
            raise RuntimeError("kaboom")

        r = await _run_with_timeout(
            fn_id="search_dd_paths",
            args_dict={"fn_id": "search_dd_paths", "query": "x"},
            sync_callable=_boom,
            timeout_s=2.0,
            normalise=lambda raw: [],
        )
        assert r.ok is False
        assert r.error is not None
        assert "RuntimeError" in r.error


# ---------------------------------------------------------------------
# Per-runner happy paths via monkeypatch
# ---------------------------------------------------------------------


class TestRunnerHappyPaths:
    @pytest.mark.asyncio
    async def test_search_existing_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake(query: str, k: int, *, gc: Any, include_superseded: bool):
            assert query == "T_e"
            assert k == 3
            assert include_superseded is False
            return [
                {
                    "id": "electron_temperature",
                    "description": "T_e",
                    "kind": "scalar",
                    "unit": "eV",
                    "score": 0.91,
                }
            ]

        monkeypatch.setattr(
            "imas_codex.standard_names.search.search_standard_names_vector",
            _fake,
        )
        args = _SearchExistingNames(fn_id="search_existing_names", query="T_e", k=3)
        r = await run_search_existing_names(
            args, gc=object(), scope=FanoutScope(), timeout_s=2.0
        )
        assert r.ok is True
        assert len(r.hits) == 1
        assert r.hits[0].kind == "standard_name"
        assert r.hits[0].id == "electron_temperature"

    @pytest.mark.asyncio
    async def test_search_dd_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Hit:
            def __init__(self) -> None:
                self.path = "core_profiles/profiles_1d/electrons/temperature"
                self.ids_name = "core_profiles"
                self.units = "eV"
                self.score = 0.83

        def _fake(gc, query, *, ids_filter, physics_domain, dd_version, k):
            assert query == "ne"
            assert ids_filter == "core_profiles"
            return [_Hit()]

        monkeypatch.setattr("imas_codex.graph.dd_search.hybrid_dd_search", _fake)
        args = _SearchDDPaths(fn_id="search_dd_paths", query="ne", k=8)
        r = await run_search_dd_paths(
            args,
            gc=object(),
            scope=FanoutScope(ids_filter="core_profiles"),
            timeout_s=2.0,
        )
        assert r.ok is True
        assert r.hits[0].kind == "dd_path"
        assert r.hits[0].id == "core_profiles/profiles_1d/electrons/temperature"

    @pytest.mark.asyncio
    async def test_find_related_dd_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _RelHit:
            def __init__(self) -> None:
                self.path = "edge_profiles/.../t_e"
                self.relationship_type = "cluster"
                self.via = "cluster_1234"

        class _Result:
            def __init__(self) -> None:
                self.hits = [_RelHit()]

        def _fake(gc, path, *, max_results, dd_version):
            assert path == "core_profiles/.../temperature"
            assert max_results == 5
            return _Result()

        monkeypatch.setattr("imas_codex.graph.dd_search.related_dd_search", _fake)
        args = _FindRelatedDDPaths(
            fn_id="find_related_dd_paths",
            path="core_profiles/.../temperature",
            max_results=5,
        )
        r = await run_find_related_dd_paths(
            args, gc=object(), scope=FanoutScope(), timeout_s=2.0
        )
        assert r.ok is True
        assert r.hits[0].kind == "dd_path"
        assert "cluster" in r.hits[0].label

    @pytest.mark.asyncio
    async def test_search_dd_clusters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Cluster:
            def __init__(self) -> None:
                self.id = "cluster_42"
                self.label = "electron temperature"
                self.description = "T_e family"
                self.score = 0.77
                self.scope = "global"

        def _fake(gc, query, *, k, dd_version):
            return [_Cluster()]

        monkeypatch.setattr("imas_codex.graph.dd_search.cluster_search", _fake)
        args = _SearchDDClusters(fn_id="search_dd_clusters", query="electron temp")
        r = await run_search_dd_clusters(
            args, gc=object(), scope=FanoutScope(), timeout_s=2.0
        )
        assert r.ok is True
        assert r.hits[0].kind == "cluster"
        assert r.hits[0].id == "cluster_42"
