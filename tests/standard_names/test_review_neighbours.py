"""Unit tests for ``fetch_review_neighbours``.

The helper returns nearest-neighbour standard names for sibling-comparison
during third-party-critic review. These tests verify:

* Graceful degradation when the graph is unavailable.
* Correct dispatch into vector / same-base / same-path lookups based on the
  fields populated on the candidate dict.
* Per-list k-limits are honoured.
* The candidate itself is excluded from the vector neighbour list.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names.context import (
    _path_ids_prefix,
    fetch_review_neighbours,
)


def test_path_ids_prefix_extracts_leading_segment():
    assert _path_ids_prefix("equilibrium/time_slice/0/global_quantities/ip") == (
        "equilibrium"
    )
    assert _path_ids_prefix("magnetics") == "magnetics"
    assert _path_ids_prefix("") is None
    assert _path_ids_prefix("/") is None


def test_fetch_review_neighbours_no_graph_returns_empty_lists():
    """When GraphClient cannot be opened, all lists are empty (no raise)."""

    def _raise(*_a, **_kw):
        raise RuntimeError("graph unavailable")

    with patch("imas_codex.graph.client.GraphClient", side_effect=_raise):
        out = fetch_review_neighbours(
            {
                "id": "electron_temperature",
                "description": "electron kinetic temperature",
                "physical_base": "temperature",
                "source_paths": ["core_profiles/profiles_1d/0/electrons/temperature"],
            }
        )
    assert out == {
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
    }


class _FakeGC:
    def __init__(self, base_rows, path_rows):
        self._base_rows = base_rows
        self._path_rows = path_rows
        self.calls: list[tuple[str, dict]] = []

    def query(self, q, **params):
        self.calls.append((q, params))
        if "sn.physical_base = $base" in q:
            return self._base_rows
        if "STARTS WITH $prefix" in q:
            return self._path_rows
        return []


def test_fetch_review_neighbours_dispatch_and_limits():
    """Same-base + same-path Cypher are issued with the right params and k."""
    gc = _FakeGC(
        base_rows=[
            {"id": "ion_temperature", "name": "T_i", "description": "ion T"},
            {"id": "neutral_temperature", "name": "T_n", "description": "neutral T"},
            {"id": "edge_temperature", "name": "T_e", "description": "edge T"},
        ],
        path_rows=[
            {
                "id": "electron_density",
                "name": "n_e",
                "description": "electron density",
            },
            {"id": "ion_density", "name": "n_i", "description": "ion density"},
        ],
    )

    # Patch the vector helper to return a deterministic list including the
    # candidate (which must be filtered out).
    fake_vector = [
        {"id": "electron_temperature", "description": "self", "score": 0.99},
        {"id": "ion_temperature", "description": "ion T", "score": 0.92},
        {"id": "edge_temperature", "description": "edge T", "score": 0.88},
    ]

    with patch(
        "imas_codex.standard_names.search.search_standard_names_vector",
        return_value=fake_vector,
    ):
        out = fetch_review_neighbours(
            {
                "id": "electron_temperature",
                "description": "electron kinetic temperature",
                "physical_base": "temperature",
                "source_paths": ["core_profiles/profiles_1d/0/electrons/temperature"],
            },
            gc=gc,
            n_vector=2,
            n_same_base=2,
            n_same_path=1,
        )

    # Vector excludes the candidate id and respects n_vector
    assert [n["id"] for n in out["vector_neighbours"]] == [
        "ion_temperature",
        "edge_temperature",
    ]
    # Same-base limit honoured (n_same_base=2 → 2 rows requested in Cypher)
    base_call = next(c for c in gc.calls if "sn.physical_base = $base" in c[0])
    assert base_call[1]["base"] == "temperature"
    assert base_call[1]["sn_id"] == "electron_temperature"
    assert base_call[1]["k"] == 2
    # Same-path: prefix passed with trailing slash
    path_call = next(c for c in gc.calls if "STARTS WITH $prefix" in c[0])
    assert path_call[1]["prefix"] == "core_profiles/"
    assert path_call[1]["k"] == 1


def test_fetch_review_neighbours_skips_lookups_for_missing_fields():
    """Missing physical_base / source_paths skip the corresponding queries."""
    gc = _FakeGC(base_rows=[], path_rows=[])
    with patch(
        "imas_codex.standard_names.search.search_standard_names_vector",
        return_value=[],
    ):
        out = fetch_review_neighbours(
            {"id": "x", "description": "y"},  # no base, no source_paths
            gc=gc,
        )
    assert out["same_base_neighbours"] == []
    assert out["same_path_neighbours"] == []
    # Neither same-base nor same-path Cypher should have been issued.
    assert all(
        "sn.physical_base = $base" not in c[0] and "STARTS WITH $prefix" not in c[0]
        for c in gc.calls
    )


def test_fetch_review_neighbours_grammar_fields_fallback():
    """``physical_base`` from grammar_fields is honoured when top-level is absent."""
    gc = _FakeGC(base_rows=[], path_rows=[])
    with patch(
        "imas_codex.standard_names.search.search_standard_names_vector",
        return_value=[],
    ):
        fetch_review_neighbours(
            {
                "id": "x",
                "description": "y",
                "grammar_fields": {"physical_base": "torque"},
            },
            gc=gc,
        )
    base_call = next(c for c in gc.calls if "sn.physical_base = $base" in c[0])
    assert base_call[1]["base"] == "torque"
