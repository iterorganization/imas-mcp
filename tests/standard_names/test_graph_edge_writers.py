"""Graph edge writer integration tests (G1–G10).

All tests are mocked — no live Neo4j required.  They verify that
``write_standard_names`` and ``_write_import_entries`` emit the correct
Cypher queries with the expected batch parameters for every structural
edge type.

Edge types covered:
  HAS_ARGUMENT   — derived from ISN parser (G1, G2, G4, G5)
  HAS_ERROR      — uncertainty siblings, inverted direction (G3)
  HAS_PREDECESSOR — from ``deprecates`` field (G7)
  HAS_SUCCESSOR   — from ``superseded_by`` field (G8)
  IN_CLUSTER      — from ``primary_cluster_id`` field (G9)
  HAS_PHYSICS_DOMAIN — from ``physics_domain`` field (G10)
  Catalog import parity (G6)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

imas_sn = pytest.importorskip("imas_standard_names")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_gc() -> MagicMock:
    """Return a fresh MagicMock for GraphClient."""
    gc = MagicMock()
    gc.query = MagicMock(return_value=[])
    return gc


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Invoke ``write_standard_names`` with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        # Patch protection and segment edges to keep tests focused
        with (
            patch(
                "imas_codex.standard_names.protection.filter_protected",
                side_effect=lambda n, **kw: (n, []),
            ),
            patch(
                "imas_codex.standard_names.graph_ops._write_segment_edges",
                return_value=[],
            ),
        ):
            from imas_codex.standard_names.graph_ops import write_standard_names

            return write_standard_names(names)


def _call_import_entries(entries: list[dict], mock_gc: MagicMock) -> int:
    """Invoke ``_write_import_entries`` directly with a mocked gc."""
    from imas_codex.standard_names.catalog_import import _write_import_entries

    return _write_import_entries(mock_gc, entries)


def _cyphers(mock_gc: MagicMock) -> list[str]:
    """Return all Cypher strings passed to gc.query."""
    return [c[0][0] for c in mock_gc.query.call_args_list]


def _batch_for(mock_gc: MagicMock, keyword: str) -> list[dict] | None:
    """Return the ``batch`` kwarg from the first query matching *keyword*."""
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if keyword in cypher:
            # batch may be positional or keyword
            kw = c[1] if len(c) > 1 else {}
            return kw.get("batch") or (c[0][1] if len(c[0]) > 1 else None)
    return None


# ---------------------------------------------------------------------------
# G1 — HAS_ARGUMENT: two names in one batch
# ---------------------------------------------------------------------------


class TestG1:
    """G1: Write temperature and maximum_of_temperature in one batch.

    ``(maximum_of_temperature)-[:HAS_ARGUMENT {operator:'maximum'}]->(temperature)``
    """

    def test_has_argument_edge_emitted(self) -> None:
        names = [
            {"id": "temperature", "unit": "eV"},
            {"id": "maximum_of_temperature", "unit": "eV"},
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        cyphers = _cyphers(mock_gc)
        assert any("HAS_ARGUMENT" in c for c in cyphers), (
            "No HAS_ARGUMENT Cypher found in queries"
        )

    def test_has_argument_batch_contains_correct_edge(self) -> None:
        names = [
            {"id": "temperature", "unit": "eV"},
            {"id": "maximum_of_temperature", "unit": "eV"},
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ARGUMENT")
        assert batch is not None, "No HAS_ARGUMENT batch found"
        edge = next(
            (
                b
                for b in batch
                if b["from_name"] == "maximum_of_temperature"
                and b["to_name"] == "temperature"
            ),
            None,
        )
        assert edge is not None, f"Expected edge not in batch: {batch}"
        assert edge["operator"] == "maximum"
        assert edge["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# G2 — HAS_ARGUMENT: forward reference (target written later)
# ---------------------------------------------------------------------------


class TestG2:
    """G2: Write maximum_of_temperature first, temperature in a later batch.

    After the second batch the edge must still be present (MERGE idempotent
    on re-run of the same name pair).
    """

    def test_forward_ref_edge_present_after_second_batch(self) -> None:
        first_batch = [{"id": "maximum_of_temperature", "unit": "eV"}]
        second_batch = [{"id": "temperature", "unit": "eV"}]

        gc1 = _make_mock_gc()
        _call_write(first_batch, gc1)

        gc2 = _make_mock_gc()
        _call_write(second_batch, gc2)

        # First batch: HAS_ARGUMENT should reference temperature as placeholder
        batch1 = _batch_for(gc1, "HAS_ARGUMENT")
        assert batch1 is not None
        assert any(b["to_name"] == "temperature" for b in batch1)

        # Second batch: temperature is a leaf, no HAS_ARGUMENT emitted
        cyphers2 = _cyphers(gc2)
        # No HAS_ARGUMENT from temperature (it's a leaf)
        ha_batches2 = [
            _batch_for(gc2, "HAS_ARGUMENT") for c in cyphers2 if "HAS_ARGUMENT" in c
        ]
        # If HAS_ARGUMENT present, it must not have temperature as from_name
        if ha_batches2 and ha_batches2[0]:
            assert not any(b["from_name"] == "temperature" for b in ha_batches2[0])

    def test_target_merged_via_merge_clause(self) -> None:
        """The HAS_ARGUMENT Cypher must MERGE the target node (forward ref)."""
        names = [{"id": "maximum_of_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        for c in _cyphers(mock_gc):
            if "HAS_ARGUMENT" in c:
                assert "MERGE" in c, "Target must be MERGEd for forward-ref support"
                break


# ---------------------------------------------------------------------------
# G3 — HAS_ERROR: uncertainty sibling, inverted direction
# ---------------------------------------------------------------------------


class TestG3:
    """G3: Write upper_uncertainty_of_temperature alone.

    ``(temperature)-[:HAS_ERROR {error_type:'upper'}]->(upper_uncertainty_of_temperature)``
    """

    def test_has_error_edge_emitted(self) -> None:
        names = [{"id": "upper_uncertainty_of_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_ERROR" in c for c in _cyphers(mock_gc))

    def test_has_error_direction_inverted(self) -> None:
        """from_name is temperature (inner), to_name is the uncertainty form."""
        names = [{"id": "upper_uncertainty_of_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ERROR")
        assert batch is not None
        edge = next(
            (
                b
                for b in batch
                if b.get("to_name") == "upper_uncertainty_of_temperature"
            ),
            None,
        )
        assert edge is not None, f"HAS_ERROR edge not found in batch: {batch}"
        assert edge["from_name"] == "temperature"
        assert edge["error_type"] == "upper"


# ---------------------------------------------------------------------------
# G4 — HAS_ARGUMENT: binary operator, two edges with role a/b
# ---------------------------------------------------------------------------


class TestG4:
    """G4: Write ratio_of_temperature_to_pressure.

    Two HAS_ARGUMENT edges: → temperature (role=a) and → pressure (role=b).
    """

    def test_two_has_argument_edges(self) -> None:
        names = [{"id": "ratio_of_temperature_to_pressure"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ARGUMENT")
        assert batch is not None

        edges = [
            b for b in batch if b["from_name"] == "ratio_of_temperature_to_pressure"
        ]
        assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}: {edges}"

        roles = {e["role"] for e in edges}
        assert roles == {"a", "b"}

    def test_binary_targets_correct(self) -> None:
        names = [{"id": "ratio_of_temperature_to_pressure"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ARGUMENT")
        assert batch is not None

        edges = {
            e["role"]: e
            for e in batch
            if e["from_name"] == "ratio_of_temperature_to_pressure"
        }
        assert edges["a"]["to_name"] == "temperature"
        assert edges["b"]["to_name"] == "pressure"
        assert edges["a"]["operator"] == "ratio"
        assert edges["a"]["operator_kind"] == "binary"


# ---------------------------------------------------------------------------
# G5 — idempotency: writing the same batch twice
# ---------------------------------------------------------------------------


class TestG5:
    """G5: Write same batch twice → edge Cypher is MERGE-based (idempotent)."""

    def test_has_argument_cypher_uses_merge(self) -> None:
        names = [{"id": "maximum_of_temperature", "unit": "eV"}]

        for _ in range(2):
            mock_gc = _make_mock_gc()
            _call_write(names, mock_gc)

            for c in _cyphers(mock_gc):
                if "HAS_ARGUMENT" in c:
                    assert "MERGE" in c, (
                        "HAS_ARGUMENT Cypher must use MERGE for idempotency"
                    )
                    break

    def test_has_error_cypher_uses_merge(self) -> None:
        names = [{"id": "upper_uncertainty_of_temperature"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        for c in _cyphers(mock_gc):
            if "HAS_ERROR" in c:
                assert "MERGE" in c
                break


# ---------------------------------------------------------------------------
# G6 — catalog import parity
# ---------------------------------------------------------------------------


class TestG6:
    """G6: Catalog import of a file containing the same names as G1.

    Same HAS_ARGUMENT edge as G1 — import path has pipeline parity.
    """

    def test_catalog_import_emits_has_argument(self) -> None:
        entries = [
            {
                "id": "temperature",
                "unit": "eV",
                "physics_domain": "core_plasma_physics",
            },
            {
                "id": "maximum_of_temperature",
                "unit": "eV",
                "physics_domain": "core_plasma_physics",
            },
        ]
        mock_gc = _make_mock_gc()
        _call_import_entries(entries, mock_gc)

        cyphers = _cyphers(mock_gc)
        assert any("HAS_ARGUMENT" in c for c in cyphers), (
            "Catalog import must emit HAS_ARGUMENT edges (pipeline parity)"
        )

    def test_catalog_import_has_argument_batch(self) -> None:
        entries = [
            {"id": "temperature", "unit": "eV"},
            {"id": "maximum_of_temperature", "unit": "eV"},
        ]
        mock_gc = _make_mock_gc()
        _call_import_entries(entries, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ARGUMENT")
        assert batch is not None
        edge = next(
            (
                b
                for b in batch
                if b["from_name"] == "maximum_of_temperature"
                and b["to_name"] == "temperature"
            ),
            None,
        )
        assert edge is not None
        assert edge["operator"] == "maximum"


# ---------------------------------------------------------------------------
# G7 — HAS_PREDECESSOR from deprecates field
# ---------------------------------------------------------------------------


class TestG7:
    """G7: Write StandardName with ``deprecates`` field → HAS_PREDECESSOR edge."""

    def test_has_predecessor_edge_pipeline(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "deprecates": "temperature_of_electrons",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_PREDECESSOR" in c for c in _cyphers(mock_gc))

    def test_has_predecessor_batch_pipeline(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "deprecates": "temperature_of_electrons",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PREDECESSOR")
        assert batch is not None
        assert any(
            b["from_name"] == "electron_temperature"
            and b["to_name"] == "temperature_of_electrons"
            for b in batch
        )

    def test_has_predecessor_edge_catalog(self) -> None:
        entries = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "deprecates": "temperature_of_electrons",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_import_entries(entries, mock_gc)

        assert any("HAS_PREDECESSOR" in c for c in _cyphers(mock_gc))


# ---------------------------------------------------------------------------
# G8 — HAS_SUCCESSOR from superseded_by field
# ---------------------------------------------------------------------------


class TestG8:
    """G8: Write StandardName with ``superseded_by`` field → HAS_SUCCESSOR edge."""

    def test_has_successor_edge_pipeline(self) -> None:
        names = [
            {
                "id": "temperature_of_electrons",
                "unit": "eV",
                "superseded_by": "electron_temperature",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_SUCCESSOR" in c for c in _cyphers(mock_gc))

    def test_has_successor_batch_pipeline(self) -> None:
        names = [
            {
                "id": "temperature_of_electrons",
                "unit": "eV",
                "superseded_by": "electron_temperature",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_SUCCESSOR")
        assert batch is not None
        assert any(
            b["from_name"] == "temperature_of_electrons"
            and b["to_name"] == "electron_temperature"
            for b in batch
        )

    def test_has_successor_edge_catalog(self) -> None:
        entries = [
            {
                "id": "temperature_of_electrons",
                "unit": "eV",
                "superseded_by": "electron_temperature",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_import_entries(entries, mock_gc)

        assert any("HAS_SUCCESSOR" in c for c in _cyphers(mock_gc))


# ---------------------------------------------------------------------------
# G9 — IN_CLUSTER from primary_cluster_id
# ---------------------------------------------------------------------------


class TestG9:
    """G9: Write StandardName with ``primary_cluster_id`` → IN_CLUSTER edge."""

    def test_in_cluster_edge_emitted(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "primary_cluster_id": "cluster:electron_temperature_global",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("IN_CLUSTER" in c for c in _cyphers(mock_gc))

    def test_in_cluster_batch(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "primary_cluster_id": "cluster:electron_temperature_global",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "IN_CLUSTER")
        assert batch is not None
        assert any(
            b["sn_id"] == "electron_temperature"
            and b["cluster_id"] == "cluster:electron_temperature_global"
            for b in batch
        )

    def test_no_in_cluster_when_no_cluster_id(self) -> None:
        """No IN_CLUSTER Cypher emitted when primary_cluster_id is absent."""
        names = [{"id": "electron_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert not any("IN_CLUSTER" in c for c in _cyphers(mock_gc))


# ---------------------------------------------------------------------------
# G10 — HAS_PHYSICS_DOMAIN from physics_domain scalar
# ---------------------------------------------------------------------------


class TestG10:
    """G10: Write StandardName with ``physics_domain='equilibrium'``
    → HAS_PHYSICS_DOMAIN edge to singleton PhysicsDomain {id:'equilibrium'}.
    """

    def test_has_physics_domain_edge_emitted(self) -> None:
        names = [
            {
                "id": "plasma_current",
                "unit": "A",
                "physics_domain": "equilibrium",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_PHYSICS_DOMAIN" in c for c in _cyphers(mock_gc))

    def test_has_physics_domain_batch(self) -> None:
        names = [
            {
                "id": "plasma_current",
                "unit": "A",
                "physics_domain": "equilibrium",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PHYSICS_DOMAIN")
        assert batch is not None
        assert any(
            b["sn_id"] == "plasma_current" and b["domain_id"] == "equilibrium"
            for b in batch
        )

    def test_has_physics_domain_catalog_import(self) -> None:
        entries = [
            {
                "id": "plasma_current",
                "unit": "A",
                "physics_domain": "equilibrium",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_import_entries(entries, mock_gc)

        assert any("HAS_PHYSICS_DOMAIN" in c for c in _cyphers(mock_gc))

    def test_no_has_physics_domain_when_absent(self) -> None:
        """No HAS_PHYSICS_DOMAIN Cypher when physics_domain is absent."""
        names = [{"id": "plasma_current", "unit": "A"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert not any("HAS_PHYSICS_DOMAIN" in c for c in _cyphers(mock_gc))
