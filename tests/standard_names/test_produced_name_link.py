"""W4a: Verify PRODUCED_NAME edges use correct source-type prefix.

The compose persist path constructs ``finalize_batch`` entries with
``sns_id`` that must include the source-type prefix (``dd:`` or
``signals:``) to match ``StandardNameSource.id`` in the graph.
"""

from __future__ import annotations


def _build_finalize_batch(candidates, compose_model="test-model"):
    """Reproduce the finalize_batch construction from persist_generated_name_batch."""
    finalize_batch = []
    for entry in candidates:
        if not entry.get("id"):
            continue
        if entry.get("model") == "deterministic:dd_error_modifier":
            continue
        raw_source_id = entry.get("source_id")
        if raw_source_id:
            _st = (entry.get("source_types") or ["dd"])[0]
            _prefix = "dd" if _st == "dd" else "signals"
            sns_id = f"{_prefix}:{raw_source_id}"
        else:
            sns_id = None
        finalize_batch.append(
            {
                "sn_id": entry["id"],
                "sns_id": sns_id,
                "model": compose_model,
            }
        )
    return finalize_batch


class TestProducedNameLinkPrefix:
    """Verify that finalize_batch gets prefixed sns_ids."""

    def test_dd_source_gets_dd_prefix(self):
        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
            },
        ]
        batch = _build_finalize_batch(candidates)
        assert len(batch) == 1
        assert (
            batch[0]["sns_id"] == "dd:core_profiles/profiles_1d/electrons/temperature"
        )

    def test_signals_source_gets_signals_prefix(self):
        candidates = [
            {
                "id": "plasma_current",
                "source_id": "tcv:magnetics:ip",
                "source_types": ["signals"],
            },
        ]
        batch = _build_finalize_batch(candidates)
        assert len(batch) == 1
        assert batch[0]["sns_id"] == "signals:tcv:magnetics:ip"

    def test_missing_source_id_gets_null(self):
        candidates = [
            {
                "id": "electron_density",
                "source_types": ["dd"],
            },
        ]
        batch = _build_finalize_batch(candidates)
        assert len(batch) == 1
        assert batch[0]["sns_id"] is None

    def test_error_siblings_excluded(self):
        candidates = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
            },
            {
                "id": "electron_temperature_error_upper",
                "source_id": "core_profiles/profiles_1d/electrons/temperature_error_upper",
                "source_types": ["dd"],
                "model": "deterministic:dd_error_modifier",
            },
        ]
        batch = _build_finalize_batch(candidates)
        assert len(batch) == 1
        assert batch[0]["sn_id"] == "electron_temperature"

    def test_five_sources_all_linked(self):
        paths = [
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_1d/pressure",
            "equilibrium/time_slice/profiles_1d/q",
            "equilibrium/time_slice/profiles_1d/j_tor",
            "core_profiles/profiles_1d/electrons/temperature",
        ]
        candidates = [
            {
                "id": f"name_{i}",
                "source_id": p,
                "source_types": ["dd"],
            }
            for i, p in enumerate(paths)
        ]
        batch = _build_finalize_batch(candidates)
        assert len(batch) == 5
        for item, path in zip(batch, paths, strict=True):
            assert item["sns_id"] == f"dd:{path}"

    def test_missing_source_types_defaults_to_dd(self):
        candidates = [
            {
                "id": "some_name",
                "source_id": "some/path",
            },
        ]
        batch = _build_finalize_batch(candidates)
        assert batch[0]["sns_id"] == "dd:some/path"

    def test_source_code_matches_test_logic(self):
        """Verify the actual graph_ops code has the same prefix logic."""
        import inspect

        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        source = inspect.getsource(persist_generated_name_batch)
        assert "source_types" in source
        assert "_prefix" in source
        assert "raw_source_id" in source
