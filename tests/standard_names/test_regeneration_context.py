"""Tests for --force regeneration context (previous name injection)."""

from __future__ import annotations

import pytest


class TestGetSourceNameMapping:
    """Tests for get_source_name_mapping graph query."""

    def test_returns_mapping(self, monkeypatch):
        """Verify mapping structure from graph query."""
        from imas_codex.standard_names import graph_ops

        mock_results = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "name": "poloidal_magnetic_flux",
                "description": "Poloidal magnetic flux",
                "kind": "scalar",
                "pipeline_status": "drafted",
            },
            {
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "name": "electron_temperature",
                "description": "Electron temperature",
                "kind": "scalar",
                "pipeline_status": "accepted",
            },
        ]

        class MockGC:
            def query(self, *args, **kwargs):
                return mock_results

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(graph_ops, "GraphClient", lambda: MockGC())
        mapping = graph_ops.get_source_name_mapping()

        assert len(mapping) == 2
        assert (
            mapping["equilibrium/time_slice/profiles_1d/psi"]["name"]
            == "poloidal_magnetic_flux"
        )
        assert (
            mapping["core_profiles/profiles_1d/electrons/temperature"][
                "pipeline_status"
            ]
            == "accepted"
        )

    def test_prefers_accepted_over_drafted(self, monkeypatch):
        """When a source has multiple names, prefer the accepted one."""
        from imas_codex.standard_names import graph_ops

        mock_results = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "name": "old_drafted_name",
                "description": "Old",
                "kind": "scalar",
                "pipeline_status": "drafted",
            },
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "name": "accepted_name",
                "description": "Accepted",
                "kind": "scalar",
                "pipeline_status": "accepted",
            },
        ]

        class MockGC:
            def query(self, *args, **kwargs):
                return mock_results

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(graph_ops, "GraphClient", lambda: MockGC())
        mapping = graph_ops.get_source_name_mapping()

        assert (
            mapping["equilibrium/time_slice/profiles_1d/psi"]["name"] == "accepted_name"
        )

    def test_empty_graph(self, monkeypatch):
        """Empty graph returns empty mapping."""
        from imas_codex.standard_names import graph_ops

        class MockGC:
            def query(self, *args, **kwargs):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(graph_ops, "GraphClient", lambda: MockGC())
        assert graph_ops.get_source_name_mapping() == {}


class TestComposePromptPreviousName:
    """Tests for previous_name rendering in compose prompt."""

    def test_renders_previous_name(self):
        """Previous name section appears in rendered prompt."""
        from imas_codex.llm.prompt_loader import render_prompt

        context = {
            "items": [
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi",
                    "description": "Poloidal magnetic flux",
                    "unit": "Wb",
                    "previous_name": {
                        "name": "poloidal_magnetic_flux",
                        "description": "The poloidal magnetic flux",
                        "pipeline_status": "drafted",
                    },
                }
            ],
            "ids_name": "equilibrium",
            "existing_names": [],
            "cluster_context": "",
            "nearby_existing_names": [],
        }
        rendered = render_prompt("sn/compose_dd", context)
        assert "Previous generation:" in rendered
        assert "poloidal_magnetic_flux" in rendered

    def test_accepted_warning(self):
        """Accepted names get a warning in the prompt."""
        from imas_codex.llm.prompt_loader import render_prompt

        context = {
            "items": [
                {
                    "path": "core_profiles/profiles_1d/electrons/temperature",
                    "description": "Electron temperature",
                    "unit": "eV",
                    "previous_name": {
                        "name": "electron_temperature",
                        "description": "Electron temperature",
                        "pipeline_status": "accepted",
                    },
                }
            ],
            "ids_name": "core_profiles",
            "existing_names": [],
            "cluster_context": "",
            "nearby_existing_names": [],
        }
        rendered = render_prompt("sn/compose_dd", context)
        assert "human-accepted" in rendered
        assert "⚠️" in rendered or "only replace" in rendered

    def test_no_previous_name(self):
        """Without previous_name, section is absent."""
        from imas_codex.llm.prompt_loader import render_prompt

        context = {
            "items": [
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi",
                    "description": "Poloidal magnetic flux",
                    "unit": "Wb",
                }
            ],
            "ids_name": "equilibrium",
            "existing_names": [],
            "cluster_context": "",
            "nearby_existing_names": [],
        }
        rendered = render_prompt("sn/compose_dd", context)
        assert "Previous generation:" not in rendered
