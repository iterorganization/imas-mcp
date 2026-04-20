"""Tests for plan-32 Phase 2 prompt tools (variant C tool-calling harness).

These tests exercise the **schema shape** and **dispatcher behaviour** of
``imas_codex.standard_names.prompt_tools``. The live-graph tool
implementations (``fetch_cluster_siblings`` et al.) are covered by the
harness run, not by unit tests — they wrap simple Cypher queries whose
correctness is driven by the graph contents rather than code logic.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names import prompt_tools


class TestToolSchemas:
    """Every tool schema must be litellm/OpenAI-compatible."""

    def test_tools_list_has_three_entries(self):
        assert len(prompt_tools.TOOLS) == 3

    @pytest.mark.parametrize(
        "tool",
        [
            prompt_tools.FETCH_CLUSTER_SIBLINGS_TOOL,
            prompt_tools.FETCH_REFERENCE_EXEMPLAR_TOOL,
            prompt_tools.FETCH_VERSION_HISTORY_TOOL,
        ],
    )
    def test_schema_shape(self, tool):
        assert tool["type"] == "function"
        fn = tool["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and len(fn["description"]) > 20
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert isinstance(params["required"], list)
        for req in params["required"]:
            assert req in params["properties"]

    def test_tool_names_unique(self):
        names = [t["function"]["name"] for t in prompt_tools.TOOLS]
        assert len(names) == len(set(names))


class TestDispatcher:
    """``dispatch_tool_call`` routes by name and rejects unknowns."""

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            prompt_tools.dispatch_tool_call("nonexistent", {})

    def test_cluster_siblings_dispatched(self, monkeypatch):
        called = {}

        def fake(cluster_id, limit=10):
            called["cluster_id"] = cluster_id
            called["limit"] = limit
            return [{"path": "a", "standard_name": "foo"}]

        monkeypatch.setattr(prompt_tools, "fetch_cluster_siblings", fake)
        out = prompt_tools.dispatch_tool_call(
            "fetch_cluster_siblings", {"cluster_id": "c1", "limit": 3}
        )
        assert called == {"cluster_id": "c1", "limit": 3}
        assert out == [{"path": "a", "standard_name": "foo"}]

    def test_reference_exemplar_dispatched(self, monkeypatch):
        called = {}

        def fake(concept):
            called["concept"] = concept
            return []

        monkeypatch.setattr(prompt_tools, "fetch_reference_exemplar", fake)
        prompt_tools.dispatch_tool_call(
            "fetch_reference_exemplar", {"concept": "electron temperature"}
        )
        assert called == {"concept": "electron temperature"}

    def test_version_history_dispatched(self, monkeypatch):
        called = {}

        def fake(path):
            called["path"] = path
            return []

        monkeypatch.setattr(prompt_tools, "fetch_version_history", fake)
        prompt_tools.dispatch_tool_call(
            "fetch_version_history",
            {"path": "equilibrium/time_slice/profiles_1d/psi"},
        )
        assert called == {"path": "equilibrium/time_slice/profiles_1d/psi"}
