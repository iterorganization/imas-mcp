"""Tests for the cli/graph/ package structure and command registration."""

from __future__ import annotations

import pytest


class TestGraphCLIPackageRegistration:
    """Verify all graph subcommands register correctly."""

    def test_all_graph_commands_registered(self):
        from imas_codex.cli.graph import graph

        cmds = sorted(graph.commands.keys())
        expected = sorted(
            [
                "clear",
                "export",
                "facility",
                "fetch",
                "init",
                "list",
                "load",
                "profiles",
                "prune",
                "pull",
                "push",
                "repair",
                "secure",
                "shell",
                "start",
                "status",
                "stop",
                "switch",
                "tags",
            ]
        )
        assert cmds == expected

    def test_server_commands_importable(self):
        from imas_codex.cli.graph.server import (
            graph_profiles,
            graph_shell,
            graph_start,
            graph_status,
            graph_stop,
        )

        assert all(
            c is not None
            for c in [
                graph_start,
                graph_stop,
                graph_status,
                graph_shell,
                graph_profiles,
            ]
        )

    def test_data_commands_importable(self):
        from imas_codex.cli.graph.data import (
            graph_clear,
            graph_export,
            graph_facility_group,
            graph_init,
            graph_list,
            graph_load,
            graph_secure,
            graph_switch,
        )

        assert all(
            c is not None
            for c in [
                graph_export,
                graph_load,
                graph_init,
                graph_switch,
                graph_list,
                graph_clear,
                graph_secure,
                graph_facility_group,
            ]
        )

    def test_registry_commands_importable(self):
        from imas_codex.cli.graph.registry import (
            graph_fetch,
            graph_prune,
            graph_pull,
            graph_push,
            graph_tags,
        )

        assert all(
            c is not None
            for c in [graph_push, graph_fetch, graph_pull, graph_tags, graph_prune]
        )

    def test_graph_help_text(self):
        from click.testing import CliRunner

        from imas_codex.cli.graph import graph

        runner = CliRunner()
        result = runner.invoke(graph, ["--help"])
        assert result.exit_code == 0
        assert "Server:" in result.output
        assert "Setup:" in result.output
        assert "Archive & Registry:" in result.output
        assert "Maintenance:" in result.output

    def test_facility_subgroup_has_commands(self):
        from imas_codex.cli.graph.data import graph_facility_group

        cmds = sorted(graph_facility_group.commands.keys())
        assert cmds == ["add", "list", "remove"]
