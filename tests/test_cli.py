"""Smoke tests for the modular CLI.

These tests verify that all CLI modules import correctly and that
the main command groups are properly registered.
"""

import pytest
from click.testing import CliRunner


class TestCLIImports:
    """Test that all CLI modules can be imported."""

    def test_import_main(self):
        """Main CLI entry point can be imported."""
        from imas_codex.cli import main

        assert main is not None
        assert hasattr(main, "commands")

    def test_import_serve(self):
        """Serve module imports."""
        from imas_codex.cli.serve import serve

        assert serve is not None

    def test_import_facilities(self):
        """Facilities module imports."""
        from imas_codex.cli.facilities import facilities

        assert facilities is not None

    def test_import_tools(self):
        """Tools module imports."""
        from imas_codex.cli.tools import tools

        assert tools is not None

    def test_import_host(self):
        """Host module imports."""
        from imas_codex.cli.host import host

        assert host is not None

    def test_import_graph_cli(self):
        """Graph CLI module imports."""
        from imas_codex.cli.graph import graph

        assert graph is not None

    def test_import_config_cli(self):
        """Config CLI module imports."""
        from imas_codex.cli.config_cli import config

        assert config is not None

    def test_import_release(self):
        """Release module imports."""
        from imas_codex.cli.release import release

        assert release is not None

    def test_import_imas_dd(self):
        """IMAS DD module imports."""
        from imas_codex.cli.imas_dd import imas

        assert imas is not None

    def test_import_discover(self):
        """Discover module imports."""
        from imas_codex.cli.discover import discover

        assert discover is not None

    def test_import_utils(self):
        """Utils module imports with expected symbols."""
        from imas_codex.cli.utils import (
            ProgressReporter,
            console,
            format_duration,
            format_size,
            run_async,
        )

        assert console is not None
        assert callable(format_size)
        assert callable(format_duration)
        assert callable(run_async)
        assert ProgressReporter is not None


class TestCLICommands:
    """Test that CLI commands are properly registered and have help text."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Main CLI shows help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "IMAS Codex" in result.output
        assert "Commands:" in result.output

    def test_main_version(self, runner):
        """--version flag works."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version string is output
        assert result.output.strip()  # Some version text present

    def test_serve_help(self, runner):
        """serve group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "MCP servers" in result.output

    def test_facilities_help(self, runner):
        """facilities group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["facilities", "--help"])
        assert result.exit_code == 0
        assert "Manage facility" in result.output

    def test_discover_help(self, runner):
        """discover group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["discover", "--help"])
        assert result.exit_code == 0
        assert "Discover facility" in result.output

    def test_imas_help(self, runner):
        """imas group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "--help"])
        assert result.exit_code == 0
        assert "IMAS" in result.output

    def test_tools_help(self, runner):
        """tools group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["tools", "--help"])
        assert result.exit_code == 0
        assert "tools" in result.output.lower()

    def test_host_help(self, runner):
        """host group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["host", "--help"])
        assert result.exit_code == 0
        assert "node" in result.output.lower() or "ssh" in result.output.lower()

    def test_host_status_help(self, runner):
        """host status command has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["host", "status", "--help"])
        assert result.exit_code == 0
        assert "ssh" in result.output.lower()

    def test_release_help(self, runner):
        """release command has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["release", "--help"])
        assert result.exit_code == 0
        assert "release" in result.output.lower()


class TestCLISubcommands:
    """Test specific subcommands are registered."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_discover_paths_help(self, runner):
        """discover paths subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["discover", "paths", "--help"])
        assert result.exit_code == 0
        assert "directory structure" in result.output.lower()

    def test_imas_dd_build_help(self, runner):
        """imas dd build subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "dd", "build", "--help"])
        assert result.exit_code == 0

    def test_imas_map_help(self, runner):
        """imas map subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "map", "--help"])
        assert result.exit_code == 0
        assert "IMAS mapping pipeline" in result.output

    def test_imas_list_help(self, runner):
        """imas list subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "list", "--help"])
        assert result.exit_code == 0
        assert "List available IDS recipes" in result.output

    def test_imas_export_help(self, runner):
        """imas export subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "export", "--help"])
        assert result.exit_code == 0

    def test_imas_epochs_help(self, runner):
        """imas epochs subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "epochs", "--help"])
        assert result.exit_code == 0

    def test_facilities_list_help(self, runner):
        """facilities list subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["facilities", "list", "--help"])
        assert result.exit_code == 0

    def test_no_top_level_ids(self, runner):
        """ids is no longer a top-level command."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["ids", "--help"])
        assert result.exit_code != 0

    def test_no_top_level_map(self, runner):
        """map is no longer a top-level command."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["map", "--help"])
        assert result.exit_code != 0


class TestHostCommand:
    """Test the host command and its helpers."""

    def test_get_load_info(self):
        """_get_load_info returns valid system info."""
        from imas_codex.cli.host import _get_load_info

        info = _get_load_info()
        assert "hostname" in info
        assert "load_1m" in info
        assert "cpu_count" in info
        assert info["cpu_count"] >= 1
        assert info["load_1m"] >= 0

    def test_colored_bar(self):
        """_colored_bar renders a bar string."""
        from imas_codex.cli.host import _colored_bar

        bar = _colored_bar(5, 10)
        assert "%" in bar

    def test_format_load_row(self):
        """_format_load_row produces 4-element list."""
        from imas_codex.cli.host import _format_load_row

        info = {
            "hostname": "test-node",
            "load_1m": 2.0,
            "load_5m": 1.5,
            "load_15m": 1.0,
            "cpu_count": 8,
            "mem_total_mb": 16384,
            "mem_used_mb": 8192,
            "users": 5,
        }
        row = _format_load_row(info)
        assert len(row) == 4
        assert row[0] == "test-node"
        assert row[3] == "5"

    def test_parse_ps_output(self):
        """_parse_ps_output filters codex-related processes."""
        from imas_codex.cli.host import _parse_ps_output

        # ps -eo user,pid,%cpu,%mem,vsz,rss,etimes,args
        ps_text = (
            "USER       PID %CPU %MEM    VSZ   RSS ELAPSED COMMAND\n"
            "user     12345  5.2  3.1 512000 32000    3600 python -m imas_codex serve\n"
            "user     12346  0.0  0.1  10000  1000     120 bash\n"
            "user     12347 80.0 15.0 800000 150000  172800 neo4j server\n"
        )
        procs = _parse_ps_output(ps_text)
        assert len(procs) == 2
        assert procs[0]["pid"] == "12345"
        assert procs[0]["age_seconds"] == 3600
        assert procs[1]["pid"] == "12347"
        assert procs[1]["age_seconds"] == 172800

    def test_format_age(self):
        """_format_age renders human-readable durations."""
        from imas_codex.cli.host import _format_age

        assert _format_age(30) == "30s"
        assert _format_age(90) == "1m"
        assert _format_age(3600) == "1h"
        assert _format_age(3720) == "1h 2m"
        assert _format_age(90000) == "1d 1h"
        assert _format_age(86400) == "1d"

    def test_parse_duration(self):
        """_parse_duration converts duration strings to seconds."""
        from imas_codex.cli.host import _parse_duration

        assert _parse_duration("30s") == 30
        assert _parse_duration("5m") == 300
        assert _parse_duration("2h") == 7200
        assert _parse_duration("1d") == 86400
        assert _parse_duration("1d12h") == 129600

    def test_host_runs_local(self):
        """host command (no args) runs successfully."""
        from click.testing import CliRunner

        from imas_codex.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["host"])
        assert result.exit_code == 0
        assert "Node:" in result.output
        assert "CPU:" in result.output

    def test_set_ssh_hostname_roundtrip(self, tmp_path):
        """_set_ssh_hostname correctly modifies SSH config."""
        from unittest.mock import patch

        from imas_codex.cli.host import _get_ssh_hostname, _set_ssh_hostname

        config_text = (
            "Host iter sdcc\n"
            "    User testuser\n"
            "    ProxyJump gateway\n"
            "\n"
            "Host iter\n"
            "    HostName 98dci4-srv-1002.iter.org\n"
            "\n"
            "Host sdcc\n"
            "    HostName 98dci4-srv-1003.iter.org\n"
        )
        fake_ssh = tmp_path / ".ssh"
        fake_ssh.mkdir()
        config_file = fake_ssh / "config"
        config_file.write_text(config_text)

        with patch("imas_codex.cli.host.Path.home", return_value=tmp_path):
            assert _get_ssh_hostname("iter") == "98dci4-srv-1002.iter.org"

            ok = _set_ssh_hostname("iter", "98dci4-srv-1005.iter.org")
            assert ok

            assert _get_ssh_hostname("iter") == "98dci4-srv-1005.iter.org"
            # sdcc should be untouched
            assert _get_ssh_hostname("sdcc") == "98dci4-srv-1003.iter.org"

    def test_host_group_routes_facility_to_survey(self):
        """Unknown subcommands are routed to the hidden survey command."""
        from click.testing import CliRunner

        from imas_codex.cli.host import host

        runner = CliRunner()
        # "fakefacility" isn't a real facility, but the routing should
        # inject "survey" and Click should parse it as the survey command
        # with facility="fakefacility". It will fail at get_facility(),
        # but we check it doesn't fail with "No such command".
        result = runner.invoke(host, ["fakefacility"])
        assert "No such command" not in (result.output or "")

    def test_host_status_is_subcommand(self):
        """'status' is dispatched as a subcommand, not a facility."""
        from click.testing import CliRunner

        from imas_codex.cli.host import host

        runner = CliRunner()
        result = runner.invoke(host, ["status", "--help"])
        assert result.exit_code == 0
        assert "SSH connectivity" in result.output

    def test_build_process_table_single_node(self):
        """_build_process_table renders single-node without Node column."""
        from imas_codex.cli.host import _build_process_table

        procs = {
            "test-node": [
                {
                    "pid": "100",
                    "cpu": "5.0",
                    "mem": "2.0",
                    "rss_mb": 50,
                    "age_seconds": 3600,
                    "command": "python serve",
                },
            ]
        }
        table = _build_process_table(procs)
        assert table is not None
        # Single node → no Node column: PID, CPU%, Mem%, RSS, Age, Command = 6
        assert len(table.columns) == 6

    def test_build_process_table_multi_node(self):
        """_build_process_table adds Node column for multiple nodes."""
        from imas_codex.cli.host import _build_process_table

        procs = {
            "node-a": [
                {
                    "pid": "1",
                    "cpu": "1.0",
                    "mem": "0.5",
                    "rss_mb": 10,
                    "age_seconds": 60,
                    "command": "cmd a",
                }
            ],
            "node-b": [
                {
                    "pid": "2",
                    "cpu": "2.0",
                    "mem": "1.0",
                    "rss_mb": 20,
                    "age_seconds": 120,
                    "command": "cmd b",
                }
            ],
        }
        table = _build_process_table(procs)
        assert table is not None
        # Multi-node → Node + PID + CPU% + Mem% + RSS + Age + Command = 7
        assert len(table.columns) == 7

    def test_build_process_table_empty(self):
        """_build_process_table returns None when no processes."""
        from imas_codex.cli.host import _build_process_table

        assert _build_process_table({}) is None
        assert _build_process_table({"node": []}) is None

    def test_host_kill_help(self):
        """kill subcommand shows help."""
        from click.testing import CliRunner

        from imas_codex.cli.host import host

        runner = CliRunner()
        result = runner.invoke(host, ["kill", "--help"])
        assert result.exit_code == 0
        assert "--include" in result.output
        assert "--exclude" in result.output
        assert "--signal" in result.output
        assert "--older-than" in result.output

    def test_build_survey_table(self):
        """_build_survey_table builds table and finds best node."""
        from imas_codex.cli.host import _build_survey_table

        results = {
            "node-a": {
                "hostname": "node-a.iter.org",
                "load_1m": 8.0,
                "load_5m": 7.0,
                "load_15m": 6.0,
                "cpu_count": 16,
                "mem_total_mb": 16384,
                "mem_used_mb": 8192,
                "users": 5,
                "codex_procs": [],
            },
            "node-b": {
                "hostname": "node-b.iter.org",
                "load_1m": 1.0,
                "load_5m": 0.5,
                "load_15m": 0.3,
                "cpu_count": 16,
                "mem_total_mb": 16384,
                "mem_used_mb": 4096,
                "users": 2,
                "codex_procs": [],
            },
            "node-c": None,
        }
        table, best_node, best_load = _build_survey_table(results, "test", None)
        assert best_node == "node-b"
        assert best_load < 10  # ~6.25%
        assert len(table.columns) == 6
