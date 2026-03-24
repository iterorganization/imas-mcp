"""Unit tests for imas_codex.graph.ghcr module.

Tests for GHCR helper functions that were extracted from graph_cli.py.
All subprocess calls and filesystem access are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from imas_codex.graph.ghcr import (
    get_git_info,
    get_package_name,
    get_registry,
    get_version_tag,
    next_dev_revision,
    resolve_token,
)

# ============================================================================
# get_git_info
# ============================================================================


class TestGetGitInfo:
    """Tests for get_git_info()."""

    def test_parses_commit(self, monkeypatch):
        """Commit hash is extracted from git rev-parse output."""
        from subprocess import CompletedProcess

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                return CompletedProcess(cmd, 0, stdout="abc1234def5678\n", stderr="")
            if cmd[:2] == ["git", "describe"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            if cmd[:2] == ["git", "status"]:
                return CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:2] == ["git", "remote"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            return CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("imas_codex.graph.ghcr.subprocess.run", mock_run)
        info = get_git_info()
        assert info["commit"] == "abc1234def5678"
        assert info["commit_short"] == "abc1234"

    def test_detects_tag(self, monkeypatch):
        """Git tag is extracted from describe output."""
        from subprocess import CompletedProcess

        def mock_run(cmd, **kwargs):
            if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                return CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
            if cmd[:2] == ["git", "describe"]:
                return CompletedProcess(cmd, 0, stdout="v1.2.3\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:2] == ["git", "remote"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            return CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("imas_codex.graph.ghcr.subprocess.run", mock_run)
        info = get_git_info()
        assert info["tag"] == "v1.2.3"

    def test_detects_dirty(self, monkeypatch):
        """Dirty working tree is detected from git status output."""
        from subprocess import CompletedProcess

        def mock_run(cmd, **kwargs):
            if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                return CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
            if cmd[:2] == ["git", "describe"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            if cmd[:2] == ["git", "status"]:
                return CompletedProcess(cmd, 0, stdout=" M file.py\n", stderr="")
            if cmd[:2] == ["git", "remote"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            return CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("imas_codex.graph.ghcr.subprocess.run", mock_run)
        info = get_git_info()
        assert info["is_dirty"] is True

    def test_detects_fork(self, monkeypatch):
        """Fork is detected when remote owner differs from iterorganization."""
        from subprocess import CompletedProcess

        def mock_run(cmd, **kwargs):
            if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                return CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
            if cmd[:2] == ["git", "describe"]:
                return CompletedProcess(cmd, 1, stdout="", stderr="")
            if cmd[:2] == ["git", "status"]:
                return CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:2] == ["git", "remote"]:
                return CompletedProcess(
                    cmd,
                    0,
                    stdout="https://github.com/myuser/imas-codex.git\n",
                    stderr="",
                )
            return CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("imas_codex.graph.ghcr.subprocess.run", mock_run)
        info = get_git_info()
        assert info["is_fork"] is True
        assert info["remote_owner"] == "myuser"


# ============================================================================
# get_registry
# ============================================================================


class TestGetRegistry:
    """Tests for get_registry()."""

    def test_default_registry(self):
        info = {"is_fork": False, "remote_owner": "iterorganization"}
        assert get_registry(info) == "ghcr.io/iterorganization"

    def test_fork_registry(self):
        info = {"is_fork": True, "remote_owner": "myuser"}
        assert get_registry(info) == "ghcr.io/myuser"

    def test_force_override(self):
        info = {"is_fork": False, "remote_owner": "iterorganization"}
        assert get_registry(info, "ghcr.io/custom") == "ghcr.io/custom"


# ============================================================================
# get_version_tag
# ============================================================================


class TestGetVersionTag:
    """Tests for get_version_tag()."""

    def test_dev_returns_revision_format(self, monkeypatch):
        monkeypatch.setattr("imas_codex.graph.ghcr.__version__", "0.5.0.dev123")
        monkeypatch.setattr(
            "imas_codex.graph.ghcr.get_local_graph_manifest", lambda: None
        )
        tag = get_version_tag({"tag": None}, dev=True)
        assert tag == "0.5.0.dev123-r1"

    def test_no_tag_no_dev_raises(self):
        with pytest.raises(click.ClickException, match="Not on a git tag"):
            get_version_tag({"tag": None}, dev=False)

    def test_uses_tag_when_present(self):
        tag = get_version_tag({"tag": "v1.0.0"}, dev=False)
        assert tag == "v1.0.0"


# ============================================================================
# get_package_name
# ============================================================================


class TestGetPackageName:
    """Tests for get_package_name()."""

    def test_default(self):
        assert get_package_name() == "imas-codex-graph"

    def test_with_facilities(self):
        assert get_package_name(["tcv", "iter"]) == "imas-codex-graph-iter-tcv"

    def test_dd_only(self):
        assert get_package_name(dd_only=True) == "imas-codex-graph-dd"

    def test_without_dd(self):
        assert get_package_name(without_dd=True) == "imas-codex-graph-without-dd"

    def test_facilities_with_without_dd(self):
        name = get_package_name(["jet"], without_dd=True)
        assert name == "imas-codex-graph-jet-without-dd"


# ============================================================================
# next_dev_revision
# ============================================================================


class TestNextDevRevision:
    """Tests for next_dev_revision()."""

    def test_fresh_returns_1(self, monkeypatch):
        monkeypatch.setattr(
            "imas_codex.graph.ghcr.get_local_graph_manifest", lambda: None
        )
        assert next_dev_revision("0.5.0.dev123") == 1

    def test_increments(self, monkeypatch):
        monkeypatch.setattr(
            "imas_codex.graph.ghcr.get_local_graph_manifest",
            lambda: {"dev_base_version": "0.5.0.dev123", "dev_revision": 3},
        )
        assert next_dev_revision("0.5.0.dev123") == 4

    def test_resets_on_version_change(self, monkeypatch):
        monkeypatch.setattr(
            "imas_codex.graph.ghcr.get_local_graph_manifest",
            lambda: {"dev_base_version": "0.5.0.dev100", "dev_revision": 5},
        )
        assert next_dev_revision("0.5.0.dev123") == 1


# ============================================================================
# resolve_token
# ============================================================================


class TestResolveToken:
    """Tests for resolve_token()."""

    def test_from_argument(self):
        assert resolve_token("my-token") == "my-token"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("GHCR_TOKEN", "env-token")
        assert resolve_token(None) == "env-token"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("GHCR_TOKEN", raising=False)
        from subprocess import CompletedProcess

        monkeypatch.setattr(
            "imas_codex.graph.ghcr.subprocess.run",
            lambda *a, **kw: CompletedProcess(a, 1, stdout="", stderr=""),
        )
        with pytest.raises(click.ClickException, match="No GitHub token found"):
            resolve_token(None)


# ============================================================================
# parse_dump_error (imported from neo4j_ops but tested here for coverage)
# ============================================================================


class TestParseDumpError:
    """Tests for parse_dump_error()."""

    def test_database_in_use(self):
        from imas_codex.graph.neo4j_ops import parse_dump_error

        msg, is_lock = parse_dump_error("Error: database is in use")
        assert is_lock is True
        assert "in use" in msg

    def test_generic_with_caused_by(self):
        from imas_codex.graph.neo4j_ops import parse_dump_error

        stderr = "Some error\nCaused by: java.io.IOException: disk full"
        msg, is_lock = parse_dump_error(stderr)
        assert is_lock is False
        assert "Caused by" in msg
