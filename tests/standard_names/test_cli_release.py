"""Tests for ``sn release`` CLI verb.

Mocks ``catalog_release`` functions to avoid filesystem/git access,
verifying that the CLI forwards flags correctly and handles exit codes.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.catalog_release import ReleaseReport

MOCK_RUN_RELEASE = "imas_codex.standard_names.catalog_release.run_release"
MOCK_GET_STATUS = "imas_codex.standard_names.catalog_release.get_release_status"
MOCK_ISNC_DIR = "imas_codex.settings.get_sn_isnc_dir"


def _success_report(**overrides) -> ReleaseReport:
    report = ReleaseReport(
        version="1.0.0rc1",
        git_tag="v1.0.0rc1",
        remote="origin",
        export_count=42,
        files_copied=21,
        commit_sha="abc123def456",
        pushed=True,
        dry_run=False,
    )
    for k, v in overrides.items():
        setattr(report, k, v)
    return report


def _error_report(errors: list[str] | None = None) -> ReleaseReport:
    return ReleaseReport(
        errors=errors or ["something went wrong"],
    )


class TestReleaseStatus:
    """sn release status subcommand."""

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_GET_STATUS)
    def test_status_displays(self, mock_status, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        mock_status.return_value = {
            "state": "rc",
            "tag": "v1.0.0rc1",
            "major": 1,
            "minor": 0,
            "patch": 0,
            "rc": 1,
            "commits_since": 3,
            "isnc_path": "/tmp/isnc",
            "isn_version": "v0.7.0rc36",
            "remotes": {"origin": "git@github.com:user/isnc.git"},
            "pages_enabled": True,
        }
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "status"])
        assert result.exit_code == 0
        assert "v1.0.0rc1" in result.output
        assert "v0.7.0rc36" in result.output


class TestReleaseMissingMessage:
    """Release without -m should fail."""

    @patch(MOCK_ISNC_DIR)
    def test_no_message_exits(self, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--bump", "minor"])
        assert result.exit_code != 0
        assert "message required" in result.output.lower()


class TestReleaseMissingIsnc:
    """Release without ISNC path should fail."""

    @patch(MOCK_ISNC_DIR, return_value=None)
    def test_no_isnc_exits(self, mock_isnc):
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--bump", "minor", "-m", "test"])
        assert result.exit_code == 2
        assert "ISNC not found" in result.output


class TestReleaseSuccess:
    """Successful release invocation."""

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_RUN_RELEASE)
    def test_exit_zero(self, mock_release, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        mock_release.return_value = _success_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--bump", "minor", "-m", "Test release"])
        assert result.exit_code == 0, result.output
        assert "v1.0.0rc1" in result.output

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_RUN_RELEASE)
    def test_dry_run_forwarded(self, mock_release, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        mock_release.return_value = _success_report(dry_run=True, pushed=False)
        runner = CliRunner()
        result = runner.invoke(
            sn,
            ["release", "--bump", "minor", "-m", "Test", "--dry-run"],
        )
        assert result.exit_code == 0
        _, kwargs = mock_release.call_args
        assert kwargs["dry_run"] is True

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_RUN_RELEASE)
    def test_skip_export_forwarded(self, mock_release, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        mock_release.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(
            sn,
            ["release", "-m", "Test", "--skip-export"],
        )
        _, kwargs = mock_release.call_args
        assert kwargs["skip_export"] is True


class TestReleaseReportErrors:
    """Release with report errors exits 2."""

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_RUN_RELEASE)
    def test_exit_two_on_report_errors(self, mock_release, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        mock_release.return_value = _error_report(["push failed"])
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--bump", "minor", "-m", "Test"])
        assert result.exit_code == 2


class TestReleaseInternalError:
    """Unexpected exceptions exit 3."""

    @patch(MOCK_ISNC_DIR)
    @patch(MOCK_RUN_RELEASE, side_effect=RuntimeError("git exploded"))
    def test_exit_three_on_internal(self, mock_release, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--bump", "minor", "-m", "Test"])
        assert result.exit_code == 3
        assert "git exploded" in result.output


class TestReleaseUnknownAction:
    """Unknown action is rejected."""

    @patch(MOCK_ISNC_DIR)
    def test_bad_action(self, mock_isnc):
        mock_isnc.return_value = "/tmp/isnc"
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "bogus"])
        assert result.exit_code != 0
        assert "Unknown action" in result.output
