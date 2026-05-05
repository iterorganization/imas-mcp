"""Tests for ``sn publish`` CLI verb.

Mocks ``run_publish`` to avoid filesystem/git access, verifying that
the CLI forwards flags correctly and handles exit codes.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.publish import PublishReport

MOCK_TARGET = "imas_codex.standard_names.publish.run_publish"


def _success_report(**overrides) -> PublishReport:
    report = PublishReport(
        staging_dir="/tmp/stg",
        isnc_path="/tmp/isnc",
        files_copied=5,
        commit_sha="abc123def456",
        pushed=False,
        dry_run=False,
    )
    for k, v in overrides.items():
        setattr(report, k, v)
    return report


def _error_report(errors: list[str] | None = None) -> PublishReport:
    return PublishReport(
        staging_dir="/tmp/stg",
        isnc_path="/tmp/isnc",
        files_copied=0,
        errors=errors or ["something went wrong"],
    )


class TestPublishMissingArgs:
    """Verify required flags."""

    def test_staging_required(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["publish"])
        assert result.exit_code != 0

    def test_isnc_required(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("staging")
            result = runner.invoke(sn, ["publish", "--staging", "staging"])
        assert result.exit_code != 0


class TestPublishSuccess:
    """Successful publish invocation."""

    @staticmethod
    def _prepare_staging() -> None:
        """Create staging dir with a minimal catalog.yml (CLI pre-validation)."""
        import os
        from pathlib import Path

        os.makedirs("staging", exist_ok=True)
        Path("staging/catalog.yml").write_text("entries: []\n")

    @patch(MOCK_TARGET)
    def test_exit_zero(self, mock_publish):
        mock_publish.return_value = _success_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            self._prepare_staging()
            os.makedirs("isnc")
            result = runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc"]
            )
        assert result.exit_code == 0, result.output
        assert "Publish complete" in result.output

    @patch(MOCK_TARGET)
    def test_push_forwarded(self, mock_publish):
        mock_publish.return_value = _success_report(pushed=True)
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            self._prepare_staging()
            os.makedirs("isnc")
            runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc", "--push"]
            )
        _, kwargs = mock_publish.call_args
        assert kwargs["push"] is True

    @patch(MOCK_TARGET)
    def test_no_push_default(self, mock_publish):
        mock_publish.return_value = _success_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            self._prepare_staging()
            os.makedirs("isnc")
            runner.invoke(sn, ["publish", "--staging", "staging", "--isnc", "isnc"])
        _, kwargs = mock_publish.call_args
        assert kwargs["push"] is False

    @patch(MOCK_TARGET)
    def test_dry_run_forwarded(self, mock_publish):
        mock_publish.return_value = _success_report(dry_run=True)
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            self._prepare_staging()
            os.makedirs("isnc")
            runner.invoke(
                sn,
                [
                    "publish",
                    "--staging",
                    "staging",
                    "--isnc",
                    "isnc",
                    "--dry-run",
                ],
            )
        _, kwargs = mock_publish.call_args
        assert kwargs["dry_run"] is True


class TestPublishReportErrors:
    """Publish with report errors exits 2."""

    @patch(MOCK_TARGET)
    def test_exit_two_on_report_errors(self, mock_publish):
        mock_publish.return_value = _error_report(["bad file"])
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("staging")
            os.makedirs("isnc")
            result = runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc"]
            )
        assert result.exit_code == 2


class TestPublishInternalError:
    """Unexpected exceptions exit 3."""

    @patch(MOCK_TARGET, side_effect=RuntimeError("git exploded"))
    def test_exit_three_on_internal(self, mock_publish):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            os.makedirs("isnc")
            result = runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc"]
            )
        assert result.exit_code == 3
        assert "git exploded" in result.output


class TestPublishSummaryOutput:
    """Summary table contains expected metrics."""

    @patch(MOCK_TARGET)
    def test_files_copied_in_output(self, mock_publish):
        mock_publish.return_value = _success_report(files_copied=42)
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            os.makedirs("isnc")
            result = runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc"]
            )
        assert "42" in result.output

    @patch(MOCK_TARGET)
    def test_commit_sha_in_output(self, mock_publish):
        mock_publish.return_value = _success_report(commit_sha="deadbeef1234")
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            os.makedirs("isnc")
            result = runner.invoke(
                sn, ["publish", "--staging", "staging", "--isnc", "isnc"]
            )
        assert "deadbeef1234" in result.output
