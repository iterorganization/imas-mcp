"""Tests for ``sn preview`` CLI verb.

Mocks ``run_preview`` to avoid launching a real server, verifying that
the CLI forwards ``--staging`` and ``--port`` correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.preview import PreviewHandle

MOCK_TARGET = "imas_codex.standard_names.preview.run_preview"


def _make_handle(*, process=None, url="http://localhost:8000") -> PreviewHandle:
    """Return a mock PreviewHandle that terminates immediately."""
    proc = process or MagicMock()
    proc.wait.return_value = 0  # process exits immediately
    return PreviewHandle(process=proc, url=url, staging_dir="/tmp/stg")


class TestPreviewMissingArgs:
    """Verify required staging content."""

    def test_staging_required(self):
        """Without --no-export, preview auto-exports; with --no-export and
        no catalog.yml in the staging dir, preview fails with exit 2."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("empty_staging")
            result = runner.invoke(
                sn, ["preview", "--staging", "empty_staging", "--no-export"]
            )
        assert result.exit_code != 0
        assert (
            "staging" in result.output.lower()
            or "catalog" in result.output.lower()
            or "Missing" in result.output
        )


class TestPreviewSuccess:
    """Successful preview invocation."""

    @staticmethod
    def _prepare_staging() -> None:
        """Create staging dir with a minimal catalog.yml (CLI pre-validation)."""
        import os
        from pathlib import Path

        os.makedirs("staging", exist_ok=True)
        Path("staging/catalog.yml").write_text("entries: []\n")

    @patch(MOCK_TARGET)
    def test_exit_zero(self, mock_preview):
        mock_preview.return_value = _make_handle()
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._prepare_staging()
            result = runner.invoke(
                sn, ["preview", "--staging", "staging", "--no-export"]
            )
        assert result.exit_code == 0, result.output

    @patch(MOCK_TARGET)
    def test_port_forwarded(self, mock_preview):
        mock_preview.return_value = _make_handle()
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._prepare_staging()
            runner.invoke(
                sn,
                [
                    "preview",
                    "--staging",
                    "staging",
                    "--no-export",
                    "--port",
                    "9090",
                ],
            )
        _, kwargs = mock_preview.call_args
        assert kwargs["port"] == 9090

    @patch(MOCK_TARGET)
    def test_url_in_output(self, mock_preview):
        mock_preview.return_value = _make_handle(url="http://localhost:9090")
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._prepare_staging()
            result = runner.invoke(
                sn, ["preview", "--staging", "staging", "--no-export"]
            )
        assert "9090" in result.output


class TestPreviewNullProcess:
    """Handle edge case: process is None."""

    @patch(MOCK_TARGET)
    def test_exit_three_null_process(self, mock_preview):
        mock_preview.return_value = PreviewHandle(
            process=None, url=None, staging_dir="/tmp/stg"
        )
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            result = runner.invoke(
                sn, ["preview", "--staging", "staging", "--no-export"]
            )
        assert result.exit_code == 3


class TestPreviewErrors:
    """Error handling."""

    @patch(MOCK_TARGET, side_effect=FileNotFoundError("isn not installed"))
    def test_exit_two_on_missing_isn(self, mock_preview):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            result = runner.invoke(
                sn, ["preview", "--staging", "staging", "--no-export"]
            )
        assert result.exit_code == 2

    @patch(MOCK_TARGET, side_effect=RuntimeError("unexpected"))
    def test_exit_three_on_internal(self, mock_preview):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            from pathlib import Path

            os.makedirs("staging")
            Path("staging/catalog.yml").write_text("entries: []\n")
            result = runner.invoke(
                sn, ["preview", "--staging", "staging", "--no-export"]
            )
        assert result.exit_code == 3
