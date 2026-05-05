"""Tests for ``sn import`` CLI verb.

Mocks ``run_import`` to avoid graph access, verifying that the CLI
forwards flags correctly and handles exit codes.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.catalog_import import ImportReport

MOCK_TARGET = "imas_codex.standard_names.catalog_import.run_import"


def _success_report(**overrides) -> ImportReport:
    report = ImportReport(
        imported=10,
        created=4,
        updated=6,
        skipped=0,
        catalog_commit_sha="abc123def456",
        pr_numbers=[42, 43],
        watermark_advanced=True,
    )
    for k, v in overrides.items():
        setattr(report, k, v)
    return report


def _error_report(errors: list[str] | None = None) -> ImportReport:
    return ImportReport(
        imported=0,
        errors=errors or ["parse error in foo.yml"],
    )


def _dry_run_report() -> ImportReport:
    return ImportReport(
        imported=5,
        created=5,
        dry_run=True,
        entries=[
            {"id": "electron_temperature", "unit": "eV"},
            {"id": "plasma_current", "unit": "A"},
        ],
    )


class TestImportMissingArgs:
    """Verify required flags."""

    def test_isnc_required(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["import"])
        assert result.exit_code != 0
        assert "isnc" in result.output.lower() or "Missing" in result.output


class TestImportSuccess:
    """Successful import invocation."""

    @patch(MOCK_TARGET)
    def test_exit_zero(self, mock_import):
        mock_import.return_value = _success_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert result.exit_code == 0, result.output
        assert "10 entries" in result.output or "Imported" in result.output

    @patch(MOCK_TARGET)
    def test_accept_unit_override_forwarded(self, mock_import):
        mock_import.return_value = _success_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            runner.invoke(sn, ["import", "--isnc", "isnc", "--accept-unit-override"])
        _, kwargs = mock_import.call_args
        assert kwargs["accept_unit_override"] is True

    @patch(MOCK_TARGET)
    def test_override_defaults_false(self, mock_import):
        mock_import.return_value = _success_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            runner.invoke(sn, ["import", "--isnc", "isnc"])
        _, kwargs = mock_import.call_args
        assert kwargs["accept_unit_override"] is False

    @patch(MOCK_TARGET)
    def test_dry_run_forwarded(self, mock_import):
        mock_import.return_value = _dry_run_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            runner.invoke(sn, ["import", "--isnc", "isnc", "--dry-run"])
        _, kwargs = mock_import.call_args
        assert kwargs["dry_run"] is True


class TestImportDryRunOutput:
    """Dry run shows preview entries."""

    @patch(MOCK_TARGET)
    def test_preview_entries_shown(self, mock_import):
        mock_import.return_value = _dry_run_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc", "--dry-run"])
        assert "electron_temperature" in result.output
        assert "plasma_current" in result.output

    @patch(MOCK_TARGET)
    def test_dry_run_says_would_import(self, mock_import):
        mock_import.return_value = _dry_run_report()
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc", "--dry-run"])
        assert "Would import" in result.output


class TestImportReportErrors:
    """Import with report errors exits 2."""

    @patch(MOCK_TARGET)
    def test_exit_two_on_errors(self, mock_import):
        mock_import.return_value = _error_report(["parse error"])
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert result.exit_code == 2

    @patch(MOCK_TARGET)
    def test_errors_printed(self, mock_import):
        mock_import.return_value = _error_report(["bad yaml in core_profiles.yml"])
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert "bad yaml" in result.output


class TestImportInternalError:
    """Unexpected exceptions exit 3."""

    @patch(MOCK_TARGET, side_effect=RuntimeError("neo4j down"))
    def test_exit_three_on_internal(self, mock_import):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert result.exit_code == 3
        assert "neo4j down" in result.output


class TestImportSummaryOutput:
    """Summary table contains expected metrics."""

    @patch(MOCK_TARGET)
    def test_pr_numbers_in_output(self, mock_import):
        mock_import.return_value = _success_report(pr_numbers=[99, 100])
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert "#99" in result.output
        assert "#100" in result.output

    @patch(MOCK_TARGET)
    def test_watermark_in_output(self, mock_import):
        mock_import.return_value = _success_report(watermark_advanced=True)
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert "watermark" in result.output.lower()

    @patch(MOCK_TARGET)
    def test_catalog_sha_in_output(self, mock_import):
        mock_import.return_value = _success_report(catalog_commit_sha="feedfacebeef")
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.makedirs("isnc")
            result = runner.invoke(sn, ["import", "--isnc", "isnc"])
        assert "feedfacebee" in result.output  # truncated to 12 chars
