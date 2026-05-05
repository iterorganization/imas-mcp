"""Tests for ``sn export`` CLI verb.

Mocks ``run_export`` to avoid graph access, verifying that the CLI:
  - requires ``--staging``
  - forwards flags correctly to ``run_export``
  - renders summary tables / gate results
  - exits 1 on gate failure, 2 on precondition error, 3 on internal error
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.export import ExportReport, GateResult


def _success_report(**overrides) -> ExportReport:
    """Return an ExportReport that passes all gates."""
    report = ExportReport(
        total_candidates=10,
        exported_count=8,
        excluded_below_score=1,
        excluded_unreviewed=1,
        excluded_by_domain=0,
        gate_failures=0,
        all_gates_passed=True,
        gate_results=[
            GateResult(gate="A", passed=True),
            GateResult(gate="B", passed=True),
            GateResult(gate="C", passed=True),
            GateResult(gate="D", passed=True),
        ],
    )
    for k, v in overrides.items():
        setattr(report, k, v)
    return report


def _failing_report(**overrides) -> ExportReport:
    """Return an ExportReport with at least one gate failure."""
    report = ExportReport(
        total_candidates=5,
        exported_count=0,
        all_gates_passed=False,
        gate_failures=1,
        gate_results=[
            GateResult(gate="A", passed=True),
            GateResult(gate="B", passed=False, issues=[{"msg": "bad"}]),
        ],
    )
    for k, v in overrides.items():
        setattr(report, k, v)
    return report


MOCK_TARGET = "imas_codex.standard_names.export.run_export"


class TestExportMissingArgs:
    """Verify required flags."""

    def test_staging_required(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["export"])
        assert result.exit_code != 0
        assert "staging" in result.output.lower() or "Missing" in result.output


class TestExportSuccess:
    """Successful export with default flags."""

    @patch(MOCK_TARGET)
    def test_exit_zero(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert result.exit_code == 0, result.output
        assert "Export complete" in result.output

    @patch(MOCK_TARGET)
    def test_default_min_score(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        _, kwargs = mock_export.call_args
        assert kwargs["min_score"] == 0.65

    @patch(MOCK_TARGET)
    def test_custom_min_score(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", "/tmp/stg", "--min-score", "0.8"])
        _, kwargs = mock_export.call_args
        assert kwargs["min_score"] == 0.8

    @patch(MOCK_TARGET)
    def test_domain_forwarded(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(
            sn, ["export", "--staging", "/tmp/stg", "--domain", "equilibrium"]
        )
        _, kwargs = mock_export.call_args
        assert kwargs["domain"] == "equilibrium"

    @patch(MOCK_TARGET)
    def test_gate_only_forwarded(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg", "--gate-only"])
        assert result.exit_code == 0
        _, kwargs = mock_export.call_args
        assert kwargs["gate_only"] is True

    @patch(MOCK_TARGET)
    def test_include_unreviewed_forwarded(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", "/tmp/stg", "--include-unreviewed"])
        _, kwargs = mock_export.call_args
        assert kwargs["include_unreviewed"] is True

    @patch(MOCK_TARGET)
    def test_gate_scope_forwarded(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", "/tmp/stg", "--gate-scope", "b"])
        _, kwargs = mock_export.call_args
        assert kwargs["gate_scope"] == "b"

    @patch(MOCK_TARGET)
    def test_override_edits_forwarded_as_list(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(
            sn,
            [
                "export",
                "--staging",
                "/tmp/stg",
                "--override-edits",
                "foo",
                "--override-edits",
                "bar",
            ],
        )
        _, kwargs = mock_export.call_args
        assert kwargs["override_edits"] == ["foo", "bar"]

    @patch(MOCK_TARGET)
    def test_override_edits_none_when_not_provided(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        _, kwargs = mock_export.call_args
        assert kwargs["override_edits"] is None

    @patch(MOCK_TARGET)
    def test_staging_passed_as_path(self, mock_export, tmp_path):
        mock_export.return_value = _success_report()
        staging = tmp_path / "staging"
        runner = CliRunner()
        runner.invoke(sn, ["export", "--staging", str(staging)])
        _, kwargs = mock_export.call_args
        assert kwargs["staging_dir"] == staging


class TestExportGateFailure:
    """Gate failures should exit 1."""

    @patch(MOCK_TARGET)
    def test_exit_one_on_gate_failure(self, mock_export):
        mock_export.return_value = _failing_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert result.exit_code == 1

    @patch(MOCK_TARGET)
    def test_failure_message(self, mock_export):
        mock_export.return_value = _failing_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert "gate failure" in result.output.lower() or "FAIL" in result.output


class TestExportPreconditionError:
    """FileExistsError maps to exit 2."""

    @patch(MOCK_TARGET, side_effect=FileExistsError("staging dir not empty"))
    def test_exit_two_on_file_exists(self, mock_export):
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert result.exit_code == 2
        assert "Precondition" in result.output or "--force" in result.output


class TestExportInternalError:
    """Unexpected errors map to exit 3."""

    @patch(MOCK_TARGET, side_effect=RuntimeError("kaboom"))
    def test_exit_three_on_internal(self, mock_export):
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert result.exit_code == 3
        assert "kaboom" in result.output


class TestExportSummaryOutput:
    """Summary table renders correct metrics."""

    @patch(MOCK_TARGET)
    def test_exported_count_in_output(self, mock_export):
        mock_export.return_value = _success_report(exported_count=42)
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert "42" in result.output

    @patch(MOCK_TARGET)
    def test_gate_results_in_output(self, mock_export):
        mock_export.return_value = _success_report()
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--staging", "/tmp/stg"])
        assert "Gate Results" in result.output or "PASS" in result.output
