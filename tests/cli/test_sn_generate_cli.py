"""Tests for ``imas-codex sn generate`` CLI flags.

Validates:
- ``sn reset`` is removed (no such command)
- ``--reset-only`` with ``--reset-to`` exits early after reset
- ``--reset-only`` without ``--reset-to`` raises UsageError
- New filter flags (``--since``, ``--before``, ``--below-score``, ``--tier``,
  ``--retry-quarantined``) plumb through to backing functions
- Backward compatibility: ``--reset-to drafted --dry-run`` works unchanged
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture
def runner():
    return CliRunner()


class TestSnResetRemoved:
    """Confirm sn reset is no longer a valid command."""

    def test_reset_command_removed(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["reset", "--help"])
        assert result.exit_code != 0
        assert "No such command 'reset'" in result.output or "Error" in result.output


class TestGenerateHelpShowsNewFlags:
    """Confirm --help shows all new flags."""

    def test_help_shows_new_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["generate", "--help"])
        assert result.exit_code == 0
        for flag in [
            "--reset-only",
            "--since",
            "--before",
            "--below-score",
            "--tier",
            "--retry-quarantined",
            "--retry-skipped",
            "--retry-vocab-gap",
            "--include-review-feedback",
        ]:
            assert flag in result.output, f"Missing flag {flag} in help output"


class TestResetOnly:
    """Test --reset-only behavior."""

    def test_reset_only_without_reset_to_raises(self, runner: CliRunner) -> None:
        """--reset-only without --reset-to should error."""
        result = runner.invoke(sn, ["generate", "--reset-only"])
        assert result.exit_code != 0
        assert "--reset-only requires --reset-to" in result.output

    def test_reset_only_with_reset_to_drafted(self, runner: CliRunner) -> None:
        """--reset-only --reset-to drafted should call reset then exit early."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query = MagicMock(return_value=[{"n": 3}])
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            result = runner.invoke(
                sn, ["generate", "--reset-to", "drafted", "--reset-only"]
            )

        assert result.exit_code == 0
        assert "--reset-only:" in result.output
        assert "reset complete, exiting without generation" in result.output

    def test_reset_only_with_reset_to_extracted(self, runner: CliRunner) -> None:
        """--reset-only --reset-to extracted should call clear then exit early."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query = MagicMock(return_value=[{"n": 2}])
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            result = runner.invoke(
                sn, ["generate", "--reset-to", "extracted", "--reset-only"]
            )

        assert result.exit_code == 0
        assert "--reset-only:" in result.output


class TestFilterPlumbing:
    """Test that new filter flags plumb through to reset/clear functions."""

    def test_since_plumbs_to_reset(self, runner: CliRunner) -> None:
        """--since should be passed to reset_standard_names."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=5,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--since",
                    "2026-04-19T10:00",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0
        mock_reset.assert_called_once()
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["since"] == "2026-04-19T10:00"

    def test_below_score_and_tier_plumb_to_reset(self, runner: CliRunner) -> None:
        """--below-score and --tier should be passed to reset_standard_names."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=3,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--below-score",
                    "0.6",
                    "--tier",
                    "poor,adequate",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0
        mock_reset.assert_called_once()
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["below_score"] == 0.6
        assert call_kwargs["tiers"] == ["poor", "adequate"]

    def test_retry_quarantined_sets_validation_status(self, runner: CliRunner) -> None:
        """--retry-quarantined should set validation_status='quarantined'."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=2,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--retry-quarantined",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["validation_status"] == "quarantined"

    def test_include_review_feedback_implies_needs_revision(
        self, runner: CliRunner
    ) -> None:
        """--include-review-feedback (without --tier) selects needs_revision names."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=3,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--include-review-feedback",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0, result.output
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["validation_status"] == "needs_revision"

    def test_explicit_tier_overrides_include_review_feedback(
        self, runner: CliRunner
    ) -> None:
        """Explicit --tier takes precedence; implicit needs_revision NOT set."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=1,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--include-review-feedback",
                    "--tier",
                    "poor",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0, result.output
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["validation_status"] is None
        assert call_kwargs["tiers"] == ["poor"]

    def test_retry_quarantined_wins_over_include_review_feedback(
        self, runner: CliRunner
    ) -> None:
        """--retry-quarantined takes precedence over --include-review-feedback."""
        with patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names",
            return_value=1,
        ) as mock_reset:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "drafted",
                    "--include-review-feedback",
                    "--retry-quarantined",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0, result.output
        call_kwargs = mock_reset.call_args[1]
        assert call_kwargs["validation_status"] == "quarantined"

    def test_before_plumbs_to_clear(self, runner: CliRunner) -> None:
        """--before should be passed to clear_standard_names when reset-to=extracted."""
        with patch(
            "imas_codex.standard_names.graph_ops.clear_standard_names",
            return_value=4,
        ) as mock_clear:
            result = runner.invoke(
                sn,
                [
                    "generate",
                    "--reset-to",
                    "extracted",
                    "--before",
                    "2026-05-01",
                    "--reset-only",
                ],
            )
        assert result.exit_code == 0
        mock_clear.assert_called_once()
        call_kwargs = mock_clear.call_args[1]
        assert call_kwargs["before"] == "2026-05-01"


class TestBackwardCompatibility:
    """Ensure existing --reset-to --dry-run workflow still works."""

    def test_reset_to_drafted_dry_run(self, runner: CliRunner) -> None:
        """--reset-to drafted --dry-run should not call reset (dry_run skips reset)."""
        # dry_run skips the reset block entirely, so no graph calls are made
        # before reaching the main pipeline — which we also need to mock
        with (
            patch("imas_codex.discovery.base.llm.set_litellm_offline_env"),
            patch(
                "imas_codex.cli.discover.common.use_rich_output",
                return_value=False,
            ),
            patch(
                "imas_codex.cli.discover.common.setup_logging",
                return_value=None,
            ),
            patch(
                "imas_codex.cli.discover.common.run_discovery",
                return_value={"extract_count": 0},
            ),
            patch(
                "imas_codex.cli.discover.common.DiscoveryConfig",
            ),
        ):
            result = runner.invoke(
                sn,
                ["generate", "--reset-to", "drafted", "--dry-run"],
            )
        # Should not crash — the pipeline may print stats or not, but exit 0
        assert result.exit_code == 0, result.output
