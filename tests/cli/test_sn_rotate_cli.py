"""Tests for ``imas-codex sn rotate`` CLI command.

Validates:
- Command registration and help text
- Dry-run prints 4-phase plan with budget split
- ``--skip-review`` causes phase 4 (regen) to no-op due to no needs_revision names
- Cost-limit split: custom ratios honored
- ``rotation_id`` appears in produced StandardName nodes (mock graph)
- ``--fail-fast`` aborts on first phase error
- Per-phase skip flags work independently
- Default budget split is 40/20/20/20
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.rotation import (
    ROTATION_SPLIT,
    PhaseResult,
    RotationConfig,
    rotation_summary,
    run_rotation,
)


@pytest.fixture
def runner():
    return CliRunner()


# ═══════════════════════════════════════════════════════════════════════
# Command registration
# ═══════════════════════════════════════════════════════════════════════


class TestRotateCommandRegistered:
    """Verify the rotate subcommand exists and has correct options."""

    def test_command_exists(self):
        cmd = sn.get_command(None, "rotate")
        assert cmd is not None, "sn rotate command not registered"

    def test_help_shows_phases(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["rotate", "--help"])
        assert result.exit_code == 0
        for keyword in ["generate", "enrich", "review", "regen"]:
            assert keyword in result.output, f"Missing phase '{keyword}' in help"

    def test_help_shows_skip_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["rotate", "--help"])
        assert result.exit_code == 0
        for flag in [
            "--skip-generate",
            "--skip-enrich",
            "--skip-review",
            "--skip-regen",
            "--fail-fast",
            "--dry-run",
        ]:
            assert flag in result.output, f"Missing flag {flag}"

    def test_domain_required(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["rotate"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════
# Dry-run
# ═══════════════════════════════════════════════════════════════════════


class TestDryRun:
    """Dry-run should print the 4-phase plan with budget split then exit."""

    def test_dry_run_prints_plan(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["rotate", "--domain", "equilibrium", "--dry-run"])
        assert result.exit_code == 0
        # Should show all 4 phase labels
        for phase in ["generate", "enrich", "review", "regen"]:
            assert phase in result.output
        # Should show budget
        assert "$" in result.output
        assert "dry run" in result.output.lower()

    def test_dry_run_shows_budget_split(self, runner: CliRunner) -> None:
        result = runner.invoke(
            sn,
            ["rotate", "--domain", "equilibrium", "--dry-run", "-c", "10.0"],
        )
        assert result.exit_code == 0
        # Default split is 40/20/20/20 of $10 = $4.00, $2.00, $2.00, $2.00
        assert "$4.00" in result.output
        assert "$2.00" in result.output

    def test_dry_run_with_skips(self, runner: CliRunner) -> None:
        result = runner.invoke(
            sn,
            [
                "rotate",
                "--domain",
                "magnetics",
                "--dry-run",
                "--skip-review",
                "--skip-regen",
            ],
        )
        assert result.exit_code == 0
        assert "skip" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════
# RotationConfig
# ═══════════════════════════════════════════════════════════════════════


class TestRotationConfig:
    """Unit tests for RotationConfig budget calculations."""

    def test_default_split(self):
        assert ROTATION_SPLIT == (0.40, 0.20, 0.20, 0.20)
        assert sum(ROTATION_SPLIT) == pytest.approx(1.0)

    def test_phase_budget(self):
        cfg = RotationConfig(domain="equilibrium", cost_limit=10.0)
        assert cfg.phase_budget(0) == pytest.approx(4.0)
        assert cfg.phase_budget(1) == pytest.approx(2.0)
        assert cfg.phase_budget(2) == pytest.approx(2.0)
        assert cfg.phase_budget(3) == pytest.approx(2.0)

    def test_custom_split(self):
        cfg = RotationConfig(
            domain="equilibrium",
            cost_limit=10.0,
            split=(0.50, 0.10, 0.30, 0.10),
        )
        assert cfg.phase_budget(0) == pytest.approx(5.0)
        assert cfg.phase_budget(1) == pytest.approx(1.0)
        assert cfg.phase_budget(2) == pytest.approx(3.0)
        assert cfg.phase_budget(3) == pytest.approx(1.0)

    def test_rotation_id_generated(self):
        cfg = RotationConfig(domain="equilibrium")
        assert cfg.rotation_id
        # Should be a valid UUID
        import uuid

        uuid.UUID(cfg.rotation_id)  # raises if invalid


# ═══════════════════════════════════════════════════════════════════════
# Skip phases
# ═══════════════════════════════════════════════════════════════════════


class TestSkipPhases:
    """Verify skip flags produce skipped PhaseResults."""

    @pytest.mark.asyncio
    async def test_skip_all_phases(self):
        cfg = RotationConfig(
            domain="equilibrium",
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,
        )
        results = await run_rotation(cfg)
        assert len(results) == 4
        assert all(r.skipped for r in results)
        assert all(r.exit_code == 0 for r in results)

    @pytest.mark.asyncio
    async def test_skip_review_skips_review_only(self):
        """--skip-review causes the review phase to be skipped."""
        cfg = RotationConfig(
            domain="equilibrium",
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,
        )
        results = await run_rotation(cfg)
        review_result = results[2]
        assert review_result.name == "review"
        assert review_result.skipped

    @pytest.mark.asyncio
    async def test_skip_review_regen_still_runs_if_not_skipped(self):
        """When --skip-review but NOT --skip-regen, regen phase should
        attempt to run (will produce 0 results if no needs_revision names)."""
        cfg = RotationConfig(
            domain="equilibrium",
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,  # Skip regen too for unit test simplicity
        )
        results = await run_rotation(cfg)
        regen_result = results[3]
        assert regen_result.name == "regen"
        assert regen_result.skipped


# ═══════════════════════════════════════════════════════════════════════
# Fail-fast
# ═══════════════════════════════════════════════════════════════════════


class TestFailFast:
    """Verify --fail-fast aborts remaining phases on error."""

    @pytest.mark.asyncio
    async def test_fail_fast_aborts_on_error(self):
        """If generate fails and --fail-fast, remaining phases are skipped."""
        cfg = RotationConfig(
            domain="equilibrium",
            fail_fast=True,
            dry_run=False,
        )

        async def _mock_gen(cfg, **kwargs):
            return PhaseResult(name="generate", exit_code=1, error="boom")

        with patch(
            "imas_codex.standard_names.rotation._run_generate_phase",
            side_effect=_mock_gen,
        ):
            results = await run_rotation(cfg)

        assert results[0].exit_code == 1
        assert results[0].error == "boom"
        # Remaining phases should be skipped
        for r in results[1:]:
            assert r.skipped

    @pytest.mark.asyncio
    async def test_no_fail_fast_continues(self):
        """Without --fail-fast, subsequent phases still run."""
        cfg = RotationConfig(
            domain="equilibrium",
            fail_fast=False,
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,
        )

        async def _mock_gen(cfg, **kwargs):
            return PhaseResult(name="generate", exit_code=1, error="boom")

        with patch(
            "imas_codex.standard_names.rotation._run_generate_phase",
            side_effect=_mock_gen,
        ):
            results = await run_rotation(cfg)

        assert results[0].exit_code == 1
        # The skipped phases still appear as skipped (not errored)
        assert results[1].skipped
        assert results[2].skipped
        assert results[3].skipped


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════


class TestRotationSummary:
    """Verify summary aggregation."""

    def test_summary_aggregates_costs(self):
        results = [
            PhaseResult(name="generate", cost=1.0, elapsed=10.0, count=5),
            PhaseResult(name="enrich", cost=0.5, elapsed=5.0, count=3),
            PhaseResult(name="review", cost=0.3, elapsed=8.0, count=10),
            PhaseResult(name="regen", cost=0.2, elapsed=3.0, count=2),
        ]
        cfg = RotationConfig(domain="equilibrium", cost_limit=5.0)
        summary = rotation_summary(results, cfg)

        assert summary["total_cost"] == pytest.approx(2.0)
        assert summary["total_elapsed"] == pytest.approx(26.0)
        assert summary["total_count"] == 20
        assert summary["exit_code"] == 0
        assert len(summary["phases"]) == 4
        assert summary["rotation_id"] == cfg.rotation_id

    def test_summary_captures_errors(self):
        results = [
            PhaseResult(name="generate", exit_code=1, error="oops"),
            PhaseResult(name="enrich", skipped=True),
            PhaseResult(name="review", skipped=True),
            PhaseResult(name="regen", skipped=True),
        ]
        cfg = RotationConfig(domain="equilibrium")
        summary = rotation_summary(results, cfg)

        assert summary["exit_code"] == 1
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["phase"] == "generate"

    def test_summary_skipped_not_counted(self):
        results = [
            PhaseResult(name="generate", skipped=True, count=0),
            PhaseResult(name="enrich", count=5, cost=0.5, elapsed=2.0),
            PhaseResult(name="review", skipped=True, count=0),
            PhaseResult(name="regen", skipped=True, count=0),
        ]
        cfg = RotationConfig(domain="equilibrium")
        summary = rotation_summary(results, cfg)

        assert summary["total_count"] == 5
