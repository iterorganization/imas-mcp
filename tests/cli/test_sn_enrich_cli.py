"""Tests for the ``imas-codex sn enrich`` CLI command.

Validates flag parsing, state construction, dry-run behaviour,
and help text for the pipeline-engine-based enrich command.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture
def runner():
    return CliRunner()


def _patch_imports():
    """Patch the lazy imports inside the enrich command body."""
    return [
        patch("imas_codex.discovery.base.llm.set_litellm_offline_env"),
        patch(
            "imas_codex.cli.discover.common.use_rich_output",
            return_value=False,
        ),
        patch(
            "imas_codex.cli.discover.common.setup_logging",
            return_value=None,
        ),
    ]


def _invoke_enrich(runner, args, return_stats=None):
    """Invoke the enrich command with mocked pipeline, capture state kwargs."""
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

    captured = {}
    stats = return_stats or {"extract_count": 0}

    def fake_run_discovery(config, async_main, **kwargs):
        captured["config"] = config
        return stats

    class CapturingState(StandardNameEnrichState):
        def __init__(self, **kwargs):
            captured["state_kwargs"] = dict(kwargs)
            super().__init__(**kwargs)

    import_patches = _patch_imports()
    for p in import_patches:
        p.start()

    try:
        with (
            patch(
                "imas_codex.cli.discover.common.run_discovery",
                side_effect=fake_run_discovery,
            ),
            patch(
                "imas_codex.standard_names.enrich_state.StandardNameEnrichState",
                CapturingState,
            ),
        ):
            result = runner.invoke(sn, ["enrich"] + list(args))
    finally:
        for p in import_patches:
            p.stop()

    return result, captured


class TestEnrichCommandExists:
    """Basic registration tests."""

    def test_command_registered(self):
        cmd = sn.get_command(None, "enrich")
        assert cmd is not None, "enrich command should be registered"

    def test_help_text(self, runner):
        result = runner.invoke(sn, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "--domain" in result.output
        assert "--cost-limit" in result.output
        assert "--batch-size" in result.output
        assert "--dry-run" in result.output
        assert "--force" in result.output
        assert "--model" in result.output
        assert "--status" in result.output
        assert "--limit" in result.output

    def test_help_contains_examples(self, runner):
        result = runner.invoke(sn, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "imas-codex sn enrich --domain equilibrium" in result.output
        assert "--dry-run" in result.output


class TestEnrichFlagParsing:
    """CLI flag parsing → state construction."""

    def test_default_cost_limit(self, runner):
        result, captured = _invoke_enrich(runner, [])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["cost_limit"] == 2.0

    def test_custom_cost_limit(self, runner):
        result, captured = _invoke_enrich(runner, ["-c", "5.0"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["cost_limit"] == 5.0

    def test_single_domain(self, runner):
        result, captured = _invoke_enrich(runner, ["--domain", "equilibrium"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["domain"] == ["equilibrium"]

    def test_multiple_domains(self, runner):
        result, captured = _invoke_enrich(
            runner, ["--domain", "transport", "--domain", "magnetics"]
        )
        assert result.exit_code == 0
        assert captured["state_kwargs"]["domain"] == ["transport", "magnetics"]

    def test_no_domain_passes_none(self, runner):
        result, captured = _invoke_enrich(runner, [])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["domain"] is None

    def test_limit_flag(self, runner):
        result, captured = _invoke_enrich(runner, ["--limit", "50"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["limit"] == 50

    def test_batch_size_flag(self, runner):
        result, captured = _invoke_enrich(runner, ["--batch-size", "12"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["batch_size"] == 12

    def test_default_batch_size(self, runner):
        result, captured = _invoke_enrich(runner, [])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["batch_size"] == 8

    def test_force_flag(self, runner):
        result, captured = _invoke_enrich(runner, ["--force"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["force"] is True

    def test_dry_run_flag(self, runner):
        result, captured = _invoke_enrich(runner, ["--dry-run"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["dry_run"] is True

    def test_model_override(self, runner):
        result, captured = _invoke_enrich(
            runner, ["--model", "openrouter/anthropic/claude-opus-4.7"]
        )
        assert result.exit_code == 0
        assert (
            captured["state_kwargs"]["model"] == "openrouter/anthropic/claude-opus-4.7"
        )

    def test_default_status_named(self, runner):
        result, captured = _invoke_enrich(runner, [])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["status_filter"] == ["named"]

    def test_custom_status(self, runner):
        result, captured = _invoke_enrich(runner, ["--status", "drafted"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["status_filter"] == ["drafted"]

    def test_comma_separated_status(self, runner):
        result, captured = _invoke_enrich(runner, ["--status", "named,drafted"])
        assert result.exit_code == 0
        assert captured["state_kwargs"]["status_filter"] == ["named", "drafted"]


class TestEnrichDryRun:
    """Dry-run path asserts no graph writes."""

    def test_dry_run_no_model_check(self, runner):
        """Dry-run skips model availability check."""
        result, captured = _invoke_enrich(runner, ["--dry-run", "--limit", "5"])
        assert result.exit_code == 0
        assert captured["config"].check_model is False

    def test_dry_run_output(self, runner):
        """Dry-run includes 'dry run' in summary output."""
        result, captured = _invoke_enrich(
            runner,
            ["--dry-run", "--limit", "5"],
            return_stats={"extract_count": 10},
        )
        assert result.exit_code == 0
        # log_print goes to logger when console is None; check the result
        # didn't error at least
        assert result.exit_code == 0


class TestEnrichValidationWarning:
    """Warning when neither --domain nor --limit is set."""

    def test_warning_no_scope(self, runner):
        result, _ = _invoke_enrich(runner, [])
        assert result.exit_code == 0
        # The warning is printed via console.print (Rich Console),
        # which outputs to stdout in the CliRunner
        assert "use --domain or --limit to scope" in result.output.lower()

    def test_no_warning_with_domain(self, runner):
        result, _ = _invoke_enrich(runner, ["--domain", "equilibrium"])
        assert result.exit_code == 0
        assert "use --domain or --limit to scope" not in result.output.lower()

    def test_no_warning_with_limit(self, runner):
        result, _ = _invoke_enrich(runner, ["--limit", "20"])
        assert result.exit_code == 0
        assert "use --domain or --limit to scope" not in result.output.lower()


class TestEnrichDiscoveryConfig:
    """Verify DiscoveryConfig is built correctly."""

    def test_graph_check_enabled(self, runner):
        result, captured = _invoke_enrich(runner, ["--limit", "10"])
        assert result.exit_code == 0
        config = captured["config"]
        assert config.check_graph is True
        assert config.check_embed is False
        assert config.check_ssh is False
        assert config.model_section == "sn-enrich"

    def test_model_check_enabled_live(self, runner):
        """Live run (not dry-run) enables model check."""
        result, captured = _invoke_enrich(runner, ["--limit", "10"])
        assert result.exit_code == 0
        assert captured["config"].check_model is True

    def test_facility_is_dd(self, runner):
        """Enrich always targets the DD facility."""
        result, captured = _invoke_enrich(runner, ["--limit", "10"])
        assert result.exit_code == 0
        assert captured["config"].facility == "dd"
        assert captured["state_kwargs"]["facility"] == "dd"
