"""Tests for ``sn generate --target {names,docs}`` routing.

Validates that the unified ``--target`` flag routes correctly:

* ``--target=names``  → single-pass compose with ``name_only=True``
* ``--target=docs``   → five-phase enrich pipeline (via ``_run_sn_docs_generation``)

No LLM calls — external dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture()
def runner():
    return CliRunner()


# Patch targets for the three routing exits in ``sn_generate``.
_ROTATOR = "imas_codex.cli.sn._run_sn_loop_cmd"
_DOCS_HELPER = "imas_codex.cli.sn._run_sn_docs_generation"


class TestTargetDocsRouting:
    """``--target=docs`` routes to the enrich pipeline, not the rotator."""

    def test_target_docs_calls_docs_helper(self, runner):
        """--target=docs → _run_sn_docs_generation called, rotator skipped."""
        with patch(_ROTATOR) as mock_rot, patch(_DOCS_HELPER) as mock_docs:
            result = runner.invoke(
                sn,
                [
                    "run",
                    "--target",
                    "docs",
                    "--physics-domain",
                    "equilibrium",
                    "--limit",
                    "1",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert mock_docs.called, (
                f"Docs helper not called for --target=docs. Output: {result.output}"
            )
            assert not mock_rot.called, "Rotator must NOT be called for --target=docs"

    def test_target_docs_forwards_domain_and_limit(self, runner):
        """--target=docs forwards --physics-domain and --limit to enrich helper."""
        with patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(
                sn,
                [
                    "run",
                    "--target",
                    "docs",
                    "--physics-domain",
                    "equilibrium",
                    "--limit",
                    "5",
                    "--dry-run",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert mock_docs.called
            kwargs = mock_docs.call_args.kwargs
            assert kwargs["domain_list"] == ["equilibrium"]
            assert kwargs["limit"] == 5
            assert kwargs["dry_run"] is True

    def test_target_docs_forwards_docs_status(self, runner):
        """--docs-status is forwarded as ``status_filter``."""
        with patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(
                sn,
                [
                    "run",
                    "--target",
                    "docs",
                    "--docs-status",
                    "named,enriched",
                    "--physics-domain",
                    "equilibrium",
                    "--limit",
                    "1",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert mock_docs.called
            assert mock_docs.call_args.kwargs["status_filter"] == "named,enriched"

    def test_target_docs_forwards_docs_batch_size(self, runner):
        """--docs-batch-size is forwarded as ``batch_size``."""
        with patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(
                sn,
                [
                    "run",
                    "--target",
                    "docs",
                    "--docs-batch-size",
                    "7",
                    "--physics-domain",
                    "equilibrium",
                    "--limit",
                    "1",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert mock_docs.called
            assert mock_docs.call_args.kwargs["batch_size"] == 7


class TestTargetNamesRouting:
    """``--target=names`` (default) routes to rotator."""

    def test_target_names_routes_to_rotator(self, runner):
        with patch(_ROTATOR) as mock_rot, patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(
                sn,
                ["run", "--target", "names", "-c", "0.01", "-q"],
            )
            assert mock_rot.called
            assert not mock_docs.called

    def test_default_target_routes_to_rotator(self, runner):
        """No --target → rotator (names default)."""
        with patch(_ROTATOR) as mock_rot, patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(sn, ["run", "-c", "0.01", "-q"])
            assert mock_rot.called
            assert not mock_docs.called


class TestTargetValidation:
    """Invalid --target values fail click validation."""

    def test_invalid_target_rejected(self, runner):
        result = runner.invoke(sn, ["run", "--target", "bogus", "-c", "0.01", "-q"])
        assert result.exit_code != 0
        assert "bogus" in result.output.lower() or "invalid" in result.output.lower()


class TestTargetNamesSkipsEnrich:
    """Bug 2 regression: --target names must set skip_enrich=True.

    Wave 9A observed Opus-4.6 ran enrich on 50-200 SNs/domain consuming
    50-60% of the $5 budget cap even though the user specified --target names.
    The fix sets skip_enrich=True when target_normalized == 'names'.
    """

    def test_target_names_passes_skip_enrich_to_loop(self, runner):
        """--target names must pass skip_enrich=True to the loop command."""
        with patch(_ROTATOR) as mock_rot:
            runner.invoke(
                sn,
                ["run", "--target", "names", "-c", "0.01", "-q"],
            )
            assert mock_rot.called, "Loop command must be invoked"
            kwargs = mock_rot.call_args.kwargs
            assert kwargs.get("skip_enrich") is True, (
                f"--target names must set skip_enrich=True; got {kwargs}"
            )

    def test_default_target_also_skips_enrich(self, runner):
        """Default target (names) must also skip enrich."""
        with patch(_ROTATOR) as mock_rot:
            runner.invoke(sn, ["run", "-c", "0.01", "-q"])
            assert mock_rot.called
            kwargs = mock_rot.call_args.kwargs
            assert kwargs.get("skip_enrich") is True

    def test_explicit_skip_enrich_flag_still_works(self, runner):
        """--skip-enrich still passes skip_enrich=True independently of target."""
        with patch(_ROTATOR) as mock_rot:
            runner.invoke(
                sn,
                ["run", "--target", "names", "--skip-enrich", "-c", "0.01", "-q"],
            )
            assert mock_rot.called
            kwargs = mock_rot.call_args.kwargs
            assert kwargs.get("skip_enrich") is True

    def test_target_docs_routes_to_docs_helper_not_loop(self, runner):
        """Symmetric: --target docs still routes to enrich helper, not loop."""
        with patch(_ROTATOR) as mock_rot, patch(_DOCS_HELPER) as mock_docs:
            runner.invoke(
                sn,
                [
                    "run",
                    "--target",
                    "docs",
                    "--physics-domain",
                    "equilibrium",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert mock_docs.called, "--target docs must call docs helper"
            assert not mock_rot.called, "--target docs must NOT call the loop"
