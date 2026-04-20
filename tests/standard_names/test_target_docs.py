"""Tests for sn generate --target docs dispatch and batching.

Covers:
* ``sn generate --target docs`` routes through ``_run_sn_docs_generation``.
* ``docs-batch-size`` resolves to the pyproject default (12) when
  ``--docs-batch-size`` is omitted and ``[sn-generate].docs-batch-size``
  is set.
* ``sn enrich`` still forwards to the same helper with equivalent args.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_generate_target_docs_invokes_docs_helper() -> None:
    """--target docs must route to _run_sn_docs_generation."""
    runner = CliRunner()
    with patch("imas_codex.cli.sn._run_sn_docs_generation") as mock_docs:
        result = runner.invoke(
            sn,
            [
                "generate",
                "--target",
                "docs",
                "--physics-domain",
                "equilibrium",
                "--limit",
                "5",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_docs.called, "expected _run_sn_docs_generation to be called"
        kwargs = mock_docs.call_args.kwargs
        assert kwargs["domain_list"] == ["equilibrium"]
        assert kwargs["limit"] == 5
        assert kwargs["dry_run"] is True


def test_generate_target_docs_picks_up_docs_batch_size_from_pyproject() -> None:
    """When --docs-batch-size is unspecified, helper must resolve from pyproject."""
    runner = CliRunner()

    # Capture the batch_size that flows into run_sn_enrich_engine.
    with patch("imas_codex.cli.sn.set_litellm_offline_env", create=True):
        pass  # not needed; helper imports it lazily

    # We stub _run_sn_docs_generation to inspect what batch_size got forwarded.
    with patch("imas_codex.cli.sn._run_sn_docs_generation") as mock_docs:
        result = runner.invoke(
            sn,
            [
                "generate",
                "--target",
                "docs",
                "--physics-domain",
                "equilibrium",
                "--limit",
                "1",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_docs.call_args.kwargs
        # CLI passes batch_size=None when --docs-batch-size is omitted;
        # the helper then resolves from pyproject.
        assert kwargs["batch_size"] is None


def test_docs_helper_resolves_batch_size_from_sn_generate_section() -> None:
    """_run_sn_docs_generation resolves batch_size from [sn-generate].docs-batch-size."""
    from imas_codex.cli import sn as sn_module

    captured: dict = {}

    def fake_state_init(self, **kwargs):
        captured.update(kwargs)
        # Don't actually construct anything heavy; raise to short-circuit.
        raise RuntimeError("short-circuit after capture")

    with patch(
        "imas_codex.standard_names.enrich_state.StandardNameEnrichState.__init__",
        fake_state_init,
    ):
        try:
            sn_module._run_sn_docs_generation(
                domain_list=["equilibrium"],
                status_filter="drafted",
                cost_limit=1.0,
                limit=1,
                batch_size=None,  # unspecified → resolve from pyproject
                dry_run=True,
                force=False,
                model_override=None,
                verbose=False,
                quiet=True,
            )
        except RuntimeError as e:
            assert "short-circuit" in str(e)

    # pyproject [sn-generate].docs-batch-size = 12 (added in Phase 1)
    assert captured.get("batch_size") == 12


def test_generate_target_docs_explicit_batch_size_overrides_pyproject() -> None:
    """--docs-batch-size=7 overrides the pyproject default of 12."""
    runner = CliRunner()
    with patch("imas_codex.cli.sn._run_sn_docs_generation") as mock_docs:
        result = runner.invoke(
            sn,
            [
                "generate",
                "--target",
                "docs",
                "--physics-domain",
                "equilibrium",
                "--docs-batch-size",
                "7",
                "--limit",
                "1",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_docs.call_args.kwargs
        assert kwargs["batch_size"] == 7


def test_sn_enrich_alias_still_forwards_to_docs_helper() -> None:
    """sn enrich remains a thin back-compat alias for sn generate --target docs."""
    runner = CliRunner()
    with patch("imas_codex.cli.sn._run_sn_docs_generation") as mock_docs:
        result = runner.invoke(
            sn,
            [
                "enrich",
                "--physics-domain",
                "equilibrium",
                "--limit",
                "3",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_docs.called
        kwargs = mock_docs.call_args.kwargs
        assert kwargs["domain_list"] == ["equilibrium"]
        assert kwargs["limit"] == 3
