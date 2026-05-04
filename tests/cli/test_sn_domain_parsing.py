"""Unit tests for the ``--domain`` CLI flag on ``imas-codex sn run``.

The flag must support three formats:
1. Repeated:        ``--domain equilibrium --domain core_profiles``
2. Space-separated: ``--domain "equilibrium core_profiles"``
3. Absent:          no flag → empty tuple → auto-seed all domains

All three cases are validated at two levels:
- The ``_split_whitespace`` callback that normalises Click values (pure unit test).
- The Click runner (invocation-level), using mocked ``_run_sn_loop_cmd`` so the
  pipeline never actually executes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import _split_whitespace, sn

# ---------------------------------------------------------------------------
# Pure unit tests for the _split_whitespace callback
# ---------------------------------------------------------------------------


class TestSplitWhitespaceCallback:
    """_split_whitespace converts each element on whitespace then flattens."""

    def test_repeated_single_values(self) -> None:
        """``--domain a --domain b`` arrives as ``('a', 'b')`` → unchanged."""
        result = _split_whitespace(None, None, ("equilibrium", "core_profiles"))
        assert result == ("equilibrium", "core_profiles")

    def test_space_separated_single_arg(self) -> None:
        """``--domain 'a b'`` arrives as ``('a b',)`` → split to ``('a', 'b')``."""
        result = _split_whitespace(None, None, ("equilibrium core_profiles",))
        assert result == ("equilibrium", "core_profiles")

    def test_mixed_repeated_and_space_separated(self) -> None:
        """``--domain 'a b' --domain c`` → ``('a', 'b', 'c')``."""
        result = _split_whitespace(
            None, None, ("equilibrium core_profiles", "magnetics")
        )
        assert result == ("equilibrium", "core_profiles", "magnetics")

    def test_empty_tuple(self) -> None:
        result = _split_whitespace(None, None, ())
        assert result == ()

    def test_none_value(self) -> None:
        """Click may pass None when the flag is absent."""
        result = _split_whitespace(None, None, None)
        assert result == ()

    def test_extra_internal_whitespace(self) -> None:
        """``str.split()`` collapses runs of whitespace."""
        result = _split_whitespace(None, None, ("  equilibrium   core_profiles  ",))
        assert result == ("equilibrium", "core_profiles")

    def test_single_domain_unchanged(self) -> None:
        result = _split_whitespace(None, None, ("equilibrium",))
        assert result == ("equilibrium",)


# ---------------------------------------------------------------------------
# CLI integration via CliRunner (mocked backend)
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _invoke_run(runner: CliRunner, extra_args: list[str]) -> tuple[int, list[str]]:
    """Invoke ``sn run`` with *extra_args*, capturing the ``domains`` tuple that
    would have been forwarded to ``_run_sn_loop_cmd``.

    Returns ``(exit_code, domains_list)``.
    """
    captured: list = []

    def fake_run_loop(**kwargs):
        captured.append(list(kwargs.get("domains", ())))

    with patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=fake_run_loop):
        # Also silence the pipeline-version clear-gate check
        with patch("imas_codex.cli.sn._check_pipeline_clear_gate"):
            result = runner.invoke(
                sn, ["run", "--dry-run"] + extra_args, catch_exceptions=False
            )

    domains = captured[0] if captured else []
    return result.exit_code, domains


class TestDomainFlagRepeated:
    """``--domain equilibrium --domain core_profiles`` → both domains in list."""

    def test_exit_code_zero(self, runner: CliRunner) -> None:
        code, _ = _invoke_run(
            runner, ["--domain", "equilibrium", "--domain", "core_profiles"]
        )
        assert code == 0

    def test_both_domains_present(self, runner: CliRunner) -> None:
        _, domains = _invoke_run(
            runner, ["--domain", "equilibrium", "--domain", "core_profiles"]
        )
        assert "equilibrium" in domains
        assert "core_profiles" in domains

    def test_no_extra_domains(self, runner: CliRunner) -> None:
        _, domains = _invoke_run(
            runner, ["--domain", "equilibrium", "--domain", "core_profiles"]
        )
        assert sorted(domains) == ["core_profiles", "equilibrium"]


class TestDomainFlagSpaceSeparated:
    """``--domain "equilibrium core_profiles"`` → both domains in list."""

    def test_exit_code_zero(self, runner: CliRunner) -> None:
        code, _ = _invoke_run(runner, ["--domain", "equilibrium core_profiles"])
        assert code == 0

    def test_both_domains_present(self, runner: CliRunner) -> None:
        _, domains = _invoke_run(runner, ["--domain", "equilibrium core_profiles"])
        assert "equilibrium" in domains
        assert "core_profiles" in domains


class TestDomainFlagAbsent:
    """No ``--domain`` flag → empty domain list (auto-seed all)."""

    def test_exit_code_zero(self, runner: CliRunner) -> None:
        code, _ = _invoke_run(runner, [])
        assert code == 0

    def test_domains_empty(self, runner: CliRunner) -> None:
        _, domains = _invoke_run(runner, [])
        assert domains == []


class TestDomainHelpText:
    """``sn run --help`` must document the ``--domain`` flag."""

    def test_help_shows_domain_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["run", "--help"])
        assert result.exit_code == 0
        assert "--domain" in result.output

    def test_help_mentions_whitespace_separated(self, runner: CliRunner) -> None:
        """Help text should hint that space-separated values work."""
        result = runner.invoke(sn, ["run", "--help"])
        # The option help mentions whitespace or space-separated
        assert "whitespace" in result.output.lower() or "space" in result.output.lower()
