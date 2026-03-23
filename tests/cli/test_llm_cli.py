"""Tests for LLM proxy CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
from click.testing import CliRunner

from imas_codex.cli.llm_cli import (
    _check_file_permissions,
    _llm_headers,
    _truncate,
    llm,
)
from imas_codex.discovery.base.llm import ensure_model_prefix, get_api_key

# httpx is imported locally inside _api_get/_api_post — patch at source
_HTTPX = "httpx"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    """Set required env vars for CLI commands."""
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test-master-key")
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-or-test")


def _mock_get(resp_json, status_code=200):
    """Create a patched httpx.get returning a mock response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = resp_json
    mock_resp.text = str(resp_json)
    return patch(f"{_HTTPX}.get", return_value=mock_resp)


def _mock_post(resp_json, status_code=200):
    """Create a patched httpx.post returning a mock response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = resp_json
    mock_resp.text = str(resp_json)
    return patch(f"{_HTTPX}.post", return_value=mock_resp)


# ── Help text tests ──────────────────────────────────────────────────────


class TestHelpText:
    """All subcommands render help without errors."""

    def test_llm_help(self, runner):
        result = runner.invoke(llm, ["--help"])
        assert result.exit_code == 0
        for cmd in ("keys", "teams", "spend", "local", "start", "stop", "status"):
            assert cmd in result.output

    def test_keys_help(self, runner):
        result = runner.invoke(llm, ["keys", "--help"])
        assert result.exit_code == 0
        for cmd in ("list", "create", "revoke", "rotate"):
            assert cmd in result.output

    def test_teams_help(self, runner):
        result = runner.invoke(llm, ["teams", "--help"])
        assert result.exit_code == 0
        for cmd in ("list", "create", "info"):
            assert cmd in result.output

    def test_local_help(self, runner):
        result = runner.invoke(llm, ["local", "--help"])
        assert result.exit_code == 0
        for cmd in ("start", "stop", "status", "models"):
            assert cmd in result.output

    def test_spend_help(self, runner):
        result = runner.invoke(llm, ["spend", "--help"])
        assert result.exit_code == 0
        assert "--team" in result.output


# ── Utility function tests ───────────────────────────────────────────────


class TestUtilities:
    def test_truncate_short(self):
        assert _truncate("short", 12) == "short"

    def test_truncate_long(self):
        assert _truncate("a-very-long-string-here", 12) == "a-very-long-..."

    def test_truncate_empty(self):
        assert _truncate("", 12) == ""

    def test_truncate_exact(self):
        assert _truncate("exactly12345", 12) == "exactly12345"

    def test_check_file_permissions_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Should not raise — no sensitive files exist
        _check_file_permissions()

    def test_check_file_permissions_secure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=x")
        env_file.chmod(0o600)
        _check_file_permissions()

    def test_llm_headers_missing_key(self, monkeypatch):
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        with pytest.raises(click.ClickException, match="LITELLM_MASTER_KEY"):
            _llm_headers()

    def test_llm_headers_with_key(self, mock_env):
        headers = _llm_headers()
        assert "Authorization" in headers
        assert "sk-test-master-key" in headers["Authorization"]


# ── ensure_model_prefix tests ────────────────────────────────────────────


class TestEnsureModelPrefix:
    def test_bare_anthropic(self):
        assert (
            ensure_model_prefix("anthropic/claude-sonnet-4-6")
            == "openrouter/anthropic/claude-sonnet-4-6"
        )

    def test_already_prefixed(self):
        assert (
            ensure_model_prefix("openrouter/anthropic/claude-sonnet-4-6")
            == "openrouter/anthropic/claude-sonnet-4-6"
        )

    def test_google(self):
        assert (
            ensure_model_prefix("google/gemini-3.1-flash-lite-preview")
            == "openrouter/google/gemini-3.1-flash-lite-preview"
        )

    def test_ollama_passthrough(self):
        assert ensure_model_prefix("ollama/qwen3:14b") == "ollama/qwen3:14b"

    def test_hosted_vllm_passthrough(self):
        assert ensure_model_prefix("hosted_vllm/model") == "hosted_vllm/model"

    def test_openai_localhost_passthrough(self):
        assert ensure_model_prefix("openai/localhost/model") == "openai/localhost/model"


# ── get_api_key tests ────────────────────────────────────────────────────


class TestGetApiKey:
    def test_imas_codex_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-imas")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert get_api_key() == "sk-imas"

    def test_no_fallback_key(self, monkeypatch):
        """No backward compat — only OPENROUTER_API_KEY_IMAS_CODEX is accepted."""
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fallback")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY_IMAS_CODEX"):
            get_api_key()

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY_IMAS_CODEX"):
            get_api_key()

    def test_imas_codex_only(self, monkeypatch):
        """Legacy OPENROUTER_API_KEY is ignored when _IMAS_CODEX is set."""
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-primary")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-legacy")
        assert get_api_key() == "sk-primary"


# ── Keys command tests ───────────────────────────────────────────────────


class TestKeysCommands:
    def test_keys_list(self, runner, mock_env):
        with _mock_get(
            {
                "keys": [
                    {
                        "key_alias": "test-key",
                        "token": "sk-test-123456789",
                        "team_alias": "imas-codex",
                        "max_budget": 100,
                        "spend": 5.50,
                        "expires": None,
                    }
                ]
            }
        ):
            result = runner.invoke(llm, ["keys", "list"])
        assert result.exit_code == 0
        assert "test-key" in result.output
        assert "1 key(s)" in result.output

    def test_keys_list_empty(self, runner, mock_env):
        with _mock_get({"keys": []}):
            result = runner.invoke(llm, ["keys", "list"])
        assert result.exit_code == 0
        assert "No virtual keys" in result.output

    def test_keys_create(self, runner, mock_env):
        with _mock_post({"key": "sk-new-key-123", "expires": None}):
            result = runner.invoke(
                llm,
                ["keys", "create", "--team", "test-team", "--alias", "test-alias"],
            )
        assert result.exit_code == 0
        assert "sk-new-key-123" in result.output
        assert "Save this key" in result.output

    def test_keys_revoke_confirm(self, runner, mock_env):
        with _mock_post({}):
            result = runner.invoke(llm, ["keys", "revoke", "sk-old-key"], input="y\n")
        assert result.exit_code == 0
        assert "Key revoked" in result.output

    def test_keys_revoke_cancel(self, runner, mock_env):
        result = runner.invoke(llm, ["keys", "revoke", "sk-old-key"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output


# ── Teams command tests ──────────────────────────────────────────────────


class TestTeamsCommands:
    def test_teams_list(self, runner, mock_env):
        with _mock_get(
            [
                {
                    "team_alias": "imas-codex",
                    "team_id": "tid-123",
                    "max_budget": 500,
                    "budget_duration": "30d",
                    "spend": 10,
                }
            ]
        ):
            result = runner.invoke(llm, ["teams", "list"])
        assert result.exit_code == 0
        assert "imas-codex" in result.output
        assert "1 team(s)" in result.output

    def test_teams_create(self, runner, mock_env):
        with _mock_post({"team_id": "tid-new-123"}):
            result = runner.invoke(
                llm, ["teams", "create", "--alias", "imas-codex-test"]
            )
        assert result.exit_code == 0
        assert "Team created" in result.output
        assert "tid-new-123" in result.output

    def test_teams_info(self, runner, mock_env):
        team_list_resp = [{"team_alias": "imas-codex", "team_id": "tid-123"}]
        team_info_resp = {
            "team_info": {
                "team_alias": "imas-codex",
                "team_id": "tid-123",
                "max_budget": 500,
                "spend": 42.50,
                "budget_duration": "30d",
            },
            "keys": [
                {"key_alias": "worker-1", "token": "sk-w1-xxx", "spend": 30},
                {"key_alias": "worker-2", "token": "sk-w2-xxx", "spend": 12.50},
            ],
        }

        def _side_effect(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/team/list" in str(url):
                mock_resp.json.return_value = team_list_resp
            else:
                mock_resp.json.return_value = team_info_resp
            mock_resp.text = ""
            return mock_resp

        with patch(f"{_HTTPX}.get", side_effect=_side_effect):
            result = runner.invoke(llm, ["teams", "info", "imas-codex"])
        assert result.exit_code == 0
        assert "imas-codex" in result.output
        assert "$42.50" in result.output
        assert "worker-1" in result.output


# ── Spend command tests ──────────────────────────────────────────────────


class TestSpendCommand:
    def test_spend_overview(self, runner, mock_env):
        with _mock_get(
            [
                {
                    "team_alias": "imas-codex",
                    "max_budget": 500,
                    "spend": 100,
                    "budget_duration": "30d",
                },
                {
                    "team_alias": "claude-code",
                    "max_budget": 200,
                    "spend": 25,
                    "budget_duration": "30d",
                },
            ]
        ):
            result = runner.invoke(llm, ["spend"])
        assert result.exit_code == 0
        assert "imas-codex" in result.output
        assert "claude-code" in result.output
        assert "Total" in result.output

    def test_spend_team_filter(self, runner, mock_env):
        team_list_resp = [{"team_alias": "imas-codex", "team_id": "tid-123"}]
        team_info_resp = {
            "team_info": {
                "team_alias": "imas-codex",
                "max_budget": 500,
                "spend": 100,
                "budget_duration": "30d",
            },
            "keys": [],
        }

        def _side_effect(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/team/list" in str(url):
                mock_resp.json.return_value = team_list_resp
            else:
                mock_resp.json.return_value = team_info_resp
            mock_resp.text = ""
            return mock_resp

        with patch(f"{_HTTPX}.get", side_effect=_side_effect):
            result = runner.invoke(llm, ["spend", "--team", "imas-codex"])
        assert result.exit_code == 0
        assert "Spend report" in result.output
        assert "$100.00" in result.output


# ── Connection error tests ───────────────────────────────────────────────


class TestConnectionErrors:
    def test_keys_list_proxy_not_running(self, runner, mock_env):
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = runner.invoke(llm, ["keys", "list"])
        assert result.exit_code != 0

    def test_teams_list_auth_failure(self, runner, mock_env):
        with _mock_get("Unauthorized", status_code=401):
            result = runner.invoke(llm, ["teams", "list"])
        assert result.exit_code != 0


# ── Setup command tests ──────────────────────────────────────────────────


class TestSetupCommand:
    def test_setup_dry_run(self, runner, mock_env):
        result = runner.invoke(llm, ["setup", "--dry-run"])
        assert result.exit_code == 0
        assert "Would create team" in result.output
        assert "imas-codex" in result.output
        assert "claude-code" in result.output

    def test_setup_creates_teams_and_keys(self, runner, mock_env):
        # Mock /team/list (empty) + /team/new + /key/generate
        call_count = {"get": 0, "post": 0}

        def mock_get(*args, **kwargs):
            call_count["get"] += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = []  # no existing teams
            resp.text = "[]"
            return resp

        def mock_post(*args, **kwargs):
            call_count["post"] += 1
            resp = MagicMock()
            resp.status_code = 200
            url = args[0] if args else kwargs.get("url", "")
            if "/team/new" in str(url):
                resp.json.return_value = {"team_id": f"tid-{call_count['post']}"}
            else:
                resp.json.return_value = {"key": f"sk-generated-{call_count['post']}"}
            resp.text = "{}"
            return resp

        with (
            patch("httpx.get", side_effect=mock_get),
            patch("httpx.post", side_effect=mock_post),
        ):
            result = runner.invoke(llm, ["setup"])
        assert result.exit_code == 0
        assert "Team created" in result.output
        assert "Key created" in result.output
        assert "SAVE THESE KEYS" in result.output

    def test_setup_skips_existing(self, runner, mock_env):
        with _mock_get(
            [
                {"team_alias": "imas-codex", "team_id": "tid-1"},
                {"team_alias": "claude-code", "team_id": "tid-2"},
            ]
        ):
            result = runner.invoke(llm, ["setup"])
        assert result.exit_code == 0
        assert "Already exists" in result.output

    def test_setup_missing_master_key(self, runner, monkeypatch):
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        result = runner.invoke(llm, ["setup"])
        assert result.exit_code != 0
        assert "LITELLM_MASTER_KEY" in result.output


# ── Security command tests ───────────────────────────────────────────────


class TestSecurityCommands:
    def test_security_audit_help(self, runner):
        result = runner.invoke(llm, ["security", "audit", "--help"])
        assert result.exit_code == 0
        assert "Audit security posture" in result.output

    def test_security_harden_help(self, runner):
        result = runner.invoke(llm, ["security", "harden", "--help"])
        assert result.exit_code == 0
        assert "Apply security hardening" in result.output

    def test_security_audit_runs(self, runner, mock_env):
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_get.return_value = mock_resp
            result = runner.invoke(llm, ["security", "audit"])
        assert result.exit_code == 0
        assert "Security Audit" in result.output
        assert "Environment Variables" in result.output

    def test_security_harden_runs(self, runner):
        result = runner.invoke(llm, ["security", "harden"])
        assert result.exit_code == 0
        assert "Security Hardening" in result.output
