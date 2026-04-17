"""Tests for LLM health check infrastructure (readiness, canary, deep)."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest


class TestProbeLitellmReadiness:
    """Tests for _probe_litellm_readiness()."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-test-worker-key")
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test-master-key")

    def _mock_readiness(self, data, status_code=200):
        """Create a mock for urllib.request.urlopen returning readiness data."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.status = status_code
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_healthy_when_connected(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.return_value = self._mock_readiness(
            {"status": "connected", "db": "connected"}
        )
        healthy, detail = _probe_litellm_readiness("iter")
        assert healthy is True
        assert "iter" in detail

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_unhealthy_when_not_connected(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.return_value = self._mock_readiness(
            {"status": "Not connected", "db": "connected"}
        )
        healthy, detail = _probe_litellm_readiness("iter")
        assert healthy is False
        assert "not ready" in detail.lower()

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_unhealthy_when_db_disconnected(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.return_value = self._mock_readiness(
            {"status": "connected", "db": "Not connected"}
        )
        healthy, detail = _probe_litellm_readiness("iter")
        assert healthy is False
        assert "db" in detail.lower()

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_unhealthy_on_connection_refused(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        healthy, detail = _probe_litellm_readiness("iter")
        assert healthy is False
        assert "refused" in detail.lower()

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_unhealthy_on_401_unauthorized(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        healthy, detail = _probe_litellm_readiness("iter")
        assert healthy is False
        assert "401" in detail

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_auth_header_uses_worker_key_over_master(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.return_value = self._mock_readiness(
            {"status": "connected", "db": "connected"}
        )
        _probe_litellm_readiness("iter")
        # Check the request object passed to urlopen
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test-worker-key"

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_auth_falls_back_to_master_key(self, mock_urlopen, mock_url, monkeypatch):
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        mock_urlopen.return_value = self._mock_readiness(
            {"status": "connected", "db": "connected"}
        )
        _probe_litellm_readiness("iter")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test-master-key"

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_no_call_to_health_endpoint(self, mock_urlopen, mock_url):
        """Readiness probe must NOT hit /health (which makes real LLM calls)."""
        from imas_codex.discovery.base.services import _probe_litellm_readiness

        mock_urlopen.return_value = self._mock_readiness(
            {"status": "connected", "db": "connected"}
        )
        _probe_litellm_readiness("iter")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/health/readiness" in req.full_url
        assert req.full_url.endswith("/health/readiness")


class TestCanaryCheck:
    """Tests for _canary_check()."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-test-key")

    def _mock_completion(self, status=200):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_all_models_healthy(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _canary_check

        mock_urlopen.return_value = self._mock_completion()
        healthy, detail = _canary_check("iter")
        assert healthy is True
        assert "canary OK" in detail

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_only_two_api_calls(self, mock_urlopen, mock_url):
        """Canary should test exactly 2 credential classes (codex + codex-anthropic)."""
        from imas_codex.discovery.base.services import _canary_check

        mock_urlopen.return_value = self._mock_completion()
        _canary_check("iter")
        assert mock_urlopen.call_count == 2

    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_partial_failure_reports_details(self, mock_urlopen, mock_url):
        from imas_codex.discovery.base.services import _canary_check

        # First call succeeds, second fails
        mock_urlopen.side_effect = [
            self._mock_completion(),
            urllib.error.HTTPError(url="", code=401, msg="", hdrs=None, fp=None),
        ]
        healthy, detail = _canary_check("iter")
        assert healthy is False
        assert "canary failed" in detail
        assert "codex-anthropic" in detail


class TestLlmDeepHealthCheck:
    """Tests for llm_deep_health_check()."""

    @patch("imas_codex.settings.get_llm_location", return_value="iter")
    @patch(
        "imas_codex.settings.get_llm_proxy_url", return_value="http://localhost:18400"
    )
    @patch("urllib.request.urlopen")
    def test_returns_three_tuple(self, mock_urlopen, mock_proxy, mock_loc, monkeypatch):
        from imas_codex.discovery.base.services import llm_deep_health_check

        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"healthy_count": 5, "unhealthy_count": 0}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = llm_deep_health_check()
        assert len(result) == 3
        healthy, detail, data = result
        assert healthy is True
        assert isinstance(data, dict)


class TestLlmHealthCheckUsesReadiness:
    """Verify llm_health_check uses readiness probe, not /health."""

    @patch("imas_codex.discovery.base.services._probe_litellm_readiness")
    @patch("imas_codex.discovery.base.services._probe_litellm_proxy")
    @patch("imas_codex.settings.get_llm_location", return_value="iter")
    def test_calls_readiness_not_proxy(self, mock_loc, mock_proxy, mock_readiness):
        from imas_codex.discovery.base.services import llm_health_check

        mock_readiness.return_value = (True, "iter")
        llm_health_check()
        mock_readiness.assert_called_once_with("iter")
        mock_proxy.assert_not_called()
