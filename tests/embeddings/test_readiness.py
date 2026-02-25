"""Tests for embedding readiness check."""

from unittest.mock import MagicMock, patch

import pytest


class TestEnsureEmbeddingReady:
    """Tests for the centralized ensure_embedding_ready function."""

    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch("imas_codex.settings.get_embed_remote_url", return_value=None)
    def test_returns_false_when_no_remote_url(self, mock_url, mock_port, mock_client):
        """Should fail fast when no remote URL is configured."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        ok, msg = ensure_embedding_ready()
        assert ok is False
        assert "not configured" in msg

    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_returns_true_when_already_healthy(self, mock_url, mock_port, mock_client):
        """Should return immediately if server is already responding."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.return_value = True
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-gpu-0002"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready()
        assert ok is True
        assert "ready" in msg.lower()
        client_instance.is_available.assert_called_once()

    @patch(
        "imas_codex.embeddings.readiness._get_embed_host",
        return_value="98dci4-srv-1003",
    )
    @patch("imas_codex.embeddings.readiness._is_on_login_node", return_value=True)
    @patch("imas_codex.embeddings.readiness._is_on_facility", return_value=True)
    @patch("imas_codex.embeddings.readiness._try_start_service", return_value=True)
    @patch(
        "imas_codex.embeddings.readiness.socket.gethostname",
        return_value="98dci4-srv-1003",
    )
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_tries_systemd_service_on_login_node(
        self,
        mock_url,
        mock_port,
        mock_client,
        mock_hostname,
        mock_start,
        mock_on_facility,
        mock_login,
        mock_embed_host,
    ):
        """On login node with embed server on same host, should try systemd service."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-srv-1003"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_start.assert_called_once()

    @patch(
        "imas_codex.embeddings.readiness._get_embed_host",
        return_value="98dci4-gpu-0002",
    )
    @patch("imas_codex.embeddings.readiness._is_on_login_node", return_value=True)
    @patch("imas_codex.embeddings.readiness._is_on_facility", return_value=True)
    @patch("imas_codex.embeddings.readiness._try_start_service", return_value=True)
    @patch(
        "imas_codex.embeddings.readiness.socket.gethostname",
        return_value="98dci4-srv-1003",
    )
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_skips_systemd_when_embed_on_titan(
        self,
        mock_url,
        mock_port,
        mock_client,
        mock_hostname,
        mock_start,
        mock_on_facility,
        mock_login,
        mock_embed_host,
    ):
        """On login node, should NOT try systemd when embed server is on Titan."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-gpu-0002"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_start.assert_not_called()

    @patch("imas_codex.embeddings.readiness._is_on_facility", return_value=False)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=True)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_creates_ssh_tunnel_off_facility(
        self,
        mock_url,
        mock_port,
        mock_client,
        mock_tunnel,
        mock_on_facility,
    ):
        """Off facility, should create SSH tunnel before retrying health check."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-gpu-0002"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_tunnel.assert_called_once_with(18765)

    @patch("imas_codex.embeddings.readiness._is_on_facility", return_value=False)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=False)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_fails_when_ssh_tunnel_fails(
        self, mock_url, mock_port, mock_client, mock_tunnel, mock_on_facility
    ):
        """Off facility, should fail if SSH tunnel cannot be established."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.return_value = False
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is False
        assert "SSH tunnel" in msg

    @patch("imas_codex.embeddings.readiness._is_on_facility", return_value=True)
    @patch("imas_codex.embeddings.readiness._is_on_login_node", return_value=True)
    @patch("imas_codex.embeddings.readiness._get_embed_host", return_value=None)
    @patch(
        "imas_codex.embeddings.readiness.socket.gethostname",
        return_value="98dci4-srv-1003",
    )
    @patch("imas_codex.embeddings.readiness._try_start_service", return_value=True)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_fails_when_server_never_responds(
        self,
        mock_url,
        mock_port,
        mock_client,
        mock_start,
        mock_hostname,
        mock_embed_host,
        mock_login,
        mock_on_facility,
    ):
        """Should fail gracefully when server never becomes available."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.return_value = False
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=3.0)
        assert ok is False
        assert "not available" in msg.lower()

    @patch("imas_codex.settings.get_embed_remote_url", return_value=None)
    def test_log_fn_callback_called(self, mock_url):
        """Should call log_fn callback with status messages."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        messages = []

        def capture_log(msg, style):
            messages.append((msg, style))

        ensure_embedding_ready(log_fn=capture_log)

    def test_resolve_source_label(self):
        """Should resolve hostname to human-readable labels."""
        from imas_codex.embeddings.readiness import _resolve_source_label

        # Returns hostname directly
        info = MagicMock()
        info.hostname = "98dci4-gpu-0003"
        assert _resolve_source_label(info) == "98dci4-gpu-0003"

        info.hostname = "98dci4-srv-1003"
        assert _resolve_source_label(info) == "98dci4-srv-1003"

        # None hostname
        info.hostname = None
        assert _resolve_source_label(info) == "remote"

        # No info
        assert _resolve_source_label(None) == "remote"


class TestEnsureSshTunnel:
    """Tests for SSH tunnel establishment."""

    @patch("imas_codex.embeddings.readiness.ensure_tunnel", return_value=True)
    def test_returns_true_when_tunnel_active(self, mock_ensure):
        """Should return True when ensure_tunnel succeeds."""
        from imas_codex.embeddings.readiness import _ensure_ssh_tunnel

        assert _ensure_ssh_tunnel(18765) is True
        mock_ensure.assert_called_once_with(port=18765, ssh_host="iter")

    @patch("imas_codex.embeddings.readiness.ensure_tunnel", return_value=False)
    def test_returns_false_when_tunnel_fails(self, mock_ensure):
        """Should return False when ensure_tunnel fails."""
        from imas_codex.embeddings.readiness import _ensure_ssh_tunnel

        assert _ensure_ssh_tunnel(18765) is False
        mock_ensure.assert_called_once()

    @patch("imas_codex.embeddings.readiness.ensure_tunnel", return_value=True)
    def test_uses_explicit_ssh_host(self, mock_ensure):
        """Should pass explicit ssh_host to ensure_tunnel."""
        from imas_codex.embeddings.readiness import _ensure_ssh_tunnel

        assert _ensure_ssh_tunnel(18765, ssh_host="tcv") is True
        mock_ensure.assert_called_once_with(port=18765, ssh_host="tcv")


class TestTryStartService:
    """Tests for systemd service start attempts."""

    @patch("imas_codex.embeddings.readiness.subprocess.run")
    def test_returns_true_when_systemctl_succeeds(self, mock_run):
        """Should return True when systemctl start succeeds."""
        from imas_codex.embeddings.readiness import _try_start_service

        mock_run.return_value = MagicMock(returncode=0)
        assert _try_start_service() is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "systemctl" in cmd
        assert "imas-codex-embed" in cmd

    @patch("imas_codex.embeddings.readiness.subprocess.run")
    def test_returns_false_when_systemctl_fails(self, mock_run):
        """Should return False when systemctl start fails."""
        from imas_codex.embeddings.readiness import _try_start_service

        mock_run.return_value = MagicMock(returncode=1, stderr="Unit not found")
        assert _try_start_service() is False

    @patch(
        "imas_codex.embeddings.readiness.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_returns_false_when_systemctl_missing(self, mock_run):
        """Should return False when systemctl is not available."""
        from imas_codex.embeddings.readiness import _try_start_service

        assert _try_start_service() is False


class TestIsOnFacility:
    """Tests for facility detection."""

    @patch("imas_codex.remote.locations.is_location_local", return_value=True)
    @patch("imas_codex.settings.get_embedding_location", return_value="titan")
    def test_detects_local_facility(self, mock_location, mock_local):
        """Should return True when on the embedding facility."""
        from imas_codex.embeddings.readiness import _is_on_facility

        assert _is_on_facility() is True
        mock_local.assert_called_once_with("titan")

    @patch("imas_codex.remote.locations.is_location_local", return_value=False)
    @patch("imas_codex.settings.get_embedding_location", return_value="titan")
    def test_detects_remote(self, mock_location, mock_local):
        """Should return False when not on the embedding facility."""
        from imas_codex.embeddings.readiness import _is_on_facility

        assert _is_on_facility() is False


class TestIsOnLoginNode:
    """Tests for login node detection via systemd."""

    @patch("imas_codex.embeddings.readiness.subprocess.run")
    def test_detects_login_node(self, mock_run):
        """Should return True when systemd user session is running."""
        from imas_codex.embeddings.readiness import _is_on_login_node

        mock_run.return_value = MagicMock(returncode=0, stdout="running")
        assert _is_on_login_node() is True

    @patch("imas_codex.embeddings.readiness.subprocess.run")
    def test_detects_compute_node(self, mock_run):
        """Should return False when no systemd user session."""
        from imas_codex.embeddings.readiness import _is_on_login_node

        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _is_on_login_node() is False

    @patch(
        "imas_codex.embeddings.readiness.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_returns_false_when_no_systemctl(self, mock_run):
        """Should return False when systemctl is not found."""
        from imas_codex.embeddings.readiness import _is_on_login_node

        assert _is_on_login_node() is False


class TestResolveUrlForCompute:
    """Tests for URL redirection to embed server host."""

    @patch(
        "imas_codex.embeddings.readiness._get_embed_host",
        return_value="98dci4-gpu-0002",
    )
    def test_rewrites_localhost(self, mock_host):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        result = _resolve_url_for_compute("http://localhost:18765")
        assert result == "http://98dci4-gpu-0002:18765"

    @patch(
        "imas_codex.embeddings.readiness._get_embed_host",
        return_value="98dci4-gpu-0002",
    )
    def test_rewrites_127_0_0_1(self, mock_host):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        result = _resolve_url_for_compute("http://127.0.0.1:18765")
        assert result == "http://98dci4-gpu-0002:18765"

    @patch(
        "imas_codex.embeddings.readiness._get_embed_host",
        return_value=None,
    )
    def test_no_rewrite_when_no_embed_host(self, mock_host):
        """Should not rewrite when embed host is not resolvable."""
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        url = "http://localhost:18765"
        assert _resolve_url_for_compute(url) == url

    def test_no_rewrite_for_remote_url(self):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        url = "http://some-server:18765"
        assert _resolve_url_for_compute(url) == url

    def test_no_rewrite_for_empty_url(self):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        assert _resolve_url_for_compute("") == ""
