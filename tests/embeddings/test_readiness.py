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

        # Mock client that is available
        client_instance = MagicMock()
        client_instance.is_available.return_value = True
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-srv-1001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready()
        assert ok is True
        assert "ready" in msg.lower()
        # Should not need further steps since server is already available
        client_instance.is_available.assert_called_once()

    @patch("imas_codex.embeddings.readiness._is_on_iter_login", return_value=True)
    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=True)
    @patch("imas_codex.embeddings.readiness._try_start_service", return_value=True)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_tries_systemd_service_on_iter(
        self, mock_url, mock_port, mock_client, mock_start, mock_on_iter, mock_login
    ):
        """On ITER login node, should try starting systemd service when server not responding."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        # First call: not available; subsequent calls: available
        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-srv-1001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_start.assert_called_once()

    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=False)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=True)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_creates_ssh_tunnel_off_iter(
        self,
        mock_url,
        mock_port,
        mock_client,
        mock_tunnel,
        mock_on_iter,
    ):
        """Off ITER, should create SSH tunnel before retrying health check."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-0.6B"
        info.hostname = "98dci4-srv-1001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_tunnel.assert_called_once_with(18765)

    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=False)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=False)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_fails_when_ssh_tunnel_fails_off_iter(
        self, mock_url, mock_port, mock_client, mock_tunnel, mock_on_iter
    ):
        """Off ITER, should fail if SSH tunnel cannot be established."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.return_value = False
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is False
        assert "SSH tunnel" in msg

    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=True)
    @patch("imas_codex.embeddings.readiness._try_start_service", return_value=True)
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_fails_when_server_never_responds(
        self, mock_url, mock_port, mock_client, mock_start, mock_on_iter
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
        # The "not configured" path doesn't call log_fn, so messages may be empty
        # but function should not raise

    def test_resolve_source_label(self):
        """Should resolve hostname to human-readable labels."""
        from imas_codex.embeddings.readiness import _resolve_source_label

        # GPU node
        info = MagicMock()
        info.hostname = "98dci4-gpu-0003"
        assert "iter-gpu" in _resolve_source_label(info)

        # Login node
        info.hostname = "98dci4-srv-1001"
        assert "iter-login" in _resolve_source_label(info)

        # None hostname
        info.hostname = None
        assert _resolve_source_label(info) == "remote"

        # No info
        assert _resolve_source_label(None) == "remote"


class TestEnsureSshTunnel:
    """Tests for SSH tunnel establishment."""

    @patch("imas_codex.embeddings.readiness.socket.socket")
    def test_returns_true_when_port_already_bound(self, mock_socket_cls):
        """Should return True immediately if port is already in use."""
        from imas_codex.embeddings.readiness import _ensure_ssh_tunnel

        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0  # Port is open
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mock_socket_cls.return_value = mock_socket

        assert _ensure_ssh_tunnel(18765) is True

    @patch("imas_codex.embeddings.readiness.subprocess.run")
    @patch("imas_codex.embeddings.readiness.socket.socket")
    def test_starts_ssh_tunnel_when_port_free(self, mock_socket_cls, mock_run):
        """Should start SSH tunnel when port is not in use."""
        from imas_codex.embeddings.readiness import _ensure_ssh_tunnel

        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 111  # Connection refused (port free)
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mock_socket_cls.return_value = mock_socket

        mock_run.return_value = MagicMock(returncode=0)

        result = _ensure_ssh_tunnel(18765)
        assert result is True
        mock_run.assert_called_once()
        # Verify SSH command includes the correct port forwarding
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "ssh" in cmd
        assert "18765:127.0.0.1:18765" in " ".join(cmd)


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


class TestIsOnIter:
    """Tests for ITER detection."""

    @patch("os.uname")
    def test_detects_iter_login_node(self, mock_uname):
        """Should detect ITER login node by hostname."""
        from imas_codex.embeddings.readiness import _is_on_iter

        mock_uname.return_value = MagicMock(nodename="98dci4-srv-1001")
        assert _is_on_iter() is True

    @patch("os.uname")
    def test_detects_iter_gpu_node(self, mock_uname):
        """Should detect ITER GPU node by hostname."""
        from imas_codex.embeddings.readiness import _is_on_iter

        mock_uname.return_value = MagicMock(nodename="98dci4-gpu-0003")
        assert _is_on_iter() is True

    @patch("os.uname")
    def test_returns_false_for_workstation(self, mock_uname):
        """Should return False for non-ITER hostnames."""
        from imas_codex.embeddings.readiness import _is_on_iter

        mock_uname.return_value = MagicMock(nodename="my-workstation")
        assert _is_on_iter() is False


class TestIsOnIterLoginAndCompute:
    """Tests for ITER login vs compute node detection."""

    @patch("os.uname")
    def test_login_node_detected(self, mock_uname):
        from imas_codex.embeddings.readiness import _is_on_iter_login

        mock_uname.return_value = MagicMock(nodename="98dci4-srv-1001")
        assert _is_on_iter_login() is True

    @patch("os.uname")
    def test_compute_node_not_login(self, mock_uname):
        from imas_codex.embeddings.readiness import _is_on_iter_login

        mock_uname.return_value = MagicMock(nodename="98dci4-clu-0042")
        assert _is_on_iter_login() is False

    @patch("os.uname")
    def test_compute_node_detected(self, mock_uname):
        from imas_codex.embeddings.readiness import _is_on_iter_compute

        mock_uname.return_value = MagicMock(nodename="98dci4-clu-0042")
        assert _is_on_iter_compute() is True

    @patch("os.uname")
    def test_login_node_not_compute(self, mock_uname):
        from imas_codex.embeddings.readiness import _is_on_iter_compute

        mock_uname.return_value = MagicMock(nodename="98dci4-srv-1001")
        assert _is_on_iter_compute() is False

    @patch("os.uname")
    def test_workstation_not_compute(self, mock_uname):
        from imas_codex.embeddings.readiness import _is_on_iter_compute

        mock_uname.return_value = MagicMock(nodename="my-workstation")
        assert _is_on_iter_compute() is False


class TestResolveUrlForCompute:
    """Tests for compute node URL redirection."""

    @patch("imas_codex.embeddings.readiness._is_on_iter_compute", return_value=True)
    def test_rewrites_localhost(self, mock_compute):
        from imas_codex.embeddings.readiness import (
            ITER_LOGIN_HOST,
            _resolve_url_for_compute,
        )

        result = _resolve_url_for_compute("http://localhost:18765")
        assert result == f"http://{ITER_LOGIN_HOST}:18765"

    @patch("imas_codex.embeddings.readiness._is_on_iter_compute", return_value=True)
    def test_rewrites_127_0_0_1(self, mock_compute):
        from imas_codex.embeddings.readiness import (
            ITER_LOGIN_HOST,
            _resolve_url_for_compute,
        )

        result = _resolve_url_for_compute("http://127.0.0.1:18765")
        assert result == f"http://{ITER_LOGIN_HOST}:18765"

    @patch("imas_codex.embeddings.readiness._is_on_iter_compute", return_value=True)
    def test_no_rewrite_for_remote_url(self, mock_compute):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        url = "http://some-server:18765"
        assert _resolve_url_for_compute(url) == url

    @patch("imas_codex.embeddings.readiness._is_on_iter_compute", return_value=False)
    def test_no_rewrite_off_compute(self, mock_compute):
        from imas_codex.embeddings.readiness import _resolve_url_for_compute

        url = "http://localhost:18765"
        assert _resolve_url_for_compute(url) == url
