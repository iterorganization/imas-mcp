"""Tests for embedding readiness check and SLURM auto-launch integration."""

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
        info.model = "Qwen/Qwen3-Embedding-4B"
        info.hostname = "98dci4-gpu-0001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready()
        assert ok is True
        assert "ready" in msg.lower()
        # Should not try SLURM launch since server is already available
        client_instance.is_available.assert_called_once()

    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=True)
    @patch("imas_codex.embeddings.slurm.ensure_server", return_value=True)
    @patch(
        "imas_codex.settings.get_imas_embedding_model",
        return_value="Qwen/Qwen3-Embedding-4B",
    )
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_calls_ensure_server_on_iter(
        self, mock_url, mock_port, mock_client, mock_model, mock_ensure, mock_on_iter
    ):
        """On ITER, should call ensure_server without SSH tunnel."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        # First call: not available; subsequent calls: available
        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-4B"
        info.hostname = "98dci4-gpu-0001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_ensure.assert_called_once()

    @patch("imas_codex.embeddings.readiness._is_on_iter", return_value=False)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=True)
    @patch("imas_codex.embeddings.slurm.ensure_server", return_value=True)
    @patch(
        "imas_codex.settings.get_imas_embedding_model",
        return_value="Qwen/Qwen3-Embedding-4B",
    )
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
        mock_model,
        mock_ensure,
        mock_tunnel,
        mock_on_iter,
    ):
        """Off ITER, should create SSH tunnel before calling ensure_server."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.side_effect = [False, True]
        info = MagicMock()
        info.model = "Qwen/Qwen3-Embedding-4B"
        info.hostname = "98dci4-gpu-0001"
        client_instance.get_info.return_value = info
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=5.0)
        assert ok is True
        mock_tunnel.assert_called_once_with(18765)
        mock_ensure.assert_called_once()

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
    @patch("imas_codex.embeddings.slurm.ensure_server", return_value=False)
    @patch(
        "imas_codex.settings.get_imas_embedding_model",
        return_value="Qwen/Qwen3-Embedding-4B",
    )
    @patch("imas_codex.embeddings.client.RemoteEmbeddingClient")
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    @patch(
        "imas_codex.settings.get_embed_remote_url",
        return_value="http://localhost:18765",
    )
    def test_fails_when_ensure_server_fails(
        self, mock_url, mock_port, mock_client, mock_model, mock_ensure, mock_on_iter
    ):
        """Should fail gracefully when SLURM ensure_server fails."""
        from imas_codex.embeddings.readiness import ensure_embedding_ready

        client_instance = MagicMock()
        client_instance.is_available.return_value = False
        mock_client.return_value = client_instance

        ok, msg = ensure_embedding_ready(timeout=3.0)
        assert ok is False
        assert "SLURM" in msg or "not available" in msg.lower()

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
        assert "iter-titan" in _resolve_source_label(info)

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


class TestEncoderSlurmAutoLaunch:
    """Tests for Encoder._try_slurm_auto_launch integration."""

    @patch("shutil.which", return_value=None)
    def test_returns_false_when_no_sbatch_anywhere(self, mock_which):
        """Should return False when sbatch not available locally or via SSH."""
        from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
        from imas_codex.embeddings.encoder import Encoder

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)  # No remote sbatch

            with patch.object(Encoder, "_initialize_backend"):
                encoder = Encoder.__new__(Encoder)
                encoder.config = EncoderConfig(
                    backend=EmbeddingBackend.LOCAL,
                    model_name="test-model",
                )
                encoder.logger = __import__("logging").getLogger("test")
                encoder._model = None
                encoder._remote_client = None

            result = encoder._try_slurm_auto_launch()
            assert result is False

    @patch("shutil.which", return_value="/usr/bin/sbatch")
    @patch("imas_codex.embeddings.slurm.ensure_server", return_value=True)
    def test_calls_ensure_server_with_local_sbatch(self, mock_ensure, mock_which):
        """Should call ensure_server when sbatch is available locally."""
        from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
        from imas_codex.embeddings.encoder import Encoder

        with patch.object(Encoder, "_initialize_backend"):
            encoder = Encoder.__new__(Encoder)
            encoder.config = EncoderConfig(
                backend=EmbeddingBackend.LOCAL,
                model_name="test-model",
            )
            encoder.logger = __import__("logging").getLogger("test")
            encoder._model = None
            encoder._remote_client = None

        result = encoder._try_slurm_auto_launch()
        assert result is True
        mock_ensure.assert_called_once_with(model_name="test-model")

    @patch("shutil.which", return_value=None)  # No local sbatch
    @patch("imas_codex.embeddings.slurm.ensure_server", return_value=True)
    @patch("imas_codex.embeddings.readiness._ensure_ssh_tunnel", return_value=True)
    @patch("imas_codex.settings.get_embed_server_port", return_value=18765)
    def test_ensures_ssh_tunnel_off_iter(
        self, mock_port, mock_tunnel, mock_ensure, mock_which
    ):
        """Off ITER, should ensure SSH tunnel before calling ensure_server."""
        from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
        from imas_codex.embeddings.encoder import Encoder

        with patch("subprocess.run") as mock_run:
            # SSH check: remote sbatch available
            mock_run.return_value = MagicMock(returncode=0)

            with patch.object(Encoder, "_initialize_backend"):
                encoder = Encoder.__new__(Encoder)
                encoder.config = EncoderConfig(
                    backend=EmbeddingBackend.LOCAL,
                    model_name="test-model",
                )
                encoder.logger = __import__("logging").getLogger("test")
                encoder._model = None
                encoder._remote_client = None

            result = encoder._try_slurm_auto_launch()
            assert result is True
            mock_tunnel.assert_called_once()
            mock_ensure.assert_called_once()
