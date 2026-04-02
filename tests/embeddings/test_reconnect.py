"""Tests for embed URL reconnection on SLURM compute node migration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.embeddings.client import RemoteEmbeddingClient
from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
from imas_codex.embeddings.encoder import Encoder


class TestRemoteClientUrlUpdate:
    """Tests for RemoteEmbeddingClient.update_base_url."""

    def test_update_base_url_changes_url(self):
        client = RemoteEmbeddingClient("http://old-node:18765")
        client.update_base_url("http://new-node:18765")
        assert client.base_url == "http://new-node:18765"

    def test_update_base_url_strips_trailing_slash(self):
        client = RemoteEmbeddingClient("http://old-node:18765")
        client.update_base_url("http://new-node:18765/")
        assert client.base_url == "http://new-node:18765"

    def test_update_base_url_noop_same_url(self):
        client = RemoteEmbeddingClient("http://node:18765")
        client._client = MagicMock()
        client.update_base_url("http://node:18765")
        client._client.close.assert_not_called()

    def test_update_base_url_closes_old_client(self):
        client = RemoteEmbeddingClient("http://old-node:18765")
        mock_http = MagicMock()
        client._client = mock_http
        client.update_base_url("http://new-node:18765")
        mock_http.close.assert_called_once()
        assert client._client is None


class TestEncoderReconnect:
    """Tests for Encoder._reconnect_remote on URL change."""

    @patch("imas_codex.embeddings.encoder.Encoder._load_model")
    def test_reconnect_updates_url_on_change(self, _mock_load):
        config = EncoderConfig(
            remote_url="http://old-node:18765",
            backend=EmbeddingBackend.REMOTE,
            model_name="test-model",
        )
        encoder = Encoder(config=config)
        encoder._remote_client = MagicMock(spec=RemoteEmbeddingClient)
        encoder._backend_validated = True

        with (
            patch("imas_codex.remote.locations._service_url_cache", {}),
            patch(
                "imas_codex.settings.get_embed_remote_url",
                return_value="http://new-node:18765",
            ),
        ):
            changed = encoder._reconnect_remote()

        assert changed is True
        assert encoder.config.remote_url == "http://new-node:18765"
        encoder._remote_client.update_base_url.assert_called_once_with(
            "http://new-node:18765"
        )
        assert encoder._backend_validated is False

    @patch("imas_codex.embeddings.encoder.Encoder._load_model")
    def test_reconnect_noop_when_url_unchanged(self, _mock_load):
        config = EncoderConfig(
            remote_url="http://same-node:18765",
            backend=EmbeddingBackend.REMOTE,
            model_name="test-model",
        )
        encoder = Encoder(config=config)
        encoder._remote_client = MagicMock(spec=RemoteEmbeddingClient)
        encoder._backend_validated = True

        with (
            patch("imas_codex.remote.locations._service_url_cache", {}),
            patch(
                "imas_codex.settings.get_embed_remote_url",
                return_value="http://same-node:18765",
            ),
        ):
            changed = encoder._reconnect_remote()

        assert changed is False
        assert encoder._backend_validated is True


class TestEncoderEmbedTextsReconnect:
    """Tests for embed_texts retry-after-reconnect on ConnectionError."""

    @patch("imas_codex.embeddings.encoder.Encoder._load_model")
    def test_embed_texts_retries_on_connection_error(self, _mock_load):
        config = EncoderConfig(
            remote_url="http://old-node:18765",
            backend=EmbeddingBackend.REMOTE,
            model_name="test-model",
        )
        encoder = Encoder(config=config)
        encoder._backend_validated = True

        mock_client = MagicMock(spec=RemoteEmbeddingClient)
        mock_client.embed.side_effect = [
            ConnectionError("refused"),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        ]
        encoder._remote_client = mock_client

        with patch.object(encoder, "_reconnect_remote", return_value=True):
            with patch.object(encoder, "_validate_remote_backend"):
                result = encoder.embed_texts(["test"])

        assert result.shape == (1, 3)
        assert mock_client.embed.call_count == 2

    @patch("imas_codex.embeddings.encoder.Encoder._load_model")
    def test_embed_texts_raises_when_reconnect_fails(self, _mock_load):
        config = EncoderConfig(
            remote_url="http://old-node:18765",
            backend=EmbeddingBackend.REMOTE,
            model_name="test-model",
        )
        encoder = Encoder(config=config)
        encoder._backend_validated = True

        mock_client = MagicMock(spec=RemoteEmbeddingClient)
        mock_client.embed.side_effect = ConnectionError("refused")
        encoder._remote_client = mock_client

        with patch.object(encoder, "_reconnect_remote", return_value=False):
            with patch.object(encoder, "_validate_remote_backend"):
                with pytest.raises(ConnectionError):
                    encoder.embed_texts(["test"])
