"""Tests for remote embedding client and server."""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.embeddings.client import RemoteEmbeddingClient
from imas_codex.embeddings.config import EncoderConfig


class TestRemoteEmbeddingClient:
    """Tests for RemoteEmbeddingClient."""

    def test_initialization(self):
        """Client initializes with URL."""
        client = RemoteEmbeddingClient("http://localhost:18765")
        assert client.base_url == "http://localhost:18765"

    def test_initialization_trailing_slash(self):
        """Client strips trailing slashes from URL."""
        client = RemoteEmbeddingClient("http://localhost:18765/")
        assert client.base_url == "http://localhost:18765"

    def test_is_available_returns_false_when_server_down(self):
        """is_available returns False when server is unreachable."""
        client = RemoteEmbeddingClient("http://localhost:99999")
        assert client.is_available() is False

    @patch("imas_codex.embeddings.client.httpx.Client")
    def test_is_available_returns_true_on_healthy_response(self, mock_client_cls):
        """is_available returns True when server responds with healthy status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = RemoteEmbeddingClient("http://localhost:18765")
        assert client.is_available() is True

    @patch("imas_codex.embeddings.client.httpx.Client")
    def test_embed_returns_embeddings(self, mock_client_cls):
        """embed returns list of embeddings from server."""
        import numpy as np

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model": "test-model",
            "dimension": 3,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = RemoteEmbeddingClient("http://localhost:18765")
        # Force client to use our mock
        client._client = mock_client
        result = client.embed(["text1", "text2"])

        assert len(result) == 2
        # Result is numpy array
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])

    @patch("imas_codex.embeddings.client.httpx.Client")
    def test_embed_sends_dimension(self, mock_client_cls):
        """When dimension is passed, it appears in the request body JSON."""
        import numpy as np

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 512],
            "model": "test-model",
            "dimension": 512,
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = RemoteEmbeddingClient("http://localhost:18765")
        client._client = mock_client
        client.embed(["text1"], dimension=512)

        call_kwargs = mock_client.post.call_args
        sent_body = call_kwargs[1]["json"]
        assert sent_body["dimension"] == 512

    @patch("imas_codex.embeddings.client.httpx.Client")
    def test_embed_omits_dimension_when_none(self, mock_client_cls):
        """When dimension is not passed, the key is absent from the request body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "model": "test-model",
            "dimension": 3,
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = RemoteEmbeddingClient("http://localhost:18765")
        client._client = mock_client
        client.embed(["text1"])

        call_kwargs = mock_client.post.call_args
        sent_body = call_kwargs[1]["json"]
        assert "dimension" not in sent_body

    @patch("imas_codex.embeddings.client.httpx.Client")
    def test_embed_sends_configured_dimension(self, mock_client_cls):
        """Encoder passes get_embedding_dimension() value through to the client."""
        import numpy as np

        from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
        from imas_codex.embeddings.encoder import Encoder
        from imas_codex.settings import get_embedding_dimension

        expected_dim = get_embedding_dimension()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [list(np.ones(expected_dim, dtype=float))],
            "model": "test-model",
            "dimension": expected_dim,
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        config = EncoderConfig(
            remote_url="http://localhost:18765",
            backend=EmbeddingBackend.REMOTE,
        )
        encoder = Encoder(config=config)
        # Inject mock directly so no real HTTP connection is made
        from imas_codex.embeddings.client import RemoteEmbeddingClient as REC

        encoder._remote_client = REC("http://localhost:18765")
        encoder._remote_client._client = mock_client
        # Skip health-check validation — we control the mock
        encoder._backend_validated = True

        encoder.embed_texts(["hello world"])

        call_kwargs = mock_client.post.call_args
        sent_body = call_kwargs[1]["json"]
        assert sent_body["dimension"] == expected_dim


class TestEncoderConfigRemote:
    """Tests for remote embedding configuration."""

    def test_config_loads_remote_url_for_remote_backend(self, monkeypatch):
        """EncoderConfig loads remote_url from settings when backend=REMOTE."""
        from imas_codex.embeddings.config import EmbeddingBackend

        # Override the conftest "local" location so get_embed_remote_url()
        # returns a URL instead of None.
        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_LOCATION", "iter")
        config = EncoderConfig(backend=EmbeddingBackend.REMOTE)
        # Should have loaded default from settings
        assert config.remote_url is not None

    def test_config_explicit_remote_url_overrides_settings(self):
        """Explicit remote_url parameter overrides settings."""
        from imas_codex.embeddings.config import EmbeddingBackend

        config = EncoderConfig(
            remote_url="http://custom:9999", backend=EmbeddingBackend.REMOTE
        )
        assert config.remote_url == "http://custom:9999"

    def test_config_local_backend_skips_url_loading(self):
        """When backend=LOCAL, remote_url stays None if not set."""
        from imas_codex.embeddings.config import EmbeddingBackend

        config = EncoderConfig(backend=EmbeddingBackend.LOCAL)
        assert config.remote_url is None

    def test_config_serialization_includes_backend_field(self):
        """to_dict includes backend configuration fields."""
        from imas_codex.embeddings.config import EmbeddingBackend

        config = EncoderConfig(
            remote_url="http://test:8080", backend=EmbeddingBackend.REMOTE
        )
        data = config.to_dict()

        assert data["remote_url"] == "http://test:8080"
        assert data["backend"] == "remote"


@pytest.mark.skipif(True, reason="Integration test - requires running server")
class TestRemoteIntegration:
    """Integration tests requiring running server."""

    def test_embed_via_remote_server(self):
        """Full integration test with running server."""
        client = RemoteEmbeddingClient("http://localhost:18765")
        if not client.is_available():
            pytest.skip("Server not running")

        result = client.embed(["test embedding text"])
        assert len(result) == 1
        assert len(result[0]) > 0  # Should have some dimensions
