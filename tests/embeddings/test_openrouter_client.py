"""Tests for embeddings/openrouter_client.py module."""

from unittest.mock import patch

import numpy as np
import pytest

from imas_mcp.embeddings.openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    create_openrouter_client,
)


class TestOpenRouterClient:
    """Tests for the OpenRouterClient class."""

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Client raises error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(OpenRouterError, match="API key required"):
            OpenRouterClient(model_name="test-model", api_key=None)

    def test_placeholder_api_key_rejected(self):
        """Client rejects placeholder API key values."""
        with pytest.raises(OpenRouterError, match="placeholder"):
            OpenRouterClient(
                model_name="test-model",
                api_key="your_api_key_here",
                base_url="https://example.com",
            )

    def test_initialization_with_valid_params(self):
        """Client initializes correctly with valid parameters."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="valid-key",
            base_url="https://openrouter.ai/api/v1",
        )

        assert client.model_name == "test-model"
        assert client.api_key == "valid-key"
        assert client.device == "api"

    def test_headers_include_authorization(self):
        """Headers include correct authorization."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_encode_single_string(self):
        """Encoding a single string returns correct shape."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        mock_embedding = [[0.1] * 384]
        with patch.object(
            client, "_make_embedding_request", return_value=mock_embedding
        ):
            result = client.encode("test text")

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 384)

    def test_encode_empty_input(self):
        """Encoding empty input returns empty array."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        result = client.encode([])

        assert result.shape == (0, 0)

    def test_save_raises_not_implemented(self):
        """Save method raises NotImplementedError."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        with pytest.raises(NotImplementedError):
            client.save("/tmp/model")

    def test_encode_multi_process_raises_not_implemented(self):
        """Multi-process encoding raises NotImplementedError."""
        client = OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        with pytest.raises(NotImplementedError):
            client.encode_multi_process([])


class TestCreateOpenRouterClient:
    """Tests for the factory function."""

    def test_factory_creates_client(self):
        """Factory function creates client correctly."""
        client = create_openrouter_client(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )

        assert client.model_name == "test-model"
