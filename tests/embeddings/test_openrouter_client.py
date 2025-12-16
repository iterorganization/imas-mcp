"""Tests for embeddings/openrouter_client.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from imas_codex.embeddings.openrouter_client import (
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

    def test_factory_passes_kwargs(self):
        """Factory passes additional kwargs to client."""
        client = create_openrouter_client(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
            batch_size=100,
            max_retries=5,
        )

        assert client.batch_size == 100
        assert client.max_retries == 5


class TestOpenRouterClientRequests:
    """Tests for OpenRouterClient request handling."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
            max_retries=1,
            retry_delay=0.01,
        )

    def test_make_embedding_request_success(self, client):
        """Embedding request returns embeddings on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 384}]}

        with patch("requests.post", return_value=mock_response):
            result = client._make_embedding_request(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 384

    def test_make_embedding_request_rate_limited(self, client):
        """Embedding request retries on rate limit."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"data": [{"embedding": [0.1] * 384}]}

        with patch("requests.post", side_effect=[mock_429, mock_200]):
            result = client._make_embedding_request(["test text"])

            assert len(result) == 1

    def test_make_embedding_request_rate_limit_exceeded(self, client):
        """Embedding request fails after max retries on rate limit."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        with patch("requests.post", return_value=mock_429):
            with pytest.raises(OpenRouterError, match="Rate limit exceeded"):
                client._make_embedding_request(["test text"])

    def test_make_embedding_request_api_error(self, client):
        """Embedding request fails on API error after retries."""
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_500):
            with pytest.raises(OpenRouterError, match="API request failed"):
                client._make_embedding_request(["test text"])

    def test_make_embedding_request_connection_error(self, client):
        """Embedding request fails on connection error."""
        with patch(
            "requests.post",
            side_effect=requests.RequestException("Connection failed"),
        ):
            with pytest.raises(OpenRouterError, match="Failed to make embedding"):
                client._make_embedding_request(["test text"])

    def test_encode_batching(self, client):
        """Encode processes texts in batches."""
        client.batch_size = 2
        mock_embeddings = [[0.1] * 384, [0.2] * 384]

        with patch.object(
            client, "_make_embedding_request", return_value=mock_embeddings
        ) as mock_request:
            result = client.encode(["text1", "text2", "text3", "text4"])

            # Should be called twice (4 texts / 2 batch size)
            assert mock_request.call_count == 2
            assert result.shape[0] == 4

    def test_encode_normalization(self, client):
        """Encode normalizes embeddings when requested."""
        mock_embedding = [[1.0, 0.0, 0.0]]

        with patch.object(
            client, "_make_embedding_request", return_value=mock_embedding
        ):
            result = client.encode(["test"], normalize_embeddings=True)

            # Normalized embedding should have unit length
            norm = np.linalg.norm(result[0])
            assert np.isclose(norm, 1.0)

    def test_encode_no_normalization(self, client):
        """Encode preserves original embeddings when normalization disabled."""
        mock_embedding = [[2.0, 0.0, 0.0]]

        with patch.object(
            client, "_make_embedding_request", return_value=mock_embedding
        ):
            result = client.encode(["test"], normalize_embeddings=False)

            assert result[0][0] == 2.0

    def test_get_sentence_embedding_dimension(self, client):
        """get_sentence_embedding_dimension returns correct dimension."""
        mock_embedding = [0.1] * 384

        with patch.object(
            client, "_make_embedding_request", return_value=[mock_embedding]
        ):
            dim = client.get_sentence_embedding_dimension()

            assert dim == 384

    def test_get_sentence_embedding_dimension_error(self, client):
        """get_sentence_embedding_dimension raises error on failure."""
        with patch.object(
            client, "_make_embedding_request", side_effect=Exception("API failed")
        ):
            with pytest.raises(OpenRouterError, match="Cannot determine"):
                client.get_sentence_embedding_dimension()

    def test_test_connection_success(self, client):
        """_test_connection succeeds on valid response."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            client._test_connection()  # Should not raise

    def test_test_connection_failure(self, client):
        """_test_connection raises error on API failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(OpenRouterError, match="connection test failed"):
                client._test_connection()


class TestOpenRouterChatCompletion:
    """Tests for OpenRouterClient chat completion."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenRouterClient(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
            max_retries=1,
            retry_delay=0.01,
        )

    def test_make_chat_request_success(self, client):
        """Chat request returns response content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, world!"}}]
        }

        with patch("requests.post", return_value=mock_response):
            result = client.make_chat_request(
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert result == "Hello, world!"

    def test_make_chat_request_with_model_override(self, client):
        """Chat request uses provided model override."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            client.make_chat_request(
                messages=[{"role": "user", "content": "Hi"}],
                model="different-model",
            )

            call_json = mock_post.call_args[1]["json"]
            assert call_json["model"] == "different-model"

    def test_make_chat_request_rate_limited(self, client):
        """Chat request retries on rate limit."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limited"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"choices": [{"message": {"content": "Response"}}]}

        with patch("requests.post", side_effect=[mock_429, mock_200]):
            result = client.make_chat_request(
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert result == "Response"

    def test_make_chat_request_api_error(self, client):
        """Chat request fails on API error after retries."""
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_500):
            with pytest.raises(OpenRouterError, match="Chat request failed"):
                client.make_chat_request(messages=[{"role": "user", "content": "Hi"}])

    def test_make_chat_request_connection_error(self, client):
        """Chat request fails on connection error."""
        with patch(
            "requests.post",
            side_effect=requests.RequestException("Connection failed"),
        ):
            with pytest.raises(OpenRouterError, match="Failed to make chat"):
                client.make_chat_request(messages=[{"role": "user", "content": "Hi"}])
