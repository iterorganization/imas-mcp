"""Tests for OpenRouter embedding client and cost tracking."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from imas_codex.embeddings.openrouter_embed import (
    EmbeddingBudgetExhaustedError,
    EmbeddingCostTracker,
    EmbeddingResult,
    EMBEDDING_MODEL_COSTS,
    MODEL_NAME_MAP,
    OpenRouterEmbeddingClient,
    OpenRouterEmbeddingError,
    OpenRouterServerInfo,
    estimate_embedding_cost,
    get_openrouter_client,
    get_openrouter_model_name,
)


class TestModelNameMapping:
    """Tests for model name conversion between HuggingFace and OpenRouter formats."""

    def test_huggingface_to_openrouter(self):
        """Test HuggingFace model name converts to OpenRouter format."""
        result = get_openrouter_model_name("Qwen/Qwen3-Embedding-0.6B")
        assert result == "qwen/qwen3-embedding-0.6b"

    def test_already_openrouter_format(self):
        """Test OpenRouter format passes through unchanged."""
        result = get_openrouter_model_name("qwen/qwen3-embedding-0.6b")
        assert result == "qwen/qwen3-embedding-0.6b"

    def test_unsupported_model_raises(self):
        """Test unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            get_openrouter_model_name("some/unsupported-model")

    def test_model_name_map_consistency(self):
        """Test MODEL_NAME_MAP values are all lowercase."""
        for hf_name, openrouter_name in MODEL_NAME_MAP.items():
            assert openrouter_name == openrouter_name.lower()


class TestEmbeddingCostTracker:
    """Tests for EmbeddingCostTracker cost tracking."""

    def test_initial_state(self):
        """Test tracker starts with zero costs."""
        tracker = EmbeddingCostTracker()
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert tracker.request_count == 0
        assert tracker.limit_usd is None

    def test_record_updates_totals(self):
        """Test recording updates all tracking fields."""
        tracker = EmbeddingCostTracker()
        cost = tracker.record(1000, "qwen/qwen3-embedding-0.6b")

        assert tracker.total_tokens == 1000
        assert tracker.request_count == 1
        assert tracker.total_cost_usd > 0
        assert cost == tracker.total_cost_usd

    def test_multiple_records_accumulate(self):
        """Test multiple records accumulate correctly."""
        tracker = EmbeddingCostTracker()
        tracker.record(1000, "qwen/qwen3-embedding-0.6b")
        tracker.record(2000, "qwen/qwen3-embedding-0.6b")

        assert tracker.total_tokens == 3000
        assert tracker.request_count == 2

    def test_budget_not_exhausted_without_limit(self):
        """Test unlimited tracker is never exhausted."""
        tracker = EmbeddingCostTracker()
        tracker.record(1_000_000, "qwen/qwen3-embedding-0.6b")

        assert not tracker.is_exhausted()
        assert tracker.remaining_usd() is None

    def test_budget_exhausted_with_limit(self):
        """Test tracker respects budget limit."""
        tracker = EmbeddingCostTracker(limit_usd=0.0001)
        tracker.record(1_000_000, "qwen/qwen3-embedding-0.6b")

        assert tracker.is_exhausted()
        assert tracker.remaining_usd() == 0.0

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        tracker = EmbeddingCostTracker(limit_usd=1.0)
        tracker.record(1000, "qwen/qwen3-embedding-0.6b")

        remaining = tracker.remaining_usd()
        assert remaining is not None
        assert remaining < 1.0
        assert remaining > 0.99  # Cost of 1000 tokens is tiny

    def test_summary_format(self):
        """Test summary returns readable string."""
        tracker = EmbeddingCostTracker(limit_usd=1.0)
        tracker.record(1000, "qwen/qwen3-embedding-0.6b")

        summary = tracker.summary()
        assert "$" in summary
        assert "spent" in summary
        assert "limit" in summary
        assert "requests" in summary
        assert "tokens" in summary


class TestEstimateEmbeddingCost:
    """Tests for cost estimation function."""

    def test_known_model_cost(self):
        """Test cost estimation for known model."""
        cost = estimate_embedding_cost(1_000_000, "qwen/qwen3-embedding-0.6b")
        expected = EMBEDDING_MODEL_COSTS["qwen/qwen3-embedding-0.6b"]
        assert cost == expected

    def test_unknown_model_uses_default(self):
        """Test unknown model uses default pricing."""
        cost = estimate_embedding_cost(1_000_000, "unknown/model")
        # Default is 0.10 per 1M tokens
        assert cost == 0.10

    def test_zero_tokens_zero_cost(self):
        """Test zero tokens has zero cost."""
        cost = estimate_embedding_cost(0, "qwen/qwen3-embedding-0.6b")
        assert cost == 0.0

    def test_case_insensitive_model_lookup(self):
        """Test model lookup is case-insensitive."""
        cost_lower = estimate_embedding_cost(1000, "qwen/qwen3-embedding-0.6b")
        cost_upper = estimate_embedding_cost(1000, "QWEN/QWEN3-EMBEDDING-0.6B")
        assert cost_lower == cost_upper


class TestOpenRouterEmbeddingClient:
    """Tests for OpenRouter embedding client."""

    def test_init_with_hf_model_name(self):
        """Test client accepts HuggingFace model name."""
        client = OpenRouterEmbeddingClient(
            api_key="test-key",
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )
        assert client.model_name == "qwen/qwen3-embedding-0.6b"

    def test_init_with_openrouter_model_name(self):
        """Test client accepts OpenRouter model name."""
        client = OpenRouterEmbeddingClient(
            api_key="test-key",
            model_name="qwen/qwen3-embedding-0.6b"
        )
        assert client.model_name == "qwen/qwen3-embedding-0.6b"

    def test_is_available_without_key(self):
        """Test is_available returns False without API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = OpenRouterEmbeddingClient(api_key=None)
            # Clear any existing env vars
            client.api_key = None
            assert not client.is_available()

    def test_is_available_with_placeholder_key(self):
        """Test is_available returns False with placeholder key."""
        client = OpenRouterEmbeddingClient(api_key="your_key_here")
        assert not client.is_available()

    def test_is_available_with_valid_key(self):
        """Test is_available returns True with valid key."""
        client = OpenRouterEmbeddingClient(api_key="sk-valid-key-12345")
        assert client.is_available()

    def test_get_info_returns_server_info(self):
        """Test get_info returns OpenRouterServerInfo."""
        client = OpenRouterEmbeddingClient(api_key="test-key")
        info = client.get_info()

        assert info is not None
        assert isinstance(info, OpenRouterServerInfo)
        assert info.model == "qwen/qwen3-embedding-0.6b"
        assert info.dimension == 1024

    def test_embed_without_key_raises(self):
        """Test embed raises error without API key."""
        client = OpenRouterEmbeddingClient(api_key=None)
        client.api_key = None

        with pytest.raises(OpenRouterEmbeddingError, match="API key not configured"):
            client.embed(["test"])

    def test_embed_empty_list_returns_empty(self):
        """Test embed with empty list returns empty array."""
        client = OpenRouterEmbeddingClient(api_key="test-key")
        result = client.embed([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_embed_with_exhausted_tracker_raises(self):
        """Test embed raises when cost tracker exhausted."""
        client = OpenRouterEmbeddingClient(api_key="test-key")
        tracker = EmbeddingCostTracker(limit_usd=0.0)
        tracker.total_cost_usd = 1.0  # Force exhausted

        with pytest.raises(EmbeddingBudgetExhaustedError):
            client.embed(["test"], cost_tracker=tracker)

    @patch("httpx.Client")
    def test_embed_with_cost_returns_result(self, mock_client_class):
        """Test embed_with_cost returns EmbeddingResult with cost info."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * 1024}],
            "usage": {"total_tokens": 100}
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=False)
        
        client = OpenRouterEmbeddingClient(api_key="test-key")
        client._client = mock_client

        result = client.embed_with_cost(["test text"])

        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (1, 1024)
        assert result.total_tokens == 100
        assert result.cost_usd > 0
        assert result.model == "qwen/qwen3-embedding-0.6b"

    @patch("httpx.Client")
    def test_embed_normalizes_by_default(self, mock_client_class):
        """Test embeddings are normalized by default."""
        # Mock response with non-normalized embedding
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [3.0, 4.0] + [0.0] * 1022}],
            "usage": {"total_tokens": 10}
        }
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        
        client = OpenRouterEmbeddingClient(api_key="test-key")
        client._client = mock_client

        result = client.embed(["test"], normalize=True)
        
        # Check L2 norm is 1.0
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.001


class TestGetOpenRouterClient:
    """Tests for get_openrouter_client factory function."""

    def test_returns_none_without_api_key(self):
        """Test returns None when API key not configured."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("imas_codex.embeddings.openrouter_embed.os.getenv", return_value=None):
                client = get_openrouter_client(api_key=None)
                # Client created but not available
                assert client is None or not client.is_available()

    def test_returns_client_with_api_key(self):
        """Test returns configured client with API key."""
        client = get_openrouter_client(api_key="test-key")
        
        assert client is not None
        assert isinstance(client, OpenRouterEmbeddingClient)
        assert client.is_available()


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_fields(self):
        """Test EmbeddingResult has all expected fields."""
        result = EmbeddingResult(
            embeddings=np.array([[0.1] * 1024]),
            total_tokens=100,
            cost_usd=0.002,
            model="qwen/qwen3-embedding-0.6b",
            elapsed_seconds=1.5
        )

        assert result.embeddings.shape == (1, 1024)
        assert result.total_tokens == 100
        assert result.cost_usd == 0.002
        assert result.model == "qwen/qwen3-embedding-0.6b"
        assert result.elapsed_seconds == 1.5
