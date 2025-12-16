"""
Integration tests for different embedding models.

This module tests the system with both local SentenceTransformer models
and API-based models (mocked and real).
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.embeddings.config import EncoderConfig
from imas_codex.embeddings.encoder import Encoder

# Define models to test
LOCAL_MODEL = "all-MiniLM-L6-v2"
API_MODEL = "qwen/qwen3-embedding-4b"


@pytest.fixture
def api_config_available():
    """Check if API configuration is available."""
    return bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(
            LOCAL_MODEL, marks=pytest.mark.skip(reason="Requires sentence-transformers")
        ),
        pytest.param(API_MODEL, marks=pytest.mark.api_embedding),
    ],
)
def test_embedding_generation_mocked(model_name):
    """
    Test embedding generation with mocked backend.

    This ensures that the integration logic works for both local and API models
    without requiring actual API keys or network access.
    """
    if "/" in model_name:
        # Mock API environment
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
            },
        ):
            config = EncoderConfig(model_name=model_name)

            # Mock requests for OpenRouterClient
            with patch(
                "imas_codex.embeddings.openrouter_client.requests.post"
            ) as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1024}]
                }
                mock_post.return_value = mock_response

                encoder = Encoder(config)
                # Encoder uses embed_texts, not encode
                embedding = encoder.embed_texts(["test text"])

                assert embedding.shape[1] == 1024
                mock_post.assert_called()

    else:
        # For local model, we can mock SentenceTransformer to avoid downloading models
        config = EncoderConfig(model_name=model_name)

        with patch("imas_codex.embeddings.encoder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            # Return numpy array to match Encoder.embed_texts return type
            mock_model.encode.return_value = np.array([[0.1] * 384])

            encoder = Encoder(config)
            # Encoder uses embed_texts, not encode
            embedding = encoder.embed_texts(["test text"])

            assert embedding.shape[1] == 384
            mock_model.encode.assert_called_once()


@pytest.mark.api_embedding
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
@pytest.mark.skipif(not os.getenv("OPENAI_BASE_URL"), reason="Requires OPENAI_BASE_URL")
def test_real_api_embedding():
    """
    Test actual API embedding generation.

    This test hits the real API and requires environment variables.
    It is marked with @pytest.mark.api_embedding so it can be selected/deselected.
    """
    # Use the original configured model from .env if available (preserved by conftest)
    # otherwise default to the hardcoded API_MODEL
    model_name = os.getenv("IMAS_CODEX_ORIGINAL_EMBEDDING_MODEL", API_MODEL)

    # Ensure we are actually testing an API model
    if "/" not in model_name and model_name != API_MODEL:
        model_name = API_MODEL

    config = EncoderConfig(model_name=model_name)

    if not config.openai_api_key:
        pytest.skip("OPENAI_API_KEY not found in configuration (checked env and .env)")

    # This will fail if keys are invalid
    encoder = Encoder(config)

    text = "This is a test sentence for IMAS MCP embedding verification."
    embedding = encoder.embed_texts([text])

    # Verify we didn't fall back to local model
    if encoder.config.model_name != model_name:
        pytest.fail(
            f"Encoder fell back to local model: {encoder.config.model_name}. "
            "Check logs for API errors (e.g. 401 Unauthorized)."
        )

    assert embedding.shape[0] == 1
    assert embedding.shape[1] > 0  # Dimension depends on model
