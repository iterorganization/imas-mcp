import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder

sentence_transformers = pytest.importorskip(
    "sentence_transformers", reason="Requires sentence-transformers extra"
)


def test_embedding_encoder_build_and_embed(tmp_path: Path):
    # Use a small batch size to exercise batching logic even for few texts
    # Force local embeddings to avoid API calls
    config = EncoderConfig(
        batch_size=2,
        use_rich=False,
        enable_cache=True,
        use_api_embeddings=False,
        model_name="all-MiniLM-L6-v2",  # Explicitly set local model to avoid env var override
    )
    encoder = Encoder(config)

    texts = ["alpha", "beta", "gamma"]
    ids = ["a", "b", "c"]
    embeddings, out_ids, was_cached = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=None, force_rebuild=True
    )
    assert not was_cached
    assert list(out_ids) == ids
    assert embeddings.shape[0] == len(texts)
    # sentence-transformers returns float32/float16 (or float64 depending on numpy version)
    assert embeddings.dtype in (np.float32, np.float16, np.float64)

    # Second call should load from cache (force_rebuild False)
    embeddings2, out_ids2, was_cached2 = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=None, force_rebuild=False
    )
    assert was_cached2
    assert np.allclose(embeddings, embeddings2)
    assert out_ids2 == out_ids


def test_embedding_encoder_ad_hoc_embed():
    config = EncoderConfig(use_rich=False, enable_cache=False)
    encoder = Encoder(config)
    vecs = encoder.embed_texts(["one", "two"])
    assert vecs.shape[0] == 2
    # sentence-transformers returns float32/float16 (or float64 depending on numpy version)
    assert vecs.dtype in (np.float32, np.float16, np.float64)


def test_api_fallback_to_local():
    """Test that Encoder falls back to local model if API fails."""
    config = EncoderConfig(
        model_name="openai/text-embedding-3-small",
        openai_api_key="fake-key",
        openai_base_url="https://fake.url",
        use_rich=False,
        enable_cache=False,
    )

    # Mock OpenRouterClient to fail
    with patch(
        "imas_mcp.embeddings.encoder.OpenRouterClient",
        side_effect=Exception("API Error"),
    ):
        # Mock SentenceTransformer to fail for API model name but succeed for fallback
        with patch("imas_mcp.embeddings.encoder.SentenceTransformer") as mock_st:

            def st_side_effect(model_name, **kwargs):
                if model_name == "openai/text-embedding-3-small":
                    raise ValueError("Not a local model")
                # Return a mock for other models (fallback)
                mock_model = MagicMock()
                mock_model.encode.return_value = np.array(
                    [[0.1, 0.2]], dtype=np.float32
                )
                mock_model.device = "cpu"
                return mock_model

            mock_st.side_effect = st_side_effect

            encoder = Encoder(config)

            # Trigger model loading
            encoder.embed_texts(["test"])

            # Verify fallback occurred
            assert config.use_api_embeddings is False
            # Should have tried to load local fallback model
            assert mock_st.call_count >= 1
            # The model name in config should have been updated to fallback
            assert config.model_name == "all-MiniLM-L6-v2"
