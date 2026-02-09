import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed (optional GPU dependency)",
)

from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
from imas_codex.embeddings.encoder import Encoder


@pytest.mark.slow
def test_embedding_encoder_build_and_embed(tmp_path: Path):
    # Use a small batch size to exercise batching logic even for few texts
    config = EncoderConfig(
        batch_size=2,
        use_rich=False,
        enable_cache=True,
        backend=EmbeddingBackend.LOCAL,
        model_name="all-MiniLM-L6-v2",  # Explicitly set small model for integration test
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


@pytest.mark.slow
def test_embedding_encoder_ad_hoc_embed():
    config = EncoderConfig(
        use_rich=False,
        enable_cache=False,
        backend=EmbeddingBackend.LOCAL,
        model_name="all-MiniLM-L6-v2",  # Explicitly set small model for integration test
    )
    encoder = Encoder(config)
    vecs = encoder.embed_texts(["one", "two"])
    assert vecs.shape[0] == 2
    # sentence-transformers returns float32/float16 (or float64 depending on numpy version)
    assert vecs.dtype in (np.float32, np.float16, np.float64)


@pytest.mark.slow
def test_model_failure_raises_error():
    """Test that Encoder raises EmbeddingBackendError if model fails to load."""
    from imas_codex.embeddings.encoder import EmbeddingBackendError

    config = EncoderConfig(
        model_name="nonexistent/model-name",
        use_rich=False,
        enable_cache=False,
        backend=EmbeddingBackend.LOCAL,
    )

    # Mock SentenceTransformer to fail for the nonexistent model
    with patch("sentence_transformers.SentenceTransformer") as mock_st:

        def st_side_effect(model_name, **kwargs):
            raise ValueError("Model not found")

        mock_st.side_effect = st_side_effect

        with pytest.raises(
            EmbeddingBackendError, match="Failed to load embedding model"
        ):
            Encoder(config)
