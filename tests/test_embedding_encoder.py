import os
from pathlib import Path

import numpy as np

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder


def test_embedding_encoder_build_and_embed(tmp_path: Path):
    # Use a small batch size to exercise batching logic even for few texts
    config = EncoderConfig(batch_size=2, use_rich=False, enable_cache=True)
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
