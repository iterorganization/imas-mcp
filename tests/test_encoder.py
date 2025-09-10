import numpy as np

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder


def test_encoder_embed_texts_basic():
    config = EncoderConfig(batch_size=8, use_rich=False)
    encoder = Encoder(config)
    texts = ["alpha", "beta", "gamma"]
    vecs = encoder.embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == len(texts)
    # dimension should be > 0
    assert vecs.shape[1] > 0


def test_encoder_build_document_embeddings_cache(tmp_path, monkeypatch):
    # Use a temp embeddings cache directory
    monkeypatch.setattr(EncoderConfig, "cache_dir", "embeddings_test")
    config = EncoderConfig(batch_size=16, use_rich=False, enable_cache=True)
    encoder = Encoder(config)
    texts = [f"text {i}" for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    cache_key = config.generate_cache_key()

    emb1, ids1, was_cached1 = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=cache_key
    )
    # First call may or may not be cached depending on test order/previous runs; just assert shape
    assert emb1.shape[0] == len(texts)

    # Build again - should load from cache
    emb2, ids2, was_cached2 = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=cache_key
    )
    assert was_cached2  # second invocation should hit cache
    assert np.array_equal(emb1, emb2)
    assert ids1 == ids2
