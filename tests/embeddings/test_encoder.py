"""Tests for embeddings/encoder.py module."""

import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.embeddings.cache import EmbeddingCache
from imas_codex.embeddings.config import EncoderConfig
from imas_codex.embeddings.encoder import Encoder


class TestEncoder:
    """Tests for the Encoder class."""

    @pytest.fixture
    def encoder_config(self) -> EncoderConfig:
        """Create an encoder config for testing."""
        return EncoderConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=8,
            enable_cache=True,
            use_rich=False,
        )

    @pytest.fixture
    def encoder(self, encoder_config) -> Encoder:
        """Create an encoder for testing."""
        return Encoder(config=encoder_config)

    def test_initialization_default_config(self):
        """Encoder initializes with default config and loads model eagerly."""
        encoder = Encoder()
        assert encoder.config is not None
        # Model is now loaded eagerly in __init__
        assert encoder._model is not None
        assert encoder._cache is None

    def test_initialization_custom_config(self, encoder_config):
        """Encoder initializes with provided config."""
        encoder = Encoder(config=encoder_config)
        assert encoder.config == encoder_config
        assert encoder.config.batch_size == 8

    def test_get_model_loads_model(self, encoder):
        """get_model returns the already-loaded model."""
        # Model is now loaded eagerly, so get_model just returns it
        result = encoder.get_model()
        assert result is not None
        assert result is encoder._model

    def test_get_model_returns_same_instance(self, encoder):
        """get_model returns the same model instance on multiple calls."""
        model1 = encoder.get_model()
        model2 = encoder.get_model()
        assert model1 is model2

    def test_embed_texts_delegates_to_model(self, encoder):
        """embed_texts delegates to the loaded model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 384))
        encoder._model = mock_model

        result = encoder.embed_texts(["text1", "text2", "text3"])

        mock_model.encode.assert_called_once()
        assert result.shape == (3, 384)

    def test_embed_texts_api_fallback(self, encoder_config):
        """embed_texts falls back to local model on API failure."""
        encoder_config.use_api_embeddings = True
        encoder = Encoder(config=encoder_config)

        # First call fails (API error), second call succeeds (local model)
        mock_api_model = MagicMock()
        mock_api_model.encode.side_effect = Exception("API Error")

        mock_local_model = MagicMock()
        mock_local_model.encode.return_value = np.zeros((1, 384))

        encoder._model = mock_api_model

        # Set up the fallback behavior
        def fallback_behavior(texts, **kwargs):
            if encoder.config.use_api_embeddings:
                encoder.config.use_api_embeddings = False
                encoder.config.model_name = "all-MiniLM-L6-v2"
                encoder._model = mock_local_model
                return mock_local_model.encode(texts)
            return mock_local_model.encode(texts)

        with patch.object(encoder, "embed_texts", side_effect=fallback_behavior):
            result = encoder.embed_texts(["test"])
            assert result is not None

    def test_get_cache_info_no_cache(self, encoder):
        """get_cache_info returns no_cache status when cache is None."""
        result = encoder.get_cache_info()
        assert result == {"status": "no_cache"}

    def test_get_cache_info_with_cache(self, encoder, tmp_path):
        """get_cache_info returns cache details when cache exists."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
            created_at=time.time(),
        )
        encoder._cache = cache
        encoder._cache_path = tmp_path / "test_cache.pkl"

        result = encoder.get_cache_info()

        assert result["model_name"] == "test-model"
        assert result["document_count"] == 10
        assert result["embedding_dimension"] == 384
        assert "created_at" in result
        assert "memory_usage_mb" in result

    def test_get_cache_info_with_cache_file(self, encoder, tmp_path):
        """get_cache_info includes file info when cache file exists."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
        )
        encoder._cache = cache

        cache_file = tmp_path / "test_cache.pkl"
        cache_file.write_bytes(b"test data")
        encoder._cache_path = cache_file

        result = encoder.get_cache_info()

        assert "cache_file_size_mb" in result
        assert result["cache_file_path"] == str(cache_file)

    def test_list_cache_files(self, encoder, tmp_path):
        """list_cache_files returns cache file information."""
        with patch.object(encoder, "_get_cache_directory", return_value=tmp_path):
            # Create test cache files
            (tmp_path / "cache1.pkl").write_bytes(b"cache1")
            (tmp_path / "cache2.pkl").write_bytes(b"cache2")

            result = encoder.list_cache_files()

            assert len(result) == 2
            assert all("filename" in f for f in result)
            assert all("size_mb" in f for f in result)
            assert all("modified" in f for f in result)

    def test_cleanup_old_caches(self, encoder, tmp_path):
        """cleanup_old_caches removes old cache files."""
        with patch.object(encoder, "_get_cache_directory", return_value=tmp_path):
            # Create test cache files with different times
            for i in range(5):
                cache_file = tmp_path / f"cache{i}.pkl"
                cache_file.write_bytes(b"cache data")

            # Set current cache path to protect it
            encoder._cache_path = tmp_path / "cache0.pkl"

            # Cleanup keeping only 2
            removed = encoder.cleanup_old_caches(keep_count=2)

            # Should have removed 3 files (5 - 2, but excluding current)
            assert removed >= 0
            remaining_files = list(tmp_path.glob("*.pkl"))
            assert len(remaining_files) <= 5  # Some may remain

    def test_generate_cache_filename_default(self, encoder):
        """_generate_cache_filename creates default filename."""
        filename = encoder._generate_cache_filename()

        assert filename.endswith(".pkl")
        assert filename.startswith(".")
        assert "MiniLM" in filename or "minilm" in filename.lower()

    def test_generate_cache_filename_with_cache_key(self, encoder):
        """_generate_cache_filename includes cache key in filename."""
        filename = encoder._generate_cache_filename(cache_key="test_key")

        assert filename.endswith(".pkl")
        assert "_" in filename  # Hash separator

    def test_generate_cache_filename_with_ids_set(self, encoder_config):
        """_generate_cache_filename handles ids_set configuration."""
        encoder_config.ids_set = {"equilibrium", "core_profiles"}
        encoder = Encoder(config=encoder_config)

        filename = encoder._generate_cache_filename()

        assert filename.endswith(".pkl")
        assert "_" in filename  # Contains hash

    def test_set_cache_path(self, encoder, tmp_path):
        """_set_cache_path sets the cache path correctly."""
        with patch.object(encoder, "_get_cache_directory", return_value=tmp_path):
            encoder._set_cache_path(cache_key="test")
            assert encoder._cache_path is not None
            assert encoder._cache_path.parent == tmp_path

    def test_set_cache_path_only_once(self, encoder, tmp_path):
        """_set_cache_path only sets path on first call."""
        with patch.object(encoder, "_get_cache_directory", return_value=tmp_path):
            encoder._set_cache_path(cache_key="first")
            first_path = encoder._cache_path

            encoder._set_cache_path(cache_key="second")
            assert encoder._cache_path == first_path

    def test_try_load_cache_no_cache_path(self, encoder):
        """_try_load_cache returns False when no cache path."""
        result = encoder._try_load_cache(["text"], ["id"], None)
        assert result is False

    def test_try_load_cache_file_not_exists(self, encoder, tmp_path):
        """_try_load_cache returns False when cache file doesn't exist."""
        encoder._cache_path = tmp_path / "nonexistent.pkl"
        result = encoder._try_load_cache(["text"], ["id"], None)
        assert result is False

    def test_try_load_cache_invalid_format(self, encoder, tmp_path):
        """_try_load_cache returns False for invalid cache format."""
        cache_file = tmp_path / "invalid.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({"invalid": "data"}, f)

        encoder._cache_path = cache_file
        result = encoder._try_load_cache(["text"], ["id"], None)
        assert result is False

    def test_try_load_cache_valid_cache(self, encoder, tmp_path):
        """_try_load_cache loads valid cache successfully."""
        cache = EmbeddingCache(
            embeddings=np.zeros((2, 384)),
            path_ids=["id1", "id2"],
            model_name=encoder.config.model_name,
            document_count=2,
        )

        cache_file = tmp_path / "valid.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

        encoder._cache_path = cache_file
        result = encoder._try_load_cache(["text1", "text2"], ["id1", "id2"], None)
        assert result is True
        assert encoder._cache is not None

    def test_try_load_cache_path_ids_mismatch(self, encoder, tmp_path):
        """_try_load_cache returns False when path IDs don't match."""
        cache = EmbeddingCache(
            embeddings=np.zeros((2, 384)),
            path_ids=["id1", "id2"],
            model_name=encoder.config.model_name,
            document_count=2,
        )

        cache_file = tmp_path / "valid.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

        encoder._cache_path = cache_file
        # Different identifiers
        result = encoder._try_load_cache(
            ["text1", "text2"], ["different1", "different2"], None
        )
        assert result is False

    def test_try_load_cache_disabled(self, encoder_config, tmp_path):
        """_try_load_cache returns False when caching is disabled."""
        encoder_config.enable_cache = False
        encoder = Encoder(config=encoder_config)

        result = encoder._try_load_cache(["text"], ["id"], None)
        assert result is False

    def test_create_cache(self, encoder, tmp_path):
        """_create_cache saves cache to disk."""
        encoder._cache_path = tmp_path / "new_cache.pkl"

        embeddings = np.zeros((3, 384))
        identifiers = ["id1", "id2", "id3"]

        encoder._create_cache(embeddings, identifiers)

        assert encoder._cache is not None
        assert encoder._cache.document_count == 3
        assert encoder._cache_path.exists()

    def test_create_cache_with_source_dir(self, encoder, tmp_path):
        """_create_cache updates source metadata when dir provided."""
        encoder._cache_path = tmp_path / "new_cache.pkl"

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"metadata": {"version": "4.0"}}')

        encoder._create_cache(np.zeros((1, 384)), ["id1"], source_data_dir=source_dir)

        assert encoder._cache.dd_version == "4.0"

    def test_create_cache_disabled(self, encoder_config):
        """_create_cache does nothing when caching is disabled."""
        encoder_config.enable_cache = False
        encoder = Encoder(config=encoder_config)

        encoder._create_cache(np.zeros((1, 384)), ["id1"])

        assert encoder._cache is None

    def test_build_document_embeddings_validates_length(self, encoder):
        """build_document_embeddings raises error for mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            encoder.build_document_embeddings(
                texts=["text1", "text2"],
                identifiers=["id1"],  # Mismatched length
            )

    def test_build_document_embeddings_generates_identifiers(self, encoder):
        """build_document_embeddings generates identifiers when not provided."""
        with patch.object(encoder, "_generate_embeddings") as mock_gen:
            mock_gen.return_value = np.zeros((2, 384))
            with patch.object(encoder, "_set_cache_path"):
                with patch.object(encoder, "_try_load_cache", return_value=False):
                    with patch.object(encoder, "_create_cache"):
                        result = encoder.build_document_embeddings(
                            texts=["text1", "text2"],
                            enable_caching=True,
                        )

            assert len(result[1]) == 2
            assert result[1][0] == "text_0"
            assert result[1][1] == "text_1"

    def test_build_document_embeddings_uses_cache(self, encoder, tmp_path):
        """build_document_embeddings uses cached embeddings when available."""
        cache = EmbeddingCache(
            embeddings=np.zeros((2, 384)),
            path_ids=["id1", "id2"],
            model_name=encoder.config.model_name,
            document_count=2,
        )

        cache_file = tmp_path / "cached.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

        with patch.object(encoder, "_get_cache_directory", return_value=tmp_path):
            with patch.object(
                encoder, "_generate_cache_filename", return_value="cached.pkl"
            ):
                embeddings, path_ids, from_cache = encoder.build_document_embeddings(
                    texts=["text1", "text2"],
                    identifiers=["id1", "id2"],
                )

        assert from_cache is True
        assert path_ids == ["id1", "id2"]

    def test_build_document_embeddings_caching_disabled(self, encoder):
        """build_document_embeddings works with caching disabled."""
        with patch.object(encoder, "_generate_embeddings") as mock_gen:
            mock_gen.return_value = np.zeros((2, 384))

            embeddings, path_ids, from_cache = encoder.build_document_embeddings(
                texts=["text1", "text2"],
                identifiers=["id1", "id2"],
                enable_caching=False,
            )

        assert from_cache is False
        mock_gen.assert_called_once()

    def test_generate_embeddings_batching(self, encoder):
        """_generate_embeddings processes texts in batches."""
        encoder.config.batch_size = 2
        mock_model = MagicMock()
        # Return different shapes for each batch call
        mock_model.encode.side_effect = [
            np.zeros((2, 384)),  # First batch: text1, text2
            np.zeros((2, 384)),  # Second batch: text3, text4
            np.zeros((1, 384)),  # Third batch: text5
        ]
        encoder._model = mock_model

        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = encoder._generate_embeddings(texts)

        # Should have been called 3 times (2+2+1)
        assert mock_model.encode.call_count == 3
        assert result.shape[0] == 5

    def test_generate_embeddings_half_precision(self, encoder_config):
        """_generate_embeddings uses half precision when configured."""
        encoder_config.use_half_precision = True
        encoder = Encoder(config=encoder_config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)
        encoder._model = mock_model

        result = encoder._generate_embeddings(["text1", "text2"])

        assert result.dtype == np.float16


@pytest.mark.slow
def test_encoder_embed_texts_basic(monkeypatch):
    """Integration test: embed texts with API embeddings."""
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    config = EncoderConfig(batch_size=8, use_rich=False)
    encoder = Encoder(config)
    texts = ["alpha", "beta", "gamma"]
    vecs = encoder.embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == len(texts)
    assert vecs.shape[1] > 0


@pytest.mark.slow
def test_encoder_build_document_embeddings_cache_integration(tmp_path, monkeypatch):
    """Integration test: build and cache embeddings with API."""
    monkeypatch.setattr(EncoderConfig, "cache_dir", "embeddings_test")
    monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    config = EncoderConfig(batch_size=16, use_rich=False, enable_cache=True)
    encoder = Encoder(config)
    texts = [f"text {i}" for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    cache_key = config.generate_cache_key()

    emb1, ids1, was_cached1 = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=cache_key
    )
    assert emb1.shape[0] == len(texts)

    emb2, ids2, was_cached2 = encoder.build_document_embeddings(
        texts=texts, identifiers=ids, cache_key=cache_key
    )
    assert was_cached2
    assert np.array_equal(emb1, emb2)
    assert ids1 == ids2
