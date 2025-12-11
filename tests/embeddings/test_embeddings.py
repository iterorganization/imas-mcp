"""Tests for embeddings/embeddings.py module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.embeddings import Embeddings
from imas_mcp.search.document_store import Document, DocumentMetadata


def create_mock_encoder_config(model_name: str = "test-model") -> MagicMock:
    """Create a mock encoder config with required attributes."""
    mock_config = MagicMock()
    mock_config.model_name = model_name
    mock_config.generate_cache_key.return_value = None
    return mock_config


class TestEmbeddings:
    """Tests for the Embeddings class."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = MagicMock()
        store._data_dir = Path("/fake/path")
        store.get_all_documents.return_value = []
        return store

    @pytest.fixture
    def mock_documents(self) -> list[Document]:
        """Create mock documents for testing."""
        docs = []
        for i in range(5):
            metadata = DocumentMetadata(
                path_id=f"test/path_{i}",
                ids_name="test_ids",
                path_name=f"path_{i}",
                units="m",
                data_type="float",
                coordinates=(),
                physics_domain="test",
                physics_phenomena=(),
            )
            doc = Document(
                metadata=metadata,
                documentation=f"Test documentation {i}",
                relationships={},
                raw_data={},
            )
            docs.append(doc)
        return docs

    def test_initialization_empty_documents(self, mock_document_store):
        """Embeddings initializes with empty document store."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            assert embeddings._embeddings is not None
            assert embeddings._embeddings.shape == (0, 0)
            assert embeddings._path_ids == []

    def test_initialization_with_documents(self, mock_document_store, mock_documents):
        """Embeddings initializes with documents."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_config = MagicMock()
            mock_config.model_name = "test-model"
            mock_config.generate_cache_key.return_value = None

            mock_encoder = MagicMock()
            mock_encoder.config = mock_config
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                [
                    "test/path_0",
                    "test/path_1",
                    "test/path_2",
                    "test/path_3",
                    "test/path_4",
                ],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            assert embeddings._embeddings is not None
            assert embeddings._embeddings.shape == (5, 384)
            assert len(embeddings._path_ids) == 5

    def test_initialization_deferred_build(self, mock_document_store):
        """Embeddings can defer loading embeddings."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings._embeddings is None
            mock_encoder.build_document_embeddings.assert_not_called()

    def test_cache_filename_default(self, mock_document_store):
        """cache_filename generates correct default filename."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("all-MiniLM-L6-v2")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="all-MiniLM-L6-v2",
            )

            filename = embeddings.cache_filename()

            assert filename.endswith(".pkl")
            assert "MiniLM" in filename or "minilm" in filename.lower()

    def test_cache_filename_with_ids_set(self, mock_document_store):
        """cache_filename includes hash for filtered dataset."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                ids_set={"equilibrium", "core_profiles"},
                load_embeddings=False,
                model_name="test-model",
            )

            filename = embeddings.cache_filename()

            assert filename.endswith(".pkl")
            assert "_" in filename  # Contains hash

    def test_cache_path(self, mock_document_store, tmp_path):
        """cache_path returns correct path."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            with patch(
                "imas_mcp.embeddings.embeddings.ResourcePathAccessor"
            ) as mock_accessor:
                mock_accessor_instance = MagicMock()
                mock_accessor_instance.embeddings_dir = tmp_path
                mock_accessor.return_value = mock_accessor_instance

                embeddings = Embeddings(
                    document_store=mock_document_store,
                    load_embeddings=False,
                    model_name="test-model",
                )

                path = embeddings.cache_path()

                assert path.parent == tmp_path

    def test_cache_exists_false(self, mock_document_store, tmp_path):
        """cache_exists returns False when cache doesn't exist."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            with patch(
                "imas_mcp.embeddings.embeddings.ResourcePathAccessor"
            ) as mock_accessor:
                mock_accessor_instance = MagicMock()
                mock_accessor_instance.embeddings_dir = tmp_path
                mock_accessor.return_value = mock_accessor_instance

                embeddings = Embeddings(
                    document_store=mock_document_store,
                    load_embeddings=False,
                    model_name="test-model",
                )

                assert embeddings.cache_exists() is False

    def test_cache_exists_true(self, mock_document_store, tmp_path):
        """cache_exists returns True when cache exists."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            with patch(
                "imas_mcp.embeddings.embeddings.ResourcePathAccessor"
            ) as mock_accessor:
                mock_accessor_instance = MagicMock()
                mock_accessor_instance.embeddings_dir = tmp_path
                mock_accessor.return_value = mock_accessor_instance

                embeddings = Embeddings(
                    document_store=mock_document_store,
                    load_embeddings=False,
                    model_name="test-model",
                )

                # Create the cache file
                cache_path = embeddings.cache_path()
                cache_path.write_bytes(b"cache data")

                assert embeddings.cache_exists() is True

    def test_effective_status_always_ready(self, mock_document_store):
        """effective_status always returns ready."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings.effective_status == "ready"

    def test_get_embeddings_matrix(self, mock_document_store, mock_documents):
        """get_embeddings_matrix returns the embeddings array."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                ["path_0", "path_1", "path_2", "path_3", "path_4"],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            matrix = embeddings.get_embeddings_matrix()

            assert matrix.shape == (5, 384)

    def test_get_embeddings_matrix_lazy_load(self, mock_document_store, mock_documents):
        """get_embeddings_matrix triggers lazy load when needed."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                ["path_0", "path_1", "path_2", "path_3", "path_4"],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,  # Defer loading
                model_name="test-model",
            )

            # Should not have loaded yet
            assert embeddings._embeddings is None

            # Access triggers load
            matrix = embeddings.get_embeddings_matrix()

            assert matrix.shape == (5, 384)

    def test_get_path_ids_empty(self, mock_document_store):
        """get_path_ids returns empty list when not loaded."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings.get_path_ids() == []

    def test_get_path_ids_loaded(self, mock_document_store, mock_documents):
        """get_path_ids returns path IDs after loading."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                [
                    "test/path_0",
                    "test/path_1",
                    "test/path_2",
                    "test/path_3",
                    "test/path_4",
                ],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            path_ids = embeddings.get_path_ids()

            assert len(path_ids) == 5
            assert "test/path_0" in path_ids

    def test_encode_texts(self, mock_document_store):
        """encode_texts delegates to encoder."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.embed_texts.return_value = np.zeros((2, 384))
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            result = embeddings.encode_texts(["text1", "text2"])

            mock_encoder.embed_texts.assert_called_once_with(["text1", "text2"])
            assert result.shape == (2, 384)

    def test_encoder_config_property(self, mock_document_store):
        """encoder_config property returns encoder's config."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            test_config = EncoderConfig(model_name="test-model", batch_size=100)
            mock_encoder = MagicMock()
            mock_encoder.config = test_config
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            config = embeddings.encoder_config

            assert config == test_config
            assert config.batch_size == 100

    def test_is_built_false(self, mock_document_store):
        """is_built returns False when embeddings not loaded."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings.is_built is False

    def test_is_built_true(self, mock_document_store, mock_documents):
        """is_built returns True when embeddings are loaded."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                ["path_0", "path_1", "path_2", "path_3", "path_4"],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            assert embeddings.is_built is True

    def test_is_built_empty_embeddings(self, mock_document_store):
        """is_built returns False for empty embeddings array."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=True,
                model_name="test-model",
            )

            # Empty embeddings
            assert embeddings.is_built is False

    def test_materialize_embeddings(self, mock_document_store, mock_documents):
        """materialize_embeddings triggers embedding load."""
        mock_document_store.get_all_documents.return_value = mock_documents

        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder.build_document_embeddings.return_value = (
                np.zeros((5, 384)),
                ["path_0", "path_1", "path_2", "path_3", "path_4"],
                False,
            )
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings._embeddings is None

            embeddings.materialize_embeddings()

            assert embeddings._embeddings is not None
            assert embeddings.is_built is True

    def test_ids_set_configuration(self, mock_document_store):
        """Embeddings stores IDS set configuration."""
        with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder.config = create_mock_encoder_config("test-model")
            mock_encoder_class.return_value = mock_encoder

            embeddings = Embeddings(
                document_store=mock_document_store,
                ids_set={"equilibrium", "core_profiles"},
                load_embeddings=False,
                model_name="test-model",
            )

            assert embeddings.ids_set == {"equilibrium", "core_profiles"}

    def test_model_name_set_from_config(self, mock_document_store):
        """model_name is set from encoder config after initialization."""
        with patch("imas_mcp.embeddings.embeddings.EncoderConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.model_name = "actual-model-from-env"
            mock_config_class.return_value = mock_config

            with patch("imas_mcp.embeddings.embeddings.Encoder") as mock_encoder_class:
                mock_encoder = MagicMock()
                mock_encoder.config = mock_config
                mock_encoder_class.return_value = mock_encoder

                embeddings = Embeddings(
                    document_store=mock_document_store,
                    load_embeddings=False,
                    model_name=None,  # Not specified, should be set from config
                )

                assert embeddings.model_name == "actual-model-from-env"
