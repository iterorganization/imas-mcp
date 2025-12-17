"""Tests for clusters/extractor.py module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.clusters.config import RelationshipExtractionConfig
from imas_codex.clusters.extractor import RelationshipExtractor
from imas_codex.clusters.models import (
    ClusterInfo,
    ClusteringStatistics,
    CrossIDSSummary,
    IntraIDSSummary,
    PathMembership,
    RelationshipMetadata,
    RelationshipSet,
)
from imas_codex.embeddings.config import EncoderConfig


@pytest.fixture
def encoder_config():
    """Create a mock encoder configuration."""
    return EncoderConfig(model_name="test-model")


@pytest.fixture
def extraction_config(encoder_config, tmp_path):
    """Create an extraction config for testing."""
    return RelationshipExtractionConfig(
        encoder_config=encoder_config,
        input_dir=tmp_path / "schemas",
        output_file=tmp_path / "clusters" / "clusters.json",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def sample_ids_data(tmp_path):
    """Create sample IDS JSON data."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)

    # Create sample equilibrium data
    equilibrium_data = {
        "paths": {
            "time_slice/profiles_1d/psi": {
                "documentation": "Poloidal flux profile data for MHD equilibrium calculations",
                "units": "Wb",
                "data_type": "FLT_1D",
            },
            "time_slice/profiles_1d/q": {
                "documentation": "Safety factor profile data for stability analysis",
                "units": "",
                "data_type": "FLT_1D",
            },
        }
    }

    with open(schema_dir / "equilibrium.json", "w") as f:
        json.dump(equilibrium_data, f)

    # Create sample core_profiles data
    core_data = {
        "paths": {
            "profiles_1d/electrons/temperature": {
                "documentation": "Electron temperature profile data in the plasma core region",
                "units": "eV",
                "data_type": "FLT_1D",
            },
        }
    }

    with open(schema_dir / "core_profiles.json", "w") as f:
        json.dump(core_data, f)

    return schema_dir


class TestRelationshipExtractor:
    """Tests for the RelationshipExtractor class."""

    def test_init_default_config(self, encoder_config):
        """Extractor initializes with encoder config."""
        config = RelationshipExtractionConfig(encoder_config=encoder_config)
        extractor = RelationshipExtractor(config)

        assert extractor.config is not None
        assert extractor.path_filter is not None
        assert extractor.unit_builder is not None
        assert extractor.clusterer is not None
        assert extractor.relationship_builder is not None

    def test_init_use_rich(self, encoder_config):
        """Extractor respects use_rich config."""
        config = RelationshipExtractionConfig(
            encoder_config=encoder_config, use_rich=False
        )
        extractor = RelationshipExtractor(config)

        assert extractor._use_rich is False

    def test_load_ids_data(self, extraction_config, sample_ids_data):
        """_load_ids_data loads JSON files."""
        extraction_config.input_dir = sample_ids_data
        extractor = RelationshipExtractor(extraction_config)

        ids_data = extractor._load_ids_data(sample_ids_data)

        assert "equilibrium" in ids_data
        assert "core_profiles" in ids_data
        assert "paths" in ids_data["equilibrium"]

    def test_load_ids_data_with_ids_filter(self, extraction_config, sample_ids_data):
        """_load_ids_data filters by ids_set."""
        extraction_config.input_dir = sample_ids_data
        extraction_config.ids_set = {"equilibrium"}
        extractor = RelationshipExtractor(extraction_config)

        ids_data = extractor._load_ids_data(sample_ids_data)

        assert "equilibrium" in ids_data
        assert "core_profiles" not in ids_data

    def test_load_ids_data_handles_invalid_json(self, extraction_config, tmp_path):
        """_load_ids_data handles invalid JSON files."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid JSON file
        with open(schema_dir / "invalid.json", "w") as f:
            f.write("not valid json {{{")

        extraction_config.input_dir = schema_dir
        extractor = RelationshipExtractor(extraction_config)

        # Should not raise, just log warning
        ids_data = extractor._load_ids_data(schema_dir)
        assert "invalid" not in ids_data


class TestBuildSummaries:
    """Tests for summary building methods."""

    @pytest.fixture
    def extractor(self, extraction_config):
        """Create extractor instance."""
        return RelationshipExtractor(extraction_config)

    def test_build_cross_ids_summary_empty(self, extractor):
        """_build_cross_ids_summary handles empty clusters."""
        clusters = []
        summary = extractor._build_cross_ids_summary(clusters)

        assert isinstance(summary, CrossIDSSummary)
        assert summary.cluster_count == 0
        assert summary.avg_similarity == 0.0
        assert summary.total_paths == 0

    def test_build_cross_ids_summary_with_clusters(self, extractor):
        """_build_cross_ids_summary summarizes cross-IDS clusters."""
        clusters = [
            ClusterInfo(
                id=1,
                similarity_score=0.85,
                size=5,
                is_cross_ids=True,
                ids_names=["equilibrium", "core_profiles"],
                paths=[
                    "equilibrium/a",
                    "equilibrium/b",
                    "core_profiles/c",
                    "core_profiles/d",
                    "core_profiles/e",
                ],
            ),
            ClusterInfo(
                id=2,
                similarity_score=0.75,
                size=3,
                is_cross_ids=True,
                ids_names=["mhd", "stability"],
                paths=["mhd/x", "mhd/y", "stability/z"],
            ),
            ClusterInfo(
                id=3,
                similarity_score=0.90,
                size=2,
                is_cross_ids=False,  # Not cross-IDS
                ids_names=["equilibrium"],
                paths=["equilibrium/m", "equilibrium/n"],
            ),
        ]

        summary = extractor._build_cross_ids_summary(clusters)

        assert summary.cluster_count == 2  # Only cross-IDS
        assert summary.cluster_index == [1, 2]
        assert summary.avg_similarity == pytest.approx(0.8)  # (0.85 + 0.75) / 2
        assert summary.total_paths == 8  # 5 + 3

    def test_build_intra_ids_summary_empty(self, extractor):
        """_build_intra_ids_summary handles empty clusters."""
        clusters = []
        summary = extractor._build_intra_ids_summary(clusters)

        assert isinstance(summary, IntraIDSSummary)
        assert summary.cluster_count == 0
        assert summary.avg_similarity == 0.0
        assert summary.total_paths == 0
        assert summary.by_ids == {}

    def test_build_intra_ids_summary_with_clusters(self, extractor):
        """_build_intra_ids_summary summarizes intra-IDS clusters."""
        clusters = [
            ClusterInfo(
                id=1,
                similarity_score=0.90,
                size=3,
                is_cross_ids=False,
                ids_names=["equilibrium"],
                paths=["equilibrium/a", "equilibrium/b", "equilibrium/c"],
            ),
            ClusterInfo(
                id=2,
                similarity_score=0.85,
                size=2,
                is_cross_ids=False,
                ids_names=["equilibrium"],
                paths=["equilibrium/x", "equilibrium/y"],
            ),
            ClusterInfo(
                id=3,
                similarity_score=0.80,
                size=2,
                is_cross_ids=False,
                ids_names=["core_profiles"],
                paths=["core_profiles/m", "core_profiles/n"],
            ),
            ClusterInfo(
                id=4,
                similarity_score=0.95,
                size=5,
                is_cross_ids=True,  # Cross-IDS, should be excluded
                ids_names=["eq", "cp"],
                paths=["eq/a", "eq/b", "cp/c", "cp/d", "cp/e"],
            ),
        ]

        summary = extractor._build_intra_ids_summary(clusters)

        assert summary.cluster_count == 3  # Only intra-IDS
        assert 1 in summary.cluster_index
        assert 2 in summary.cluster_index
        assert 3 in summary.cluster_index
        assert 4 not in summary.cluster_index  # Cross-IDS excluded
        assert "equilibrium" in summary.by_ids
        assert "core_profiles" in summary.by_ids
        assert summary.by_ids["equilibrium"]["path_count"] == 5  # 3 + 2


class TestFallbackLabels:
    """Tests for fallback label generation."""

    @pytest.fixture
    def extractor(self, extraction_config):
        """Create extractor instance."""
        return RelationshipExtractor(extraction_config)

    def test_generate_fallback_labels(self, extractor):
        """_generate_fallback_labels creates labels from paths."""
        clusters = [
            ClusterInfo(
                id=1,
                similarity_score=0.85,
                size=3,
                is_cross_ids=True,
                ids_names=["equilibrium", "core_profiles"],
                paths=[
                    "equilibrium/temperature",
                    "equilibrium/density",
                    "core_profiles/pressure",
                ],
            ),
        ]

        labels = extractor._generate_fallback_labels(clusters)

        assert 1 in labels
        assert "label" in labels[1]
        assert "description" in labels[1]
        assert "cross-IDS" in labels[1]["description"]

    def test_generate_fallback_labels_with_single_path(self, extractor):
        """_generate_fallback_labels handles minimal paths (size must be >= 1)."""
        clusters = [
            ClusterInfo(
                id=1,
                similarity_score=0.85,
                size=1,
                is_cross_ids=False,
                ids_names=["equilibrium"],
                paths=["equilibrium/single_path"],
            ),
        ]

        labels = extractor._generate_fallback_labels(clusters)

        assert 1 in labels
        assert "label" in labels[1]
        # Single path should generate label from that path
        assert labels[1]["label"] != ""

    def test_generate_fallback_labels_from_dicts(self, extractor):
        """_generate_fallback_labels_from_dicts creates labels from dicts."""
        cluster_dicts = [
            {
                "id": 1,
                "is_cross_ids": True,
                "ids_names": ["equilibrium", "transport"],
                "paths": ["equilibrium/flux", "transport/conductivity"],
            },
            {
                "id": 2,
                "is_cross_ids": False,
                "ids_names": ["mhd"],
                "paths": [],
            },
        ]

        labels = extractor._generate_fallback_labels_from_dicts(cluster_dicts)

        assert 1 in labels
        assert 2 in labels
        assert "label" in labels[1]
        assert "description" in labels[1]
        assert "cross-IDS" in labels[1]["description"]
        assert labels[2]["label"] == "Cluster 2"  # Empty paths


class TestLabelGeneration:
    """Tests for label generation with cache."""

    @pytest.fixture
    def extractor(self, extraction_config):
        """Create extractor instance."""
        return RelationshipExtractor(extraction_config)

    @patch("os.getenv")
    def test_generate_cluster_labels_no_api_key(self, mock_getenv, extractor):
        """Falls back when no API key available."""
        mock_getenv.return_value = None

        clusters = [
            {"id": 1, "is_cross_ids": False, "ids_names": ["eq"], "paths": ["eq/path"]}
        ]

        labels = extractor._generate_cluster_labels_for_batch(clusters)

        assert 1 in labels
        assert "label" in labels[1]

    @patch("os.getenv")
    def test_generate_cluster_labels_placeholder_key(self, mock_getenv, extractor):
        """Falls back when placeholder API key."""
        mock_getenv.return_value = "your_api_key_here"

        clusters = [
            {"id": 1, "is_cross_ids": False, "ids_names": ["eq"], "paths": ["eq/path"]}
        ]

        labels = extractor._generate_cluster_labels_for_batch(clusters)

        assert 1 in labels

    def test_get_labeling_model(self, extractor):
        """_get_labeling_model returns model name."""
        model = extractor._get_labeling_model()
        assert isinstance(model, str)


class TestSaveEmbeddingsNpz:
    """Tests for saving embeddings to NPZ format."""

    @pytest.fixture
    def extractor(self, extraction_config):
        """Create extractor instance."""
        return RelationshipExtractor(extraction_config)

    def test_save_embeddings_npz_basic(self, extractor, tmp_path):
        """_save_embeddings_npz saves and returns hash."""
        embeddings_file = tmp_path / "embeddings.npz"

        centroid_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        centroid_ids = [1, 2]
        label_embeddings = [np.array([0.7, 0.8, 0.9])]
        label_ids = [1]

        file_hash = extractor._save_embeddings_npz(
            embeddings_file,
            centroid_embeddings,
            centroid_ids,
            label_embeddings,
            label_ids,
        )

        assert embeddings_file.exists()
        assert isinstance(file_hash, str)
        assert len(file_hash) == 16

        # Verify contents
        data = np.load(embeddings_file)
        assert "centroids" in data
        assert "centroid_cluster_ids" in data
        assert "label_embeddings" in data
        assert "label_cluster_ids" in data
        assert data["centroids"].shape == (2, 3)
        assert data["label_embeddings"].shape == (1, 3)

    def test_save_embeddings_npz_empty(self, extractor, tmp_path):
        """_save_embeddings_npz handles empty data."""
        embeddings_file = tmp_path / "empty.npz"

        file_hash = extractor._save_embeddings_npz(embeddings_file, [], [], [], [])

        assert embeddings_file.exists()
        assert isinstance(file_hash, str)


class TestGenerateLabelEmbeddings:
    """Tests for generating label embeddings."""

    @pytest.fixture
    def extractor(self, extraction_config):
        """Create extractor instance."""
        return RelationshipExtractor(extraction_config)

    def test_generate_label_embeddings_empty(self, extractor):
        """_generate_label_embeddings handles empty input."""
        result = extractor._generate_label_embeddings({})
        assert result == {}

    @patch.object(RelationshipExtractor, "_generate_label_embeddings")
    def test_generate_label_embeddings_with_data(self, mock_gen, extractor):
        """_generate_label_embeddings generates embeddings for labels."""
        mock_gen.return_value = {1: [0.1, 0.2, 0.3]}

        labels_map = {1: {"label": "Test Label", "description": "Test description"}}
        result = extractor._generate_label_embeddings(labels_map)

        assert isinstance(result, dict)
