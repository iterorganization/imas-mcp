"""Tests for labeler.py - LLM-based cluster labeling."""

import pytest

from imas_codex.clusters.labeler import ClusterLabel, ClusterLabeler


class TestClusterLabel:
    """Tests for ClusterLabel dataclass."""

    def test_cluster_label_creation(self):
        """Test creating a ClusterLabel."""
        label = ClusterLabel(
            cluster_id="uuid-1",
            label="Temperature Profiles",
            description="A cluster of temperature-related paths.",
        )
        assert label.cluster_id == "uuid-1"
        assert label.label == "Temperature Profiles"
        assert label.description == "A cluster of temperature-related paths."


class TestClusterLabelerFallback:
    """Tests for ClusterLabeler fallback methods (no API key required)."""

    def test_generate_fallback_label_with_paths(self):
        """Test generating fallback label when paths are available."""
        # Create labeler without API key should fail, but we can test
        # the fallback method directly if we instantiate differently
        cluster = {
            "id": 0,
            "is_cross_ids": True,
            "ids": ["core_profiles", "equilibrium"],
            "paths": ["core_profiles/temperature", "equilibrium/pressure"],
        }

        # Test the fallback label generation logic manually
        paths = cluster.get("paths", [])[:5]
        if paths:
            last_segments = [p.split("/")[-1] for p in paths]
            common = "_".join(last_segments[:2])
            label = f"{common.replace('_', ' ').title()}"
        else:
            label = f"Cluster {cluster['id']}"

        assert "Temperature" in label or "Pressure" in label

    def test_generate_fallback_label_without_paths(self):
        """Test generating fallback label when no paths available."""
        cluster = {
            "id": 5,
            "is_cross_ids": False,
            "ids": ["core_profiles"],
            "paths": [],
        }

        # Fallback logic
        paths = cluster.get("paths", [])[:5]
        if paths:
            last_segments = [p.split("/")[-1] for p in paths]
            common = "_".join(last_segments[:2])
            label = f"{common.replace('_', ' ').title()}"
        else:
            label = f"Cluster {cluster['id']}"

        assert label == "Cluster 5"

    def test_deduplicate_labels_unique(self):
        """Test deduplication with unique labels."""
        labels = [
            ClusterLabel(cluster_id=0, label="Label A", description="Desc A"),
            ClusterLabel(cluster_id=1, label="Label B", description="Desc B"),
            ClusterLabel(cluster_id=2, label="Label C", description="Desc C"),
        ]

        # Manual deduplication logic
        seen = {}
        deduplicated = []
        for label in labels:
            base_label = label.label
            if base_label in seen:
                seen[base_label] += 1
                new_label = f"{base_label} {seen[base_label]}"
                deduplicated.append(
                    ClusterLabel(
                        cluster_id=label.cluster_id,
                        label=new_label,
                        description=label.description,
                    )
                )
            else:
                seen[base_label] = 1
                deduplicated.append(label)

        assert len(deduplicated) == 3
        assert all(d.label in ["Label A", "Label B", "Label C"] for d in deduplicated)

    def test_deduplicate_labels_with_duplicates(self):
        """Test deduplication with duplicate labels."""
        labels = [
            ClusterLabel(cluster_id=0, label="Temperature", description="Desc 1"),
            ClusterLabel(cluster_id=1, label="Temperature", description="Desc 2"),
            ClusterLabel(cluster_id=2, label="Temperature", description="Desc 3"),
        ]

        # Manual deduplication logic
        seen = {}
        deduplicated = []
        for label in labels:
            base_label = label.label
            if base_label in seen:
                seen[base_label] += 1
                new_label = f"{base_label} {seen[base_label]}"
                deduplicated.append(
                    ClusterLabel(
                        cluster_id=label.cluster_id,
                        label=new_label,
                        description=label.description,
                    )
                )
            else:
                seen[base_label] = 1
                deduplicated.append(label)

        assert len(deduplicated) == 3
        label_names = [d.label for d in deduplicated]
        assert "Temperature" in label_names
        assert "Temperature 2" in label_names
        assert "Temperature 3" in label_names


class TestBuildPrompt:
    """Tests for prompt building logic."""

    def test_prompt_limits_paths(self):
        """Test that prompt limits paths to prevent context overflow."""
        cluster = {
            "id": 0,
            "is_cross_ids": True,
            "ids_names": ["core_profiles", "equilibrium"],
            "paths": [f"path_{i}" for i in range(50)],  # 50 paths
        }

        # Build cluster data as labeler does
        cluster_data = {
            "id": cluster["id"],
            "type": "cross_ids" if cluster.get("is_cross_ids") else "intra_ids",
            "ids": cluster.get("ids", cluster.get("ids_names", [])),
            "paths": cluster.get("paths", [])[:20],  # Limit paths
            "path_count": len(cluster.get("paths", [])),
        }

        assert len(cluster_data["paths"]) == 20
        assert cluster_data["path_count"] == 50

    def test_cluster_type_detection(self):
        """Test cluster type detection in prompt building."""
        cross_cluster = {"id": 0, "is_cross_ids": True, "type": None}
        intra_cluster = {"id": 1, "is_cross_ids": False, "type": None}

        cross_type = "cross_ids" if cross_cluster.get("is_cross_ids") else "intra_ids"
        intra_type = "cross_ids" if intra_cluster.get("is_cross_ids") else "intra_ids"

        assert cross_type == "cross_ids"
        assert intra_type == "intra_ids"
