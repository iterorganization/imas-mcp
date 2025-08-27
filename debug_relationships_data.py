#!/usr/bin/env python3
"""
Debug script to analyze relationships.json clustering data.
Specifically examines how density and temperature paths are categorized.
"""

import json
from pathlib import Path
from typing import Any


def load_relationships_data() -> dict[str, Any]:
    """Load the relationships.json file."""
    relationships_file = Path("imas_mcp/resources/schemas/relationships.json")
    with open(relationships_file) as f:
        return json.load(f)


def find_clusters_containing_path(
    data: dict[str, Any], target_path: str
) -> list[dict[str, Any]]:
    """Find all clusters that contain a specific path."""
    matching_clusters = []

    for cluster in data.get("clusters", []):
        if target_path in cluster.get("paths", []):
            matching_clusters.append(cluster)

    return matching_clusters


def analyze_density_temperature_clustering(data: dict[str, Any]):
    """Analyze how density and temperature paths are clustered."""

    print("=" * 80)
    print("RELATIONSHIPS.JSON CLUSTERING ANALYSIS")
    print("=" * 80)

    # Metadata
    metadata = data.get("metadata", {})
    print(f"Generation timestamp: {metadata.get('generation_timestamp')}")
    print(f"Total paths processed: {metadata.get('total_paths_processed')}")
    print()

    # Clustering statistics
    stats = metadata.get("statistics", {})
    cross_ids_stats = stats.get("cross_ids_clustering", {})
    intra_ids_stats = stats.get("intra_ids_clustering", {})

    print("CLUSTERING STATISTICS:")
    print(f"  Cross-IDS clusters: {cross_ids_stats.get('total_clusters')}")
    print(f"  Cross-IDS paths in clusters: {cross_ids_stats.get('paths_in_clusters')}")
    print(f"  Intra-IDS clusters: {intra_ids_stats.get('total_clusters')}")
    print(f"  Intra-IDS paths in clusters: {intra_ids_stats.get('paths_in_clusters')}")
    print()

    # Target paths to analyze
    target_paths = [
        "core_profiles/profiles_1d/electrons/density",
        "core_profiles/profiles_1d/electrons/density_thermal",
        "core_profiles/profiles_1d/electrons/density_fast",
        "core_profiles/profiles_1d/electrons/temperature",
        "core_profiles/profiles_1d/electrons/temperature_fit",
        "core_profiles/profiles_1d/electrons/temperature_validity",
    ]

    print("ANALYZING TARGET PATHS:")
    print("-" * 40)

    for path in target_paths:
        print(f"\nPath: {path}")
        clusters = find_clusters_containing_path(data, path)

        if not clusters:
            print("  ❌ Not found in any cluster")
            continue

        for i, cluster in enumerate(clusters):
            cluster_type = "CROSS-IDS" if cluster.get("is_cross_ids") else "INTRA-IDS"
            print(f"  📊 Cluster {i + 1}: ID={cluster.get('id')}, Type={cluster_type}")
            print(f"      Size: {cluster.get('size')} paths")
            print(f"      Similarity: {cluster.get('similarity_score', 0):.3f}")
            print(f"      IDS names: {cluster.get('ids_names', [])}")

            # Show a few other paths in the same cluster for context
            other_paths = [p for p in cluster.get("paths", []) if p != path][:5]
            if other_paths:
                print(f"      Sample other paths: {other_paths}")


def analyze_cross_ids_vs_intra_ids_logic(data: dict[str, Any]):
    """Analyze the logic behind cross-IDS vs intra-IDS classification."""

    print("\n" + "=" * 80)
    print("CROSS-IDS vs INTRA-IDS CLASSIFICATION ANALYSIS")
    print("=" * 80)

    cross_ids_clusters = []
    intra_ids_clusters = []

    for cluster in data.get("clusters", []):
        if cluster.get("is_cross_ids"):
            cross_ids_clusters.append(cluster)
        else:
            intra_ids_clusters.append(cluster)

    print(f"Total Cross-IDS clusters: {len(cross_ids_clusters)}")
    print(f"Total Intra-IDS clusters: {len(intra_ids_clusters)}")
    print()

    # Analyze a few cross-IDS clusters to see if they really span multiple IDS
    print("SAMPLE CROSS-IDS CLUSTERS:")
    print("-" * 30)

    for _i, cluster in enumerate(cross_ids_clusters[:3]):
        print(f"\nCluster {cluster.get('id')} (Cross-IDS):")
        print(f"  IDS names: {cluster.get('ids_names', [])}")
        print(f"  Size: {cluster.get('size')} paths")

        # Check if paths actually span multiple IDS
        paths = cluster.get("paths", [])[:10]  # First 10 paths
        unique_ids = set()
        for path in paths:
            if "/" in path:
                ids_name = path.split("/")[0]
                unique_ids.add(ids_name)

        print(f"  Actual IDS names from paths: {sorted(unique_ids)}")
        print(f"  Sample paths: {paths}")

        # Check if this is truly cross-IDS
        if len(unique_ids) <= 1:
            print(
                f"  ⚠️  WARNING: This 'cross-IDS' cluster only contains paths from {unique_ids}"
            )


def find_problematic_clusters():
    """Find clusters that are misclassified."""

    data = load_relationships_data()

    print("\n" + "=" * 80)
    print("FINDING MISCLASSIFIED CLUSTERS")
    print("=" * 80)

    misclassified = []

    for cluster in data.get("clusters", []):
        paths = cluster.get("paths", [])
        is_cross_ids = cluster.get("is_cross_ids", False)

        # Determine actual IDS span
        unique_ids = set()
        for path in paths:
            if "/" in path:
                ids_name = path.split("/")[0]
                unique_ids.add(ids_name)

        actual_cross_ids = len(unique_ids) > 1

        if is_cross_ids != actual_cross_ids:
            misclassified.append(
                {
                    "cluster_id": cluster.get("id"),
                    "labeled_as": "cross-IDS" if is_cross_ids else "intra-IDS",
                    "actually_is": "cross-IDS" if actual_cross_ids else "intra-IDS",
                    "unique_ids": sorted(unique_ids),
                    "size": cluster.get("size"),
                    "sample_paths": paths[:5],
                }
            )

    print(f"Found {len(misclassified)} misclassified clusters:")

    for mc in misclassified[:10]:  # Show first 10
        print(f"\n  Cluster {mc['cluster_id']}:")
        print(f"    Labeled as: {mc['labeled_as']}")
        print(f"    Actually is: {mc['actually_is']}")
        print(f"    IDS names: {mc['unique_ids']}")
        print(f"    Size: {mc['size']} paths")
        print(f"    Sample paths: {mc['sample_paths']}")


if __name__ == "__main__":
    try:
        data = load_relationships_data()
        analyze_density_temperature_clustering(data)
        analyze_cross_ids_vs_intra_ids_logic(data)
        find_problematic_clusters()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
