#!/usr/bin/env python3
"""
Optimize clustering parameters for cross-IDS and intra-IDS clustering.
"""

import json
import logging
import sys
from pathlib import Path

# Add numpy for Latin Hypercube sampling
import numpy as np

# =============================================================================
# OPTIMAL CLUSTERING PARAMETERS (Found via Latin Hypercube Optimization)
# =============================================================================
# Final optimal parameters discovered through systematic optimization:
# - cross_eps = 0.0751  (cross-IDS clustering epsilon)
# - cross_min = 2       (cross-IDS minimum samples)
# - intra_eps = 0.0319  (intra-IDS clustering epsilon)
# - intra_min = 2       (intra-IDS minimum samples)
#
# Optimization History:
# - Initial Grid: cross_eps=0.050, intra_eps=0.030 → Score: 3041.73
# - LHC Round 1: cross_eps=0.0698, intra_eps=0.0311 → Score: 5173.85 (+70%)
# - LHC Round 2: cross_eps=0.0751, intra_eps=0.0319 → Score: 5436.17 (+79%)
#
# Best Result Metrics:
# - 95 cross-IDS clusters (0.925 similarity, 17.8 avg size)
# - 1792 intra-IDS clusters (0.980 similarity, 3.5 avg size)
# - 1100 multi-membership paths, 4687 isolated paths
# - Cross quality: 847.30, Intra quality: 2223.12
# =============================================================================

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imas_mcp.relationships.config import RelationshipExtractionConfig
from imas_mcp.relationships.extractor import RelationshipExtractor


def evaluate_clustering_quality(relationships):
    """Evaluate the quality of clustering results."""
    stats = relationships.metadata.statistics
    cross_stats = stats.cross_ids_clustering
    intra_stats = stats.intra_ids_clustering

    clusters = relationships.clusters
    cross_clusters = [c for c in clusters if c.is_cross_ids]
    intra_clusters = [c for c in clusters if not c.is_cross_ids]

    # Quality metrics - statistics are stored as dictionaries
    cross_cluster_count = cross_stats["total_clusters"]
    cross_avg_similarity = cross_stats["avg_similarity"]
    cross_paths_clustered = cross_stats["paths_in_clusters"]

    intra_cluster_count = intra_stats["total_clusters"]
    intra_avg_similarity = intra_stats["avg_similarity"]
    intra_paths_clustered = intra_stats["paths_in_clusters"]

    multi_membership = stats.multi_membership_paths
    isolated_paths = stats.isolated_paths

    # Calculate cluster size diversity for both cross and intra clusters
    if cross_clusters:
        cross_sizes = [c.size for c in cross_clusters]
        cross_avg_size = sum(cross_sizes) / len(cross_sizes)
    else:
        cross_avg_size = 0

    if intra_clusters:
        intra_sizes = [c.size for c in intra_clusters]
        intra_size_std = (
            sum((x - sum(intra_sizes) / len(intra_sizes)) ** 2 for x in intra_sizes)
            / len(intra_sizes)
        ) ** 0.5
        intra_avg_size = sum(intra_sizes) / len(intra_sizes)
    else:
        intra_size_std = 0
        intra_avg_size = 0

    # Enhanced composite quality score
    # Cross-IDS quality (most important - finding meaningful physics connections across IDS)
    cross_quality = (
        cross_cluster_count * 5.0  # Number of cross-IDS clusters (very important)
        + cross_avg_similarity * 20.0  # Quality of cross-IDS clusters (very important)
        + (cross_paths_clustered / 5)  # Paths successfully clustered
        + min(cross_avg_size, 8) * 2.0  # Prefer moderate-sized clusters (2-8 paths)
    )

    # Intra-IDS quality (secondary - organizing within IDS)
    intra_quality = (
        intra_cluster_count * 1.0  # Number of intra-IDS clusters
        + intra_avg_similarity * 8.0  # Quality of intra-IDS clusters
        + (intra_paths_clustered / 15)  # Paths successfully clustered
        + intra_size_std * 1.5  # Diversity in cluster sizes is good
        + min(intra_avg_size, 6) * 1.0  # Prefer moderate-sized clusters
    )

    # Bonuses and penalties
    multi_membership_bonus = (
        multi_membership * 3.0
    )  # Paths in both cross and intra clusters
    isolation_penalty = isolated_paths * 0.2  # Small penalty for isolated paths

    # Balance penalty - prefer balanced clustering (not all cross or all intra)
    total_clustered = cross_paths_clustered + intra_paths_clustered
    if total_clustered > 0:
        cross_ratio = cross_paths_clustered / total_clustered
        # Prefer 20-60% cross-IDS, 40-80% intra-IDS
        balance_score = 1.0 - abs(cross_ratio - 0.4) * 2.0  # Optimal at 40% cross
        balance_bonus = max(0, balance_score) * 5.0
    else:
        balance_bonus = 0

    total_score = (
        cross_quality
        + intra_quality
        + multi_membership_bonus
        + balance_bonus
        - isolation_penalty
    )

    return {
        "total_score": total_score,
        "cross_quality": cross_quality,
        "intra_quality": intra_quality,
        "cross_clusters": cross_cluster_count,
        "intra_clusters": intra_cluster_count,
        "cross_similarity": cross_avg_similarity,
        "intra_similarity": intra_avg_similarity,
        "cross_avg_size": cross_avg_size,
        "intra_avg_size": intra_avg_size,
        "multi_membership": multi_membership,
        "isolated": isolated_paths,
        "cross_paths": cross_paths_clustered,
        "intra_paths": intra_paths_clustered,
        "balance_bonus": balance_bonus,
    }


def test_parameters(
    cross_eps,
    cross_min,
    intra_eps,
    intra_min,
    ids_filter=None,
):
    """Test a specific parameter combination."""
    config = RelationshipExtractionConfig(
        cross_ids_eps=cross_eps,
        cross_ids_min_samples=cross_min,
        intra_ids_eps=intra_eps,
        intra_ids_min_samples=intra_min,
        ids_set={name.strip() for name in ids_filter.split()} if ids_filter else None,
        use_rich=False,  # Disable rich output for cleaner optimization
    )

    # DEBUG: Print the config values to verify they're set correctly
    print(
        f"  DEBUG: Config - cross_eps={config.cross_ids_eps}, intra_eps={config.intra_ids_eps}"
    )

    # Set up logging to reduce noise but show key debug messages
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        extractor = RelationshipExtractor(config)
        relationships = extractor.extract_relationships()
        return evaluate_clustering_quality(relationships)
    except Exception as e:
        print(
            f"Error with params ({cross_eps}, {cross_min}, {intra_eps}, {intra_min}): {e}"
        )
        return None


def optimize_parameters():
    """Run focused parameter optimization using Latin Hypercube sampling."""
    print("=== LATIN HYPERCUBE CLUSTERING OPTIMIZATION (ROUND 3) ===")
    print("Previous best: cross_eps=0.0751, intra_eps=0.0319 (score=5436.17)")
    print("Strategy: Fine-tune around optimal region for final validation")
    print()

    # Define parameter ranges based on previous results
    # Current optimal: cross_eps=0.0751, intra_eps=0.0319 (score=5436.17)
    # Fine-tuning around the optimal region
    cross_eps_min, cross_eps_max = 0.070, 0.080  # Narrow range around optimal
    intra_eps_min, intra_eps_max = 0.028, 0.035  # Narrow range around optimal

    # Fixed parameters
    cross_min = 2
    intra_min = 2

    # Number of samples (trials)
    n_samples = 4

    # Generate Latin Hypercube samples
    np.random.seed(42)  # For reproducibility

    # Generate LHS samples in [0,1] space
    lhs_samples = np.random.rand(n_samples, 2)

    # Apply stratification (Latin Hypercube property)
    for i in range(2):  # For each parameter dimension
        # Sort the random values
        sorted_indices = np.argsort(lhs_samples[:, i])
        # Assign stratified values
        for j, idx in enumerate(sorted_indices):
            lhs_samples[idx, i] = (j + np.random.rand()) / n_samples

    # Scale to actual parameter ranges
    cross_eps_samples = cross_eps_min + lhs_samples[:, 0] * (
        cross_eps_max - cross_eps_min
    )
    intra_eps_samples = intra_eps_min + lhs_samples[:, 1] * (
        intra_eps_max - intra_eps_min
    )

    print("Parameter ranges:")
    print(f"  cross_eps: [{cross_eps_min:.3f}, {cross_eps_max:.3f}]")
    print(f"  intra_eps: [{intra_eps_min:.3f}, {intra_eps_max:.3f}]")
    print(f"\nLatin Hypercube samples ({n_samples} trials):")
    for i in range(n_samples):
        print(
            f"  Trial {i + 1}: cross_eps={cross_eps_samples[i]:.4f}, intra_eps={intra_eps_samples[i]:.4f}"
        )
    print()

    best_score = -float("inf")
    best_params = None
    best_results = None
    all_results = []

    for i in range(n_samples):
        cross_eps = round(cross_eps_samples[i], 4)
        intra_eps = round(intra_eps_samples[i], 4)

        print(
            f"Testing {i + 1}/{n_samples}: cross_eps={cross_eps}, intra_eps={intra_eps}"
        )

        result = test_parameters(cross_eps, cross_min, intra_eps, intra_min)
        if result:
            result["params"] = {
                "cross_eps": cross_eps,
                "cross_min": cross_min,
                "intra_eps": intra_eps,
                "intra_min": intra_min,
            }
            all_results.append(result)

            if result["total_score"] > best_score:
                best_score = result["total_score"]
                best_params = result["params"]
                best_results = result
                print(f"  *** NEW BEST SCORE: {best_score:.2f} ***")

            print(
                f"  Score: {result['total_score']:.2f} "
                f"(Cross: {result['cross_clusters']} clusters, {result['cross_similarity']:.3f} sim, {result['cross_avg_size']:.1f} avg | "
                f"Intra: {result['intra_clusters']} clusters, {result['intra_similarity']:.3f} sim, {result['intra_avg_size']:.1f} avg | "
                f"Multi: {result['multi_membership']}, Isolated: {result['isolated']}, Balance: {result['balance_bonus']:.1f})"
            )

            # DEBUG: Show detailed breakdown
            print(
                f"    DEBUG: Cross quality={result['cross_quality']:.2f}, Intra quality={result['intra_quality']:.2f}"
            )
            print(
                f"    DEBUG: Cross paths={result['cross_paths']}, Intra paths={result['intra_paths']}"
            )

    print()
    print("=== LATIN HYPERCUBE OPTIMIZATION RESULTS ===")
    if best_params and best_results:
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.2f}")
        print()
        print("Best results breakdown:")
        for key, value in best_results.items():
            if key != "params":
                print(f"  {key}: {value}")
    else:
        print("No valid results found!")

    # Show all results ranked
    print()
    print(f"=== ALL {len(all_results)} LATIN HYPERCUBE SAMPLES RANKED ===")
    sorted_results = sorted(all_results, key=lambda x: x["total_score"], reverse=True)
    for i, result in enumerate(sorted_results):
        params = result["params"]
        print(f"{i + 1}. Score: {result['total_score']:.2f}")
        print(
            f"   Params: cross_eps={params['cross_eps']:.4f}, intra_eps={params['intra_eps']:.4f}"
        )
        print(
            f"   Cross: {result['cross_clusters']} clusters ({result['cross_similarity']:.3f} sim, {result['cross_paths']} paths, {result['cross_avg_size']:.1f} avg size)"
        )
        print(
            f"   Intra: {result['intra_clusters']} clusters ({result['intra_similarity']:.3f} sim, {result['intra_paths']} paths, {result['intra_avg_size']:.1f} avg size)"
        )
        print(
            f"   Multi-membership: {result['multi_membership']}, Isolated: {result['isolated']}, Balance bonus: {result['balance_bonus']:.1f}"
        )
        print()

    return best_params, sorted_results


if __name__ == "__main__":
    best_params, all_results = optimize_parameters()

    # Save results to file
    results_file = Path("clustering_optimization_results.json")
    with open(results_file, "w") as f:
        json.dump({"best_params": best_params, "all_results": all_results}, f, indent=2)

    print(f"Results saved to {results_file}")
