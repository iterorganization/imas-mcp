#!/usr/bin/env python3
"""
Debug single parameter test to see the exact logging output.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from imas_mcp.relationships.config import RelationshipExtractionConfig
from imas_mcp.relationships.extractor import RelationshipExtractor

# Enable all logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")


def test_single_params():
    """Test a single parameter combination with debug output."""
    print("=== TESTING SINGLE PARAMETER COMBINATION ===")

    # Test just one set to see detailed breakdown
    print("\n--- Testing intra_eps=0.02 vs 0.03 ---")

    for intra_eps in [0.02, 0.03]:
        print(f"\n=== TESTING intra_eps={intra_eps} ===")

        config = RelationshipExtractionConfig(
            cross_ids_eps=0.04,
            cross_ids_min_samples=2,
            intra_ids_eps=intra_eps,
            intra_ids_min_samples=2,
            use_rich=False,
        )

        print(f"Config created with intra_ids_eps={config.intra_ids_eps}")

        try:
            extractor = RelationshipExtractor(config)
            relationships = extractor.extract_relationships()

            # Detailed analysis
            cross_clusters = [c for c in relationships.clusters if c.is_cross_ids]
            intra_clusters = [c for c in relationships.clusters if not c.is_cross_ids]

            print("\nRESULTS:")
            print(f"  Total clusters: {len(relationships.clusters)}")
            print(f"  Cross-IDS clusters: {len(cross_clusters)}")
            print(f"  Intra-IDS clusters: {len(intra_clusters)}")

            # Check cluster IDs
            if intra_clusters:
                print(f"  Intra cluster IDs: {[c.id for c in intra_clusters[:5]]}")
                print(f"  Intra cluster sizes: {[c.size for c in intra_clusters[:5]]}")
                print(
                    f"  Intra cluster IDS names: {[c.ids_names for c in intra_clusters[:5]]}"
                )

            # Check statistics
            stats = relationships.metadata.statistics
            print("\nSTATISTICS:")
            print(f"  Cross-IDS stats: {stats.cross_ids_clustering}")
            print(f"  Intra-IDS stats: {stats.intra_ids_clustering}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_single_params()
