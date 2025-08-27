#!/usr/bin/env python3
"""
Debug path filtering to understand why certain paths are missing from embeddings.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from imas_mcp.relationships.config import RelationshipExtractionConfig
from imas_mcp.relationships.preprocessing import PathFilter
from imas_mcp.search.document_store import DocumentStore


def debug_path_filtering():
    """Debug path filtering differences."""
    print("=== DEBUGGING PATH FILTERING ===")

    # Load dataset_description.json directly
    schema_file = Path("imas_mcp/resources/schemas/detailed/dataset_description.json")
    with open(schema_file) as f:
        dataset_data = json.load(f)

    print(f"\nDataset description has {len(dataset_data['paths'])} total paths")

    # Get simulation-related paths
    simulation_paths = {
        path: data
        for path, data in dataset_data["paths"].items()
        if "simulation" in path
    }
    print(f"Found {len(simulation_paths)} simulation-related paths:")
    for path in simulation_paths:
        print(f"  - {path}")

    # Test PathFilter
    config = RelationshipExtractionConfig()
    path_filter = PathFilter(config)

    # Create mock IDS data structure
    ids_data = {"dataset_description": dataset_data}

    print("\n=== PATHFILTER ANALYSIS ===")
    print(f"Skip patterns: {config.skip_patterns}")
    print(f"Min documentation length: {config.min_documentation_length}")
    print(f"Generic docs to skip: {config.generic_docs}")

    # Filter paths
    filtered_paths = path_filter.filter_meaningful_paths(ids_data)

    print(f"\nPathFilter included {len(filtered_paths)} total paths")

    # Check simulation paths specifically
    simulation_filtered = {
        path: data for path, data in filtered_paths.items() if "simulation" in path
    }
    print(f"PathFilter included {len(simulation_filtered)} simulation paths:")
    for path in simulation_filtered:
        print(f"  + {path}")

    # Check what was excluded
    excluded_simulation = set(simulation_paths.keys()) - set(simulation_filtered.keys())
    print(f"\nPathFilter excluded {len(excluded_simulation)} simulation paths:")
    for path in excluded_simulation:
        print(f"  - {path}")
        # Check why it was excluded
        path_data = simulation_paths[path]
        reason = []

        # Check skip patterns
        for pattern_str in config.skip_patterns:
            import re

            pattern = re.compile(pattern_str)
            if pattern.match(path):
                reason.append(f"matches skip pattern: {pattern_str}")

        # Check documentation length
        doc = path_data.get("documentation", "")
        if len(doc.strip()) < config.min_documentation_length:
            reason.append(
                f"documentation too short ({len(doc.strip())} < {config.min_documentation_length})"
            )

        # Check generic docs
        if doc.strip() in config.generic_docs:
            reason.append("generic documentation")

        if reason:
            print(f"    Reason: {', '.join(reason)}")
        else:
            print("    Reason: unknown")

    # Test DocumentStore
    print("\n=== DOCUMENTSTORE ANALYSIS ===")
    document_store = DocumentStore(ids_set={"dataset_description"})
    all_docs = document_store.get_all_documents()

    print(f"DocumentStore included {len(all_docs)} total documents")

    # Check simulation paths in DocumentStore
    simulation_docs = [doc for doc in all_docs if "simulation" in doc.metadata.path_id]
    print(f"DocumentStore included {len(simulation_docs)} simulation documents:")
    for doc in simulation_docs:
        print(f"  + {doc.metadata.path_id}")

    # Find the discrepancy
    doc_paths = {doc.metadata.path_id for doc in all_docs}
    filtered_paths_set = set(filtered_paths.keys())

    in_docs_not_filtered = doc_paths - filtered_paths_set
    in_filtered_not_docs = filtered_paths_set - doc_paths

    print("\n=== DISCREPANCY ANALYSIS ===")
    print(f"Paths in DocumentStore but not in PathFilter: {len(in_docs_not_filtered)}")
    for path in sorted(in_docs_not_filtered):
        if "simulation" in path:
            print(f"  ! {path}")

    print(f"Paths in PathFilter but not in DocumentStore: {len(in_filtered_not_docs)}")
    for path in sorted(in_filtered_not_docs):
        if "simulation" in path:
            print(f"  ! {path}")


if __name__ == "__main__":
    debug_path_filtering()
