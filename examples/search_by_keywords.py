#!/usr/bin/env python
"""
Example script demonstrating how to use the keyword search functionality
of the IMAS PathIndex to find paths using natural language descriptions.
"""

import logging
from imas_mcp_server.path_index_cache import PathIndexCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating keyword search capabilities."""  # Initialize the path index cache
    path_index_cache = PathIndexCache()
    path_index = path_index_cache.path_index

    keyword_count = len(getattr(path_index, "keyword_index", {}))
    print(
        f"Loaded path index with {len(path_index.paths)} paths and {keyword_count} keywords"
    )

    # Example searches
    example_queries = [
        "plasma current measurement",
        "electron temperature profile",
        "magnetic flux surfaces",
        "ion density",
        "safety factor q profile",
    ]

    for query in example_queries:
        print(f"\n\n*** Searching for: '{query}' ***\n")
        results = path_index.search_by_keywords(query, limit=5)

        if not results:
            print(f"No results found for '{query}'")
            continue

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            path = result["path"]
            score = result["score"]
            doc = result["doc"]

            # Truncate documentation for display
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            doc_preview = doc_preview.replace("\n", " ")

            print(f"{i}. Path: {path}")
            print(f"   Score: {score}")
            print(f"   Documentation: {doc_preview}")

    # Interactive search
    print("\n\n*** Interactive Search ***")
    while True:
        query = input("\nEnter search query (or 'q' to quit): ")
        if query.lower() in ("q", "quit", "exit"):
            break

        results = path_index.search_by_keywords(query, limit=10)

        if not results:
            print(f"No results found for '{query}'")
            continue

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            path = result["path"]
            score = result["score"]
            doc = result["doc"]

            # Truncate documentation for display
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            doc_preview = doc_preview.replace("\n", " ")

            print(f"{i}. Path: {path}")
            print(f"   Score: {score}")
            print(f"   Documentation: {doc_preview}")


if __name__ == "__main__":
    main()
