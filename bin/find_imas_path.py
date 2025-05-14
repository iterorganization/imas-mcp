#!/usr/bin/env python
"""
Command-line tool for finding IMAS data dictionary paths using natural language queries.
"""

import argparse
import sys
from imas_mcp_server.path_index_cache import PathIndexCache


def main():
    parser = argparse.ArgumentParser(
        description="Find IMAS DD paths using natural language descriptions"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Natural language description of what you're looking for",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=10,
        help="Maximum number of results to show (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full documentation for each result",
    )
    args = parser.parse_args()

    # Combine all parts of the query
    query = " ".join(args.query)

    # Get the path index
    try:
        print("Loading IMAS data dictionary index...")
        path_index_cache = PathIndexCache()
        path_index = path_index_cache.path_index
    except Exception as e:
        print(f"Error loading IMAS data dictionary: {e}", file=sys.stderr)
        return 1

    # Search for paths
    print(f"Searching for: {query}")
    results = path_index.search_by_keywords(query, limit=args.num_results)

    # Display results
    if not results:
        print("No matching paths found")
        return 1

    print(f"Found {len(results)} matching paths:")
    for i, result in enumerate(results, 1):
        path = result["path"]
        score = result["score"]
        print(f"{i}. {path} (relevance: {score})")

        if args.verbose:
            doc = result["doc"].replace("\n", "\n   ")
            print(f"   Documentation: {doc}")
            print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
