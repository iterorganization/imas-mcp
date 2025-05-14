#!/usr/bin/env python
"""
Command-line tool for searching IMAS DD paths using natural language queries.
"""

import argparse
import logging
import sys
from imas_mcp_server.path_index_cache import PathIndexCache


def setup_logging(verbose=False):
    """Set up logging with appropriate verbosity."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")


def search_paths(query, limit=10, verbose=False):
    """Search for paths matching the given query."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Load the path index
    logger.debug("Loading path index...")
    path_index_cache = PathIndexCache()
    path_index = path_index_cache.path_index

    logger.debug(
        f"Loaded {len(path_index.paths)} paths and {len(path_index.keyword_index)} keywords"
    )

    # Search for paths
    results = path_index.search_by_keywords(query, limit=limit)

    return results


def main():
    """Main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(
        description="Search for IMAS DD paths using natural language."
    )
    parser.add_argument("query", nargs="+", help="Natural language query to search for")
    parser.add_argument(
        "-l", "--limit", type=int, default=10, help="Maximum number of results to show"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Combine all arguments into a single query string
    query = " ".join(args.query)

    results = search_paths(query, limit=args.limit, verbose=args.verbose)

    if not results:
        print(f"No paths found matching '{query}'")
        return 1

    print(f"Found {len(results)} paths matching '{query}':")
    for i, result in enumerate(results, 1):
        path = result["path"]
        score = result["score"]
        doc = result["doc"]

        # Truncate documentation for display
        doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
        doc_preview = doc_preview.replace("\n", " ")

        print(f"{i}. {path} (score: {score})")
        if args.verbose:
            print(f"   Documentation: {doc_preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
