#!/usr/bin/env python3
"""
Build the LexicographicSearch index for the Docker container.
This script initializes the LexicographicSearch class and builds the index.
"""

import logging
from imas_mcp_server.lexicographic_search import LexicographicSearch


def main():
    """Build the lexicographic search index."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize and build the index
    logger = logging.getLogger(__name__)
    logger.info("Initializing LexicographicSearch and building index...")

    # This will automatically build the index if it's empty
    search = LexicographicSearch()

    logger.info(f"Index built successfully with {len(search)} documents")
    logger.info(f"Index location: {search.dirname}")
    logger.info(f"Index name: {search.indexname}")


if __name__ == "__main__":
    main()
