#!/usr/bin/env python3
"""
Get the exact index name for the LexicographicSearch class.
This script prints the exact index name that should be copied.
"""

import sys
from imas_mcp_server.lexicographic_search import LexicographicSearch


def main():
    """Get the exact index name."""
    try:
        # Initialize the search class (without building)
        search = LexicographicSearch(build=False)
        print(search.indexname)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
