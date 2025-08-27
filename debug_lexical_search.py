#!/usr/bin/env python3
"""
Debug script to test lexical search behavior with exact paths.
"""

import asyncio
import logging
from pathlib import Path

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.search_strategy import SearchConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_lexical_search_debug():
    """Test lexical search with various path queries."""
    # Initialize document store and engine
    data_dir = Path(__file__).parent / "imas_mcp" / "database" / "processed"
    document_store = DocumentStore(data_dir=data_dir, ids_set=None)
    lexical_engine = LexicalSearchEngine(document_store)

    # Test queries
    test_queries = [
        "equilibrium/time_slice/profiles_1d/elongation",
        "equilibrium/time_slice/profiles_1d/mass_density",
        "core_profiles/profiles_1d/electrons/temperature_validity",
        "equilibrium time_slice profiles_1d elongation",  # without slashes
        "equilibrium",  # just IDS name
        "elongation",  # just field name
    ]

    config = SearchConfig(max_results=5)

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Testing query: '{query}'")
        print(f"{'=' * 60}")

        try:
            # Test with lexical engine
            response = await lexical_engine.search(query, config)

            print(f"Found {len(response.hits)} results:")
            for i, hit in enumerate(response.hits):
                print(
                    f"  {i + 1}. {hit.document.metadata.path_id} (score: {hit.score:.3f})"
                )

            # Test direct FTS search
            print("\nDirect FTS search results:")
            fts_results = document_store.search_full_text(query, max_results=5)
            print(f"Found {len(fts_results)} FTS results:")
            for i, doc in enumerate(fts_results):
                print(f"  {i + 1}. {doc.metadata.path_id}")

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(test_lexical_search_debug())
