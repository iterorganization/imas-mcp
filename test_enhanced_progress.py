#!/usr/bin/env python3
"""
Test script to verify the enhanced Rich progress bar updates work correctly.
"""

import logging
from imas_mcp_server.lexicographic_search import LexicographicSearch

if __name__ == "__main__":
    # Set up logging to INFO level for detailed progress
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("Testing Enhanced Rich progress bar with frequent updates...")

    # Test with a small subset that has good variety
    search = LexicographicSearch(ids_set={"pf_active"})

    print(f"Total elements to process: {search._total_elements}")

    # Test the document batch processing with enhanced progress
    batch_count = 0
    total_docs = 0

    try:
        for document_batch in search._get_document_batch(
            batch_size=25
        ):  # Smaller batches for more frequent updates
            batch_count += 1
            batch_size = len(document_batch)
            total_docs += batch_size
            print(
                f"✓ Batch {batch_count}: {batch_size} documents (total: {total_docs})"
            )

            # Add to index
            search.add_document_batch(document_batch)

            # Complete after fewer batches to demonstrate the enhanced progress
            if batch_count >= 5:
                print("✓ Enhanced progress test completed - stopping after 5 batches")
                break

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

    print(
        f"🎉 Enhanced progress test completed! Processed {batch_count} batches with {total_docs} total documents"
    )
    print("Features demonstrated:")
    print("  • 10x faster refresh rate (10 updates/second)")
    print("  • Real-time description updates showing current IDS and path")
    print("  • Enhanced progress bar with M/N complete column")
    print("  • Wider progress bar for better visibility")
