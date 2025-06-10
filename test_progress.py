#!/usr/bin/env python3
"""
Test script to verify the Rich progress bar works correctly and doesn't hang.
"""

import logging
from imas_mcp_server.lexicographic_search import LexicographicSearch

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("Testing Rich progress bar with small dataset...")

    # Test with a small subset first
    search = LexicographicSearch(ids_set={"pf_active"})

    print(f"Total elements to process: {search._total_elements}")

    # Test the document batch processing
    batch_count = 0
    total_docs = 0

    try:
        for document_batch in search._get_document_batch(batch_size=50):
            batch_count += 1
            batch_size = len(document_batch)
            total_docs += batch_size
            print(
                f"Processed batch {batch_count}: {batch_size} documents (total: {total_docs})"
            )

            # Add the batch to the index
            search.add_document_batch(document_batch)

            # Break after a few batches for testing
            if batch_count >= 3:
                print("Test completed successfully - stopping after 3 batches")
                break

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

    print(
        f"Test completed! Processed {batch_count} batches with {total_docs} total documents"
    )
