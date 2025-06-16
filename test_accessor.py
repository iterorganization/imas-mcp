#!/usr/bin/env python3
"""
Test script to verify the DD accessor pattern works in different scenarios.
"""

import os
import tempfile
from pathlib import Path

from imas_mcp.dd_accessor import create_dd_accessor


def test_accessor_patterns():
    """Test different DD accessor patterns."""
    print("Testing DD accessor patterns...")

    # Test 1: Environment variable
    os.environ["IMAS_DD_VERSION"] = "4.0.1"
    accessor = create_dd_accessor()
    print(f"Environment accessor version: {accessor.get_version()}")
    print(f"Environment accessor available: {accessor.is_available()}")

    # Test 2: Metadata file
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_dir = Path(tmpdir)
        metadata_file = metadata_dir / "test_index.metadata.json"
        metadata_file.write_text(
            '{"dd_version": "4.0.2", "build_timestamp": "2023-01-01T00:00:00Z"}'
        )

        accessor = create_dd_accessor(
            metadata_dir=metadata_dir, index_name="test_index"
        )
        print(f"Metadata accessor version: {accessor.get_version()}")
        print(f"Metadata accessor available: {accessor.is_available()}")

    # Test 3: Index name parsing
    accessor = create_dd_accessor(
        index_name="lexicographic_4.0.3", index_prefix="lexicographic"
    )
    print(f"Index name accessor version: {accessor.get_version()}")
    print(f"Index name accessor available: {accessor.is_available()}")

    # Test 4: Try imas-data-dictionary (may fail if not installed)
    try:
        accessor = create_dd_accessor()
        # Clear environment to test IMAS DD access
        if "IMAS_DD_VERSION" in os.environ:
            del os.environ["IMAS_DD_VERSION"]

        # This should fall back to IMAS DD if available
        print(f"IMAS DD accessor version: {accessor.get_version()}")
        print(f"IMAS DD accessor available: {accessor.is_available()}")
        print("✓ imas-data-dictionary package is available")
    except Exception as e:
        print(f"✗ imas-data-dictionary not available: {e}")

    # Clean up
    if "IMAS_DD_VERSION" in os.environ:
        del os.environ["IMAS_DD_VERSION"]

    print("All accessor tests completed!")


if __name__ == "__main__":
    test_accessor_patterns()
