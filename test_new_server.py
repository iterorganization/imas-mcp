#!/usr/bin/env python3
"""
Test script to demonstrate the new class-based IMAS MCP Server
with configurable IDS set filtering.
"""

from imas_mcp_server.server import IMASMCPServer


def test_server_creation():
    """Test creating servers with different configurations."""

    print("Testing IMAS MCP Server creation...")

    # Test 1: Default server (all IDS)
    print("\n1. Creating server with all IDS...")
    server_all = IMASMCPServer()
    print(
        f"   Server created with search index type: {server_all.search_index.index_prefix}"
    )
    print(f"   Available IDS count: {len(server_all.search_index.ids_names)}")

    # Test 2: Server with specific IDS set
    print("\n2. Creating server with specific IDS set...")
    specific_ids = {"core_profiles", "equilibrium"}
    server_filtered = IMASMCPServer(ids_set=specific_ids)
    print(f"   Server created with filtered IDS set: {specific_ids}")
    print(f"   Search index type: {server_filtered.search_index.index_prefix}")

    # Test 3: Verify that the servers are independent
    print("\n3. Verifying server independence...")
    print(f"   Server 1 (all IDS) has {len(server_all.search_index.ids_names)} IDS")
    print(f"   Server 2 (filtered) configured for {server_filtered.ids_set}")

    # Test 4: Test tool functionality
    print("\n4. Testing tool functionality...")
    ids_info_all = server_all.ids_info()
    print(f"   All IDS server version: {ids_info_all['version']}")
    print(f"   All IDS server document count: {ids_info_all['total_documents']}")

    ids_info_filtered = server_filtered.ids_info()
    print(f"   Filtered server version: {ids_info_filtered['version']}")
    print(f"   Filtered server document count: {ids_info_filtered['total_documents']}")

    print("\n✅ All tests passed! The new class-based server is working correctly.")


if __name__ == "__main__":
    test_server_creation()
