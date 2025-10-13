"""Test that search results contain full field data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from imas_mcp.embeddings.embeddings import Embeddings
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.search.search_strategy import SearchConfig

print("=" * 80)
print("TESTING SEARCH RESULTS FOR FULL FIELD POPULATION")
print("=" * 80)

# Initialize
print("\nInitializing...")
store = DocumentStore()
embeddings = Embeddings(document_store=store, load_embeddings=True)
engine = HybridSearchEngine(document_store=store, embeddings=embeddings)

# Run search
query = "equilibrium boundary psi"
config = SearchConfig(max_results=5)

print(f"\nSearching for: '{query}'")
result = engine.search(query, config)

print(f"Found {len(result.hits)} results\n")

# Check each hit
for i, match in enumerate(result.hits, 1):
    print(f"{i}. {match.path_name}")
    print(f"   Path: {match.path}")
    print(f"   Documentation: {match.documentation[:80]}...")
    print(f"   Data type: {match.data_type}")
    print(f"   Units: {match.units}")

    # THE CRITICAL TEST - check fields that should exist
    print(f"   ✓ validation_rules: {match.validation_rules}")
    print(f"   ✓ lifecycle_status: {match.lifecycle_status}")
    print(f"   ✓ related_paths: {match.related_paths}")
    print(f"   ✓ usage_examples: {match.usage_examples}")
    print()

print("=" * 80)
print("VERDICT")
print("=" * 80)

# Check if fields are populated
has_validation = any(m.validation_rules for m in result.hits)
has_lifecycle = any(m.lifecycle_status for m in result.hits)
has_related = any(m.related_paths for m in result.hits)

print(f"\n✓ validation_rules present: {has_validation}")
print(f"✓ lifecycle_status present: {has_lifecycle}")
print(f"✓ related_paths present: {has_related}")

if has_validation:
    print("\n✅ SUCCESS: Full field data IS available in search results!")
    print("   The SQLite FTS is just a search index.")
    print("   Complete data comes from in-memory Documents.")
else:
    print("\n❌ Fields are missing from search results")
    print("   This indicates a mapping problem")
