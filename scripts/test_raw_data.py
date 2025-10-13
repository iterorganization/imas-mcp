"""Test if raw_data is preserved in Document objects."""

from imas_mcp.search.document_store import DocumentStore

# Create document store
print("Initializing DocumentStore...")
store = DocumentStore()

# Load a specific IDS
print("\nLoading equilibrium IDS...")
docs = store.get_documents_by_ids("equilibrium")
print(f"Loaded {len(docs)} documents")

# Check first few documents
print("\nChecking raw_data content for first 5 documents:\n")
for i, doc in enumerate(docs[:5]):
    print(f"{i + 1}. {doc.metadata.path_name}")
    print(f"   raw_data keys: {list(doc.raw_data.keys())}")

    # Check specific fields
    print(f"   lifecycle_status: {doc.raw_data.get('lifecycle_status')}")
    print(f"   validation_rules: {doc.raw_data.get('validation_rules')}")
    print(f"   related_paths: {doc.raw_data.get('related_paths')}")
    print(f"   usage_examples: {doc.raw_data.get('usage_examples')}")
    print()

# Now test to_datapath conversion
print("\nTesting to_datapath() conversion:")
print("-" * 80)

test_doc = docs[10]  # Pick a random doc
print(f"Document: {test_doc.metadata.path_name}\n")

print("Before conversion - raw_data fields:")
print(f"  validation_rules: {test_doc.raw_data.get('validation_rules')}")
print(f"  lifecycle_status: {test_doc.raw_data.get('lifecycle_status')}")

idsnode = test_doc.to_datapath()
print("\nAfter conversion - IdsNode fields:")
print(f"  validation_rules: {idsnode.validation_rules}")
print(f"  lifecycle_status: {idsnode.lifecycle_status}")
print(f"  related_paths: {idsnode.related_paths}")
print(f"  usage_examples: {idsnode.usage_examples}")

print("\n✓ Test complete!")
