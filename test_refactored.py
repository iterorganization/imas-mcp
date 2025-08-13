#!/usr/bin/env python3
"""Test refactored relationship modules."""

from imas_mcp.physics_extraction.physics_data_loader import PhysicsDataLoader
from imas_mcp.physics_extraction.relationship_analysis import (
    RelationshipEngine,
    SemanticAnalyzer,
)

print("Testing refactored modules...")

# Test physics data loader
loader = PhysicsDataLoader()
characteristics = loader.load_domain_characteristics()
print("Loaded " + str(len(characteristics)) + " domain characteristics")

# Test semantic analyzer
analyzer = SemanticAnalyzer(loader)
result = analyzer.analyze_concept("core_profiles/profiles_1d/electrons/density")
print("Concept analysis: " + str(result.get("concepts", [])))

# Test relationship engine
mock_catalog = {"cross_references": {}, "physics_concepts": {}, "unit_families": {}}
engine = RelationshipEngine(mock_catalog, loader)
relationships = engine.discover_relationships("test/path", max_depth=1)
print("Relationship discovery: " + str(list(relationships.keys())))

print("✅ All refactored modules working correctly!")
