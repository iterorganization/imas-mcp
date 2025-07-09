#!/usr/bin/env python3
"""
Test script for the Physics Context Module

This script demonstrates the capabilities of the physics context module
and its integration with IMAS MCP tools.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imas_mcp.physics_context import (
    get_physics_engine,
    concept_to_imas_paths,
    concept_to_units,
    concept_to_symbol,
    search_physics_concepts,
    get_quantities_by_units,
    get_quantities_by_domain,
    PhysicsDomain,
)

from imas_mcp.physics_integration import (
    get_physics_integration,
    physics_enhanced_search,
    explain_physics_concept,
    get_concept_imas_mapping,
)


def test_basic_physics_context():
    """Test basic physics context functionality."""
    print("=" * 60)
    print("TESTING BASIC PHYSICS CONTEXT MODULE")
    print("=" * 60)

    engine = get_physics_engine()

    # Test 1: Poloidal flux mapping
    print("\n1. Testing poloidal flux concept mapping:")
    print("   Query: 'poloidal flux'")
    paths = concept_to_imas_paths("poloidal flux")
    units = concept_to_units("poloidal flux")
    symbol = concept_to_symbol("poloidal flux")

    print(f"   Symbol: {symbol}")
    print(f"   Units: {units}")
    print(f"   IMAS paths ({len(paths)} total):")
    for i, path in enumerate(paths[:5]):  # Show first 5
        print(f"     {i + 1}. {path}")
    if len(paths) > 5:
        print(f"     ... and {len(paths) - 5} more")

    # Test 2: Temperature concepts
    print("\n2. Testing temperature concept search:")
    print("   Query: 'temperature'")
    temp_results = search_physics_concepts("temperature")
    print(f"   Found {len(temp_results)} matches:")
    for concept, quantity in temp_results[:3]:
        print(f"     - {concept}: {quantity.symbol} [{quantity.units}]")

    # Test 3: Units-based search
    print("\n3. Testing units-based search:")
    print("   Query: units = 'eV'")
    ev_quantities = get_quantities_by_units("eV")
    print(f"   Found {len(ev_quantities)} quantities with eV units:")
    for q in ev_quantities:
        print(f"     - {q.concept} ({q.symbol})")

    # Test 4: Domain-based search
    print("\n4. Testing domain-based search:")
    print("   Query: domain = 'EQUILIBRIUM'")
    eq_quantities = get_quantities_by_domain(PhysicsDomain.EQUILIBRIUM)
    print(f"   Found {len(eq_quantities)} equilibrium quantities:")
    for q in eq_quantities:
        print(f"     - {q.concept}: {q.symbol} [{q.units}]")

    # Test 5: Summary statistics
    print("\n5. Physics context summary:")
    stats = engine.get_summary_stats()
    print(f"   Total quantities: {stats['total_quantities']}")
    print(f"   Total concepts: {stats['total_concepts']}")
    print(f"   Unique units: {stats['unique_units']}")
    print(f"   Physics domains: {stats['physics_domains']}")
    print("   Quantities by domain:")
    for domain, count in stats["quantities_by_domain"].items():
        print(f"     - {domain}: {count}")


def test_physics_integration():
    """Test physics integration functionality."""
    print("\n" + "=" * 60)
    print("TESTING PHYSICS INTEGRATION MODULE")
    print("=" * 60)

    integration = get_physics_integration()

    # Test 1: Enhanced search
    print("\n1. Testing enhanced physics search:")
    print("   Query: 'electron temperature'")
    search_result = physics_enhanced_search("electron temperature")

    if search_result.get("physics_matches"):
        match = search_result["physics_matches"][0]
        print(f"   Best match: {match['concept']}")
        print(f"   Symbol: {match['symbol']}")
        print(f"   Units: {match['units']}")
        print(f"   Domain: {match['domain']}")
        print(f"   Relevance: {match['relevance_score']:.2f}")
        print(f"   IMAS paths: {len(match['imas_paths'])} shown")
        for path in match["imas_paths"]:
            print(f"     - {path}")

    print(f"   Concept suggestions: {search_result.get('concept_suggestions', [])}")

    # Test 2: Concept explanation
    print("\n2. Testing concept explanation:")
    print("   Query: 'safety factor' (intermediate level)")
    explanation = explain_physics_concept("safety factor", "intermediate")

    if "error" not in explanation:
        q = explanation["quantity"]
        print(f"   Concept: {explanation['concept']}")
        print(f"   Symbol: {q['symbol']}")
        print(f"   Units: {q['units']}")
        print(f"   Domain: {q['domain']}")
        print(f"   Description: {q['description']}")
        print(f"   IMAS paths: {explanation['imas_integration']['total_paths']} total")
        print(
            f"   Coordinate type: {explanation['imas_integration']['coordinate_type']}"
        )
        print(f"   Alternative names: {explanation['alternative_names']}")

        if explanation.get("physics_context"):
            ctx = explanation["physics_context"]
            print(f"   Physics significance: {ctx.get('significance', 'N/A')}")
            print(f"   Measurement context: {ctx.get('measurement_context', 'N/A')}")
    else:
        print(f"   Error: {explanation['error']}")

    # Test 3: Concept to IMAS mapping
    print("\n3. Testing concept to IMAS mapping:")
    print("   Query: 'plasma current'")
    mapping = get_concept_imas_mapping("plasma current")

    if "error" not in mapping:
        print(f"   Concept: {mapping['concept']}")
        print(f"   Symbol: {mapping['symbol']}")
        print(f"   Units: {mapping['units']}")
        print(f"   Domain: {mapping['domain']}")
        print(f"   IMAS paths ({len(mapping['imas_paths'])} total):")
        for path in mapping["imas_paths"]:
            print(f"     - {path}")

        if mapping.get("usage_examples"):
            print("   Usage examples:")
            for example in mapping["usage_examples"][:2]:
                print(f"     {example['scenario']}: {example['code']}")
    else:
        print(f"   Error: {mapping['error']}")

    # Test 4: Domain overview
    print("\n4. Testing domain overview:")
    print("   Query: domain = 'equilibrium'")
    domain_result = integration.get_domain_overview(PhysicsDomain.EQUILIBRIUM)

    print(f"   Domain: {domain_result['domain']}")
    print(f"   Description: {domain_result['description']}")
    print(f"   Quantity count: {domain_result['quantity_count']}")
    print(f"   Total IMAS paths: {domain_result['imas_coverage']['total_paths']}")
    print(f"   Unique IDS: {domain_result['imas_coverage']['unique_ids']}")
    print(f"   Common units: {domain_result['common_units']}")
    print("   Key quantities:")
    for q in domain_result["key_quantities"][:3]:
        print(
            f"     - {q['concept']}: {q['symbol']} [{q['units']}] ({q['path_count']} paths)"
        )

    # Test 5: Query validation
    print("\n5. Testing query validation:")
    queries_to_test = ["psi", "temperature", "xyz123", "magnetic flux"]

    for query in queries_to_test:
        validation = integration.validate_physics_query(query)
        print(f"   Query: '{query}'")
        print(f"     Valid concept: {validation['is_valid_physics_concept']}")

        if validation.get("best_match"):
            match = validation["best_match"]
            print(
                f"     Best match: {match['concept']} (confidence: {match['confidence']})"
            )

        if validation.get("suggestions"):
            print(f"     Suggestions: {validation['suggestions'][:3]}")

        if validation.get("alternative_queries"):
            print(f"     Alternatives: {validation['alternative_queries']}")
        print()


def test_real_world_scenarios():
    """Test real-world usage scenarios."""
    print("\n" + "=" * 60)
    print("TESTING REAL-WORLD SCENARIOS")
    print("=" * 60)

    # Scenario 1: User wants to find temperature data
    print("\n1. Scenario: User wants electron temperature profile data")
    print("   User query: 'How do I get electron temperature profiles?'")

    # Search for electron temperature
    result = physics_enhanced_search("electron temperature")
    if result["physics_matches"]:
        match = result["physics_matches"][0]
        print(f"   → Found concept: {match['concept']}")
        print(f"   → Symbol: {match['symbol']}")
        print(f"   → Units: {match['units']}")
        print("   → IMAS access:")
        for path in match["imas_paths"][:2]:
            print(f"     ids.{path}")

    # Scenario 2: User has symbol, wants IMAS path
    print("\n2. Scenario: User knows symbol 'psi', wants IMAS paths")
    print("   User query: 'Where is psi stored in IMAS?'")

    # Search for psi
    psi_result = physics_enhanced_search("psi")
    if psi_result["physics_matches"]:
        match = psi_result["physics_matches"][0]
        print(f"   → psi = {match['concept']}")
        print(f"   → Units: {match['units']}")
        print("   → Primary IMAS locations:")
        for path in match["imas_paths"][:3]:
            print(f"     {path}")

    # Scenario 3: User wants to understand physics concept
    print("\n3. Scenario: User wants to understand 'beta'")
    print("   User query: 'What is beta in fusion physics?'")

    # Search for beta concepts
    beta_result = physics_enhanced_search("beta")
    if beta_result["physics_matches"]:
        print("   → Found beta-related concepts:")
        for match in beta_result["physics_matches"][:3]:
            print(f"     {match['concept']}: {match['symbol']} [{match['units']}]")

    # Get detailed explanation
    beta_explanation = explain_physics_concept("toroidal beta")
    if "error" not in beta_explanation:
        ctx = beta_explanation.get("physics_context", {})
        print(f"   → Physics significance: {ctx.get('significance', 'N/A')}")

    # Scenario 4: User wants all quantities with specific units
    print("\n4. Scenario: User wants all pressure-related quantities")
    print("   User query: 'What quantities have pressure units (Pa)?'")

    pa_quantities = get_quantities_by_units("Pa")
    print(f"   → Found {len(pa_quantities)} quantities with Pa units:")
    for q in pa_quantities:
        paths_count = len(q.imas_paths)
        print(f"     {q.concept} ({q.symbol}) - {paths_count} IMAS paths")


def main():
    """Run all tests."""
    print("IMAS PHYSICS CONTEXT MODULE - COMPREHENSIVE TEST")
    print("=" * 60)
    print("This test demonstrates the physics context module capabilities:")
    print("- Mapping physics concepts to IMAS attributes")
    print("- Symbol and unit lookup")
    print("- Enhanced search with physics awareness")
    print("- Concept explanations with IMAS integration")
    print("- Real-world usage scenarios")

    try:
        # Run all test modules
        test_basic_physics_context()
        test_physics_integration()
        test_real_world_scenarios()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey capabilities demonstrated:")
        print("✓ Physics concept to IMAS path mapping")
        print("✓ Symbol and unit lookup (e.g., 'psi' → poloidal flux, Wb)")
        print("✓ Enhanced search with physics context")
        print("✓ Comprehensive concept explanations")
        print("✓ Domain-based quantity organization")
        print("✓ Query validation and suggestions")
        print("✓ Real-world usage scenarios")

        print("\nExample usage in IMAS MCP tools:")
        print("- concept_to_imas_paths('poloidal flux') → IMAS paths for psi")
        print("- concept_to_symbol('electron temperature') → 'Te'")
        print("- concept_to_units('plasma current') → 'A'")
        print("- search_physics_concepts('temperature') → all temp concepts")

    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
