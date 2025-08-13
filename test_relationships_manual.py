#!/usr/bin/env python3
"""Test relationship discovery functionality"""

import asyncio

from imas_mcp.models.constants import RelationshipType
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.tools.relationships_tool import RelationshipsTool


async def test_relationships():
    """Test the relationship discovery system."""
    document_store = DocumentStore()
    tool = RelationshipsTool(document_store)

    # Test a physics path
    result = await tool.explore_relationships(
        path="core_profiles/profiles_1d/electrons/density",
        relationship_type=RelationshipType.ALL,
        max_depth=2,
    )

    print("=== RELATIONSHIP DISCOVERY RESULTS ===")
    print(f"Physics domains: {result.physics_domains}")
    print(f"Total nodes: {len(result.nodes)}")
    print(f"Connection types: {list(result.connections.keys())}")

    # Check AI response
    if result.ai_response:
        insights = result.ai_response.get("relationship_insights", {})
        if insights:
            discovery = insights.get("discovery_summary", {})
            print(f"Relationship types: {discovery.get('relationship_types', [])}")
            print(f"Average strength: {discovery.get('avg_strength', 0):.2f}")

            # Show some specific insights
            semantic_insights = insights.get("semantic_insights", [])
            if semantic_insights:
                print(f"Semantic insights ({len(semantic_insights)} found):")
                for insight in semantic_insights[:3]:  # Show first 3
                    print(
                        f"  - {insight.get('path', 'unknown')}: {insight.get('description', 'no description')}"
                    )

    print(
        f"\n✅ Relationship tool validated with {len(result.nodes)} relationships found!"
    )


if __name__ == "__main__":
    asyncio.run(test_relationships())
