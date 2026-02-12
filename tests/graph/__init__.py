"""Graph quality and compliance tests.

These tests require a live Neo4j connection and are excluded from
default test runs. Run explicitly with:

    uv run pytest tests/graph/ -m graph -v

Connection resolves via:
    1. Environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    2. pyproject.toml [tool.imas-codex.graph] section
    3. Built-in defaults (bolt://localhost:7687, neo4j, imas-codex)
"""
