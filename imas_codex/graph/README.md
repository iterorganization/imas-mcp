# Knowledge Graph Module

This module provides Neo4j-based storage for facility knowledge discovered through exploration.

## Architecture: Schema-Driven Graph Ontology

### Single Source of Truth: LinkML

The **LinkML schema** (`imas_codex/schemas/facility.yaml`) is the authoritative source for:
- Class definitions → Neo4j **node labels**
- Slots with class ranges → Neo4j **relationships**
- Enumerations → Constrained property values
- Identifier/required fields → Database constraints

### Module Structure

```
graph/
├── __init__.py      # Public API exports
├── client.py        # Neo4j client for CRUD operations
├── schema.py        # ✅ Schema-driven graph ontology
└── models.py        # ✅ Auto-generated Pydantic models
```

### Generated from Schema

From the LinkML schema, we auto-generate:
- **Pydantic models** (`models.py`) via `gen-pydantic`
- **Node labels** derived at runtime from class names
- **Relationship types** derived from slot names with class ranges
- **Constraints** derived from identifier fields

## Usage

### GraphSchema - Schema Introspection

```python
from imas_codex.graph import GraphSchema, get_schema

# Get schema singleton (lazy-loaded)
schema = get_schema()

# Get all node labels (non-abstract classes)
print(schema.node_labels)
# ['Facility', 'MDSplusServer', 'MDSplusTree', 'TreeNode', ...]

# Get all relationships (slots with class ranges)
for rel in schema.relationships:
    print(f"{rel.from_class} -[:{rel.cypher_type}]-> {rel.to_class}")
# MDSplusServer -[:FACILITY_ID]-> Facility
# MDSplusTree -[:FACILITY_ID]-> Facility
# ...

# Get identifier field for a class
print(schema.get_identifier("Facility"))  # 'id'
print(schema.get_identifier("MDSplusServer"))  # 'hostname'

# Generate constraint statements
for stmt in schema.constraint_statements():
    print(stmt)
# CREATE CONSTRAINT facility_id IF NOT EXISTS FOR (n:Facility) REQUIRE n.id IS UNIQUE
# CREATE CONSTRAINT mdsplusserver_hostname IF NOT EXISTS FOR (n:MDSplusServer) REQUIRE n.hostname IS UNIQUE
```

### GraphClient - Neo4j Operations

```python
from imas_codex.graph import GraphClient

with GraphClient() as client:
    # Initialize schema (creates constraints and indexes)
    client.initialize_schema()
    
    # Create nodes using string labels (derived from schema)
    client.create_node("Facility", "epfl", {"name": "EPFL/TCV", "ssh_host": "epfl"})
    
    # Create relationships using SCREAMING_SNAKE_CASE types
    client.create_relationship(
        "MDSplusServer", "tcv.epfl.ch",
        "Facility", "epfl",
        "FACILITY_ID",
        from_id_field="hostname"
    )
    
    # High-level methods
    client.create_facility("iter", name="ITER", machine="ITER")
    client.create_tool("epfl", name="git", available=True, category="vcs")
    
    # Query
    facilities = client.get_facilities()
    tools = client.get_tools("epfl")
```

### Utility Functions

```python
from imas_codex.graph import to_cypher_props, merge_node_query, merge_relationship_query

# Convert Pydantic model to Neo4j-safe dict
props = to_cypher_props(my_model, exclude={"internal_field"})

# Generate Cypher MERGE queries
node_query = merge_node_query("Facility", id_field="id")
# "MERGE (n:Facility {id: $id}) SET n += $props"

rel_query = merge_relationship_query("Tool", "Facility", "FACILITY_ID")
# "MATCH (a:Tool {id: $from_id}), (b:Facility {id: $to_id}) MERGE (a)-[r:FACILITY_ID]->(b)"
```

## Schema Patterns for Neo4j

### Defining Relationships in LinkML

Slots with class ranges automatically become relationships:

```yaml
classes:
  MDSplusServer:
    attributes:
      facility_id:            # Creates FACILITY_ID relationship
        range: Facility       # Target node type
        required: true
```

At runtime, `GraphSchema` detects this and exposes:
```python
Relationship(from_class="MDSplusServer", slot_name="facility_id", to_class="Facility")
```

### Identifier Fields → Unique Constraints

Fields marked `identifier: true` automatically get unique constraints:

```yaml
classes:
  Facility:
    attributes:
      id:
        identifier: true    # → UNIQUE constraint on Facility.id
        required: true
```

### Indexes

Common query patterns get indexes via `schema.index_statements()`:
- `Facility.ssh_host`
- `Tool.category`, `Tool.available`
- `Diagnostic.category`

## Regenerating Models

When the LinkML schema changes:

```bash
uv run build-models --force
```

This regenerates `models.py` from `schemas/facility.yaml`.

## Testing

```bash
# Start Neo4j
docker-compose up neo4j -d

# Run graph tests
uv run pytest tests/graph/ -v
```

## References

- [LinkML Schema](../schemas/facility.yaml) - Source of truth
- [linkml-store docs](https://linkml.io/linkml-store/) - Alternative abstraction
- [LinkML Runtime SchemaView](https://linkml.io/linkml/developers/schemaview.html)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
